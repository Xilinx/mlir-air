//===- AIRLoweringPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Conversion/AIRLoweringPass.h"
#include "air/Conversion/AIRPipeline.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

#define DEBUG_TYPE "air-lowering-pass"

using namespace mlir;
using namespace xilinx;

namespace {

class AIRLaunchConversion : public ConversionPattern {
public:
  explicit AIRLaunchConversion(MLIRContext *context)
      : ConversionPattern(air::LaunchOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    air::LaunchOp launch = cast<air::LaunchOp>(op);

    std::string launch_name("launch");
    if (auto attr =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      launch_name = attr.getValue().str();

    SmallVector<Value> lbs, ubs, steps;
    auto c0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);

    // make scf.parallel to replace air.launch
    for (auto d : launch.getSizeOperands()) {
      lbs.push_back(c0);
      ubs.push_back(d);
      steps.push_back(c1);
    }
    if (lbs.empty()) {
      lbs.push_back(c0);
      ubs.push_back(c1);
      steps.push_back(c1);
    }
    auto scfPar =
        rewriter.create<scf::ParallelOp>(op->getLoc(), lbs, ubs, steps);

    // map launch iteration space to scf.parallel ivs
    for (auto p : llvm::zip(launch.getIds(), scfPar.getInductionVars()))
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));

    // map launch size to scf.parallel upper bounds
    for (auto p : llvm::zip(launch.getSizeOperands(), scfPar.getUpperBound()))
      if (std::get<0>(p) != std::get<1>(p))
        std::get<0>(p).replaceAllUsesWith(std::get<1>(p));

    int i = 0;
    for (auto arg : launch.getKernelArguments())
      arg.replaceAllUsesWith(launch.getKernelOperand(i++));

    auto &body = launch.getBody().front().getOperations();
    scfPar.getBody()->getOperations().splice(scfPar.getBody()->begin(), body,
                                             body.begin(), --body.end());

    if (op->getNumResults()) {
      rewriter.setInsertionPoint(scfPar);
      SmallVector<Value> deps;
      for (auto &o : operands)
        if (o.getType().isa<airrt::EventType>())
          deps.push_back(o);
      rewriter.replaceOpWithNewOp<airrt::WaitAllOp>(
          op, airrt::EventType::get(op->getContext()), deps);
    } else
      rewriter.eraseOp(launch);
    return success();
  }
};

class AIRSegmentConversion : public ConversionPattern {
public:
  explicit AIRSegmentConversion(MLIRContext *context)
      : ConversionPattern(air::SegmentOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    air::SegmentOp segment = cast<air::SegmentOp>(op);
    if (auto attr =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      auto segment_name = attr.getValue().str();
      rewriter.create<airrt::SegmentLoadOp>(op->getLoc(), rewriter.getI64Type(),
                                            segment_name);
    }

    SmallVector<Value> deps;
    for (auto &o : operands)
      if (o.getType().isa<airrt::EventType>())
        deps.push_back(o);
    if (op->getNumResults()) {
      auto w = rewriter.create<airrt::WaitAllOp>(
          op->getLoc(), airrt::EventType::get(op->getContext()), deps);
      segment.getResult(0).replaceAllUsesWith(w.getResult(0));
    }

    SmallVector<Value> lbs, ubs, steps;
    auto c0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);

    // make scf.parallel to replace air.segment
    for (auto d : segment.getSizeOperands()) {
      lbs.push_back(c0);
      ubs.push_back(d);
      steps.push_back(c1);
    }
    if (lbs.empty()) {
      lbs.push_back(c0);
      ubs.push_back(c1);
      steps.push_back(c1);
    }
    auto scfPar =
        rewriter.create<scf::ParallelOp>(op->getLoc(), lbs, ubs, steps);

    // map segment iteration space to scf.parallel ivs
    for (auto p : llvm::zip(segment.getIds(), scfPar.getInductionVars()))
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));

    // map segment size to scf.parallel upper bounds
    for (auto p : llvm::zip(segment.getSizeOperands(), scfPar.getUpperBound()))
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));

    int i = 0;
    for (auto arg : segment.getKernelArguments())
      arg.replaceAllUsesWith(segment.getKernelOperand(i++));

    auto &body = segment.getBody().front().getOperations();
    scfPar.getBody()->getOperations().splice(scfPar.getBody()->begin(), body,
                                             body.begin(), --body.end());

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRHerdConversion : public ConversionPattern {
public:
  explicit AIRHerdConversion(MLIRContext *context)
      : ConversionPattern(air::HerdOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    air::HerdOp herd = cast<air::HerdOp>(op);

    auto herd_name_attr =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (!herd_name_attr) {
      emitError(op->getLoc(),
                "error lowering air.herd: herd name is undefined.\n");
      return failure();
    }

    rewriter.create<airrt::HerdLoadOp>(op->getLoc(), rewriter.getI64Type(),
                                       herd_name_attr.getValue().str());

    SmallVector<Value, 4> deps;
    for (auto &o : operands)
      if (o.getType().isa<airrt::EventType>())
        deps.push_back(o);
    if (op->getNumResults()) {
      auto w = rewriter.create<airrt::WaitAllOp>(
          op->getLoc(), airrt::EventType::get(op->getContext()), deps);
      herd.getResult(0).replaceAllUsesWith(w.getResult(0));
    }

    // If the herd doesn't contain a memcpy op, then it can be deleted
    SmallVector<Operation *> herdOps;
    herd.walk([&](air::MemcpyInterface op) { herdOps.push_back(op); });

    if (herdOps.size()) {
      auto herd_size = herd.getSizeOperands();
      int64_t herd_size_x = herd.getNumCols();
      int64_t herd_size_y = herd.getNumRows();

      auto outer = rewriter.create<AffineForOp>(herd.getLoc(), 0, herd_size_x);
      auto outer_builder = OpBuilder::atBlockBegin(outer.getBody());
      auto inner =
          outer_builder.create<AffineForOp>(herd.getLoc(), 0, herd_size_y);

      outer->setAttr("air.herd", StringAttr::get(op->getContext(), "outer"));
      inner->setAttr("air.herd", StringAttr::get(op->getContext(), "inner"));

      herd.getSize()[0].replaceAllUsesWith(herd_size[0]);
      herd.getSize()[1].replaceAllUsesWith(herd_size[1]);
      herd.getIds()[0].replaceAllUsesWith(outer.getInductionVar());
      herd.getIds()[1].replaceAllUsesWith(inner.getInductionVar());

      int i = 0;
      for (auto arg : herd.getKernelArguments())
        arg.replaceAllUsesWith(herd.getKernelOperand(i++));

      auto &body = herd.getBody().front().getOperations();
      inner.getBody()->getOperations().splice(inner.getBody()->begin(), body,
                                              body.begin(), --body.end());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRPipelineConversion : public ConversionPattern {
public:
  explicit AIRPipelineConversion(MLIRContext *context)
      : ConversionPattern(air::HerdPipelineOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto pipeOp = cast<air::HerdPipelineOp>(op);
    Block &bb = pipeOp.getBody().front();
    rewriter.eraseOp(pipeOp.getBody().back().getTerminator());
    bb.getOperations().splice(Block::iterator(op), bb.getOperations());
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRPipelinePutConversion : public ConversionPattern {
public:
  explicit AIRPipelinePutConversion(MLIRContext *context)
      : ConversionPattern(air::PipelinePutOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRPipelineGetConversion : public ConversionPattern {
public:
  explicit AIRPipelineGetConversion(MLIRContext *context)
      : ConversionPattern(air::PipelineGetOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto getOp = cast<air::PipelineGetOp>(op);
    SmallVector<Value, 2> gets;
    for (auto r : getOp.getResults()) {
      if (auto ty = r.getType().dyn_cast<RankedTensorType>())
        gets.push_back(rewriter.create<bufferization::AllocTensorOp>(
            op->getLoc(), ty, ValueRange{}));
      else
        return failure();
    }
    rewriter.replaceOp(op, gets);
    return success();
  }
};

class AIRWaitAllToAIRRtConversion : public OpConversionPattern<air::WaitAllOp> {
public:
  using OpConversionPattern<air::WaitAllOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::WaitAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> tys;
    if (op->getNumResults())
      tys.push_back(airrt::EventType::get(op->getContext()));

    rewriter.replaceOpWithNewOp<airrt::WaitAllOp>(op, tys,
                                                  adaptor.getOperands());
    return success();
  }
};

class AIRDmaMemcpyNdToAIRRtConversion
    : public OpConversionPattern<air::DmaMemcpyNdOp> {
public:
  using OpConversionPattern<air::DmaMemcpyNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    SmallVector<Value, 4> deps;
    for (auto o : adaptor.getOperands())
      if (o.getType().isa<airrt::EventType>())
        deps.push_back(o);
    if (deps.size())
      rewriter.create<airrt::WaitAllOp>(
          op->getLoc(), airrt::EventType::get(op->getContext()), deps);

    MemRefType src = op.getSrcMemref().getType().cast<MemRefType>();
    MemRefType dst = op.getDstMemref().getType().cast<MemRefType>();
    bool isFromTile = false;
    bool isFullMemcpy = false;
    if (src.getMemorySpaceAsInt() == (int)air::MemorySpace::L1 &&
        dst.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) {
      isFromTile = true;
    } else if (dst.getMemorySpaceAsInt() == (int)air::MemorySpace::L1 &&
               src.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) {
      isFromTile = false;
    } else if (src.getMemorySpaceAsInt() == (int)air::MemorySpace::L1 &&
               dst.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      isFromTile = true;
    } else if (dst.getMemorySpaceAsInt() == (int)air::MemorySpace::L1 &&
               src.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      isFromTile = false;
    } else if (src.getMemorySpaceAsInt() == (int)air::MemorySpace::L3 &&
               dst.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      isFullMemcpy = true;
    } else if (dst.getMemorySpaceAsInt() == (int)air::MemorySpace::L3 &&
               src.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      isFromTile = true;
      isFullMemcpy = true;
    } else
      return failure();

    SmallVector<Value, 16> opers;

    if (!isFullMemcpy) {
      auto idTy = IntegerType::get(op->getContext(), 32);
      if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
        opers.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
      } else {
        opers.push_back(rewriter.create<arith::ConstantOp>(
            loc, idTy, IntegerAttr::get(idTy, 0)));
      }

      air::HerdOp launch = op->getParentOfType<air::HerdOp>();
      if (!launch) {

        AffineForOp afo = op->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());

        afo = afo->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());
      } else {
        auto tileIds = launch.getIds();
        opers.push_back(tileIds[0]);
        opers.push_back(tileIds[1]);
      }
      opers[1] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[1]);
      opers[2] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[2]);

      if (isFromTile)
        opers.push_back(op.getDstMemref());
      else
        opers.push_back(op.getSrcMemref());
    } else {
      opers.push_back(op.getDstMemref());
      opers.push_back(op.getSrcMemref());
    }
    auto i64Ty = rewriter.getI64Type();
    auto zero = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                   IntegerAttr::get(i64Ty, 0));
    auto one = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                  IntegerAttr::get(i64Ty, 1));

    SmallVector<Value, 4> offsets(4, zero);
    SmallVector<Value, 4> lengths(4, one);
    SmallVector<Value, 3> strides(3, zero);

    int idx = 4 - src.getRank();
    for (auto o : isFromTile ? op.getDstOffsets() : op.getSrcOffsets())
      offsets[idx++] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), o);
    idx = 4 - dst.getRank();
    auto op_strides = isFromTile ? op.getDstStrides() : op.getSrcStrides();
    if (op_strides.size())
      for (auto o : op_strides.drop_back())
        strides[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), o);
    idx = 4 - src.getRank();
    for (auto o : isFromTile ? op.getDstSizes() : op.getSrcSizes())
      lengths[idx++] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), o);

    opers.append(offsets);
    opers.append(lengths);
    opers.append(strides);

    Operation *airrtOp = nullptr;
    SmallVector<Type, 1> tys;
    if (op->getNumResults())
      tys.push_back(airrt::EventType::get(ctx));
    if (isFullMemcpy) {
      airrtOp = rewriter.create<airrt::MemcpyNdOp>(loc, tys, opers);
    } else {
      airrtOp = rewriter.create<airrt::DmaMemcpyNdOp>(loc, tys, opers);
    }
    rewriter.replaceOp(op, airrtOp->getResults());
    return success();
  }
};

void remapExternalPutGet(OpBuilder rewriter, Value herd_x, Value herd_y,
                         air::ChannelInterface op,
                         air::ChannelInterface externalOp, IRMapping &remap) {

  if (auto par = externalOp->getParentOfType<scf::ParallelOp>()) {
    // TODO: What if some scf::par dims get canonicalized away
    remap.map(par.getInductionVars()[0],
              herd_x.getDefiningOp<arith::IndexCastOp>().getIn());
    remap.map(par.getInductionVars()[1],
              herd_y.getDefiningOp<arith::IndexCastOp>().getIn());
  }
  if (auto for_op = externalOp->getParentOfType<scf::ForOp>()) {
    remap.map(for_op.getInductionVar(),
              op->getParentOfType<scf::ForOp>().getInductionVar());
  }
  for (auto o : externalOp.getOffsets()) {
    if (auto constOp = o.getDefiningOp<arith::ConstantIndexOp>()) {
      auto newConstOp = rewriter.create<arith::ConstantIndexOp>(
          op->getLoc(), constOp.value());
      remap.map(constOp.getResult(), newConstOp.getResult());
    } else if (auto muliOp = o.getDefiningOp<arith::MulIOp>()) {
      for (auto operand : muliOp.getOperands()) {
        if (auto constOp = operand.getDefiningOp<arith::ConstantIndexOp>()) {
          remap.map(constOp.getResult(),
                    rewriter.create<arith::ConstantIndexOp>(op->getLoc(),
                                                            constOp.value()));
        }
      }
      auto newMulIOp = rewriter.clone(*muliOp.getOperation(), remap);
      remap.map(muliOp->getResult(0), newMulIOp->getResult(0));
    } else if (auto execOp = o.getDefiningOp<air::ExecuteOp>()) {
      assert(false);
    }
  }
}

class AIRChannelPutToAIRRtConversion
    : public OpConversionPattern<xilinx::air::ChannelPutOp> {
public:
  using OpConversionPattern<xilinx::air::ChannelPutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::ChannelPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    if (op->getParentOfType<air::HerdOp>())
      return failure();

    if (op->getParentOfType<AIE::CoreOp>())
      return failure();

    SmallVector<Value, 4> deps;
    xilinx::airrt::WaitAllOp placeholder = nullptr;
    for (auto o : adaptor.getOperands())
      if (o.getType().isa<xilinx::airrt::EventType>())
        deps.push_back(o);
    if (deps.size())
      placeholder = rewriter.create<xilinx::airrt::WaitAllOp>(
          op->getLoc(), xilinx::airrt::EventType::get(op->getContext()), deps);

    auto getOps = getTheOtherChannelOpThroughSymbol(op);
    if (getOps.size() > 1)
      return failure();
    auto getOp = getOps[0];

    MemRefType srcType = op.getSrc().getType().cast<MemRefType>();
    MemRefType dstType = getOp.getDst().getType().cast<MemRefType>();

    bool isFromTile =
        srcType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1;
    bool isFullMemcpy = false;
    if (srcType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3 &&
        dstType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      isFullMemcpy = true;
    } else if (dstType.getMemorySpaceAsInt() ==
                   (int)xilinx::air::MemorySpace::L3 &&
               srcType.getMemorySpaceAsInt() ==
                   (int)xilinx::air::MemorySpace::L2) {
      isFullMemcpy = true;
    }
    if (!isFromTile && !isFullMemcpy) {
      if (!placeholder)
        placeholder = rewriter.create<xilinx::airrt::WaitAllOp>(
            op->getLoc(), xilinx::airrt::EventType::get(op->getContext()),
            deps);
      rewriter.replaceOp(op, placeholder->getResults());
      return success();
    }

    SmallVector<Value, 16> opers;

    if (!isFullMemcpy) {
      auto idTy = IntegerType::get(op->getContext(), 32);
      if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
        opers.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
      } else {
        opers.push_back(rewriter.create<arith::ConstantOp>(
            loc, idTy, IntegerAttr::get(idTy, 0)));
      }

      air::HerdOp launch = op->getParentOfType<air::HerdOp>();
      if (!launch) {

        AffineForOp afo = op->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());

        afo = afo->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());
      } else {
        auto tileIds = launch.getIds();
        opers.push_back(tileIds[0]);
        opers.push_back(tileIds[1]);
      }
      opers[1] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[1]);
      opers[2] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[2]);

      if (isFromTile)
        opers.push_back(getOp.getDstMemref());
      else
        opers.push_back(op.getSrcMemref());
    } else {
      opers.push_back(getOp.getDstMemref());
      opers.push_back(op.getSrcMemref());
    }
    auto i64Ty = rewriter.getI64Type();
    auto zero = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                   IntegerAttr::get(i64Ty, 0));
    auto one = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                  IntegerAttr::get(i64Ty, 1));

    SmallVector<Value, 4> offsets(4, zero);
    SmallVector<Value, 4> lengths(4, one);
    SmallVector<Value, 3> strides(3, zero);

    int idx = 4 - srcType.getRank();
    if (isFromTile) {
      // Hoisting the external get back into the herd
      // TODO: this is not promising, and only makes sense when we only have one
      // queue and one centralized controller
      IRMapping remap;
      // Note: toggled herd x and y during remap, because of tracking affine.for
      // from inner to outer
      remapExternalPutGet(rewriter, opers[2], opers[1], op, getOp, remap);
      for (auto o : getOp.getDstOffsets()) {
        offsets[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), remap.lookupOrDefault(o));
      }
    } else {
      for (auto o : op.getSrcOffsets())
        offsets[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), o);
    }

    idx = 4 - dstType.getRank();
    auto op_strides = isFromTile ? getOp.getDstStrides() : op.getSrcStrides();
    if (op_strides.size())
      for (auto o : op_strides.drop_back())
        strides[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), o);
    idx = 4 - srcType.getRank();
    for (auto o : isFromTile ? getOp.getDstSizes() : op.getSrcSizes())
      lengths[idx++] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), o);

    opers.append(offsets);
    opers.append(lengths);
    opers.append(strides);

    Operation *airrtOp = nullptr;
    SmallVector<Type, 1> tys;
    if (op->getNumResults())
      tys.push_back(airrt::EventType::get(ctx));
    if (isFullMemcpy) {
      airrtOp = rewriter.create<airrt::MemcpyNdOp>(loc, tys, opers);
    } else {
      airrtOp = rewriter.create<airrt::DmaMemcpyNdOp>(loc, tys, opers);
    }

    rewriter.replaceOp(op, airrtOp->getResults());
    return success();
  }
};

class AIRChannelGetToAIRRtConversion
    : public OpConversionPattern<xilinx::air::ChannelGetOp> {
public:
  using OpConversionPattern<xilinx::air::ChannelGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::ChannelGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    if (op->getParentOfType<air::HerdOp>())
      return failure();

    if (op->getParentOfType<AIE::CoreOp>())
      return failure();

    SmallVector<Value, 4> deps;
    xilinx::airrt::WaitAllOp placeholder = nullptr;
    for (auto o : adaptor.getOperands())
      if (o.getType().isa<xilinx::airrt::EventType>())
        deps.push_back(o);
    if (deps.size())
      placeholder = rewriter.create<xilinx::airrt::WaitAllOp>(
          op->getLoc(), xilinx::airrt::EventType::get(op->getContext()), deps);

    auto putOps = getTheOtherChannelOpThroughSymbol(op);
    if (putOps.size() > 1)
      return failure();
    auto putOp = putOps[0];

    MemRefType srcType = putOp.getSrc().getType().cast<MemRefType>();
    MemRefType dstType = op.getDst().getType().cast<MemRefType>();

    bool isToTile =
        dstType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1;
    bool isFromTile = !isToTile;
    bool isFullMemcpy = false;
    if (srcType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3 &&
        dstType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      isFullMemcpy = true;
    } else if (dstType.getMemorySpaceAsInt() ==
                   (int)xilinx::air::MemorySpace::L3 &&
               srcType.getMemorySpaceAsInt() ==
                   (int)xilinx::air::MemorySpace::L2) {
      isFullMemcpy = true;
    }
    if (!isToTile && !isFullMemcpy) {
      if (!placeholder)
        placeholder = rewriter.create<xilinx::airrt::WaitAllOp>(
            op->getLoc(), xilinx::airrt::EventType::get(op->getContext()),
            deps);
      rewriter.replaceOp(op, placeholder->getResults());
      return success();
    }

    SmallVector<Value, 16> opers;

    if (!isFullMemcpy) {
      auto idTy = IntegerType::get(op->getContext(), 32);
      if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
        opers.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
      } else {
        opers.push_back(rewriter.create<arith::ConstantOp>(
            loc, idTy, IntegerAttr::get(idTy, 0)));
      }

      air::HerdOp launch = op->getParentOfType<air::HerdOp>();
      if (!launch) {

        AffineForOp afo = op->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());

        afo = afo->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());
      } else {
        auto tileIds = launch.getIds();
        opers.push_back(tileIds[0]);
        opers.push_back(tileIds[1]);
      }
      opers[1] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[1]);
      opers[2] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[2]);

      if (isFromTile)
        opers.push_back(op.getDstMemref());
      else
        opers.push_back(putOp.getSrcMemref());
    } else {
      opers.push_back(op.getDstMemref());
      opers.push_back(putOp.getSrcMemref());
    }
    auto i64Ty = rewriter.getI64Type();
    auto zero = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                   IntegerAttr::get(i64Ty, 0));
    auto one = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                  IntegerAttr::get(i64Ty, 1));

    SmallVector<Value, 4> offsets(4, zero);
    SmallVector<Value, 4> lengths(4, one);
    SmallVector<Value, 3> strides(3, zero);

    int idx = 4 - srcType.getRank();

    if (isToTile) {
      IRMapping remap;
      // Note: toggled herd x and y during remap, because of tracking affine.for
      // from inner to outer
      remapExternalPutGet(rewriter, opers[2], opers[1], op, putOp, remap);
      for (auto o : putOp.getSrcOffsets()) {
        offsets[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), remap.lookupOrDefault(o));
      }
    } else {
      for (auto o : op.getDstOffsets())
        offsets[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), o);
    }

    idx = 4 - dstType.getRank();
    auto op_strides = isFromTile ? op.getDstStrides() : putOp.getSrcStrides();
    if (op_strides.size())
      for (auto o : op_strides.drop_back())
        strides[idx++] = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), o);
    idx = 4 - srcType.getRank();
    for (auto o : isFromTile ? op.getDstSizes() : putOp.getSrcSizes())
      lengths[idx++] = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), o);

    opers.append(offsets);
    opers.append(lengths);
    opers.append(strides);

    Operation *airrtOp = nullptr;
    SmallVector<Type, 1> tys;
    if (op->getNumResults())
      tys.push_back(airrt::EventType::get(ctx));
    if (isFullMemcpy) {
      airrtOp = rewriter.create<airrt::MemcpyNdOp>(loc, tys, opers);
    } else {
      airrtOp = rewriter.create<airrt::DmaMemcpyNdOp>(loc, tys, opers);
    }

    rewriter.replaceOp(op, airrtOp->getResults());
    return success();
  }
};

class L2AllocToAIRRtConversion : public ConversionPattern {
public:
  explicit L2AllocToAIRRtConversion(MLIRContext *context)
      : ConversionPattern(memref::AllocOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto alloc = cast<memref::AllocOp>(op);
    auto type = alloc.getType();
    if (type.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      rewriter.replaceOpWithNewOp<airrt::AllocOp>(op, type);
      return success();
    }
    return failure();
  }
};

class L2DeallocToAIRRtConversion : public ConversionPattern {
public:
  explicit L2DeallocToAIRRtConversion(MLIRContext *context)
      : ConversionPattern(memref::DeallocOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dealloc = cast<memref::DeallocOp>(op);
    auto type = dealloc.getMemref().getType().cast<MemRefType>();
    if (type.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      rewriter.replaceOpWithNewOp<airrt::DeallocOp>(op, SmallVector<Type>{},
                                                    op->getOperands());
      return success();
    }
    return failure();
  }
};

LogicalResult lowerAirExecute(Operation *op) {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  SmallVector<Operation *, 8> erased;
  module->walk([&](air::ExecuteOp exe) {
    auto &bb = exe.getBody().front();
    unsigned idx = 0;

    OpBuilder builder(exe);
    if (exe.getAsyncDependencies().size())
      builder.create<air::WaitAllOp>(op->getLoc(), Type{},
                                     exe.getAsyncDependencies());

    for (auto &arg : bb.getArguments()) {
      arg.replaceAllUsesWith(exe.getOperand(idx));
      idx++;
    }
    exe.walk([&](air::ExecuteTerminatorOp t) {
      int resultIdx = 1;
      for (auto r : t->getOperands())
        exe.getResult(resultIdx++).replaceAllUsesWith(r);
      erased.push_back(t);
    });
    exe->getBlock()->getOperations().splice(Block::iterator(exe),
                                            bb.getOperations());
    if (exe.getNumResults() > 0) {
      auto w = builder.create<air::WaitAllOp>(
          op->getLoc(), air::AsyncTokenType::get(exe->getContext()),
          SmallVector<Value>{});
      exe.getResult(0).replaceAllUsesWith(w.getResult(0));
    }
    erased.push_back(exe);
  });
  for (auto a : erased)
    a->erase();
  return success();
}

class ScfYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    SmallVector<Type, 2> retTys;
    for (auto t : op->getResultTypes()) {
      if (t.isa<air::AsyncTokenType>()) {
        retTys.push_back(airrt::EventType::get(op->getContext()));
      } else {
        retTys.push_back(t);
      }
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, retTys, operands);
    return success();
  }
};

class ScfReduceOpConversion : public OpConversionPattern<scf::ReduceOp> {
public:
  using OpConversionPattern<scf::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        rewriter.replaceOpWithNewOp<scf::ReduceOp>(op, adaptor.getOperand());
    auto body = &op.getRegion().front();
    auto newBody = &newOp.getRegion().front();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), ops.end());
    return success();
  }
};

class ScfReduceReturnOpConversion
    : public OpConversionPattern<scf::ReduceReturnOp> {
public:
  using OpConversionPattern<scf::ReduceReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    SmallVector<Type, 2> retTys;
    for (auto t : op->getResultTypes()) {
      if (t.isa<air::AsyncTokenType>()) {
        retTys.push_back(airrt::EventType::get(op->getContext()));
      } else {
        retTys.push_back(t);
      }
    }
    rewriter.replaceOpWithNewOp<scf::ReduceReturnOp>(op, retTys, operands);
    return success();
  }
};

class ScfIfOpConversion : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type, 2> retTys;
    for (auto t : op->getResultTypes()) {
      if (t.isa<air::AsyncTokenType>()) {
        retTys.push_back(airrt::EventType::get(op->getContext()));
      } else {
        retTys.push_back(t);
      }
    }

    bool hasElseBlock = op.elseBlock() != nullptr;
    auto newIf = rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, retTys, op.getCondition(), hasElseBlock);

    auto &thenOps = op.thenBlock()->getOperations();
    auto &newThenOps = newIf.thenBlock()->getOperations();
    newThenOps.splice(newThenOps.begin(), thenOps, thenOps.begin(),
                      thenOps.end());

    if (!hasElseBlock)
      return success();

    auto &elseOps = op.elseBlock()->getOperations();
    auto &newElseOps = newIf.elseBlock()->getOperations();
    newElseOps.splice(newElseOps.begin(), elseOps, elseOps.begin(),
                      elseOps.end());

    return success();
  }
};

class ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep(),
        adaptor.getInitArgs());
    auto body = op.getBody();
    auto newBody = newOp.getBody();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), ops.end());
    return success();
  }
};

class ScfParOpConversion : public OpConversionPattern<scf::ParallelOp> {
public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<scf::ParallelOp>(
        op, adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep(),
        adaptor.getInitVals());
    auto body = op.getBody();
    auto newBody = newOp.getBody();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), --ops.end());
    return success();
  }
};

class AIRLoweringPass : public air::AIRLoweringBase<AIRLoweringPass> {

public:
  AIRLoweringPass() = default;
  AIRLoweringPass(const AIRLoweringPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect, airrt::AIRRtDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter converter;
    converter.addConversion([&](Type type) -> Optional<Type> {
      // convert !air.async.token to !airrt.event
      if (auto t = type.dyn_cast<air::AsyncTokenType>())
        return airrt::EventType::get(context);
      else
        return type;
    });
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           arith::ArithDialect, AffineDialect, scf::SCFDialect,
                           linalg::LinalgDialect, memref::MemRefDialect,
                           bufferization::BufferizationDialect,
                           airrt::AIRRtDialect>();

    // AIR ExecuteOp conversion
    if (failed(lowerAirExecute(module))) {
      emitError(UnknownLoc::get(context), "error lowering air.execute\n");
      signalPassFailure();
    }

    // Replace the PipelineStageOps first, followed by the
    // HerdPipelineOps, then run the rest of the patterns.
    // This avoids creating invalid intermediate code with respect
    // to the herd->pipeline->stages nesting requirements.

    // PipelineStageOp conversion
    RewritePatternSet air_pipe_stage_patterns(context);
    air_pipe_stage_patterns.insert<air::AIRPipeStageConversion>(
        context, air::AIRPipeStageConversion::LoweringType::AllocBuffer);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_pipe_stage_patterns)))) {
      emitError(UnknownLoc::get(context),
                "error lowering air.pipeline.stage\n");
      signalPassFailure();
    }

    // HerdPipelineOp conversion
    RewritePatternSet air_pipe_patterns(context);
    air_pipe_patterns.insert<AIRPipelineConversion, AIRPipelineGetConversion,
                             AIRPipelinePutConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_pipe_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline\n");
      signalPassFailure();
    }

    // DMA and HerdOp conversion
    RewritePatternSet air_patterns(context);

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() != (int)air::MemorySpace::L2);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (
          op.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
          (int)air::MemorySpace::L2);
    });

    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      for (auto o : op.getRegionIterArgs()) {
        if (o.getType().isa<air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ParallelOp>([&](scf::ParallelOp op) {
      for (auto o : op.getInitVals()) {
        if (o.getType().isa<air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      for (auto v : op.getResults()) {
        if (v.getType().isa<air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ReduceOp>([&](scf::ReduceOp op) {
      if (op.getOperand().getType().isa<air::AsyncTokenType>())
        return false;
      else
        return true;
    });

    target.addDynamicallyLegalOp<scf::ReduceReturnOp>(
        [&](scf::ReduceReturnOp op) {
          if (op.getResult().getType().isa<air::AsyncTokenType>())
            return false;
          else
            return true;
        });

    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      for (auto v : op.getResults()) {
        if (v.getType().isa<air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    air_patterns.add<
        ScfYieldOpConversion, ScfIfOpConversion, ScfParOpConversion,
        ScfReduceReturnOpConversion, ScfReduceOpConversion, ScfForOpConversion,
        L2AllocToAIRRtConversion, L2DeallocToAIRRtConversion,
        AIRLaunchConversion, AIRSegmentConversion, AIRHerdConversion>(context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(air_patterns,
                                                                   converter);

    air_patterns
        .add<AIRDmaMemcpyNdToAIRRtConversion, AIRChannelPutToAIRRtConversion,
             AIRChannelGetToAIRRtConversion, AIRWaitAllToAIRRtConversion>(
            converter, context);

    if (failed(
            applyPartialConversion(module, target, std::move(air_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }
  }
};

class AIRPipelineToAffinePass
    : public air::AIRPipelineToAffineBase<AIRPipelineToAffinePass> {

public:
  AIRPipelineToAffinePass() = default;
  AIRPipelineToAffinePass(const AIRPipelineToAffinePass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           arith::ArithDialect, AffineDialect, scf::SCFDialect,
                           linalg::LinalgDialect, memref::MemRefDialect,
                           bufferization::BufferizationDialect,
                           airrt::AIRRtDialect, air::airDialect>();

    target.addIllegalOp<air::PipelineStageOp, air::PipelineYieldOp>();

    // PipelineStageOp conversion
    RewritePatternSet air_pipe_stage_patterns(context);
    auto loweringType =
        air::AIRPipeStageConversion::LoweringType::PipelineGetPut;
    if (clLoweringType == "buffer")
      loweringType = air::AIRPipeStageConversion::LoweringType::AllocBuffer;
    air_pipe_stage_patterns.insert<air::AIRPipeStageConversion>(context,
                                                                loweringType);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_pipe_stage_patterns)))) {
      emitError(UnknownLoc::get(context),
                "error lowering air.pipeline.stage\n");
      signalPassFailure();
    }

    SmallVector<Operation *, 8> pipelines;
    module.walk([&](air::HerdPipelineOp p) { pipelines.push_back(p); });

    for (auto p : pipelines) {
      auto pipeOp = cast<air::HerdPipelineOp>(p);
      OpBuilder b(p);
      Block &bb = pipeOp.getBody().front();
      IRMapping remap;
      bb.getTerminator()->erase();
      for (auto &o : bb)
        b.clone(o, remap);
      p->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> xilinx::air::createAIRLoweringPass() {
  return std::make_unique<AIRLoweringPass>();
}

std::unique_ptr<mlir::Pass> xilinx::air::createAIRPipelineToAffinePass() {
  return std::make_unique<AIRPipelineToAffinePass>();
}
