//===- AIRLoweringPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRLoweringPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
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

    IRMapping remap;

    // map launch iteration space to scf.parallel ivs
    for (auto p : llvm::zip(launch.getIds(), scfPar.getInductionVars()))
      remap.map(std::get<0>(p), std::get<1>(p));

    // map launch size to scf.parallel upper bounds
    for (auto p : llvm::zip(launch.getSizeOperands(), scfPar.getUpperBound()))
      remap.map(std::get<0>(p), std::get<1>(p));

    // remap isolated from above launch operands
    auto launchOperands =
        operands.drop_front(operands.size() - launch.getNumKernelOperands());
    for (auto p : llvm::zip(launch.getKernelArguments(), launchOperands))
      remap.map(std::get<0>(p), std::get<1>(p));

    // clone the body
    rewriter.setInsertionPointToStart(scfPar.getBody());
    auto &launchOps = launch.getBody().front().getOperations();
    for (auto bi = launchOps.begin(), be = --launchOps.end(); bi != be; ++bi)
      rewriter.clone(*bi, remap);

    // replace output events with airrt.wait_all
    if (op->getNumResults()) {
      SmallVector<Value> deps;
      for (auto &o : operands)
        if (llvm::isa<airrt::EventType>(o.getType()))
          deps.push_back(o);
      rewriter.setInsertionPoint(scfPar);
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

    IRMapping remap;

    // map segment iteration space to scf.parallel ivs
    for (auto p : llvm::zip(segment.getIds(), scfPar.getInductionVars()))
      remap.map(std::get<0>(p), std::get<1>(p));

    // map segment size to scf.parallel upper bounds
    for (auto p : llvm::zip(segment.getSizeOperands(), scfPar.getUpperBound()))
      remap.map(std::get<0>(p), std::get<1>(p));

    int i = 0;
    for (auto arg : segment.getKernelArguments()) {
      auto oper = operands[segment.getAsyncDependencies().size() +
                           segment.getNumDims() + i++];
      remap.map(arg, oper);
    }

    // clone the body
    rewriter.setInsertionPointToStart(scfPar.getBody());
    for (auto &o : segment.getBody().front().getOperations()) {
      if (!isa<air::ChannelGetOp, air::ChannelPutOp, air::SegmentTerminatorOp>(
              o)) {
        rewriter.clone(o, remap);
      } else if (auto chanOp = dyn_cast<air::ChannelInterface>(o)) {
        // clone L3 get/put
        BaseMemRefType memrefTy =
            llvm::cast<BaseMemRefType>(chanOp.getMemref().getType());
        if (memrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) {
          rewriter.clone(o, remap);
          continue;
        }
        auto async = cast<air::AsyncOpInterface>(o);
        if (o.getNumResults()) {
          auto tok = o.getResult(0);
          SmallVector<Value> deps;
          for (auto d : async.getAsyncDependencies())
            deps.push_back(remap.lookupOrDefault(d));
          auto w = rewriter.create<air::WaitAllOp>(op->getLoc(), tok.getType(),
                                                   deps);
          remap.map(tok, w.getResult(0));
        }
      }
    }

    if (op->getNumResults()) {
      SmallVector<Value> deps;
      for (auto &o : operands)
        if (llvm::isa<airrt::EventType>(o.getType()))
          deps.push_back(o);
      rewriter.setInsertionPoint(scfPar);
      rewriter.replaceOpWithNewOp<airrt::WaitAllOp>(
          op, airrt::EventType::get(op->getContext()), deps);
    } else {
      rewriter.eraseOp(op);
    }
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

    // Integer kernel operands are passed as arguments (runtime parameters) to
    // the herd load op.
    SmallVector<Value> args;
    for (int i = 0, e = herd.getNumKernelOperands(); i < e; i++) {
      Value o = herd.getKernelOperand(i);
      Value arg = herd.getKernelArgument(i);
      if (arg.use_empty())
        continue;
      if (llvm::isa<IntegerType, IndexType, FloatType>(o.getType()))
        args.push_back(o);
    }

    rewriter.create<airrt::HerdLoadOp>(op->getLoc(), rewriter.getI64Type(),
                                       herd_name_attr.getValue().str(), args);

    SmallVector<Value, 4> deps;
    for (auto &o : operands)
      if (llvm::isa<airrt::EventType>(o.getType()))
        deps.push_back(o);
    if (op->getNumResults()) {
      auto w = rewriter.create<airrt::WaitAllOp>(
          op->getLoc(), airrt::EventType::get(op->getContext()), deps);
      herd.getResult(0).replaceAllUsesWith(w.getResult(0));
    }

    // If the herd doesn't contain a dma op, then it can be deleted
    SmallVector<air::DmaMemcpyNdOp> dmaOps;
    herd.walk([&](air::DmaMemcpyNdOp op) { dmaOps.push_back(op); });
    if (!dmaOps.size()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto herd_size = herd.getSizeOperands();
    int64_t herd_size_x = herd.getNumCols();
    int64_t herd_size_y = herd.getNumRows();

    auto outer =
        rewriter.create<affine::AffineForOp>(herd.getLoc(), 0, herd_size_x);
    rewriter.setInsertionPointToStart(outer.getBody());
    auto inner =
        rewriter.create<affine::AffineForOp>(herd.getLoc(), 0, herd_size_y);

    outer->setAttr("air.herd", StringAttr::get(op->getContext(), "outer"));
    inner->setAttr("air.herd", StringAttr::get(op->getContext(), "inner"));

    IRMapping remap;
    remap.map(herd.getSize()[0], herd_size[0]);
    remap.map(herd.getSize()[1], herd_size[1]);
    remap.map(herd.getIds()[0], outer.getInductionVar());
    remap.map(herd.getIds()[1], inner.getInductionVar());

    // remap isolated from above herd operands
    auto herdOperands =
        operands.drop_front(operands.size() - herd.getNumKernelOperands());
    for (auto p : llvm::zip(herd.getKernelArguments(), herdOperands))
      remap.map(std::get<0>(p), std::get<1>(p));

    rewriter.setInsertionPointToStart(inner.getBody());
    for (auto &o : herd.getBody().front().getOperations())
      if (!isa<air::HerdTerminatorOp>(o))
        rewriter.clone(o, remap);

    rewriter.eraseOp(op);
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
      if (llvm::isa<airrt::EventType>(o.getType()))
        deps.push_back(o);
    if (deps.size())
      rewriter.create<airrt::WaitAllOp>(
          op->getLoc(), airrt::EventType::get(op->getContext()), deps);

    BaseMemRefType src =
        llvm::cast<BaseMemRefType>(op.getSrcMemref().getType());
    BaseMemRefType dst =
        llvm::cast<BaseMemRefType>(op.getDstMemref().getType());
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

        affine::AffineForOp afo = op->getParentOfType<affine::AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<affine::AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());

        afo = afo->getParentOfType<affine::AffineForOp>();
        while (afo && !afo->getAttr("air.herd"))
          afo = afo->getParentOfType<affine::AffineForOp>();
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
    // Copy over discardable attrs assigned in -air-to-aie pass.
    op->removeAttr("id"); // Op's id is no longer useful. Airrt.dma op's id has
                          // been assigned.
    airrtOp->setAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, airrtOp->getResults());
    return success();
  }
};

// AIR channel to AIRRT impl.
FailureOr<Operation *>
AIRChannelInterfaceToAIRRtConversionImpl(OpBuilder builder,
                                         air::ChannelInterface thisOp,
                                         air::ChannelInterface theOtherOp) {
  auto loc = thisOp->getLoc();
  auto ctx = thisOp->getContext();

  BaseMemRefType thisMemrefType =
      llvm::cast<BaseMemRefType>(thisOp.getMemref().getType());

  bool thisOpIsInShim =
      thisMemrefType.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3;
  if (!thisOpIsInShim)
    return nullptr;

  SmallVector<Value, 16> opers;
  Operation *airrtOp = nullptr;
  auto i64Ty = builder.getI64Type();
  auto zero =
      builder.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 0));
  auto zero_idx = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one_idx = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto idTy = IntegerType::get(ctx, 32);
  // Get op id of the internal put/get op
  if (auto id_attr = theOtherOp->getAttrOfType<IntegerAttr>("id")) {
    opers.push_back(builder.create<arith::ConstantOp>(loc, idTy, id_attr));
  } else {
    opers.push_back(zero);
  }

  scf::ParallelOp launch = thisOp->getParentOfType<scf::ParallelOp>();
  if (!launch) {
    if (auto for_op = thisOp->getParentOfType<scf::ForOp>()) {
      // Broadcast channel control loop
      if (!theOtherOp->hasAttr("tile")) {
        theOtherOp->emitOpError(
            "missing 'tile' attribute as compile-time flag.");
        return failure();
      }
      ArrayAttr tiles = theOtherOp->getAttrOfType<ArrayAttr>("tile");
      auto tile_dict = llvm::cast<DictionaryAttr>(tiles[0]);
      auto row = llvm::cast<IntegerAttr>(tile_dict.get("row")).getInt();
      auto col = llvm::cast<IntegerAttr>(tile_dict.get("col")).getInt();
      opers.push_back(builder.create<arith::ConstantOp>(
          loc, i64Ty, IntegerAttr::get(i64Ty, col)));
      opers.push_back(builder.create<arith::ConstantOp>(
          loc, i64Ty, IntegerAttr::get(i64Ty, row)));
    } else {
      opers.push_back(zero);
      opers.push_back(zero);
    }
  } else {
    opers.push_back(builder.create<arith::IndexCastOp>(
        loc, IntegerType::get(ctx, 64), launch.getInductionVars()[0]));
    if (launch.getNumLoops() == 2)
      opers.push_back(builder.create<arith::IndexCastOp>(
          loc, IntegerType::get(ctx, 64), launch.getInductionVars()[1]));
    else if (launch.getNumLoops() == 1)
      opers.push_back(zero);
    else
      opers.push_back(zero);
  }

  opers.push_back(thisOp.getMemref());

  SmallVector<Value> offsets = thisOp.getOffsets();
  SmallVector<Value> wraps = thisOp.getSizes();
  SmallVector<Value> strides = thisOp.getStrides();

  auto memrefType = thisOp.getMemref().getType();

  // If empty offsets/sizes/strides, then populate the lists with default
  // values.
  if (offsets.empty() && wraps.empty() && strides.empty()) {
    offsets.push_back(zero_idx);
    auto memref_volume = air::getTensorVolume(memrefType);
    wraps.push_back(builder.create<arith::ConstantIndexOp>(loc, memref_volume));
    strides.push_back(one_idx);
  }
  // Stride field implicit last element one
  auto lastStrideConst = getConstantIntValue(strides.back());
  if (!lastStrideConst) {
    thisOp->emitOpError("last stride is not static.");
    return failure();
  }

  strides.pop_back();
  while (offsets.size() < 4) {
    offsets.insert(offsets.begin(), zero_idx);
  }
  while (wraps.size() < 4) {
    wraps.insert(wraps.begin(), one_idx);
  }
  while (strides.size() < 3) {
    strides.insert(strides.begin(), zero_idx);
  }

  for (unsigned i = 0; i < offsets.size(); i++)
    offsets[i] = builder.create<arith::IndexCastOp>(
        loc, IntegerType::get(ctx, 64), offsets[i]);

  for (unsigned i = 0; i < strides.size(); i++)
    strides[i] = builder.create<arith::IndexCastOp>(
        loc, IntegerType::get(ctx, 64), strides[i]);

  for (unsigned i = 0; i < wraps.size(); i++)
    wraps[i] = builder.create<arith::IndexCastOp>(
        loc, IntegerType::get(ctx, 64), wraps[i]);

  opers.append(offsets);
  opers.append(wraps);
  opers.append(strides);

  SmallVector<Type, 1> tys;
  if (thisOp->getNumResults())
    tys.push_back(airrt::EventType::get(ctx));

  airrtOp = builder.create<airrt::DmaMemcpyNdOp>(loc, tys, opers);
  // Copy over discardable attrs assigned in -air-to-aie pass.
  thisOp->removeAttr("id"); // Op's id is no longer useful. Airrt.dma op's id
                            // has been assigned.
  airrtOp->setAttrs(thisOp->getDiscardableAttrDictionary());
  return airrtOp;
}

template <typename OpT>
class AIRChannelGetPutToAIRRtConversion : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (llvm::isa<air::HerdOp>(op->getParentOp()))
      return failure();

    if (llvm::isa<AIE::CoreOp>(op->getParentOp()))
      return failure();

    auto otherOps = getTheOtherChannelOpThroughSymbol(op);
    if (otherOps.empty())
      return op->emitOpError("failed to find the other side of air.channel");
    auto otherOp = otherOps[0];

    auto airrtOp =
        AIRChannelInterfaceToAIRRtConversionImpl(rewriter, op, otherOp);

    if (failed(airrtOp))
      return failure();

    if (*airrtOp != nullptr) {
      // Resolve channel op's dependency list
      if ((*airrtOp)->getNumResults()) {
        SmallVector<Value, 4> deps = {(*airrtOp)->getResult(0)};
        for (auto o : adaptor.getOperands())
          if (llvm::isa<xilinx::airrt::EventType>(o.getType()))
            deps.push_back(o);
        auto wa = rewriter.create<xilinx::airrt::WaitAllOp>(
            op->getLoc(), xilinx::airrt::EventType::get(op->getContext()),
            deps);
        rewriter.replaceOp(op, wa);
        return success();
      } else {
        rewriter.replaceOp(op, *airrtOp);
        return success();
      }
    }

    if (op->getNumResults()) {
      // Resolve channel op's dependency list
      SmallVector<Value, 4> deps;
      for (auto o : adaptor.getOperands())
        if (llvm::isa<xilinx::airrt::EventType>(o.getType()))
          deps.push_back(o);
      if (deps.size())
        rewriter.replaceOpWithNewOp<xilinx::airrt::WaitAllOp>(
            op, xilinx::airrt::EventType::get(op->getContext()), deps);
      else
        rewriter.eraseOp(op);
      return success();
    }

    rewriter.eraseOp(op);
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
    auto type = llvm::cast<BaseMemRefType>(dealloc.getMemref().getType());
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

  llvm::SmallSet<Operation *, 8> erased;
  module->walk([&](air::ExecuteOp exe) {
    auto &bb = exe.getRegion().front();
    unsigned idx = 0;

    // If the execute op only contains pure ops (ops that do not touch memory),
    // or memref.alloc/dealloc, then we shouldn't enforce blocking wait_alls to
    // it.
    auto isNonBlockingOp = [](Operation *o) {
      return air::isPure(o) ||
             isa<air::WaitAllOp, memref::AllocOp, memref::DeallocOp>(o);
    };
    bool isNonBlockingAsyncExecute = false;
    if (llvm::all_of(exe.getChildOps(), [isNonBlockingOp](Operation &o) {
          return isNonBlockingOp(&o);
        }))
      isNonBlockingAsyncExecute = true;

    OpBuilder builder(exe);
    if (!isNonBlockingAsyncExecute && exe.getAsyncDependencies().size())
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
      erased.insert(t);
    });
    exe->getBlock()->getOperations().splice(Block::iterator(exe),
                                            bb.getOperations());
    if (exe.getNumResults() > 0) {
      auto w = builder.create<air::WaitAllOp>(
          op->getLoc(), air::AsyncTokenType::get(exe->getContext()),
          SmallVector<Value>{});
      exe.getResult(0).replaceAllUsesWith(w.getResult(0));
    }
    erased.insert(exe);
  });
  for (auto a : erased)
    a->erase();
  return success();
}

template <typename hierTy, typename loadTy>
LogicalResult generateLoadForHierarchy(Operation *op) {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  module->walk([&](hierTy hier) {
    if (auto attr = hier->template getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName())) {
      OpBuilder b(hier);
      auto hierarchy_name = attr.getValue().str();
      if (auto launch = hier->template getParentOfType<air::LaunchOp>())
        b.setInsertionPointToStart(&launch.getBody().front());
      else
        b.setInsertionPointToStart(hier->getBlock());
      if constexpr (std::is_same<loadTy, airrt::SegmentLoadOp>::value)
        b.create<airrt::SegmentLoadOp>(hier->getLoc(), b.getI64Type(),
                                       hierarchy_name);

      else if constexpr (std::is_same<loadTy, airrt::HerdLoadOp>::value)
        b.create<airrt::HerdLoadOp>(hier->getLoc(), b.getI64Type(),
                                    hierarchy_name,
                                    /* operands */ SmallVector<Value>());
    }
  });
  return success();
}

class ScfYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
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
        rewriter.replaceOpWithNewOp<scf::ReduceOp>(op, adaptor.getOperands());
    auto body = &op.getRegion(0).front();
    auto newBody = &newOp.getRegion(0).front();
    rewriter.setInsertionPointToStart(newBody);

    IRMapping remap;
    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      auto arg = body->getArgument(i);
      auto newArg = newBody->getArgument(i);
      if (isa<airrt::EventType>(newArg.getType())) {
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            op->getLoc(), arg.getType(), newArg);
        remap.map(arg, cast.getResult(0));
      } else {
        remap.map(arg, newArg);
      }
    }
    for (auto &o : body->getOperations()) {
      if (isa<scf::ReduceReturnOp>(o)) {
        SmallVector<Value> opers;
        for (int i = 0, e = o.getNumOperands(); i < e; i++) {
          auto oper = remap.lookupOrDefault(o.getOperand(i));
          if (llvm::isa<air::AsyncTokenType>(oper.getType())) {
            auto ty = airrt::EventType::get(o.getContext());
            auto cast = rewriter.create<UnrealizedConversionCastOp>(
                op->getLoc(), ty, oper);
            remap.map(o.getOperand(i), cast->getResult(0));
          }
        }
      }
      rewriter.clone(o, remap);
    }
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
    SmallVector<Type, 2> retTys;
    for (auto t : op->getResultTypes()) {
      if (llvm::isa<air::AsyncTokenType>(t)) {
        retTys.push_back(airrt::EventType::get(op->getContext()));
      } else {
        retTys.push_back(t);
      }
    }
    rewriter.replaceOpWithNewOp<scf::ReduceReturnOp>(op, retTys,
                                                     adaptor.getOperands());
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
      if (llvm::isa<air::AsyncTokenType>(t)) {
        retTys.push_back(airrt::EventType::get(op->getContext()));
      } else {
        retTys.push_back(t);
      }
    }

    bool hasElseBlock = op.elseBlock() != nullptr;
    auto newIf = rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, retTys, op.getCondition(), hasElseBlock);

    IRMapping remap;
    rewriter.setInsertionPointToStart(newIf.thenBlock());
    for (auto &o : op.thenBlock()->getOperations())
      rewriter.clone(o, remap);

    if (!hasElseBlock)
      return success();

    rewriter.setInsertionPointToStart(newIf.elseBlock());
    for (auto &o : op.elseBlock()->getOperations())
      rewriter.clone(o, remap);

    return success();
  }
};

class ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    scf::ForOp newOp =
        dyn_cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Set operands and update block argument and result types.
    newOp->setOperands(adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter)))
      return failure();
    for (auto result : newOp.getResults())
      result.setType(typeConverter->convertType(result.getType()));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class ScfParOpConversion : public OpConversionPattern<scf::ParallelOp> {
public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    scf::ParallelOp newOp = dyn_cast<scf::ParallelOp>(
        rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Set operands and update block argument and result types.
    newOp->setOperands(adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter)))
      return failure();
    for (auto result : newOp.getResults())
      result.setType(typeConverter->convertType(result.getType()));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

LogicalResult ScfParToAffineForConversion(Operation *op) {
  func::FuncOp f = dyn_cast<func::FuncOp>(op);
  if (!f)
    return failure();

  llvm::SetVector<Operation *> erased;
  SmallVector<scf::ParallelOp> scf_pars;
  f.walk([&](scf::ParallelOp scf_par) { scf_pars.push_back(scf_par); });
  for (auto scf_par : scf_pars) {
    if (!llvm::all_of(scf_par.getLowerBound(), [](Value v) {
          auto constV = getConstantIntValue(v);
          if (!constV)
            return false;
          if (*constV != 0)
            return false;
          return true;
        })) {
      scf_par->emitOpError("has non-zero lower bound.");
      return failure();
    }
    if (!llvm::all_of(scf_par.getStep(), [](Value v) {
          auto constV = getConstantIntValue(v);
          if (!constV)
            return false;
          if (*constV != 1)
            return false;
          return true;
        })) {
      scf_par->emitOpError("has non-unit step size.");
      return failure();
    }
    std::vector<int> par_sizes = {};
    for (auto v : scf_par.getUpperBound())
      par_sizes.push_back(
          dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp()).value());

    OpBuilder builder(scf_par);
    SmallVector<affine::AffineForOp> loops;
    for (unsigned i = 0; i < par_sizes.size(); i++) {
      if (i == 0)
        loops.push_back(builder.create<affine::AffineForOp>(scf_par.getLoc(), 0,
                                                            par_sizes[0]));
      else {
        auto inner_builder = OpBuilder::atBlockBegin(loops[i - 1].getBody());
        loops.push_back(inner_builder.create<affine::AffineForOp>(
            scf_par.getLoc(), 0, par_sizes[i]));
      }
    }

    builder.setInsertionPointToStart(loops.back().getBody());
    IRMapping remap;
    for (unsigned i = 0; i < par_sizes.size(); i++)
      remap.map(scf_par.getInductionVars()[i], loops[i].getInductionVar());
    for (auto &o : scf_par.getBody()->getOperations()) {
      if (!isa<scf::ReduceOp>(o) && !isa<scf::YieldOp>(o) &&
          !isa<scf::ParallelOp>(o)) {
        builder.clone(o, remap);
      }
    }
    erased.insert(scf_par);
  }
  for (auto a : erased) {
    if (a->getNumResults())
      for (auto token : a->getResults())
        for (auto user : token.getUsers())
          for (unsigned i = 0; i < user->getNumOperands(); i++)
            if (user->getOperand(i) == token)
              user->eraseOperand(i);
    a->erase();
  }
  return success();
}

class AIRLoweringPass : public air::impl::AIRLoweringBase<AIRLoweringPass> {

public:
  AIRLoweringPass() = default;
  AIRLoweringPass(const AIRLoweringPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, airrt::AIRRtDialect,
                    LLVM::LLVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter converter;
    converter.addConversion([&](Type type) -> std::optional<Type> {
      // convert !air.async.token to !airrt.event
      if (auto t = llvm::dyn_cast<air::AsyncTokenType>(type))
        return airrt::EventType::get(context);
      else
        return type;
    });
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return cast.getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    ConversionTarget target(*context);

    target.addLegalDialect<
        LLVM::LLVMDialect, func::FuncDialect, arith::ArithDialect,
        affine::AffineDialect, scf::SCFDialect, linalg::LinalgDialect,
        memref::MemRefDialect, bufferization::BufferizationDialect,
        airrt::AIRRtDialect>();

    // AIR ExecuteOp conversion
    if (failed(lowerAirExecute(module))) {
      emitError(UnknownLoc::get(context), "error lowering air.execute\n");
      signalPassFailure();
    }

    // Generate airrt load op as handle for reloading binaries
    if (failed(generateLoadForHierarchy<air::SegmentOp, airrt::SegmentLoadOp>(
            module))) {
      emitError(UnknownLoc::get(context),
                "error generating airrt.segment_load\n");
      signalPassFailure();
    }

    // DMA and HerdOp conversion
    RewritePatternSet air_patterns(context);

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() != (int)air::MemorySpace::L2);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (llvm::cast<BaseMemRefType>(op.getMemref().getType())
                  .getMemorySpaceAsInt() != (int)air::MemorySpace::L2);
    });

    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      for (auto o : op.getRegionIterArgs()) {
        if (llvm::isa<air::AsyncTokenType>(o.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ParallelOp>([&](scf::ParallelOp op) {
      for (auto v : op.getResults()) {
        if (llvm::isa<air::AsyncTokenType>(v.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      for (auto v : op.getResults()) {
        if (llvm::isa<air::AsyncTokenType>(v.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ReduceOp>([&](scf::ReduceOp op) {
      for (auto o : op.getOperands())
        if (llvm::isa<air::AsyncTokenType>(o.getType()))
          return false;
      return true;
    });

    target.addDynamicallyLegalOp<scf::ReduceReturnOp>(
        [&](scf::ReduceReturnOp op) {
          if (llvm::isa<air::AsyncTokenType>(op.getResult().getType()))
            return false;
          else
            return true;
        });

    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      for (auto v : op.getResults()) {
        if (llvm::isa<air::AsyncTokenType>(v.getType()))
          return false;
      }
      return true;
    });
    target.addIllegalOp<air::WaitAllOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    air_patterns
        .add<L2AllocToAIRRtConversion, L2DeallocToAIRRtConversion,
             AIRLaunchConversion, AIRSegmentConversion, AIRHerdConversion>(
            context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(air_patterns,
                                                                   converter);

    air_patterns.add<
        ScfYieldOpConversion, ScfIfOpConversion, ScfForOpConversion,
        ScfParOpConversion, ScfReduceReturnOpConversion, ScfReduceOpConversion,
        AIRDmaMemcpyNdToAIRRtConversion, AIRWaitAllToAIRRtConversion,
        AIRChannelGetPutToAIRRtConversion<air::ChannelGetOp>,
        AIRChannelGetPutToAIRRtConversion<air::ChannelPutOp>>(converter,
                                                              context);

    if (failed(
            applyPartialConversion(module, target, std::move(air_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }

    // If scf parallel loops containing memcpy ops exist in the same scope as
    // herd load, then attempt to serialize the asynchronous control programs.
    module.walk([&](func::FuncOp f) {
      bool hasCandidateSCFParallel = false;
      for (auto par : f.getBody().getOps<scf::ParallelOp>()) {
        par.walk(
            [&](airrt::DmaMemcpyNdOp c) { hasCandidateSCFParallel = true; });
      }
      if (hasCandidateSCFParallel)
        if (failed(serializeAsyncControlFlows(f)))
          signalPassFailure();

      // SCF parallel to affine for conversion
      if (failed(ScfParToAffineForConversion(f))) {
        emitError(UnknownLoc::get(context), "error lowering air.execute\n");
        signalPassFailure();
      }

      // Label root of perfectly nested affine for loops for affine
      // optimizations.
      std::vector<SmallVector<mlir::affine::AffineForOp, 6>> bands;
      mlir::affine::getTileableBands(f, &bands);
      for (auto &band : bands) {
        band[0]->setAttr("affine_opt_label",
                         StringAttr::get(f.getContext(), "tiling"));
      }
    });

    // HACK: Remove scf.for loops containing only pure ops
    // Details: Currently, the scf for canonicalizer is not able to remove empty
    // loop body with nothing but wait_all, if any of them is merging multiple
    // iter_arg air async tokens
    SmallVector<scf::ForOp> forOps;
    module.walk([&forOps](scf::ForOp fop) { forOps.push_back(fop); });
    for (auto fop : forOps) {
      if (!llvm::all_of(fop.getOps(), [](Operation &o) {
            return air::isPure(&o) || isa<airrt::WaitAllOp>(o);
          }))
        continue;
      IRRewriter rewriter(module.getContext());
      rewriter.replaceOp(fop, fop.getInitArgs());
    }
  }

private:
  // Remap memcpy
  Operation *remapOpAndOperands(OpBuilder builder, Operation *op,
                                IRMapping &remap) const {
    for (auto operand : op->getOperands()) {
      if (operand.getDefiningOp()) {
        if (auto index_cast =
                dyn_cast<arith::IndexCastOp>(operand.getDefiningOp())) {
          remapOpAndOperands(builder, operand.getDefiningOp(), remap);
          builder.clone(*index_cast, remap);
        } else if (auto const_op =
                       dyn_cast<arith::ConstantOp>(operand.getDefiningOp())) {
          builder.clone(*const_op, remap);
        } else if (auto muli_op =
                       dyn_cast<arith::MulIOp>(operand.getDefiningOp())) {
          remapOpAndOperands(builder, operand.getDefiningOp(), remap);
          builder.clone(*muli_op, remap);
        }
      }
    }
    return builder.clone(*op, remap);
  }

  // Remap loop nests
  void remapLoop(scf::ForOp src_for, scf::ForOp dst_for,
                 IRMapping &remap) const {
    remap.map(src_for.getInductionVar(), dst_for.getInductionVar());
    for (unsigned i = 0; i < src_for.getRegionIterArgs().size(); i++) {
      remap.map(src_for.getRegionIterArgs()[i], dst_for.getRegionIterArgs()[i]);
    }
  }
  void remapLoop(scf::ParallelOp src_par, scf::ParallelOp dst_par,
                 IRMapping &remap) const {
    for (unsigned i = 0; i < src_par.getInductionVars().size(); i++) {
      remap.map(src_par.getInductionVars()[i], dst_par.getInductionVars()[i]);
    }
    for (unsigned i = 0; i < src_par.getInitVals().size(); i++) {
      remap.map(src_par.getInitVals()[i], dst_par.getInitVals()[i]);
    }
  }
  void remapLoop(Operation *src, Operation *dst, IRMapping &remap) const {
    auto src_for = dyn_cast<scf::ForOp>(src);
    auto dst_for = dyn_cast<scf::ForOp>(dst);
    auto src_par = dyn_cast<scf::ParallelOp>(src);
    auto dst_par = dyn_cast<scf::ParallelOp>(dst);
    if (src_for && dst_for) {
      remapLoop(src_for, dst_for, remap);
    } else if (src_par && dst_par) {
      remapLoop(src_par, dst_par, remap);
    }
  }

  // Get parent loop nest
  std::vector<Operation *> getParentLoopNest(Operation *op,
                                             Operation *outermost) const {
    if (!op)
      return std::vector<Operation *>();
    std::vector<Operation *> output;
    for (auto parent = op->getParentOp(); parent != outermost;
         parent = parent->getParentOp()) {
      output.push_back(parent);
    }
    output.push_back(outermost);
    return output;
  }

  // Get (the first) memcpy op from loop nest
  Operation *getInnerMostMemcpyFromLoopNest(Operation *op) const {
    if (auto scf_par = dyn_cast<scf::ParallelOp>(op))
      return getInnerMostMemcpyFromLoopNest(scf_par);
    else if (auto scf_for = dyn_cast<scf::ForOp>(op))
      return getInnerMostMemcpyFromLoopNest(scf_for);
    // else return nullptr;
    else {
      llvm_unreachable("unhandled op");
    }
  }
  Operation *getInnerMostMemcpyFromLoopNest(scf::ForOp op) const {
    Operation *output = nullptr;
    op.walk([&](airrt::DmaMemcpyNdOp o) { output = o; });
    return output;
  }
  Operation *getInnerMostMemcpyFromLoopNest(scf::ParallelOp op) const {
    Operation *output = nullptr;
    op.walk([&](airrt::DmaMemcpyNdOp o) { output = o; });
    return output;
  }

  // This function is a workaround for vck190 having one single control
  // processor, where all the async. control programs are serialized here.
  LogicalResult serializeAsyncControlFlows(func::FuncOp func) const {

    // Collect async scf loops in line-by-line order
    std::vector<Operation *> scf_loops;
    for (auto scf_loop : func.getBody().getOps<scf::ForOp>()) {
      if (getInnerMostMemcpyFromLoopNest(scf_loop)) {
        scf_loops.push_back(scf_loop);
      }
    }
    for (auto scf_loop : func.getBody().getOps<scf::ParallelOp>()) {
      if (getInnerMostMemcpyFromLoopNest(scf_loop)) {
        scf_loops.push_back(scf_loop);
      }
    }

    // Adjacent loops which contain for loop should merge into one
    std::vector<std::vector<Operation *>> scf_loop_buckets;
    scf_loop_buckets.push_back(std::vector<Operation *>());
    int bucket_id = 0;
    bool prev_loop_in_bucket = false;
    for (auto scf_loop : scf_loops) {
      bool merge_candidate_loop = false;
      if (isa<scf::ForOp>(scf_loop))
        merge_candidate_loop = true;
      else if (auto scf_par = dyn_cast<scf::ParallelOp>(scf_loop)) {
        merge_candidate_loop = false;
        for (auto child_scf_for_loop :
             scf_par.getBody()->getOps<scf::ForOp>()) {
          (void)child_scf_for_loop;
          merge_candidate_loop = true;
        }
      }

      if (merge_candidate_loop) {
        scf_loop_buckets[bucket_id].push_back(scf_loop);
        prev_loop_in_bucket = true;
      } else {
        if (prev_loop_in_bucket == true) {
          bucket_id++;
          scf_loop_buckets.push_back(std::vector<Operation *>());
        }
        prev_loop_in_bucket = false;
      }
    }

    // Merge each bucket of loops into one
    for (auto bucket : scf_loop_buckets) {
      if (!bucket.empty()) {
        OpBuilder builder(bucket[0]);
        auto new_ctrl_loop = builder.clone(*bucket[0]);
        Operation *first_chan_op =
            getInnerMostMemcpyFromLoopNest(new_ctrl_loop);
        auto dst_loop_nest = getParentLoopNest(first_chan_op, new_ctrl_loop);
        for (unsigned i = 1; i < bucket.size(); i++) {
          IRMapping remap;
          Operation *chan_op = getInnerMostMemcpyFromLoopNest(bucket[i]);
          if (!chan_op) {
            func->emitOpError("memcpy in innermost loop body not found.");
            return failure();
          }
          auto src_loop_nest = getParentLoopNest(chan_op, bucket[i]);
          for (auto [src_loop, dst_loop] :
               llvm::zip_equal(src_loop_nest, dst_loop_nest)) {
            remapLoop(src_loop, dst_loop, remap);
          }
          auto yield_op = dst_loop_nest[0]
                              ->getRegions()
                              .front()
                              .getBlocks()
                              .front()
                              .getTerminator();
          builder.setInsertionPoint(yield_op);
          remapOpAndOperands(builder, chan_op, remap);
          if (i == bucket.size() - 1) {
            SmallVector<Value, 8> operands{};
            if (auto new_ctrl_loop_par =
                    dyn_cast<scf::ParallelOp>(dst_loop_nest[0])) {
              operands.push_back(new_ctrl_loop_par.getInitVals()[0]);
            } else if (auto new_ctrl_loop_for =
                           dyn_cast<scf::ForOp>(dst_loop_nest[0])) {
              operands.push_back(new_ctrl_loop_for.getRegionIterArgs()[0]);
            }
            builder.create<scf::YieldOp>(yield_op->getLoc(), operands);
            yield_op->erase();
          }
        }
      }
      for (unsigned i = 0; i < bucket.size(); i++) {
        bucket[i]->erase();
      }
    }
    return success();
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLoweringPass() {
  return std::make_unique<AIRLoweringPass>();
}

} // namespace air
} // namespace xilinx
