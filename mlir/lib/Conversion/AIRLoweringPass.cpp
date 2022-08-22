// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"

#include "air/Conversion/AIRPipeline.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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
using namespace mlir::arith;
using namespace xilinx::air;

namespace {

class AIRPartitionConversion : public ConversionPattern {
public:
  explicit AIRPartitionConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::PartitionOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xilinx::air::PartitionOp partition = cast<xilinx::air::PartitionOp>(op);
    if (auto attr =
    op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      auto partition_name = attr.getValue().str();
      rewriter.create<xilinx::airrt::PartitionLoadOp>(op->getLoc(),
      rewriter.getI64Type(), partition_name);
    }

    SmallVector<Value, 4> deps;
    for (auto &o : operands)
      if (o.getType().isa<xilinx::airrt::EventType>())
        deps.push_back(o);
    if (op->getNumResults()) {
      auto w = rewriter.create<xilinx::airrt::WaitAllOp>(
          op->getLoc(), xilinx::airrt::EventType::get(op->getContext()), deps);
      partition.getResult(0).replaceAllUsesWith(w.getResult(0));
    }

    SmallVector<Value, 2> lbs, ubs, steps;
    auto c0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);

    // make scf.parallel to replace air.partition
    for (auto d : partition.getSizeOperands()) {
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

    // map partition iteration space to scf.parallel ivs
    for (auto p : llvm::zip(partition.getIds(), scfPar.getInductionVars()))
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));

    // map partition size to scf.parallel upper bounds
    for (auto p :
         llvm::zip(partition.getSizeOperands(), scfPar.getUpperBound()))
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));

    int i = 0;
    for (auto arg : partition.getKernelArguments())
      arg.replaceAllUsesWith(partition.getKernelOperand(i++));

    auto &body = partition.body().front().getOperations();
    scfPar.getBody()->getOperations().splice(scfPar.getBody()->begin(), body,
                                             body.begin(), --body.end());

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRHerdConversion : public ConversionPattern {
public:
  explicit AIRHerdConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::HerdOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    xilinx::air::HerdOp launch = cast<xilinx::air::HerdOp>(op);
    if (auto attr = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      auto herd_name = attr.getValue().str();
      rewriter.create<xilinx::airrt::HerdLoadOp>(op->getLoc(), rewriter.getI64Type(), herd_name);
    }

    SmallVector<Value, 4> deps;
    for (auto &o : operands)
      if (o.getType().isa<xilinx::airrt::EventType>())
        deps.push_back(o);
    if (op->getNumResults()) {
      auto w = rewriter.create<xilinx::airrt::WaitAllOp>(
          op->getLoc(), xilinx::airrt::EventType::get(op->getContext()), deps);
      launch.getResult(0).replaceAllUsesWith(w.getResult(0));
    }

    auto herd_size = launch.getSizeOperands();
    int64_t herd_size_x = cast<ConstantIndexOp>(herd_size[0].getDefiningOp()).value();
    int64_t herd_size_y = cast<ConstantIndexOp>(herd_size[1].getDefiningOp()).value();

    auto outer = rewriter.create<AffineForOp>(launch.getLoc(), 0, herd_size_x);
    auto outer_builder = OpBuilder::atBlockBegin(outer.getBody());
    auto inner = outer_builder.create<AffineForOp>(launch.getLoc(), 0, herd_size_y);

    outer->setAttr("air.herd", StringAttr::get(op->getContext(), "outer"));
    inner->setAttr("air.herd", StringAttr::get(op->getContext(), "inner"));

    launch.getSize()[0].replaceAllUsesWith(herd_size[0]);
    launch.getSize()[1].replaceAllUsesWith(herd_size[1]);
    launch.getIds()[0].replaceAllUsesWith(outer.getInductionVar());
    launch.getIds()[1].replaceAllUsesWith(inner.getInductionVar());

    int i=0;
    for (auto arg : launch.getKernelArguments())
      arg.replaceAllUsesWith(launch.getKernelOperand(i++));

    auto &body = launch.body().front().getOperations();
    inner.getBody()->getOperations().splice(inner.getBody()->begin(), body,
                                            body.begin(), --body.end());

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRPipelineConversion : public ConversionPattern {
public:
  explicit AIRPipelineConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::HerdPipelineOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto pipeOp = cast<xilinx::air::HerdPipelineOp>(op);
    Block &bb = pipeOp.body().front();
    rewriter.eraseOp(pipeOp.body().back().getTerminator());
    bb.getOperations().splice(Block::iterator(op),
                              bb.getOperations());
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRPipelinePutConversion : public ConversionPattern {
public:
  explicit AIRPipelinePutConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::PipelinePutOp::getOperationName(), 1,
                          context) {}

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
      : ConversionPattern(xilinx::air::PipelineGetOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto getOp = cast<xilinx::air::PipelineGetOp>(op);
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

class AIRWaitAllToAIRRtConversion: public OpConversionPattern<xilinx::air::WaitAllOp> {
public:
  using OpConversionPattern<xilinx::air::WaitAllOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::WaitAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    SmallVector<Type, 1> tys;
    if (op->getNumResults())
      tys.push_back(xilinx::airrt::EventType::get(op->getContext()));

    rewriter.replaceOpWithNewOp<xilinx::airrt::WaitAllOp>(op, tys, adaptor.getOperands());
    return success();
  }
};

static
Operation* convertDmaMemcpyToAirRt(Operation *op, ArrayRef<Value > operands,
                                   ConversionPatternRewriter &rewriter) {
  auto dmaif = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op);
  auto loc = op->getLoc();
  auto ctx = op->getContext();

  MemRefType src = dmaif.getSrcMemref().getType().cast<MemRefType>();
  MemRefType dst = dmaif.getDstMemref().getType().cast<MemRefType>();
  bool isFromTile;
  if (src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
      dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3) {
    isFromTile = true;
  }
  else if (dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
           src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3) {
    isFromTile = false;
  }
  else if (src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
           dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
    isFromTile = true;
  }
  else if (dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
           src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
    isFromTile = false;
  }
  else
    return nullptr;

  SmallVector<Value, 16> opers;

  auto idTy = IntegerType::get(ctx, 32);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    opers.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  }

  xilinx::air::HerdOp launch = op->getParentOfType<xilinx::air::HerdOp>();
  if (!launch) {

    AffineForOp afo = op->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd"))
      afo = afo->getParentOfType<AffineForOp>();
    if (!afo) return nullptr;
    opers.push_back(afo.getInductionVar());

    afo = afo->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd"))
      afo = afo->getParentOfType<AffineForOp>();
    if (!afo) return nullptr;
    opers.push_back(afo.getInductionVar());
  }
  else {
    auto tileIds = launch.getIds();
    opers.push_back(tileIds[0]);
    opers.push_back(tileIds[1]);
  }

  SmallVector<Value, 4> deps;
  for (auto o : operands) {
    if (o.getType().isa<xilinx::airrt::EventType>())
      deps.push_back(o);
    else
      opers.push_back(o);
  }
  if (deps.size())
    rewriter.create<xilinx::airrt::WaitAllOp>(loc, SmallVector<Type, 1>{}, deps);

  if (isFromTile) {
    opers.erase(opers.begin() + 4);
    for (unsigned int dim = 0; dim<dmaif.getNumDims(); dim++)
      opers.erase(opers.begin() + dmaif.getNumDims()+4);
  }
  else {
    opers.erase(opers.begin() + 3);
    for (unsigned int dim = 0; dim<dmaif.getNumDims(); dim++)
      opers.erase(opers.begin() + 4);
  }

  for (int i=0,e=opers.size(); i<e; i++) {
    if (opers[i].getType().isa<IndexType>()) {
      opers[i] = rewriter.create<IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), opers[i]);
    }
  }

  SmallVector<Type, 1> tys;
  if (op->getNumResults())
    tys.push_back(xilinx::airrt::EventType::get(ctx));

  Operation* airrtOp = nullptr;
  if (dmaif.getNumDims() == 1)
    airrtOp = rewriter.create<xilinx::airrt::DmaMemcpyOp>(loc, tys, opers);
  else if (dmaif.getNumDims() == 2)
    airrtOp = rewriter.create<xilinx::airrt::DmaMemcpy2dOp>(loc, tys, opers);
  else if (dmaif.getNumDims() == 4)
    airrtOp = rewriter.create<xilinx::airrt::DmaMemcpy4dOp>(loc, tys, opers);
  if (airrtOp)
    rewriter.replaceOp(op, airrtOp->getResults());
  return airrtOp;
}

class AIRDmaMemcpyToAIRRtConversion : public OpConversionPattern<xilinx::air::DmaMemcpyOp> {
public:
  using OpConversionPattern<xilinx::air::DmaMemcpyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::DmaMemcpyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    auto call = convertDmaMemcpyToAirRt(op, operands, rewriter);
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpy2dToAIRRtConversion : public OpConversionPattern<xilinx::air::DmaMemcpy2dOp> {
public:
  using OpConversionPattern<xilinx::air::DmaMemcpy2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::DmaMemcpy2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto call = convertDmaMemcpyToAirRt(op, operands, rewriter);
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpy4dToAIRRtConversion : public OpConversionPattern<xilinx::air::DmaMemcpy4dOp> {
public:
  using OpConversionPattern<xilinx::air::DmaMemcpy4dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::DmaMemcpy4dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto call = convertDmaMemcpyToAirRt(op, operands, rewriter);
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpyNdToAIRRtConversion : public OpConversionPattern<xilinx::air::DmaMemcpyNdOp> {
public:
  using OpConversionPattern<xilinx::air::DmaMemcpyNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    SmallVector<Value, 4> deps;
    for (auto o : adaptor.getOperands())
      if (o.getType().isa<xilinx::airrt::EventType>())
        deps.push_back(o);
    if (deps.size())
      rewriter.create<xilinx::airrt::WaitAllOp>(
          op->getLoc(), xilinx::airrt::EventType::get(op->getContext()), deps);

    MemRefType src = op.getSrcMemref().getType().cast<MemRefType>();
    MemRefType dst = op.getDstMemref().getType().cast<MemRefType>();
    bool isFromTile = false;
    bool isFullMemcpy = false;
    if (src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
        dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3) {
      isFromTile = true;
    } else if (dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
               src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3) {
      isFromTile = false;
    } else if (src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
               dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      isFromTile = true;
    } else if (dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1 &&
               src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      isFromTile = false;
    } else if (src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3 &&
               dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      isFullMemcpy = true;
    } else if (dst.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L3 &&
               src.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
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
        opers.push_back(
            rewriter.create<arith::ConstantOp>(loc, idTy, IntegerAttr::get(idTy, 0)));
      }

      xilinx::air::HerdOp launch =
          op->getParentOfType<xilinx::air::HerdOp>();
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
      opers[1] = rewriter.create<IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[1]);
      opers[2] = rewriter.create<IndexCastOp>(
          op->getLoc(), IntegerType::get(op->getContext(), 64), opers[2]);

      if (isFromTile)
        opers.push_back(op.getDstMemref());
      else
        opers.push_back(op.getSrcMemref());
    }
    else {
      opers.push_back(op.getDstMemref());
      opers.push_back(op.getSrcMemref());
    }
    auto i64Ty = rewriter.getI64Type();
    auto zero =
        rewriter.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 1));

    SmallVector<Value, 4> offsets(4, zero);
    SmallVector<Value, 4> lengths(4, one);
    SmallVector<Value, 3> strides(3, zero);

    int idx = 4 - src.getRank();
    for (auto o : isFromTile ? op.dst_offsets() : op.src_offsets())
      offsets[idx++] = rewriter.create<IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), o);
    idx = 4 - dst.getRank();
    auto op_strides = isFromTile ? op.dst_strides() : op.src_strides();
    if (op_strides.size())
      for (auto o : op_strides.drop_back())
        strides[idx++] = rewriter.create<IndexCastOp>(
            op->getLoc(), IntegerType::get(ctx, 64), o);
    idx = 4 - src.getRank();
    for (auto o : isFromTile ? op.dst_sizes() : op.src_sizes())
      lengths[idx++] = rewriter.create<IndexCastOp>(
          op->getLoc(), IntegerType::get(ctx, 64), o);

    opers.append(offsets);
    opers.append(lengths);
    opers.append(strides);

    Operation *airrtOp = nullptr;
    SmallVector<Type, 1> tys;
      if (op->getNumResults())
        tys.push_back(xilinx::airrt::EventType::get(ctx));
    if (isFullMemcpy) {
      airrtOp = rewriter.create<xilinx::airrt::MemcpyNdOp>(loc, tys, opers);
    } else {
      airrtOp = rewriter.create<xilinx::airrt::DmaMemcpyNdOp>(loc, tys, opers);
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
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto alloc = cast<memref::AllocOp>(op);
    auto type = alloc.getType();
    if (type.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      rewriter.replaceOpWithNewOp<xilinx::airrt::AllocOp>(op, type);
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
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto dealloc = cast<memref::DeallocOp>(op);
    auto type = dealloc.memref().getType().cast<MemRefType>();
    if (type.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L2) {
      rewriter.replaceOpWithNewOp<xilinx::airrt::DeallocOp>(op, SmallVector<Type>{}, op->getOperands());
      return success();
    }
    return failure();
  }
};

LogicalResult lowerAirRegions(Operation *op) {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  SmallVector<Operation *, 8> erased;
  module->walk([&](xilinx::air::RegionOp rop) {
    auto &bb = rop.body().front();
    unsigned idx = 0;
    for (auto &arg : bb.getArguments()) {
      arg.replaceAllUsesWith(rop.getOperand(idx));
      idx++;
    }
    if (rop.getNumResults() > 0) {
      OpBuilder builder(rop);
      auto w = builder.create<xilinx::air::WaitAllOp>(
          op->getLoc(), xilinx::air::AsyncTokenType::get(rop->getContext()),
          rop.asyncDependencies());
      rop.getResult(0).replaceAllUsesWith(w.getResult(0));
    }
    rop.walk([&](xilinx::air::RegionTerminatorOp t) {
      int resultIdx = 1;
      for (auto r : t->getOperands())
        rop.getResult(resultIdx++).replaceAllUsesWith(r);
      erased.push_back(t);
    });
    rop->getBlock()->getOperations().splice(Block::iterator(rop),
                                            bb.getOperations());
    erased.push_back(rop);
  });
  for (auto a : erased)
    a->erase();
  return success();
}

class AIRLoweringPass : public AIRLoweringBase<AIRLoweringPass> {

public:

  AIRLoweringPass() = default;
  AIRLoweringPass(const AIRLoweringPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect,
                    xilinx::airrt::AIRRtDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter converter;
    converter.addConversion([&](Type type) -> Optional<Type> {
      // convert !air.async.token to !airrt.event
      if (auto t = type.dyn_cast<xilinx::air::AsyncTokenType>())
        return xilinx::airrt::EventType::get(context);
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

    target.addLegalDialect<
        LLVM::LLVMDialect, func::FuncDialect, arith::ArithmeticDialect,
        AffineDialect, scf::SCFDialect, linalg::LinalgDialect,
        memref::MemRefDialect, bufferization::BufferizationDialect,
        xilinx::airrt::AIRRtDialect>();

    // AIR RegionOp conversion
    if (failed(lowerAirRegions(module))) {
      emitError(UnknownLoc::get(context), "error lowering air.region\n");
      signalPassFailure();
    }

    // Replace the PipelineStageOps first, followed by the 
    // HerdPipelineOps, then run the rest of the patterns.
    // This avoids creating invalid intermediate code with respect
    // to the herd->pipeline->stages nesting requirements.

    // PipelineStageOp conversion
    RewritePatternSet air_pipe_stage_patterns(context);
    air_pipe_stage_patterns.insert<AIRPipeStageConversion>(context, AIRPipeStageConversion::LoweringType::AllocBuffer);
    if (failed(applyPartialConversion(module, target, std::move(air_pipe_stage_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline.stage\n");
      signalPassFailure();
    }

    // HerdPipelineOp conversion
    RewritePatternSet air_pipe_patterns(context);
    air_pipe_patterns.insert<AIRPipelineConversion, AIRPipelineGetConversion,
                             AIRPipelinePutConversion>(context);
    if (failed(applyPartialConversion(module, target, std::move(air_pipe_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline\n");
      signalPassFailure();
    }

    // DMA and HerdOp conversion
    RewritePatternSet air_patterns(context);

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L2);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
              (int)xilinx::air::MemorySpace::L2);
    });

    air_patterns.add<L2AllocToAIRRtConversion, L2DeallocToAIRRtConversion,
                     AIRPartitionConversion, AIRHerdConversion>(context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(air_patterns,
                                                                   converter);

    air_patterns.add<AIRDmaMemcpy2dToAIRRtConversion,
        AIRDmaMemcpy4dToAIRRtConversion, AIRDmaMemcpyNdToAIRRtConversion,
       AIRDmaMemcpyToAIRRtConversion, AIRWaitAllToAIRRtConversion>(converter, context);

    if (failed(applyPartialConversion(module, target, std::move(air_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }
  }
};

class AIRPipelineToAffinePass : public AIRPipelineToAffineBase<AIRPipelineToAffinePass> {

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

    target.addLegalDialect<
        LLVM::LLVMDialect, func::FuncDialect, arith::ArithmeticDialect,
        AffineDialect, scf::SCFDialect, linalg::LinalgDialect,
        memref::MemRefDialect, bufferization::BufferizationDialect,
        xilinx::airrt::AIRRtDialect, xilinx::air::airDialect>();

    target.addIllegalOp<xilinx::air::PipelineStageOp, xilinx::air::PipelineYieldOp>();

    // PipelineStageOp conversion
    RewritePatternSet air_pipe_stage_patterns(context);
    auto loweringType = AIRPipeStageConversion::LoweringType::PipelineGetPut;
    if (clLoweringType == "buffer")
      loweringType = AIRPipeStageConversion::LoweringType::AllocBuffer;
    air_pipe_stage_patterns.insert<AIRPipeStageConversion>(context,
                                                           loweringType);
    if (failed(applyPartialConversion(module, target, std::move(air_pipe_stage_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline.stage\n");
      signalPassFailure();
    }

    SmallVector<Operation *, 8> pipelines;
    module.walk([&](xilinx::air::HerdPipelineOp p) {
      pipelines.push_back(p);
    });

    for (auto p : pipelines) {
      auto pipeOp = cast<xilinx::air::HerdPipelineOp>(p);
      OpBuilder b(p);
      Block &bb = pipeOp.body().front();
      BlockAndValueMapping remap;
      bb.getTerminator()->erase();
      for (auto &o : bb)
        b.clone(o, remap);
      p->erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLoweringPass() {
  return std::make_unique<AIRLoweringPass>();
}

std::unique_ptr<mlir::Pass> createAIRPipelineToAffinePass() {
  return std::make_unique<AIRPipelineToAffinePass>();
}

} // namespace air
} // namespace xilinx
