// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.


#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include <vector>

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#define DEBUG_TYPE "air-lowering-pass"

using namespace mlir;
using namespace mlir::arith;
using namespace xilinx::air;

namespace {

class AIRHerdLaunchConversion : public ConversionPattern {
public:
  explicit AIRHerdLaunchConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::HerdLaunchOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    xilinx::air::HerdLaunchOp launch = cast<xilinx::air::HerdLaunchOp>(op);
    if (auto attr = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      auto herd_name = attr.getValue().str();
      rewriter.create<xilinx::airrt::HerdLoadOp>(op->getLoc(), rewriter.getI64Type(), herd_name);
    }
    auto herd_size = launch.getHerdSizeOperands();
    int64_t herd_size_x = cast<ConstantIndexOp>(herd_size.x.getDefiningOp()).value();
    int64_t herd_size_y = cast<ConstantIndexOp>(herd_size.y.getDefiningOp()).value();

    auto outer = rewriter.create<AffineForOp>(launch.getLoc(), 0, herd_size_x);
    auto outer_builder = OpBuilder::atBlockBegin(outer.getBody());
    auto inner = outer_builder.create<AffineForOp>(launch.getLoc(), 0, herd_size_y);

    outer->setAttr("air.herd_launch", StringAttr::get(op->getContext(), "outer"));
    inner->setAttr("air.herd_launch", StringAttr::get(op->getContext(), "inner"));

    launch.getHerdSize().x.replaceAllUsesWith(herd_size.x);
    launch.getHerdSize().y.replaceAllUsesWith(herd_size.y);
    launch.getTileIds().x.replaceAllUsesWith(outer.getInductionVar());
    launch.getTileIds().y.replaceAllUsesWith(inner.getInductionVar());

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

class AIRPipeStageConversion : public ConversionPattern {
public:
  explicit AIRPipeStageConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::PipelineStageOp::getOperationName(), 10, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    // xilinx::air::HerdPipelineOp pipeline =
    //   op->getParentOfType<xilinx::air::HerdPipelineOp>();
    xilinx::air::HerdLaunchOp launch = 
      op->getParentOfType<xilinx::air::HerdLaunchOp>();
    if (!launch) {
      LLVM_DEBUG(llvm::errs() << "Failed to find herd op for air.pipeline\n");
      return failure();
    }

    Value x = launch.getTileIds().x;
    Value y = launch.getTileIds().y;

    auto ctx = op->getContext();
    auto stage = cast<xilinx::air::PipelineStageOp>(op);

    // Create an affine.if to contain the code for this pipeline stage.
    unsigned id = stage.getStageId();
    SmallVector<AffineExpr, 2> constraints{getAffineDimExpr(0, ctx) -
                                            getAffineConstantExpr(id, ctx),
                                           getAffineDimExpr(1, ctx)};
    SmallVector<bool,2> eqflags{true, false};
    auto int_set = IntegerSet::get(2, 0, constraints, eqflags);
    SmallVector<Value, 2> int_set_args{x, y};
    AffineIfOp aif = rewriter.create<AffineIfOp>(stage->getLoc(), int_set,
                                                  int_set_args, false);

    auto &stageBlock = stage.body().front();
    auto &yield = stageBlock.getOperations().back();

    // For each output of the pipeline stage, create a buffer + store
    SmallVector<Value, 4> bufs;
    for (auto o: yield.getOperands()) {
      if (RankedTensorType tt = o.getType().dyn_cast<RankedTensorType>()) {
        auto memrefTy = MemRefType::get(tt.getShape(), tt.getElementType());
        rewriter.setInsertionPoint(aif);
        auto buf = rewriter.create<memref::AllocOp>(op->getLoc(), memrefTy);
        rewriter.setInsertionPoint(&yield);
        rewriter.create<memref::TensorStoreOp>(yield.getLoc(), o, buf);
        rewriter.setInsertionPointAfter(aif);
        bufs.push_back(
          rewriter.create<bufferization::ToTensorOp>(aif.getLoc(),buf).getResult());
      }
    }
    rewriter.replaceOp(stage, bufs);

    // Clone the region into the affine.if while remapping the args
    BlockAndValueMapping remap;
    for (int i = 0, e=stageBlock.getNumArguments(); i<e; i++)
      remap.map(stageBlock.getArgument(i), operands[i]);
    stage.body().cloneInto(&aif.getBodyRegion(),
                            aif.getBodyRegion().begin(), remap);
    rewriter.eraseBlock(&aif.getBodyRegion().back());

    // replace the pipeline.yield with affine.yield
    rewriter.eraseOp(aif.getBodyRegion().front().getTerminator());
    rewriter.setInsertionPointToEnd(&aif.getBodyRegion().front());
    rewriter.create<AffineYieldOp>(aif.getLoc());

    return success();
  }
};

static
CallOp convertDmaMemcpyToMemcpyFn(Operation *op, ArrayRef<Value > operands,
                                  ConversionPatternRewriter &rewriter, StringRef fnName) {
  auto loc = op->getLoc();

  SmallVector<Value, 16> callops;
  SmallVector<Type, 1> retTys{};

  auto idTy = IntegerType::get(op->getContext(), 32);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    callops.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  } else {
    callops.push_back(rewriter.create<arith::ConstantOp>(loc, idTy,
                                                  IntegerAttr::get(idTy, 0)));
  }

  xilinx::air::HerdLaunchOp launch = op->getParentOfType<xilinx::air::HerdLaunchOp>();
  if (!launch) {
    AffineForOp afo = op->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd_launch"))
      afo = afo->getParentOfType<AffineForOp>();
    if (!afo) return nullptr;
    callops.push_back(afo.getInductionVar());

    afo = afo->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd_launch"))
      afo = afo->getParentOfType<AffineForOp>();
    if (!afo) return nullptr;
    callops.push_back(afo.getInductionVar());
  }
  else {
    auto tileIds = launch.getTileIds();
    callops.push_back(tileIds.x);
    callops.push_back(tileIds.y);
  }

  for (auto o : operands)
    callops.push_back(o);

  SmallVector<Type, 16> tys;
  for (auto o : callops)
    tys.push_back(o.getType());

  auto fn = xilinx::air::getATenFn(op->getParentOfType<ModuleOp>(),
                                   fnName.str(), callops, {});
  auto call = rewriter.create<CallOp>(loc, retTys, SymbolRefAttr::get(fn), callops);
  rewriter.eraseOp(op);
  return call;
}

static
Operation* convertDmaMemcpyToAirRt(Operation *op, ArrayRef<Value > operands,
                                   ConversionPatternRewriter &rewriter) {
  auto dmaif = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op);
  auto loc = op->getLoc();

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

  auto idTy = IntegerType::get(op->getContext(), 32);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    opers.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  } else {
    opers.push_back(rewriter.create<arith::ConstantOp>(loc, idTy,
                                                IntegerAttr::get(idTy, 0)));
  }

  xilinx::air::HerdLaunchOp launch = op->getParentOfType<xilinx::air::HerdLaunchOp>();
  if (!launch) {

    AffineForOp afo = op->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd_launch"))
      afo = afo->getParentOfType<AffineForOp>();
    if (!afo) return nullptr;
    opers.push_back(afo.getInductionVar());

    afo = afo->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd_launch"))
      afo = afo->getParentOfType<AffineForOp>();
    if (!afo) return nullptr;
    opers.push_back(afo.getInductionVar());
  }
  else {
    auto tileIds = launch.getTileIds();
    opers.push_back(tileIds.x);
    opers.push_back(tileIds.y);
  }

  for (auto o : operands)
    opers.push_back(o);

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

  SmallVector<Type, 16> tys;
  for (int i=0,e=opers.size(); i<e; i++) {
    if (opers[i].getType().isa<IndexType>()) {
      opers[i] = rewriter.create<IndexCastOp>(
        op->getLoc(), opers[i], IntegerType::get(op->getContext(), 64));
    }
  }

  Operation* airrtOp = nullptr;
  if (dmaif.getNumDims() == 1)
    airrtOp = rewriter.create<xilinx::airrt::DmaMemcpyOp>(loc, tys, opers);
  else if (dmaif.getNumDims() == 2)
    airrtOp = rewriter.create<xilinx::airrt::DmaMemcpy2dOp>(loc, tys, opers);
  else if (dmaif.getNumDims() == 4)
    airrtOp = rewriter.create<xilinx::airrt::DmaMemcpy4dOp>(loc, tys, opers);
  rewriter.eraseOp(op);
  return airrtOp;
}

class AIRDmaMemcpyToAIRRtConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpyToAIRRtConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpyOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto call = convertDmaMemcpyToAirRt(op, operands, rewriter);
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpy2dToAIRRtConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpy2dToAIRRtConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpy2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto call = convertDmaMemcpyToAirRt(op, operands, rewriter);
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpy4dToAIRRtConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpy4dToAIRRtConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpy4dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto call = convertDmaMemcpyToAirRt(op, operands, rewriter);
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpyNdToAIRRtConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpyNdToAIRRtConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpyNdOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dmaif = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op);
    auto dmaOp = dyn_cast<xilinx::air::DmaMemcpyNdOp>(op);
    auto loc = op->getLoc();

    MemRefType src = dmaif.getSrcMemref().getType().cast<MemRefType>();
    MemRefType dst = dmaif.getDstMemref().getType().cast<MemRefType>();
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

      xilinx::air::HerdLaunchOp launch =
          op->getParentOfType<xilinx::air::HerdLaunchOp>();
      if (!launch) {

        AffineForOp afo = op->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd_launch"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());

        afo = afo->getParentOfType<AffineForOp>();
        while (afo && !afo->getAttr("air.herd_launch"))
          afo = afo->getParentOfType<AffineForOp>();
        if (!afo)
          return failure();
        opers.push_back(afo.getInductionVar());
      } else {
        auto tileIds = launch.getTileIds();
        opers.push_back(tileIds.x);
        opers.push_back(tileIds.y);
      }
      opers[1] = rewriter.create<IndexCastOp>(
          op->getLoc(), opers[1], IntegerType::get(op->getContext(), 64));
      opers[2] = rewriter.create<IndexCastOp>(
          op->getLoc(), opers[2], IntegerType::get(op->getContext(), 64));

      if (isFromTile)
        opers.push_back(dmaif.getDstMemref());
      else
        opers.push_back(dmaif.getSrcMemref());
    }
    else {
      opers.push_back(dmaif.getDstMemref());
      opers.push_back(dmaif.getSrcMemref());
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
    for (auto o : isFromTile ? dmaOp.dst_offsets() : dmaOp.src_offsets())
      offsets[idx++] = rewriter.create<IndexCastOp>(
          op->getLoc(), o, IntegerType::get(op->getContext(), 64));
    idx = 4 - dst.getRank();
    for (auto o : isFromTile ? dmaOp.dst_strides().drop_back()
                            : dmaOp.src_strides().drop_back())
      strides[idx++] = rewriter.create<IndexCastOp>(
          op->getLoc(), o, IntegerType::get(op->getContext(), 64));
    idx = 4 - src.getRank();
    for (auto o : isFromTile ? dmaOp.dst_sizes() : dmaOp.src_sizes())
      lengths[idx++] = rewriter.create<IndexCastOp>(
          op->getLoc(), o, IntegerType::get(op->getContext(), 64));

    opers.append(offsets);
    opers.append(lengths);
    opers.append(strides);

    SmallVector<Type, 1> tys;
    if (isFullMemcpy) {
      rewriter.create<xilinx::airrt::MemcpyNdOp>(loc, tys, opers);
    } else {
      rewriter.create<xilinx::airrt::DmaMemcpyNdOp>(loc, tys, opers);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRDmaMemcpyToMemcpyConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpyToMemcpyConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpyOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto call = convertDmaMemcpyToMemcpyFn(op, operands, rewriter, "air_memcpy");
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpy2dToMemcpyConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpy2dToMemcpyConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpy2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto call = convertDmaMemcpyToMemcpyFn(op, operands, rewriter, "air_memcpy2d");
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRDmaMemcpy4dToMemcpyConversion : public ConversionPattern {
public:
  explicit AIRDmaMemcpy4dToMemcpyConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::DmaMemcpy4dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto call = convertDmaMemcpyToMemcpyFn(op, operands, rewriter, "air_memcpy4d");
    if (call)
      return success();
    else
      return failure();
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

class AIRLoweringPass : public AIRLoweringBase<AIRLoweringPass> {

  MemRefType convertTensorType(TensorType tensor) {
    return mlir::MemRefType::get(tensor.getShape(), tensor.getElementType(), {}, 0);
  }

public:

  AIRLoweringPass() = default;
  AIRLoweringPass(const AIRLoweringPass &pass) {}

  Option<bool>
  lowerToCpu{*this, "lower-to-cpu",
              llvm::cl::desc("Lower for cpu emulation"),
              llvm::cl::init(false)};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect,
                    xilinx::airrt::AIRRtDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect,
                          StandardOpsDialect,
                          arith::ArithmeticDialect,
                          AffineDialect,
                          scf::SCFDialect,
                          linalg::LinalgDialect,
                          memref::MemRefDialect,
                          bufferization::BufferizationDialect,
                          xilinx::airrt::AIRRtDialect>();

    // Replace the PipelineStageOps first, followed by the 
    // HerdPipelineOps, then run the rest of the patterns.
    // This avoids creating invalid intermediate code with respect
    // to the herd->pipeline->stages nesting requirements.

    // PipelineStageOp conversion
    OwningRewritePatternList air_pipe_stage_patterns(context);
    air_pipe_stage_patterns.insert<AIRPipeStageConversion>(context);
    if (failed(applyPartialConversion(module, target, std::move(air_pipe_stage_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline.stage\n");
      signalPassFailure();
    }

    // HerdPipelineOp conversion
    OwningRewritePatternList air_pipe_patterns(context);
    air_pipe_patterns.insert<AIRPipelineConversion>(context);
    if (failed(applyPartialConversion(module, target, std::move(air_pipe_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline\n");
      signalPassFailure();
    }

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L2);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
              (int)xilinx::air::MemorySpace::L2);
    });

    // DMA and HerdLaunchOp conversion
    OwningRewritePatternList air_patterns(context);

    if (lowerToCpu) {
      // lower to cpu memcpy
      air_patterns.insert<AIRDmaMemcpyToMemcpyConversion,
                          AIRDmaMemcpy2dToMemcpyConversion,
                          AIRDmaMemcpy4dToMemcpyConversion>(context);
    }
    else {
      // lower to air runtime
      air_patterns.insert<
          AIRDmaMemcpyToAIRRtConversion, AIRDmaMemcpy2dToAIRRtConversion,
          AIRDmaMemcpy4dToAIRRtConversion, AIRDmaMemcpyNdToAIRRtConversion,
          L2AllocToAIRRtConversion, L2DeallocToAIRRtConversion>(context);
    }

    air_patterns.insert<AIRHerdLaunchConversion>(context);

    TypeConverter typeConverter;
    mlir::populateFuncOpTypeConversionPattern(air_patterns,
                                              typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(air_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }
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
