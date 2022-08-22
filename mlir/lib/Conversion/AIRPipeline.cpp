#include "air/Conversion/AIRPipeline.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-pipeline-conversion"

using namespace mlir;

namespace xilinx {
namespace air {

LogicalResult AIRPipeStageConversion::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  xilinx::air::HerdPipelineOp pipeline =
      op->getParentOfType<xilinx::air::HerdPipelineOp>();

  auto direction = pipeline->getAttrOfType<StringAttr>("direction");

  xilinx::air::HerdOp herd = op->getParentOfType<xilinx::air::HerdOp>();
  if (!herd) {
    LLVM_DEBUG(llvm::errs() << "Failed to find herd op for air.pipeline\n");
    return failure();
  }

  Value x = herd.getIds()[0];
  Value y = herd.getIds()[1];

  auto ctx = op->getContext();
  auto stage = cast<xilinx::air::PipelineStageOp>(op);

  // Create an affine.if to contain the code for this pipeline stage.
  unsigned id = stage.getStageId();

  bool dir = (direction.str() == "horiz");

  SmallVector<AffineExpr, 2> constraints{getAffineDimExpr(dir ? 0 : 1, ctx) -
                                             getAffineConstantExpr(id, ctx),
                                         getAffineDimExpr(dir ? 1 : 0, ctx)};
  SmallVector<bool, 2> eqflags{true, false};
  auto int_set = IntegerSet::get(2, 0, constraints, eqflags);
  SmallVector<Value, 2> int_set_args{x, y};
  AffineIfOp aif = rewriter.create<AffineIfOp>(stage->getLoc(), int_set,
                                               int_set_args, false);

  auto &stageBlock = stage.body().front();
  auto &yield = stageBlock.getOperations().back();

  if (loweringType == LoweringType::AllocBuffer) {
    // For each output of the pipeline stage, create a buffer + store
    SmallVector<Value, 4> bufs;
    for (auto o : yield.getOperands()) {
      if (RankedTensorType tt = o.getType().dyn_cast<RankedTensorType>()) {
        auto memrefTy = MemRefType::get(tt.getShape(), tt.getElementType());
        rewriter.setInsertionPoint(aif);
        auto buf = rewriter.create<memref::AllocOp>(op->getLoc(), memrefTy);
        rewriter.setInsertionPoint(&yield);
        auto to_memref = rewriter.create<bufferization::ToMemrefOp>(
            yield.getLoc(), buf.getType(), o);
        rewriter.create<memref::CopyOp>(yield.getLoc(), to_memref, buf);
        rewriter.setInsertionPointAfter(aif);
        bufs.push_back(
            rewriter.create<bufferization::ToTensorOp>(aif.getLoc(), buf)
                .getResult());
      }
    }
    rewriter.replaceOp(stage, bufs);
  } else if (loweringType == LoweringType::PipelineGetPut) {
    SmallVector<Value, 4> bufs;
    rewriter.setInsertionPoint(aif);
    for (auto o : yield.getOperands()) {
      if (RankedTensorType tt = o.getType().dyn_cast<RankedTensorType>()) {
        rewriter.setInsertionPoint(&yield);
        auto idValPlus =
            rewriter.create<arith::ConstantIndexOp>(op->getLoc(), id + 1);
        rewriter.create<xilinx::air::PipelinePutOp>(
            yield.getLoc(), dir ? idValPlus : x, dir ? y : idValPlus, o);
        bufs.push_back(o);
      }
    }
    rewriter.replaceOp(stage, bufs);
  }

  // Clone the region into the affine.if while remapping the args
  BlockAndValueMapping remap;
  rewriter.setInsertionPoint(aif);
  auto idVal = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), id);
  remap.map(dir ? x : y, idVal);

  for (int i = 0, e = stageBlock.getNumArguments(); i < e; i++) {
    if (loweringType == LoweringType::AllocBuffer) {
      remap.map(stageBlock.getArgument(i), operands[i]);
    } else if (loweringType == LoweringType::PipelineGetPut) {
      auto idValMinus =
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), id - 1);
      rewriter.setInsertionPointToStart(&aif.getBodyRegion().front());
      auto get = rewriter.create<xilinx::air::PipelineGetOp>(
          stage->getLoc(), operands[i].getType(), dir ? idValMinus : x,
          dir ? y : idValMinus);
      remap.map(stageBlock.getArgument(i), get.getResult(0));
    }
  }

  auto &body_region = aif.getBodyRegion();
  stage.body().cloneInto(&body_region, body_region.begin(), remap);
  body_region.back().getOperations().back().erase();
  body_region.front().getOperations().splice(
      body_region.front().getOperations().begin(),
      body_region.back().getOperations());
  rewriter.eraseBlock(&body_region.back());

  // replace the pipeline.yield with affine.yield
  rewriter.eraseOp(body_region.front().getTerminator());
  rewriter.setInsertionPointToEnd(&body_region.front());
  rewriter.create<AffineYieldOp>(aif.getLoc());

  return success();
}

} // namespace air
} // namespace xilinx