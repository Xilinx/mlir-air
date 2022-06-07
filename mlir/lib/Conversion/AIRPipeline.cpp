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
        rewriter.create<memref::TensorStoreOp>(yield.getLoc(), o, buf);
        rewriter.setInsertionPointAfter(aif);
        bufs.push_back(
            rewriter.create<bufferization::ToTensorOp>(aif.getLoc(), buf)
                .getResult());
      }
    }
    rewriter.replaceOp(stage, bufs);
  }

  // Clone the region into the affine.if while remapping the args
  BlockAndValueMapping remap;
  for (int i = 0, e = stageBlock.getNumArguments(); i < e; i++)
    remap.map(stageBlock.getArgument(i), operands[i]);

  rewriter.setInsertionPoint(aif);
  auto idVal = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), id);
  remap.map(dir ? x : y, idVal);

  stage.body().cloneInto(&aif.getBodyRegion(), aif.getBodyRegion().begin(),
                         remap);
  rewriter.eraseBlock(&aif.getBodyRegion().back());

  // replace the pipeline.yield with affine.yield
  rewriter.eraseOp(aif.getBodyRegion().front().getTerminator());
  rewriter.setInsertionPointToEnd(&aif.getBodyRegion().front());
  rewriter.create<AffineYieldOp>(aif.getLoc());

  return success();
}

} // namespace air
} // namespace xilinx