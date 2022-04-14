// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"
#include "air/Transform/AIRLowerLinalgTensors.h"
#include "aie/AIEDialect.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-lower-linalg-tensors"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

// Remove tensor_load followed by buffer_cast
struct RemoveBufferCastPattern
    : public OpRewritePattern<bufferization::ToMemrefOp> {
  using OpRewritePattern<bufferization::ToMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToMemrefOp op,
                                PatternRewriter &rewriter) const override {
                                
    auto load = op.getOperand().getDefiningOp<bufferization::ToTensorOp>();
    if (!load)
      return failure();
      
    auto buffer = load.memref();
    if (!buffer)
      return failure();
    rewriter.replaceOp(op, buffer);
    return success();
  }
};

// Replace a pattern like this:
//  %1 = memref.alloc() : memref<32x32xi32>
//  linalg.copy(%2, %1) : memref<32x32xi32, 2>, memref<32x32xi32> 
//  use %1
// with this:
//  use %2
struct RemoveAllocCopyPattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {

    Value memref;
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    for (auto u : op->getUsers())
      if (auto copy = dyn_cast<linalg::CopyOp>(u)) {
        memref = copy.inputs()[0];
        rewriter.eraseOp(copy);
      }

    if (!memref)
      return failure();

    rewriter.replaceOp(op, memref);
    return success();
  }
};

// Rewrite of this pattern:
//   %1 = AIE.buffer(%0) {..} : memref<?, 2>
//   %2 = memref.alloc() : memref<?>
//   <use of %2>
//   %3 = memref.tensor_load %2 : memref<?>
//   memref.tensor_store %3, %1 : memref<?, 2>
// to this:
//   <use of %1>
struct RemoveTensorLoadStorePattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {

    auto alloc = op.getOperand().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    if (!op->hasOneUse())
      return failure();

    auto store = dyn_cast<memref::TensorStoreOp>(*op->user_begin());
    if (!store)
      return failure();

    rewriter.replaceOp(alloc, store.memref());
    rewriter.eraseOp(store);
    rewriter.eraseOp(op);

    return success();
  }
};

struct AIRLowerLinalgTensors : public AIRLowerLinalgTensorsBase<AIRLowerLinalgTensors> {
  void runOnOperation() override;
};

void AIRLowerLinalgTensors::runOnOperation() {
  ModuleOp aie_module = getOperation();
  MLIRContext &context = getContext();

  ConversionTarget target(context);
  bufferization::BufferizeTypeConverter typeConverter;
  target.addLegalDialect<AIE::AIEDialect, AffineDialect, math::MathDialect,
                         memref::MemRefDialect, func::FuncDialect,
                         arith::ArithmeticDialect>();
  target.addIllegalOp<linalg::InitTensorOp, tensor::ExtractSliceOp, tensor::InsertSliceOp>();

  // Mark all Linalg operations illegal as long as they work on tensors.
  auto isLegalOperation = [&](Operation *op) {
    return typeConverter.isLegal(op);
  };
  target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOperation);

  bufferization::BufferizationOptions options =
      bufferization::getPartialBufferizationOptions();
  options.allowDialectInFilter<linalg::LinalgDialect>();

  if (failed(bufferizeOp(getOperation(), options)))
    signalPassFailure();

  RewritePatternSet patterns1(&context);
  patterns1.add<RemoveBufferCastPattern,
              //RemoveAllocCopyPattern,
              RemoveTensorLoadStorePattern>(&context);
  (void)applyPatternsAndFoldGreedily(aie_module, std::move(patterns1));

  RewritePatternSet patterns2(&context);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns2);
  if (1/*lower to loops*/) {
    patterns2.add<linalg::LinalgLoweringPattern<linalg::GenericOp>>(
        &context, linalg::LinalgLoweringType::AffineLoops);
  } 
  // else lower to function call
  else {
    linalg::populateLinalgToStandardConversionPatterns(patterns2);
  }
  (void)applyPatternsAndFoldGreedily(aie_module, std::move(patterns2));
}

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLowerLinalgTensorsPass() {
  return std::make_unique<AIRLowerLinalgTensors>();
}

} // namespace air
} // namespace xilinx