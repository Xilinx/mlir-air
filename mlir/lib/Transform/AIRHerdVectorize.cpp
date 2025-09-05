//===- AIRHerdVectorize.cpp ----------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRHerdVectorize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/PassDetail.h"

using namespace mlir;

namespace {
/// This is an helper only to call vectorize via a pattern inside of
/// AIRHerdVectorizePass::runOnOperation.
struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context,
                                bool vectorizeExtract = false,
                                bool flattenConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorizeNDExtract(vectorizeExtract),
        flatten1DDepthwiseConv(flattenConv) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::hasVectorizationImpl(op))
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Op, cannot vectorize");
    FailureOr<linalg::VectorizationResult> vectorResults =
        linalg::vectorize(rewriter, op, /*inputVectorSizes=*/{},
                          /*inputScalableVecDims=*/{}, vectorizeNDExtract,
                          flatten1DDepthwiseConv);
    if (failed(vectorResults))
      return failure();
    rewriter.replaceOp(op, vectorResults->replacements);
    return success();
  }

private:
  /// Controls whether to vectorize `tensor.extract` when the input tensor is
  /// rank >= 2.
  bool vectorizeNDExtract = false;
  /// Controls whether to "flatten" the channel dimension when vectorising 1D
  /// depthwise convolutions. This should lead to bette vectorization for
  /// tensors with a low number of channel dimensions.
  bool flatten1DDepthwiseConv = false;
};

} // namespace

namespace xilinx {
namespace air {

class AIRHerdVectorizePass
    : public air::impl::AIRHerdVectorizePassBase<AIRHerdVectorizePass> {

public:
  AIRHerdVectorizePass() = default;
  AIRHerdVectorizePass(const AIRHerdVectorizePass &pass) {}
  AIRHerdVectorizePass(bool vectorizeNdExtract, bool flatten1dDepthwiseConv,
                       bool disableTransferPermutationMapLoweringPatterns,
                       bool disableMultiReductionToContractPatterns,
                       bool vectorizePadding)
      : vectorizeNdExtract(vectorizeNdExtract),
        flatten1dDepthwiseConv(flatten1dDepthwiseConv),
        disableTransferPermutationMapLoweringPatterns(
            disableTransferPermutationMapLoweringPatterns),
        disableMultiReductionToContractPatterns(
            disableMultiReductionToContractPatterns),
        vectorizePadding(vectorizePadding) {}

  StringRef getArgument() const override { return "air-herd-vectorize"; }
  StringRef getDescription() const override {
    return "Vectorize operations inside air.herd operations";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

private:
  bool vectorizeNdExtract = false;
  bool flatten1dDepthwiseConv = false;
  bool disableTransferPermutationMapLoweringPatterns = false;
  bool disableMultiReductionToContractPatterns = false;
  bool vectorizePadding = false;
};

void AIRHerdVectorizePass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *ctx = &getContext();

  func.walk([&](air::HerdOp herdOp) {
    if (!herdOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      herdOp->emitOpError("requires isolated-from-above targets");
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    patterns.add<VectorizationPattern>(ctx, vectorizeNdExtract,
                                       flatten1dDepthwiseConv);

    if (!disableTransferPermutationMapLoweringPatterns)
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

    if (!disableMultiReductionToContractPatterns)
      vector::populateVectorReductionToContractPatterns(patterns);

    vector::populateSinkVectorOpsPatterns(patterns);

    patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                 linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                         /*benefit=*/2);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

    patterns.add<linalg::CopyVectorizationPattern>(ctx);

    if (vectorizePadding) {
      linalg::populatePadOpVectorizationPatterns(patterns);
      // This creates an alternative path for lowering tensor.pad - by
      // decomposing it into e.g. linalg.fill.
      linalg::populateDecomposePadPatterns(patterns);
    }
    vector::populateVectorStepLoweringPatterns(patterns);

    if (failed(applyPatternsGreedily(herdOp, std::move(patterns))))
      return signalPassFailure();
  });
}

std::unique_ptr<mlir::Pass> createAIRHerdVectorizePass() {
  return std::make_unique<AIRHerdVectorizePass>();
}

std::unique_ptr<mlir::Pass>
createAIRHerdVectorizePass(bool vectorizeNdExtract, bool flatten1dDepthwiseConv,
                           bool disableTransferPermutationMapLoweringPatterns,
                           bool disableMultiReductionToContractPatterns,
                           bool vectorizePadding) {
  return std::make_unique<AIRHerdVectorizePass>(
      vectorizeNdExtract, flatten1dDepthwiseConv,
      disableTransferPermutationMapLoweringPatterns,
      disableMultiReductionToContractPatterns, vectorizePadding);
}

} // namespace air
} // namespace xilinx
