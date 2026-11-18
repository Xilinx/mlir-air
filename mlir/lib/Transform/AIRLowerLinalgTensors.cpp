//===- AIRLowerLinalgTensors.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRLowerLinalgTensors.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-lower-linalg-tensors"

using namespace mlir;

struct LowerLinalgOpPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    if (succeeded(linalg::linalgOpToAffineLoops(rewriter, op))) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct AIRLowerLinalgTensors
    : public xilinx::air::impl::AIRLowerLinalgTensorsBase<
          AIRLowerLinalgTensors> {
  void runOnOperation() override;
};

void AIRLowerLinalgTensors::runOnOperation() {
  ModuleOp aie_module = getOperation();
  MLIRContext &context = getContext();

  bufferization::BufferizationOptions options;
  options.opFilter.allowDialect<linalg::LinalgDialect>();

  bufferization::BufferizationState state;
  if (failed(bufferizeOp(getOperation(), options, state)))
    signalPassFailure();

  RewritePatternSet patterns1(&context);
  bufferization::ToBufferOp::getCanonicalizationPatterns(patterns1, &context);
  (void)applyPatternsGreedily(aie_module, std::move(patterns1));

  RewritePatternSet patterns2(&context);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns2);
  patterns2.add<LowerLinalgOpPattern>(&context);
  (void)applyPatternsGreedily(aie_module, std::move(patterns2));
}

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLowerLinalgTensorsPass() {
  return std::make_unique<AIRLowerLinalgTensors>();
}

} // namespace air
} // namespace xilinx