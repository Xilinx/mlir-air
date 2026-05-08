//===- AIRMatmulBufferizationPasses.h ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M2 (Group A tail) passes: bufferization, post-bufferize cleanup, ping-pong
// loop fusion, and bf16-output truncf fusion. See MATMUL_CODEGEN_PIPELINE_PLAN.md.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_BUFFERIZATION_PASSES_H
#define AIR_MATMUL_BUFFERIZATION_PASSES_H

#include "air/Transform/PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeOutputL2Pass();
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeOutputL2Pass(
    const AIRMatmulBufferizeOutputL2Options &);

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1InputsPass();
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1InputsPass(
    const AIRMatmulBufferizeL1InputsOptions &);

// Free-function bodies for the now-internal pass impls. Called from
// option-driven steps in parametric passes (pack-and-transpose,
// prologue-epilogue, tile-for-vectorize, bufferize-output-l2).
mlir::LogicalResult
runFusePingpongLoopsImpl(mlir::func::FuncOp f,
                         mlir::RewriterBase &rewriter);
void runFuseOutputTruncfImpl(mlir::func::FuncOp f,
                             mlir::RewriterBase &rewriter);
void runHoistStaticAllocImpl(mlir::func::FuncOp f,
                             mlir::RewriterBase &rewriter);
mlir::LogicalResult
runBufferizeL1OutputImpl(mlir::func::FuncOp f, int64_t memorySpace,
                         llvm::StringRef packedMatmulMarker,
                         mlir::RewriterBase &rewriter);
mlir::LogicalResult
runPostBufferizeCleanupImpl(mlir::func::FuncOp f,
                            mlir::RewriterBase &rewriter);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_BUFFERIZATION_PASSES_H
