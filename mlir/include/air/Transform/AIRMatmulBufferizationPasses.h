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

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeOutputL2Pass();
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeOutputL2Pass(
    const AIRMatmulBufferizeOutputL2Options &);

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1OutputPass();
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1OutputPass(
    const AIRMatmulBufferizeL1OutputOptions &);

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1InputsPass();
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1InputsPass(
    const AIRMatmulBufferizeL1InputsOptions &);

std::unique_ptr<mlir::Pass> createAIRMatmulCleanupBufferizePass();

std::unique_ptr<mlir::Pass> createAIRMatmulFusePingpongLoopsPass();

std::unique_ptr<mlir::Pass> createAIRMatmulFuseOutputTruncfPass();

std::unique_ptr<mlir::Pass> createAIRHoistStaticAllocPass();

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_BUFFERIZATION_PASSES_H
