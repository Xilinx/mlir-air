//===- AIRMatmulVectorizePasses.h -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M1a passes of the matmul codegen pipeline. See MATMUL_CODEGEN_PIPELINE_PLAN.md.
// These wrap (by copy) the C++ logic backing the existing transform.air.* ops
// in AIRLinalgCodegen.cpp, exposing it as ordinary func-level passes.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_VECTORIZE_PASSES_H
#define AIR_MATMUL_VECTORIZE_PASSES_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulTileForVectorizePass();
std::unique_ptr<mlir::Pass>
createAIRMatmulTileForVectorizePass(const AIRMatmulTileForVectorizeOptions &);

std::unique_ptr<mlir::Pass> createAIRFoldUnitExtentDimsPass();

std::unique_ptr<mlir::Pass> createAIREliminateRedundantVectorTransfersPass();

std::unique_ptr<mlir::Pass> createAIRFlattenForIterArgsPass();

std::unique_ptr<mlir::Pass> createAIRHoistLoopInvariantTransfersPass();

std::unique_ptr<mlir::Pass> createAIRHoistVectorTransferPointersPass();

std::unique_ptr<mlir::Pass> createAIRVectorCastForEmulationPass();
std::unique_ptr<mlir::Pass>
createAIRVectorCastForEmulationPass(const AIRVectorCastForEmulationOptions &);

std::unique_ptr<mlir::Pass> createAIRHoistCastPairsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_VECTORIZE_PASSES_H
