//===- AIRMatmulTilePasses.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M2 Phase 4 / Phase 5: tile-k-and-fuse-packs and tile-cores. Drive the
// reduction-loop and per-core forall tiling of the packed matmul, plus
// fusion of the LHS/RHS L1 pack producers into the new loops. See
// MATMUL_CODEGEN_PIPELINE_PLAN.md.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_TILE_PASSES_H
#define AIR_MATMUL_TILE_PASSES_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulTileKAndFusePacksPass();
std::unique_ptr<mlir::Pass> createAIRMatmulTileKAndFusePacksPass(
    const AIRMatmulTileKAndFusePacksOptions &);

std::unique_ptr<mlir::Pass> createAIRMatmulTileCoresPass();
std::unique_ptr<mlir::Pass>
createAIRMatmulTileCoresPass(const AIRMatmulTileCoresOptions &);

std::unique_ptr<mlir::Pass> createAIRMatmulPrologueEpiloguePass();
std::unique_ptr<mlir::Pass> createAIRMatmulPrologueEpiloguePass(
    const AIRMatmulPrologueEpilogueOptions &);

std::unique_ptr<mlir::Pass> createAIRMatmulSetCodegenConfigPass();
std::unique_ptr<mlir::Pass> createAIRMatmulSetCodegenConfigPass(
    const AIRMatmulSetCodegenConfigOptions &);

std::unique_ptr<mlir::Pass> createAIRMatmulTileLaunchTilePass();
std::unique_ptr<mlir::Pass> createAIRMatmulTileLaunchTilePass(
    const AIRMatmulTileLaunchTileOptions &);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_TILE_PASSES_H
