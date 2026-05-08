//===- AIRMatmulTilePasses.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Free-function bodies invoked by the air-matmul-codegen orchestrator:
// launch-tile, tile-k-and-fuse-packs, tile-cores, and prologue/epilogue
// tiling. Each drives a discrete tiling step on the packed matmul (and,
// where applicable, fuses the LHS/RHS pack producers into the new loop).
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_TILE_PASSES_H
#define AIR_MATMUL_TILE_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace air {

mlir::LogicalResult
runTileLaunchTileImpl(mlir::func::FuncOp f, llvm::ArrayRef<int64_t> tileSizes,
                      llvm::StringRef launchTileForallMarker,
                      mlir::RewriterBase &rewriter);

mlir::LogicalResult runTileKAndFusePacksImpl(
    mlir::func::FuncOp f, int64_t kTileFactor, int64_t kIterIndex,
    llvm::StringRef packedMatmulMarker, llvm::StringRef kReductionLoopMarker,
    llvm::StringRef lhsPackMarker, llvm::StringRef rhsPackMarker,
    llvm::StringRef lhsL2PackMarker, llvm::StringRef rhsL2PackMarker,
    mlir::RewriterBase &rewriter);

mlir::LogicalResult runTileCoresImpl(
    mlir::func::FuncOp f, llvm::ArrayRef<int64_t> tileSizes,
    llvm::StringRef packedMatmulMarker, llvm::StringRef lhsPackInKMarker,
    llvm::StringRef rhsPackInKMarker, llvm::StringRef computeForallMarker,
    llvm::StringRef matmulComputeMarker, llvm::StringRef lhsL1PackMarker,
    llvm::StringRef rhsL1PackMarker, mlir::RewriterBase &rewriter);

mlir::LogicalResult runPrologueEpilogueImpl(
    mlir::func::FuncOp f, llvm::ArrayRef<int64_t> prologueTileSizes,
    llvm::ArrayRef<int64_t> epilogueTileSizes,
    llvm::ArrayRef<int64_t> fillIteratorInterchange,
    llvm::StringRef initFillMarker, llvm::StringRef prologueForallMarker,
    llvm::StringRef epilogueForallMarker, bool hoistStaticAllocFirst,
    mlir::RewriterBase &rewriter);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_TILE_PASSES_H
