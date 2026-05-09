//===- AIRMatmulVectorizePasses.h -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Vectorization-prep free functions invoked by the air-matmul-codegen
// orchestrator: tile-for-vectorize and the vec-prep composite (eliminate-
// redundant-transfers, vector-cast-for-emulation, hoist-loop-invariant,
// flatten-for-iter-args, hoist-vector-transfer-pointers, hoist-cast-pairs).
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_VECTORIZE_PASSES_H
#define AIR_MATMUL_VECTORIZE_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace xilinx {
namespace air {

mlir::LogicalResult runTileForVectorizeImpl(
    mlir::func::FuncOp f, llvm::ArrayRef<int64_t> matmulTileSizes,
    llvm::ArrayRef<int64_t> matmulUnrollTileSizes, int64_t matmulUnrollFactor,
    llvm::ArrayRef<int64_t> fillTileSizes, bool doPostBufferizeCleanupFirst,
    mlir::RewriterBase &rewriter);

mlir::LogicalResult runCodegenVecPrepImpl(
    mlir::func::FuncOp f, llvm::StringRef cast1TargetElementType,
    llvm::ArrayRef<int64_t> cast1InputIndices,
    llvm::ArrayRef<int64_t> cast1OutputIndices,
    llvm::StringRef cast2TargetElementType,
    llvm::ArrayRef<int64_t> cast2InputIndices,
    llvm::ArrayRef<int64_t> cast2OutputIndices, bool doHoistCastPairs,
    int64_t hoistCastPairsMaxIterations, mlir::RewriterBase &rewriter);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_VECTORIZE_PASSES_H
