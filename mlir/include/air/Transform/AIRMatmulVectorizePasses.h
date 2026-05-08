//===- AIRMatmulVectorizePasses.h -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M1a passes of the matmul codegen pipeline. See
// MATMUL_CODEGEN_PIPELINE_PLAN.md. These wrap (by copy) the C++ logic backing
// the existing transform.air.* ops in AIRLinalgCodegen.cpp, exposing it as
// ordinary func-level passes.
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

std::unique_ptr<mlir::Pass> createAIRFoldUnitExtentDimsPass();

mlir::LogicalResult runTileForVectorizeImpl(
    mlir::func::FuncOp f, llvm::ArrayRef<int64_t> matmulTileSizes,
    llvm::ArrayRef<int64_t> matmulUnrollTileSizes, int64_t matmulUnrollFactor,
    llvm::ArrayRef<int64_t> fillTileSizes, bool doPostBufferizeCleanupFirst,
    mlir::RewriterBase &rewriter);

mlir::LogicalResult runCodegenVecPrepImpl(
    mlir::func::FuncOp f, bool doFoldUnitExtentDims,
    bool doEliminateRedundantVectorTransfers,
    llvm::StringRef cast1TargetElementType,
    llvm::ArrayRef<int64_t> cast1InputIndices,
    llvm::ArrayRef<int64_t> cast1OutputIndices,
    llvm::StringRef cast2TargetElementType,
    llvm::ArrayRef<int64_t> cast2InputIndices,
    llvm::ArrayRef<int64_t> cast2OutputIndices,
    bool doHoistLoopInvariantTransfers, bool doFlattenForIterArgs,
    bool doHoistVectorTransferPointers, bool doHoistCastPairs,
    int64_t hoistCastPairsMaxIterations, mlir::RewriterBase &rewriter);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_VECTORIZE_PASSES_H
