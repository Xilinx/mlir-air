//===- AIRMatmulPackAndTranspose.h ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_PACK_AND_TRANSPOSE_H
#define AIR_MATMUL_PACK_AND_TRANSPOSE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace air {

mlir::LogicalResult runPackAndTransposeImpl(
    mlir::func::FuncOp f, llvm::ArrayRef<int64_t> packSizes,
    llvm::ArrayRef<int64_t> lhsOuter, llvm::ArrayRef<int64_t> lhsInner,
    llvm::ArrayRef<int64_t> rhsOuter, llvm::ArrayRef<int64_t> rhsInner,
    llvm::ArrayRef<int64_t> accOuter, llvm::ArrayRef<int64_t> accInner,
    llvm::StringRef packedMatmulMarker, bool doBufferizeL1Output,
    int64_t bufferizeL1OutputMemorySpace, mlir::RewriterBase &rewriter);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_PACK_AND_TRANSPOSE_H
