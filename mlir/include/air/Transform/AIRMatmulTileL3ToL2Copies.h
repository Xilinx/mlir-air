//===- AIRMatmulTileL3ToL2Copies.h ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Free-function body for the former `air-matmul-tile-l3-to-l2-copies` pass.
// Now invoked from `air-matmul-bufferize-output-l2` when its
// `do-tile-l3-to-l2-copies` option is set.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_TILE_L3_TO_L2_COPIES_H
#define AIR_MATMUL_TILE_L3_TO_L2_COPIES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace air {

mlir::LogicalResult
runTileL3ToL2CopiesImpl(mlir::func::FuncOp func, int64_t kL2Tile,
                        llvm::StringRef copyAMarker = "copy_a_loop",
                        llvm::StringRef copyBMarker = "copy_b_loop");

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_TILE_L3_TO_L2_COPIES_H
