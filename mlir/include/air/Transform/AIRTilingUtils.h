//===- AIRTilingUtils.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===- AIRTilingUtils.h - AIR Loop tiling utilities
//------------------------===//
//
// This header file defines utility functions that are commonly used in passes,
// primarily AIR automatic loop tiling passes.
//===-----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace xilinx {
namespace air {
/// Identify valid and profitable bands of loops to tile. This is currently just
/// a temporary placeholder to test the mechanics of tiled code generation.
/// Returns all maximal outermost perfect loop nests that has been attached with
/// the given label to tile.
void getTileableBands(func::FuncOp f,
                      std::vector<SmallVector<AffineForOp, 6>> &bands,
                      const char *attrName, StringRef label);

/// Get the loop band that has been attached with the given label.
AffineForOp getLabel(AffineForOp root, StringRef label, const char *attrName);

} // namespace air
} // namespace xilinx
