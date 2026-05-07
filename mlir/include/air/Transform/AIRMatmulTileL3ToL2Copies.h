//===- AIRMatmulTileL3ToL2Copies.h ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_TILE_L3_TO_L2_COPIES_H
#define AIR_MATMUL_TILE_L3_TO_L2_COPIES_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulTileL3ToL2CopiesPass();
std::unique_ptr<mlir::Pass>
createAIRMatmulTileL3ToL2CopiesPass(const AIRMatmulTileL3ToL2CopiesOptions &);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_TILE_L3_TO_L2_COPIES_H
