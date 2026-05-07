//===- AIRMatmulPackAndTranspose.h ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_PACK_AND_TRANSPOSE_H
#define AIR_MATMUL_PACK_AND_TRANSPOSE_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulPackAndTransposePass();
std::unique_ptr<mlir::Pass>
createAIRMatmulPackAndTransposePass(const AIRMatmulPackAndTransposeOptions &);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_PACK_AND_TRANSPOSE_H
