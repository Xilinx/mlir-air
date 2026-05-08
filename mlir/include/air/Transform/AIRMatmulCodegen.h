//===- AIRMatmulCodegen.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// AIRMatmulCodegen: single public matmul codegen pass. Orchestrates the
// internal phases (launch tile, pack, K-tile, core tile, prologue/epilogue,
// bufferization, vectorize) in fixed order. Internal phases are exposed as
// free functions in their respective headers.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_CODEGEN_H
#define AIR_MATMUL_CODEGEN_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRMatmulCodegenPass();
std::unique_ptr<mlir::Pass>
createAIRMatmulCodegenPass(const AIRMatmulCodegenOptions &);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_CODEGEN_H
