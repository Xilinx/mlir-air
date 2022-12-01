//===- AIRLinalgCodegen.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_LINALG_CODEGEN_H
#define AIR_LINALG_CODEGEN_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLinalgCodegenPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LINALG_CODEGEN_H