//===- AIRLoweringPass.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TO_STD_H
#define AIR_TO_STD_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLoweringPass();
std::unique_ptr<mlir::Pass> createAIRPipelineToAffinePass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_STD_H