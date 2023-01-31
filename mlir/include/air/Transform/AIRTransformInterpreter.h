//===- AIRMiscPasses.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TRANSFORM_INTERPRETER_H
#define AIR_TRANSFORM_INTERPRETER_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRTransformInterpreterPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TRANSFORM_INTERPRETER_H