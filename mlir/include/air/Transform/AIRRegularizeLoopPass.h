//===- AIRRegularizeLoopPass.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_REGULARIZE_LOOP_PASS_H
#define AIR_REGULARIZE_LOOP_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRRegularizeLoopPass();

} // namespace air
} // namespace xilinx

#endif // AIR_REGULARIZE_LOOP_PASS_H