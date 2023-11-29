//===- AIRToAsyncPass.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TO_ASYNC_PASS_H
#define AIR_TO_ASYNC_PASS_H

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToAsyncPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_ASYNC_PASS_H
