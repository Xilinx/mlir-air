//===- AIRToROCDLPass.h -------------------------------------------*- C++
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------===//

#ifndef CONVERT_TO_AIR_ROCDL
#define CONVERT_TO_AIR_ROCDL

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToROCDLPass();

} // namespace air
} // namespace xilinx
#endif // CONVERT_TO_AIR_ROCDL
