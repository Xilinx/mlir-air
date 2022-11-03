//===- AIRHerdPlacementPass.h -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
#ifndef AIR_PLACE_HERDS_H
#define AIR_PLACE_HERDS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHerdPlacementPass();

} // namespace air
} // namespace xilinx

#endif // AIR_PLACE_HERD_H