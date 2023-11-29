//===- AIRLowerLinalgTensors.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_LOWER_LINALG_TENSORS_H
#define AIR_LOWER_LINALG_TENSORS_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLowerLinalgTensorsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LOWER_LINALG_TENSORS_H