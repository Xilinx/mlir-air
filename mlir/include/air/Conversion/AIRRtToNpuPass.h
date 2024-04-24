//===- AIRRtToNpuPass.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_TO_NPU_H
#define AIRRT_TO_NPU_H

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace airrt {

using namespace mlir;

std::unique_ptr<mlir::Pass> createAIRRtToNpuPass();

} // namespace airrt
} // namespace xilinx

#endif // AIRRT_TO_NPU_H
