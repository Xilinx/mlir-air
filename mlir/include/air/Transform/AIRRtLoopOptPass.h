//===- AIRRtLoopOptPass.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_LOOP_OPT_H
#define AIRRT_LOOP_OPT_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"

#include <memory>
#include <vector>

namespace xilinx {
namespace airrt {

using namespace mlir;

std::unique_ptr<mlir::Pass> createAIRRtLoopOptPass();

} // namespace airrt
} // namespace xilinx

#endif // AIRRT_LOOP_OPT_H