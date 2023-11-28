//===- AIRRtToIpuPass.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_TO_IPU_H
#define AIRRT_TO_IPU_H

// #include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace airrt {

using namespace mlir;
#define GEN_PASS_DEF_AIRRTTOIPU
#include "air/Conversion/Passes.h.inc"
std::unique_ptr<mlir::Pass> createAIRRtToIpuPass();

} // namespace airrt
} // namespace xilinx

#endif // AIRRT_TO_IPU_H
