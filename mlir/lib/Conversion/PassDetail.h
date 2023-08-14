//===- PassDetail.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_PASSDETAIL_H
#define AIR_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

using namespace mlir;

#define GEN_PASS_CLASSES
#include "air/Conversion/Passes.h.inc"

#ifdef BUILD_WITH_AIE
#define GEN_PASS_CLASSES
#include "air/Conversion/AIRToAIEPass.h.inc"
#endif

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_PASSDETAIL_H