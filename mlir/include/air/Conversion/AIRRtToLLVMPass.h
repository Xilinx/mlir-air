//===- AIRRtToLLVMPass.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_TO_LLVM_H
#define AIRRT_TO_LLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToLLVMPass();

} // namespace airrt
} // namespace xilinx

#endif // AIRRT_TO_LLVM_H