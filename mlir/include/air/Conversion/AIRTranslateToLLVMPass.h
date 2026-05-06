//===- AIRTranslateToLLVMPass.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_AIR_TRANSLATE_TO_LLVM_PASS_H
#define AIR_CONVERSION_AIR_TRANSLATE_TO_LLVM_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRTranslateToLLVMPass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_AIR_TRANSLATE_TO_LLVM_PASS_H
