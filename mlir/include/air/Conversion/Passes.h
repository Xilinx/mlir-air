//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_PASSES_H
#define AIR_CONVERSION_PASSES_H

#include <memory>

// Core passes (always available)
#include "air/Conversion/AIRRtToLLVMPass.h"
#include "air/Conversion/AIRToAsyncPass.h"
#include "air/Conversion/ConvertToAIRPass.h"

// AIE-specific passes (only available when AIE is enabled)
#ifdef AIR_ENABLE_AIE
#include "air/Conversion/AIRLoweringPass.h"
#include "air/Conversion/AIRRtToNpuPass.h"
#include "air/Conversion/AIRToAIEPass.h"
#endif
namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace air {

void registerConversionPasses();

// GPU passes (always available)
std::unique_ptr<mlir::Pass> createAIRGPUOutliningPass();
std::unique_ptr<mlir::Pass> createConvertAIRToROCDLPass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_PASSES_H
