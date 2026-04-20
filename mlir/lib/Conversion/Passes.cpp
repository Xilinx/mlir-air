//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/Passes.h"

#if AIR_ENABLE_GPU
#include "air/Conversion/AIRToROCDLPass.h"
#include "air/Conversion/GPUKernelOutlinePass.h"
#endif

namespace air_conv_passes {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"
} // namespace air_conv_passes

#if AIR_ENABLE_GPU
namespace air_gpu_passes {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/GPUPasses.h.inc"
} // namespace air_gpu_passes
#endif

void xilinx::air::registerConversionPasses() {
#if AIR_ENABLE_AIE
  air_conv_passes::registerPasses();
#else
  // Register only non-AIE conversion passes.
  air_conv_passes::registerParallelToHerd();
  air_conv_passes::registerParallelToLaunch();
  air_conv_passes::registerParallelToSegment();
  air_conv_passes::registerCopyToDma();
  air_conv_passes::registerAIRToAsync();
  air_conv_passes::registerInsertEmptyLaunchOverHerd();
  air_conv_passes::registerAIRRankToLaunch();
  air_conv_passes::registerAIRWrapFuncWithParallelPass();
#endif
#if AIR_ENABLE_GPU
  air_gpu_passes::registerPasses();
#endif
}
