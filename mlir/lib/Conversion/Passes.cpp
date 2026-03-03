//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/Passes.h"

namespace {

// Core passes (always available)
#define GEN_PASS_REGISTRATION_PARALLELTOHERD
#define GEN_PASS_REGISTRATION_PARALLELTOLAUNCH
#define GEN_PASS_REGISTRATION_PARALLELTOSEGMENT
#define GEN_PASS_REGISTRATION_COPYTODMA
#define GEN_PASS_REGISTRATION_AIRTOASYNC
#define GEN_PASS_REGISTRATION_INSERTEMPTYLAUNCHOVERHERD
#define GEN_PASS_REGISTRATION_AIRWRAPFUNCWITHPARALLELPASS

#if AIR_ENABLE_GPU
// GPU passes
#define GEN_PASS_REGISTRATION_CONVERTAIRTOROCDL
#define GEN_PASS_REGISTRATION_CONVERTGPUKERNELOUTLINE
#endif

#if AIR_ENABLE_AIE
// AIE passes
#define GEN_PASS_REGISTRATION_AIRLOWERING
#define GEN_PASS_REGISTRATION_AIRLINALGTOFUNC
#define GEN_PASS_REGISTRATION_AIRTOAIE
#define GEN_PASS_REGISTRATION_AIRRTTOLLVM
#define GEN_PASS_REGISTRATION_AIRRTTONPU
#define GEN_PASS_REGISTRATION_AIRSPLITDEVICES
#endif

#include "air/Conversion/Passes.h.inc"

}

void xilinx::air::registerConversionPasses() {
  // Core passes (always available)
  registerParallelToHerd();
  registerParallelToLaunch();
  registerParallelToSegment();
  registerCopyToDma();
  registerAIRToAsync();
  registerInsertEmptyLaunchOverHerd();
  registerAIRWrapFuncWithParallelPass();

#if AIR_ENABLE_GPU
  // GPU passes
  registerConvertAIRToROCDL();
  registerConvertGPUKernelOutline();
#endif

#if AIR_ENABLE_AIE
  // AIE passes
  registerAIRLowering();
  registerAIRLinalgToFunc();
  registerAIRToAIE();
  registerAIRRtToLLVM();
  registerAIRRtToNpu();
  registerAIRSplitDevices();
#endif
}
