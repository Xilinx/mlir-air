# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_library(
  AIRConversionPasses
  ConvertToAIRPass.cpp
  AIRLoweringPass.cpp
  AIRRtToLLVMPass.cpp
  AIRToAIEPass.cpp
  AIRToAIESchedulingUtils.cpp
  AIRToAsyncPass.cpp
  AIRPipeline.cpp
  AIRRtToIpuPass.cpp
  Passes.cpp

  DEPENDS
  AIRConversionPassIncGen
  AIRTransformOpsIncGen
  AIRTransformPasses
  AIRDialect
  AIRRtDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIE
  AIEX
  MLIRIR
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRSupport
  MLIRTransforms
  MLIRPass)
