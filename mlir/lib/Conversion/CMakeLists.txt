# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(CONVERSION_SOURCES
  ConvertToAIRPass.cpp
  AIRToAsyncPass.cpp
  Passes.cpp
)

set(CONVERSION_LINK_LIBS
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIRIR
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRSupport
  MLIRTransforms
  MLIRPass
)

# Add GPU-specific sources and libraries if GPU is enabled
if(AIR_ENABLE_GPU)
  list(APPEND CONVERSION_SOURCES
    AIRToROCDLPass.cpp
    GPUKernelOutlingPass.cpp
  )
  list(APPEND CONVERSION_LINK_LIBS
    MLIRGPUDialect
    MLIRGPUToROCDLTransforms
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRControlFlowDialect
    MLIRROCDLDialect
  )
endif()

# Add AIE-specific sources and libraries if AIE is enabled
if(AIR_ENABLE_AIE)
  list(APPEND CONVERSION_SOURCES
    AIRToAIEPass.cpp
    AIRToAIESchedulingUtils.cpp
    AIRLoweringPass.cpp
    AIRRtToLLVMPass.cpp
    AIRRtToNpuPass.cpp
  )
  list(APPEND CONVERSION_LINK_LIBS
    AIE
    AIEX
  )
endif()

add_mlir_library(
  AIRConversionPasses
  ${CONVERSION_SOURCES}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  AIRConversionPassIncGen
  AIRTransformOpsIncGen
  AIRTransformPasses
  AIRDialect
  AIRRtDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS
  ${CONVERSION_LINK_LIBS}
)
