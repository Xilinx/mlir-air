# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Source order matches origin/main for consistent archive ordering.
# When AIE is enabled, AIE-specific sources are interleaved at their
# original positions to preserve the archive object order that the
# MLIR shared library symbol resolution depends on.
set(CONVERSION_SOURCES
  ConvertToAIRPass.cpp
)
if(AIR_ENABLE_AIE)
  list(APPEND CONVERSION_SOURCES
    AIRLoweringPass.cpp
    AIRRtToLLVMPass.cpp
    AIRToAIEPass.cpp
    AIRToAIESchedulingUtils.cpp
  )
endif()
list(APPEND CONVERSION_SOURCES
  AIRToAsyncPass.cpp
)
if(AIR_ENABLE_AIE)
  list(APPEND CONVERSION_SOURCES
    AIRRtToNpuPass.cpp
  )
endif()
list(APPEND CONVERSION_SOURCES
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

# Add AIE-specific libraries if AIE is enabled
if(AIR_ENABLE_AIE)
  list(APPEND CONVERSION_LINK_LIBS
    AIE
    AIEX
  )
endif()

# Add GPU-specific sources, libraries, and dependencies if GPU is enabled
set(GPU_PASS_DEPENDS "")
if(AIR_ENABLE_GPU)
  set(GPU_PASS_DEPENDS AIRGPUConversionPassIncGen)
  list(APPEND CONVERSION_SOURCES
    AIRToROCDLPass.cpp
    GPUKernelOutlinePass.cpp
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

add_mlir_library(
  AIRConversionPasses
  ${CONVERSION_SOURCES}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  AIRConversionPassIncGen
  ${GPU_PASS_DEPENDS}
  AIRTransformOpsIncGen
  AIRTransformPasses
  AIRDialect
  AIRRtDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS
  ${CONVERSION_LINK_LIBS}
)
