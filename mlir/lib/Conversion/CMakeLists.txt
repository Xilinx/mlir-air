# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Core conversion pass sources (always built)
set(AIR_CONVERSION_SOURCES
  ConvertToAIRPass.cpp
  AIRRtToLLVMPass.cpp
  AIRToAsyncPass.cpp
  AIRToROCDLPass.cpp
  GPUKernelOutlingPass.cpp
  Passes.cpp
)

# AIE-dependent sources (only built when AIE is enabled)
if(AIR_ENABLE_AIE)
  list(APPEND AIR_CONVERSION_SOURCES
    AIRLoweringPass.cpp
    AIRToAIEPass.cpp
    AIRToAIESchedulingUtils.cpp
    AIRRtToNpuPass.cpp
  )
endif()

# Link libraries - core libs always needed
set(AIR_CONVERSION_LINK_LIBS
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

# Add AIE libs only when enabled
if(AIR_ENABLE_AIE)
  list(APPEND AIR_CONVERSION_LINK_LIBS
    AIE
    AIEX
  )
endif()

add_mlir_library(
  AIRConversionPasses
  ${AIR_CONVERSION_SOURCES}

  # Allow partial sources since we conditionally exclude AIE files
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
  ${AIR_CONVERSION_LINK_LIBS}
)

# Add compile definition for conditional compilation in source files
if(AIR_ENABLE_AIE)
  target_compile_definitions(AIRConversionPasses PRIVATE AIR_ENABLE_AIE=1)
endif()
