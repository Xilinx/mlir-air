# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Core target sources (always built)
set(AIR_TARGET_SOURCES
  AIRHerdToJSON.cpp
)

# AIE-dependent target sources
if(AIR_ENABLE_AIE)
  list(APPEND AIR_TARGET_SOURCES
    AIRTargets.cpp
  )
endif()

add_mlir_library(
  AIRTargets
  ${AIR_TARGET_SOURCES}

  # Allow partial sources since we conditionally exclude AIE files
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  AIRRtDialect
  AIRDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms)

# Add compile definition for conditional compilation in source files
if(AIR_ENABLE_AIE)
  target_compile_definitions(AIRTargets PRIVATE AIR_ENABLE_AIE=1)
endif()
