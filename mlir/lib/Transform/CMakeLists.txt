# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(TRANSFORM_SOURCES
  AIRAutomaticTilingPass.cpp
  AIRDependency.cpp
  AIRDependencyCanonicalize.cpp
  AIRDependencyParseGraph.cpp
  AIRDmaToChannel.cpp
  AIRHerdAssignPass.cpp
  AIRHerdPlacementPass.cpp
  AIRHerdVectorize.cpp
  AIRLinalgBufferize.cpp
  AIRLinalgCodegen.cpp
  AIRLinalgOpStats.cpp
  AIRLoopMergingPass.cpp
  AIRLoopPermutationPass.cpp
  AIRMiscPasses.cpp
  AIRRegularizeLoopPass.cpp
  AIRTilingUtils.cpp
  AIRTransformInterpreter.cpp
  AffineLoopOptPass.cpp
  Passes.cpp
  ReturnEliminationPass.cpp
)

set(TRANSFORM_LINK_LIBS
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIRIR
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRSupport
)

# Add AIE-specific sources and libraries if AIE is enabled
if(AIR_ENABLE_AIE)
  list(APPEND TRANSFORM_SOURCES
    AIRDependencyScheduleOpt.cpp
    AIRLowerLinalgTensors.cpp
  )
  list(APPEND TRANSFORM_LINK_LIBS
    AIE
    AIEX
  )
endif()

add_mlir_library(
  AIRTransformPasses
  ${TRANSFORM_SOURCES}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  AIRTransformOpsIncGen
  AIRTransformPassIncGen
  AIRDialect
  AIRRtDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${TRANSFORM_LINK_LIBS}
)
