# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Source order matches origin/main for consistent archive ordering.
set(TRANSFORM_SOURCES
  AIRAutomaticTilingPass.cpp
  AIRDependency.cpp
  AIRDependencyCanonicalize.cpp
  AIRDependencyParseGraph.cpp
)
if(AIR_ENABLE_AIE)
  list(APPEND TRANSFORM_SOURCES
    AIRDependencyScheduleOpt.cpp
  )
endif()
list(APPEND TRANSFORM_SOURCES
  AIRDmaToChannel.cpp
  AIRHerdAssignPass.cpp
  AIRHerdPlacementPass.cpp
  AIRHerdVectorize.cpp
  AIRLinalgBufferize.cpp
  AIRLinalgCodegen.cpp
  AIRLinalgOpStats.cpp
  AIRLoopMergingPass.cpp
)
if(AIR_ENABLE_AIE)
  list(APPEND TRANSFORM_SOURCES
    AIRMergeUnrolledDevicesPass.cpp
  )
endif()
list(APPEND TRANSFORM_SOURCES
  AIRLoopPermutationPass.cpp
)
if(AIR_ENABLE_AIE)
  list(APPEND TRANSFORM_SOURCES
    AIRLowerLinalgTensors.cpp
  )
endif()
list(APPEND TRANSFORM_SOURCES
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

if(AIR_ENABLE_AIE)
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
