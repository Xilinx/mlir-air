# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(_depends
  AIRTransformOpsIncGen
  AIRTransformPassIncGen
  AIRDialect
  AIRRtDialect
)

if(BUILD_WITH_AIE)
  list(APPEND _depends AIRToAIETransformOpsIncGen AIRToAIEConversionPassIncGen)
endif()

add_air_library(AIRTransformPasses
  AffineLoopOptPass.cpp
  AIRAutomaticTilingPass.cpp
  AIRHerdAssignPass.cpp
  AIRHerdPlacementPass.cpp
  AIRLinalgCodegen.cpp
  AIRLinalgOpStats.cpp
  AIRLoopMergingPass.cpp
  AIRLoopPermutationPass.cpp
  AIRLowerLinalgTensors.cpp
  AIRMiscPasses.cpp
  AIRRegularizeLoopPass.cpp
  AIRTilingUtils.cpp
  AIRTransformInterpreter.cpp
  AIRDependency.cpp
  AIRDependencyScheduleOpt.cpp
  AIRDependencyCanonicalize.cpp
  AIRDependencyParseGraph.cpp
  Passes.cpp
  ReturnEliminationPass.cpp

  DEPENDS
  ${_depends}

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIRIR
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRSupport
)