# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_library(
  AIRTransformPasses
  AIRAutomaticTilingPass.cpp
  AIRDependency.cpp
  AIRDependencyCanonicalize.cpp
  AIRDependencyParseGraph.cpp
  AIRDependencyScheduleOpt.cpp
  AIRDmaToChannel.cpp
  AIRHerdAssignPass.cpp
  AIRHerdPlacementPass.cpp
  AIRHerdVectorize.cpp
  AIRLinalgBufferize.cpp
  AIRLinalgCodegen.cpp
  AIRLinalgOpStats.cpp
  AIRLoopMergingPass.cpp
  AIRMergeUnrolledDevicesPass.cpp
  AIRLoopPermutationPass.cpp
  AIRLowerLinalgTensors.cpp
  AIRMiscPasses.cpp
  AIRRegularizeLoopPass.cpp
  AIRTilingUtils.cpp
  AIRTransformInterpreter.cpp
  AffineLoopOptPass.cpp
  Passes.cpp
  ReturnEliminationPass.cpp

  DEPENDS
  AIRTransformOpsIncGen
  AIRTransformPassIncGen
  AIRDialect
  AIRRtDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIE
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIRIR
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRSupport)
