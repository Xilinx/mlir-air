//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/Passes.h"

#if AIR_ENABLE_AIE
namespace {
#define GEN_PASS_REGISTRATION
#include "air/Transform/Passes.h.inc"
} // namespace

void xilinx::air::registerTransformPasses() { registerPasses(); }
#else
// When AIE is disabled, register only passes whose implementations are
// compiled (i.e., those that don't depend on AIE headers).
// TODO: Split Passes.td into AIE-independent and AIE-dependent .td files
// so that registerPasses() can remain the single source of truth.
namespace {
#define GEN_PASS_REGISTRATION
#include "air/Transform/Passes.h.inc"
} // namespace

void xilinx::air::registerTransformPasses() {
  registerAffineLoopOptPass();
  registerAIRDependency();
  registerAIRDependencyCanonicalize();
  registerAIRDependencyParseGraph();
  registerDmaToChannel();
  registerAIRExamplePass();
  registerAIRFuseNestedHerdPass();
  registerAIRFuseParallelHerdPass();
  registerAIRHerdPlacementPass();
  registerAIRHerdVectorizePass();
  registerAIRLinalgCodegen();
  registerAIRLinalgNamePass();
  registerAIRLinalgOpStats();
  registerAIRLabelBroadcastChannelWithTilePass();
  registerAIRLowerHerdParallelPass();
  registerAIROverrideMemRefMemorySpace();
  registerAIRPipelineReducePass();
  registerAIRRenumberDmaIdPass();
  registerAIRresolveTensorOpOperandConflictsWithNewTensors();
  registerAIRSpecializeDmaBroadcast();
  registerAIRSplitL2MemrefForBufferConstraintPass();
  registerAIRSplitLaunchForPadding();
  registerAIRTransformInterpreterPass();
  registerAIRUnrollOuterPerfectlyNestedLoopsPass();
  registerAIRVerifyHierarchyLocality();
}
#endif
