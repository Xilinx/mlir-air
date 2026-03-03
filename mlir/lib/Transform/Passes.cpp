//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/Passes.h"

namespace {

// Core passes (always available)
#define GEN_PASS_REGISTRATION_AFFINELOOPOPTPASS
#define GEN_PASS_REGISTRATION_AIRAUTOMATICTILING
#define GEN_PASS_REGISTRATION_AIRCOLLAPSEHERDPASS
#define GEN_PASS_REGISTRATION_AIRDEPENDENCY
#define GEN_PASS_REGISTRATION_AIRDEPENDENCYCANONICALIZE
#define GEN_PASS_REGISTRATION_AIRDEPENDENCYPARSEGRAPH
#define GEN_PASS_REGISTRATION_AIREXAMPLEPASS
#define GEN_PASS_REGISTRATION_AIRFUSENESTEDHERDPASS
#define GEN_PASS_REGISTRATION_AIRFUSEPARALLELHERDPASS
#define GEN_PASS_REGISTRATION_AIRHERDASSIGN
#define GEN_PASS_REGISTRATION_AIRHERDPLACEMENTPASS
#define GEN_PASS_REGISTRATION_AIRHERDVECTORIZEPASS
#define GEN_PASS_REGISTRATION_AIRLABELBROADCASTCHANNELWITHTILEPASS
#define GEN_PASS_REGISTRATION_AIRLINALGCODEGEN
#define GEN_PASS_REGISTRATION_AIRLINALGNAMEPASS
#define GEN_PASS_REGISTRATION_AIRLINALGOPSTATS
#define GEN_PASS_REGISTRATION_AIRLOOPMERGINGPASS
#define GEN_PASS_REGISTRATION_AIRLOOPPERMUTATION
#define GEN_PASS_REGISTRATION_AIRLOWERHERDPARALLELPASS
#define GEN_PASS_REGISTRATION_AIROVERRIDEMEMREFMEMORYSPACE
#define GEN_PASS_REGISTRATION_AIRPIPELINEREDUCEPASS
#define GEN_PASS_REGISTRATION_AIRREGULARIZELOOP
#define GEN_PASS_REGISTRATION_AIRREMOVELINALGNAMEPASS
#define GEN_PASS_REGISTRATION_AIRRENUMBERDMAIDPASS
#define GEN_PASS_REGISTRATION_AIRRESOLVETENSOROPOPERANDCONFLICTSWITHNEWTENSORS
#define GEN_PASS_REGISTRATION_AIRRETURNELIMINATION
#define GEN_PASS_REGISTRATION_AIRSPECIALIZEDMABROADCAST
#define GEN_PASS_REGISTRATION_AIRSPLITL2MEMREFFORBUFFERCONSTRAINTPASS
#define GEN_PASS_REGISTRATION_AIRTRANSFORMINTERPRETERPASS
#define GEN_PASS_REGISTRATION_AIRUNROLLOUTERPERFECTLYNESTEDLOOPSPASS
#define GEN_PASS_REGISTRATION_DMATOCHANNEL

// AIE-specific passes - only define when AIE is enabled
#if AIR_ENABLE_AIE
// From AIRDependencyScheduleOpt.cpp
#define GEN_PASS_REGISTRATION_AIRANNOTATEFRONTANDBACKOPSINFORPATTERN
#define GEN_PASS_REGISTRATION_AIRBROADCASTDETECTION
#define GEN_PASS_REGISTRATION_AIRCONSTRUCTPINGPONGDEPENDENCYPATTERN
#define GEN_PASS_REGISTRATION_AIRFUSEALLOCDEALLOC
#define GEN_PASS_REGISTRATION_AIRFUSECHANNELS
#define GEN_PASS_REGISTRATION_AIRHOISTDMAINACCUMPATTERN
#define GEN_PASS_REGISTRATION_AIRHOISTMEMALLOCINFORPATTERN
#define GEN_PASS_REGISTRATION_AIRHOISTOPSNOTUSINGPINGPONGPATTERN
#define GEN_PASS_REGISTRATION_AIRISOLATEASYNCDMALOOPNESTS
#define GEN_PASS_REGISTRATION_AIRLABELSCFFORLOOPFORPINGPONGPATTERN
#define GEN_PASS_REGISTRATION_AIRLABELSCFFORLOOPINAIRSEGMENTPATTERN
#define GEN_PASS_REGISTRATION_AIRLOOPFUSION
#define GEN_PASS_REGISTRATION_AIROPTIMIZEMEMTILEDMABDS
#define GEN_PASS_REGISTRATION_AIROPTIMIZESHIMDMABDS
#define GEN_PASS_REGISTRATION_AIRPINGPONGTRANSFORMATIONPATTERN
#define GEN_PASS_REGISTRATION_AIRPRUNELINALGGENERICINPUTDMA
#define GEN_PASS_REGISTRATION_AIRSHRINKMEMREFSIZESBYACCESS
#define GEN_PASS_REGISTRATION_AIRSPECIALIZECHANNELWRAPANDSTRIDEPATTERN
#define GEN_PASS_REGISTRATION_AIRUNROLLCHANNELBYFACTORPATTERN
#define GEN_PASS_REGISTRATION_AIRUNROLLLOOPFORPIPELININGPATTERN
// From AIRLowerLinalgTensors.cpp
#define GEN_PASS_REGISTRATION_AIRLOWERLINALGTENSORS
#endif

#include "air/Transform/Passes.h.inc"

}

void xilinx::air::registerTransformPasses() {
  // Core passes (always available)
  registerAffineLoopOptPass();
  registerAIRAutomaticTiling();
  registerAIRCollapseHerdPass();
  registerAIRDependency();
  registerAIRDependencyCanonicalize();
  registerAIRDependencyParseGraph();
  registerAIRExamplePass();
  registerAIRFuseNestedHerdPass();
  registerAIRFuseParallelHerdPass();
  registerAIRHerdAssign();
  registerAIRHerdPlacementPass();
  registerAIRHerdVectorizePass();
  registerAIRLabelBroadcastChannelWithTilePass();
  registerAIRLinalgCodegen();
  registerAIRLinalgNamePass();
  registerAIRLinalgOpStats();
  registerAIRLoopMergingPass();
  registerAIRLoopPermutation();
  registerAIRLowerHerdParallelPass();
  registerAIROverrideMemRefMemorySpace();
  registerAIRPipelineReducePass();
  registerAIRRegularizeLoop();
  registerAIRRemoveLinalgNamePass();
  registerAIRRenumberDmaIdPass();
  registerAIRresolveTensorOpOperandConflictsWithNewTensors();
  registerAIRReturnElimination();
  registerAIRSpecializeDmaBroadcast();
  registerAIRSplitL2MemrefForBufferConstraintPass();
  registerAIRTransformInterpreterPass();
  registerAIRUnrollOuterPerfectlyNestedLoopsPass();
  registerDmaToChannel();

#if AIR_ENABLE_AIE
  // AIE-specific passes from AIRDependencyScheduleOpt.cpp
  registerAIRAnnotateFrontAndBackOpsInForPattern();
  registerAIRBroadcastDetection();
  registerAIRConstructPingPongDependencyPattern();
  registerAIRFuseAllocDealloc();
  registerAIRFuseChannels();
  registerAIRHoistDmaInAccumPattern();
  registerAIRHoistMemallocInForPattern();
  registerAIRHoistOpsNotUsingPingPongPattern();
  registerAIRIsolateAsyncDmaLoopNests();
  registerAIRLabelScfForLoopForPingPongPattern();
  registerAIRLabelScfForLoopInAIRSegmentPattern();
  registerAIRLoopFusion();
  registerAIROptimizeMemtileDMABDs();
  registerAIROptimizeShimDMABDs();
  registerAIRPingPongTransformationPattern();
  registerAIRPruneLinalgGenericInputDma();
  registerAIRShrinkMemrefSizesByAccess();
  registerAIRSpecializeChannelWrapAndStridePattern();
  registerAIRUnrollChannelByFactorPattern();
  registerAIRUnrollLoopForPipeliningPattern();
  // From AIRLinalgCodegen.cpp
  registerAIRLinalgCodegen();
  // From AIRLowerLinalgTensors.cpp
  registerAIRLowerLinalgTensors();
#endif
}
