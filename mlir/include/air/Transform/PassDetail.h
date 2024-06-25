//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TRANSFORM_PASSDETAIL_H_
#define AIR_TRANSFORM_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace xilinx {
namespace air {

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AIRANNOTATEFRONTANDBACKOPSINFORPATTERN
#define GEN_PASS_DEF_AIRAUTOMATICTILING
#define GEN_PASS_DEF_AIRBROADCASTDETECTION
#define GEN_PASS_DEF_AIRCOLLAPSEHERDPASS
#define GEN_PASS_DEF_AIRUNROLLOUTERPERFECTLYNESTEDLOOPSPASS
#define GEN_PASS_DEF_AIRCONSTRUCTPINGPONGDEPENDENCYPATTERN
#define GEN_PASS_DEF_AIRDEALIASMEMREF
#define GEN_PASS_DEF_AIRDEPENDENCY
#define GEN_PASS_DEF_AIRDEPENDENCYCANONICALIZE
#define GEN_PASS_DEF_AIRDEPENDENCYPARSEGRAPH
#define GEN_PASS_DEF_AIRDEPENDENCYSCHEDULEOPT
#define GEN_PASS_DEF_AIRENFORCELOOPCARRIEDMEMREFDEALLOCPATTERN
#define GEN_PASS_DEF_AIREXAMPLEPASS
#define GEN_PASS_DEF_AIRFUSECHANNELS
#define GEN_PASS_DEF_AIRFUSEPARALLELHERDPASS
#define GEN_PASS_DEF_AIRHERDASSIGN
#define GEN_PASS_DEF_AIRHERDPLACEMENTPASS
#define GEN_PASS_DEF_AIRHOISTDMAINACCUMPATTERN
#define GEN_PASS_DEF_AIRHOISTMEMALLOCINFORPATTERN
#define GEN_PASS_DEF_AIRHOISTOPSNOTUSINGPINGPONGPATTERN
#define GEN_PASS_DEF_AIRHOISTSCFCHANNELGETPUTPASS
#define GEN_PASS_DEF_AIRISOLATEASYNCDMALOOPNESTS
#define GEN_PASS_DEF_AIRLABELBROADCASTCHANNELWITHTILEPASS
#define GEN_PASS_DEF_AIRLABELSCFFORLOOPFORPINGPONGPATTERN
#define GEN_PASS_DEF_AIRLABELSCFFORLOOPINAIRSEGMENTPATTERN
#define GEN_PASS_DEF_AIRSPECIALIZECHANNELWRAPANDSTRIDEPATTERN
#define GEN_PASS_DEF_AIRLINALGCODEGEN
#define GEN_PASS_DEF_AIRLINALGNAMEPASS
#define GEN_PASS_DEF_AIRLINALGOPSTATS
#define GEN_PASS_DEF_AIRLOOPMERGINGPASS
#define GEN_PASS_DEF_AIRLOOPPERMUTATION
#define GEN_PASS_DEF_AIRLOWERHERDPARALLELPASS
#define GEN_PASS_DEF_AIRLOWERLINALGTENSORS
#define GEN_PASS_DEF_AIRPINGPONGTRANSFORMATIONPATTERN
#define GEN_PASS_DEF_AIRPIPELINEREDUCEPASS
#define GEN_PASS_DEF_AIRPROMOTEUNIFORML1DMA
#define GEN_PASS_DEF_AIRPRUNELINALGGENERICINPUTDMA
#define GEN_PASS_DEF_AIRREGULARIZELOOP
#define GEN_PASS_DEF_AIRREMOVELINALGNAMEPASS
#define GEN_PASS_DEF_AIRRENUMBERDMAIDPASS
#define GEN_PASS_DEF_AIRRETURNELIMINATION
#define GEN_PASS_DEF_AIRSPECIALIZEDMA
#define GEN_PASS_DEF_AIRSPECIALIZEDMABROADCAST
#define GEN_PASS_DEF_AIRTRANSFORMINTERPRETERPASS
#define GEN_PASS_DEF_AIRUNROLLCHANNELBYFACTORPATTERN
#define GEN_PASS_DEF_AIRUNROLLLOOPFORPIPELININGPATTERN
#define GEN_PASS_DEF_AFFINELOOPOPTPASS
#define GEN_PASS_DEF_AIRSEGMENTLOOPFUSION
#define GEN_PASS_DEF_AIRSPLITL2MEMREFFORBUFFERCONSTRAINTPASS
#define GEN_PASS_DEF_DMATOCHANNEL
#include "air/Transform/Passes.h.inc"

} // namespace air
} // namespace xilinx

#endif
