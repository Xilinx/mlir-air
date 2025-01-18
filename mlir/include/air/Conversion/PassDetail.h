//===- PassDetail.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_PASSDETAIL_H
#define AIR_CONVERSION_PASSDETAIL_H

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx {
namespace air {

using namespace mlir;

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AIRLINALGTOFUNC
#define GEN_PASS_DEF_AIRLOWERING
#define GEN_PASS_DEF_AIRPIPELINETOAFFINE
#define GEN_PASS_DEF_AIRRTTONPU
#define GEN_PASS_DEF_AIRRTTOLLVM
#define GEN_PASS_DEF_AIRSPLITDEVICES
#define GEN_PASS_DEF_AIRTOAIE
#define GEN_PASS_DEF_AIRTOASYNC
#define GEN_PASS_DEF_COPYTODMA
#define GEN_PASS_DEF_INSERTEMPTYLAUNCHOVERHERD
#define GEN_PASS_DEF_PARALLELTOHERD
#define GEN_PASS_DEF_PARALLELTOLAUNCH
#define GEN_PASS_DEF_PARALLELTOSEGMENT
#include "air/Conversion/Passes.h.inc"

} // namespace air
} // namespace xilinx

#endif
