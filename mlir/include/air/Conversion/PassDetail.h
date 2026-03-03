//===- PassDetail.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_PASSDETAIL_H
#define AIR_CONVERSION_PASSDETAIL_H

#if AIR_ENABLE_AIE
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx {
namespace air {

using namespace mlir;

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AIRTOASYNC
#define GEN_PASS_DEF_COPYTODMA
#define GEN_PASS_DEF_INSERTEMPTYLAUNCHOVERHERD
#define GEN_PASS_DEF_PARALLELTOHERD
#define GEN_PASS_DEF_PARALLELTOLAUNCH
#define GEN_PASS_DEF_PARALLELTOSEGMENT
#define GEN_PASS_DEF_AIRWRAPFUNCWITHPARALLELPASS

// AIE-specific pass definitions
#if AIR_ENABLE_AIE
#define GEN_PASS_DEF_AIRLINALGTOFUNC
#define GEN_PASS_DEF_AIRLOWERING
#define GEN_PASS_DEF_AIRPIPELINETOAFFINE
#define GEN_PASS_DEF_AIRRTTONPU
#define GEN_PASS_DEF_AIRRTTOLLVM
#define GEN_PASS_DEF_AIRSPLITDEVICES
#define GEN_PASS_DEF_AIRTOAIE
#endif

#include "air/Conversion/Passes.h.inc"

} // namespace air
} // namespace xilinx

#endif
