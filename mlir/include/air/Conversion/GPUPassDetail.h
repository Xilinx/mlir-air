//===- GPUPassDetail.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_GPU_PASSDETAIL_H
#define AIR_CONVERSION_GPU_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx {
namespace air {

using namespace mlir;

#define GEN_PASS_DECL
#define GEN_PASS_DEF_CONVERTAIRTOROCDL
#define GEN_PASS_DEF_CONVERTGPUKERNELOUTLINE
#include "air/Conversion/GPUPasses.h.inc"

} // namespace air
} // namespace xilinx

#endif
