//===- GPUPasses.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_GPU_PASSES
#define AIR_CONVERSION_GPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAIRToROCDL : Pass<"air-to-rocdl", "ModuleOp"> {
  let summary = "Lower AIR dialect to ROCDL dialect";
  let constructor = "xilinx::air::createAIRToROCDLPass()";
  let description = [{
  }];
  let dependentDialects = [
    "gpu::GPUDialect", "ROCDL::ROCDLDialect", "LLVM::LLVMDialect"
  ];
  let options = [];
}

def ConvertGPUKernelOutline : Pass<"air-gpu-outlining", "ModuleOp"> {
  let summary = "Outline GPU Kernel Func from GPU Launch";
  let constructor = "xilinx::air::createGPUKernelOutlinePass()";
  let description = [{
  }];
  let dependentDialects = [
    "gpu::GPUDialect", "ROCDL::ROCDLDialect", "LLVM::LLVMDialect"
  ];
  let options = [];
}

#endif // AIR_CONVERSION_GPU_PASSES
