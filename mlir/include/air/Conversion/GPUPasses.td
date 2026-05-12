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

def AIRTranslateToLLVM : Pass<"air-translate-to-llvm", "ModuleOp"> {
  let summary = "Lower air.translate to memref.reinterpret_cast + LLVM-dialect address arithmetic";
  let description = [{
    Expands each `air.translate` op into the pointer-rebase computation:
    `bases[to_rank] - bases[from_rank]`, converted from bytes to elements
    of the source memref's element type, then applied as a new offset
    via `memref.reinterpret_cast`. The expansion is pure arithmetic; it
    works identically on host functions and inside `gpu.func`.
  }];
  let constructor = "xilinx::air::createAIRTranslateToLLVMPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::LLVM::LLVMDialect"
  ];
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
