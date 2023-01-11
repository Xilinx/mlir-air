//===- air_linalg_pipeline_reduce.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-pipeline-reduce %s | FileCheck %s
// CHECK: air.pipeline

// XFAIL: *

module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x256xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<256x256xf32>)
    memref.copy %0, %1 : memref<256x256xf32> to memref<256x256xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<256x256xf32>, memref<256x256xf32>) outs(%1 : memref<256x256xf32>)
    memref.copy %1, %arg2 : memref<256x256xf32> to memref<256x256xf32>
    return
  }
}