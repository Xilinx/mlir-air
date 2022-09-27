//===- air_linalg_pipeline_reduce.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-pipeline-reduce %s | FileCheck %s
// CHECK: air.pipeline

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