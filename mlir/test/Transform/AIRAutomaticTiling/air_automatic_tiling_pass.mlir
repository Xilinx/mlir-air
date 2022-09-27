//===- air_automatic_tiling_pass.mlir --------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-automatic-tiling="air-label=xten.binary_op" -affine-simplify-structures -cse | FileCheck %s
// CHECK: affine.for {{.*}} = 0 to 28
// CHECK: affine.for {{.*}} = 0 to 10
// CHECK: {affine_opt_label = "affine_opt"}
// CHECK: affine.for {{.*}} = 0 to 7
// CHECK: affine.for {{.*}} = 0 to 5
// CHECK: {affine_opt_label = "xten.binary_op"}

module  {
  func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
    %0 = memref.alloc() : memref<28x10xf32>
    %1 = bufferization.to_memref %arg0 : memref<28x10xf32>
    affine.for %arg2 = 0 to 28 {
      affine.for %arg3 = 0 to 10 {
        %7 = affine.load %1[%arg2, %arg3] : memref<28x10xf32>
        %cst = arith.constant 1.000000e+00 : f32
        %8 = arith.addf %7, %cst : f32
        affine.store %8, %0[%arg2, %arg3] : memref<28x10xf32>
      }
    } {affine_opt_label = "affine_opt"}
    %2 = bufferization.to_tensor %0 : memref<28x10xf32>
    %3 = memref.alloc() : memref<28x10xf32>
    %4 = bufferization.to_memref %2 : memref<28x10xf32>
    %5 = bufferization.to_memref %arg1 : memref<28x10xf32>
    affine.for %arg2 = 0 to 28 {
      affine.for %arg3 = 0 to 10 {
        %7 = affine.load %4[%arg2, %arg3] : memref<28x10xf32>
        %8 = affine.load %5[%arg2, %arg3] : memref<28x10xf32>
        %9 = arith.mulf %7, %8 : f32
        affine.store %9, %3[%arg2, %arg3] : memref<28x10xf32>
      }
    } {affine_opt_label = "xten.binary_op"}
    %6 = bufferization.to_tensor %3 : memref<28x10xf32>
    return %6 : tensor<28x10xf32>
  }
}
