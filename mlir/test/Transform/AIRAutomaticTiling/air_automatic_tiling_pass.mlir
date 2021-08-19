//===- air_automatic_tiling_pass.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-automatic-tiling="air-label=xten.binary_op" -simplify-affine-structures -cse | FileCheck %s
// CHECK: affine.for {{.*}} = 0 to 28
// CHECK: affine.for {{.*}} = 0 to 10
// CHECK: {affine_opt_label = "affine_opt"}
// CHECK: affine.for {{.*}} = 0 to 7
// CHECK: affine.for {{.*}} = 0 to 5
// CHECK: {affine_opt_label = "xten.binary_op"}

module  {
  func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
    %0 = memref.alloc() : memref<28x10xf32>
    %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
    affine.for %arg2 = 0 to 28 {
      affine.for %arg3 = 0 to 10 {
        %7 = affine.load %1[%arg2, %arg3] : memref<28x10xf32>
        %cst = constant 1.000000e+00 : f32
        %8 = addf %7, %cst : f32
        affine.store %8, %0[%arg2, %arg3] : memref<28x10xf32>
      }
    } {affine_opt_label = "affine_opt"}
    %2 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
    %3 = memref.alloc() : memref<28x10xf32>
    %4 = "aten.type_cast"(%2) : (tensor<28x10xf32>) -> memref<28x10xf32>
    %5 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
    affine.for %arg2 = 0 to 28 {
      affine.for %arg3 = 0 to 10 {
        %7 = affine.load %4[%arg2, %arg3] : memref<28x10xf32>
        %8 = affine.load %5[%arg2, %arg3] : memref<28x10xf32>
        %9 = mulf %7, %8 : f32
        affine.store %9, %3[%arg2, %arg3] : memref<28x10xf32>
      }
    } {affine_opt_label = "xten.binary_op"}
    %6 = "aten.type_cast"(%3) : (memref<28x10xf32>) -> tensor<28x10xf32>
    return %6 : tensor<28x10xf32>
  }
}
