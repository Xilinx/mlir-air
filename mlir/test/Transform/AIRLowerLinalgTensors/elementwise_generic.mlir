//===- elementwise_generic.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// XFAIL:*
// RUN: air-opt %s -air-lower-linalg-tensors | FileCheck %s
// CHECK: %[[TILE0:.*]] = aie.tile(1, 1)
// CHECK: %[[BUF2:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "buf2"} : memref<32x32xi32, 2>
// CHECK: %[[BUF1:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "buf1"} : memref<32x32xi32, 2>
// CHECK: %[[BUF0:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "buf0"} : memref<32x32xi32, 2>
// CHECK: affine.for %arg0 = 0 to 32 {
// CHECK:     affine.for %arg1 = 0 to 32 {
// CHECK:       %{{.*}} = affine.load %[[BUF0]][%arg0, %arg1] : memref<32x32xi32, 2>
// CHECK:       %{{.*}} = affine.load %[[BUF1]][%arg0, %arg1] : memref<32x32xi32, 2>
// CHECK:       %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:       affine.store %{{.*}}, %[[BUF2]][%arg0, %arg1] : memref<32x32xi32, 2>
// CHECK:     }
// CHECK:   }
#map = affine_map<(d0, d1) -> (d0, d1)>
module @aie.0  {
  %0 = aie.tile(1, 1)
  %1 = aie.lock(%0, 2)
  %2 = aie.lock(%0, 1)
  %3 = aie.lock(%0, 0)
  %4 = aie.buffer(%0) {sym_name = "buf2"} : memref<32x32xi32, 2>
  %5 = aie.buffer(%0) {sym_name = "buf1"} : memref<32x32xi32, 2>
  %6 = aie.buffer(%0) {sym_name = "buf0"} : memref<32x32xi32, 2>
  %7 = aie.core(%0)  {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    aie.use_lock(%3, Acquire, 1)
    aie.use_lock(%2, Acquire, 1)
    aie.use_lock(%1, Acquire, 0)
    %8 = tensor.empty () : tensor<32x32xi32>
    %9 = bufferization.to_tensor %6 : memref<32x32xi32, 2>
    %10 = bufferization.to_tensor %5 : memref<32x32xi32, 2>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : tensor<32x32xi32>, tensor<32x32xi32>) outs(%8 : tensor<32x32xi32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
      %12 = arith.muli %arg0, %arg1 : i32
      linalg.yield %12 : i32
    } -> tensor<32x32xi32>
    memref.tensor_store %11, %4 : memref<32x32xi32, 2>
    aie.use_lock(%3, Release, 0)
    aie.use_lock(%2, Release, 0)
    aie.use_lock(%1, Release, 1)
    aie.end
  }
}
