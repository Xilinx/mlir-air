//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// An epilogue herd handed slices of an L1 accumulator made outside it takes the
// accumulator itself and rebuilds the views inside; it also takes the name of the
// herd it shares that L1 buffer with (one physical herd, two time phases).

// CHECK-LABEL: @views
// CHECK: air.herd @acc_herd {{.*}} args(%{{.*}}=%[[ACC:.*]]) : memref<4x2x8x8xf32, 2>
// CHECK: air.herd @acc_herd {{.*}} args(%[[A0:.*]]=%[[ACC]], %[[A1:.*]]=%[[ACC]]) : memref<4x2x8x8xf32, 2>, memref<4x2x8x8xf32, 2> attributes {epilogue_herd}
// CHECK-DAG: %[[E0:.*]] = memref.expand_shape %[[A0]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-DAG: memref.subview %[[E0]][0, 0, 0, 0, 0] [4, 2, 8, 4, 1]
// CHECK-DAG: %[[E1:.*]] = memref.expand_shape %[[A1]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-DAG: memref.subview %[[E1]][0, 0, 0, 0, 1] [4, 2, 8, 4, 1]
func.func @views() {
  %c1 = arith.constant 1 : index
  %acc = memref.alloc() : memref<4x2x8x8xf32, 2>
  air.herd @acc_herd tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%a = %acc) : memref<4x2x8x8xf32, 2> {
    %z = arith.constant 0.0 : f32
    linalg.fill ins(%z : f32) outs(%a : memref<4x2x8x8xf32, 2>)
  }
  %e = memref.expand_shape %acc [[0], [1], [2], [3, 4]] output_shape [4, 2, 8, 4, 2] : memref<4x2x8x8xf32, 2> into memref<4x2x8x4x2xf32, 2>
  %g = memref.subview %e[0, 0, 0, 0, 0] [4, 2, 8, 4, 1] [1, 1, 1, 1, 1] : memref<4x2x8x4x2xf32, 2> to memref<4x2x8x4xf32, strided<[128, 64, 8, 2]>, 2>
  %u = memref.subview %e[0, 0, 0, 0, 1] [4, 2, 8, 4, 1] [1, 1, 1, 1, 1] : memref<4x2x8x4x2xf32, 2> to memref<4x2x8x4xf32, strided<[128, 64, 8, 2], offset: 1>, 2>
  air.herd tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%ga = %g, %ua = %u) : memref<4x2x8x4xf32, strided<[128, 64, 8, 2]>, 2>, memref<4x2x8x4xf32, strided<[128, 64, 8, 2], offset: 1>, 2> attributes {epilogue_herd} {
    %c0 = arith.constant 0 : index
    %v = memref.load %ga[%c0, %c0, %c0, %c0] : memref<4x2x8x4xf32, strided<[128, 64, 8, 2]>, 2>
    memref.store %v, %ua[%c0, %c0, %c0, %c0] : memref<4x2x8x4xf32, strided<[128, 64, 8, 2], offset: 1>, 2>
  }
  return
}
