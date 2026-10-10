//===- scf_parallel_to_herd_l1_views.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-par-to-herd | FileCheck %s

// A parallel loop reading views of an L1 buffer made outside it becomes a herd
// that takes the buffer itself and builds the views inside. The two herds then
// share the buffer, and so one name (two time phases of one physical herd).

// CHECK-LABEL: @l1_views
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<4x2x8x8xf32, 2>
// CHECK: air.herd @[[NAME:.*]] tile {{.*}} args(%{{.*}}=%[[ACC]]) : memref<4x2x8x8xf32, 2>
// CHECK: air.herd @[[NAME]] tile {{.*}} args(%[[A:.*]]=%[[ACC]]) : memref<4x2x8x8xf32, 2>
// CHECK-DAG: memref.expand_shape %[[A]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-DAG: memref.subview %{{.*}}[0, 0, 0, 0, 0] [4, 2, 8, 4, 1]
// CHECK-DAG: memref.subview %{{.*}}[0, 0, 0, 0, 1] [4, 2, 8, 4, 1]
// CHECK: memref.load
func.func @l1_views() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc = memref.alloc() : memref<4x2x8x8xf32, 2>
  scf.parallel (%x, %y) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
    %z = arith.constant 0.0 : f32
    linalg.fill ins(%z : f32) outs(%acc : memref<4x2x8x8xf32, 2>)
    scf.reduce
  }
  %e = memref.expand_shape %acc [[0], [1], [2], [3, 4]] output_shape [4, 2, 8, 4, 2] : memref<4x2x8x8xf32, 2> into memref<4x2x8x4x2xf32, 2>
  %g = memref.subview %e[0, 0, 0, 0, 0] [4, 2, 8, 4, 1] [1, 1, 1, 1, 1] : memref<4x2x8x4x2xf32, 2> to memref<4x2x8x4xf32, strided<[128, 64, 8, 2]>, 2>
  %u = memref.subview %e[0, 0, 0, 0, 1] [4, 2, 8, 4, 1] [1, 1, 1, 1, 1] : memref<4x2x8x4x2xf32, 2> to memref<4x2x8x4xf32, strided<[128, 64, 8, 2], offset: 1>, 2>
  scf.parallel (%x, %y) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
    %v = memref.load %g[%c0, %c0, %c0, %c0] : memref<4x2x8x4xf32, strided<[128, 64, 8, 2]>, 2>
    memref.store %v, %u[%c0, %c0, %c0, %c0] : memref<4x2x8x4xf32, strided<[128, 64, 8, 2], offset: 1>, 2>
    scf.reduce
  }
  return
}

// A view with a non-constant operand stays an operand.
// CHECK-LABEL: @dynamic_view
// CHECK: %[[S:.*]] = memref.subview
// CHECK: air.herd {{.*}} args(%{{.*}}=%[[S]]
func.func @dynamic_view(%i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc = memref.alloc() : memref<8x8xf32, 2>
  %s = memref.subview %acc[%i, 0] [1, 8] [1, 1] : memref<8x8xf32, 2> to memref<1x8xf32, strided<[8, 1], offset: ?>, 2>
  scf.parallel (%x, %y) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
    %z = arith.constant 0.0 : f32
    memref.store %z, %s[%c0, %c0] : memref<1x8xf32, strided<[8, 1], offset: ?>, 2>
    scf.reduce
  }
  return
}
