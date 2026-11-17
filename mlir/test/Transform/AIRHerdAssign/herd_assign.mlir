//===- herd_assign.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-herd-assign | FileCheck %s

// The labeled loop pair becomes one affine.parallel over normalized ranges.
// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0 * 2 + 2)>
// CHECK-LABEL: func.func @constant_bounds
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 8) {
// CHECK:   %[[R:.*]] = affine.apply #[[MAP]](%[[I]])
// CHECK:   affine.store %{{.*}}, %{{.*}}[%[[R]], %[[J]]]
func.func @constant_bounds(%m: memref<8x8xi32>) {
  %c = arith.constant 1 : i32
  affine.for %i = 2 to 10 step 2 {
    affine.for %j = 0 to 8 {
      affine.store %c, %m[%i, %j] : memref<8x8xi32>
    }
  } {affine_opt_label = "x"}
  return
}

// -----

// Bounds that use a symbol keep it in the rebuilt index.
// CHECK: #[[MAP:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL: func.func @symbolic_bounds
// CHECK-SAME: %[[N:[a-zA-Z0-9]+]]: index
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 8) {
// CHECK:   %[[R:.*]] = affine.apply #[[MAP]](%[[I]])[%[[N]]]
// CHECK:   affine.store %{{.*}}, %{{.*}}[%[[R]], %[[J]]]
func.func @symbolic_bounds(%m: memref<8x8xi32>, %n: index) {
  %c = arith.constant 1 : i32
  affine.for %i = %n to affine_map<()[s0] -> (s0 + 4)>()[%n] {
    affine.for %j = 0 to 8 {
      affine.store %c, %m[%i, %j] : memref<8x8xi32>
    }
  } {affine_opt_label = "x"}
  return
}
