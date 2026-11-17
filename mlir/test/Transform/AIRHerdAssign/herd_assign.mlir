//===- herd_assign.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-herd-assign %s -split-input-file -verify-diagnostics | FileCheck %s

// The two outer loops of a labelled band are normalized to a zero lower bound
// and unit step, then turned into one affine.parallel. A trip count that the
// step does not divide is rounded up.

// CHECK-DAG: #[[OUTER:.*]] = affine_map<(d0) -> (d0 * 2 + 4)>
// CHECK-DAG: #[[INNER:.*]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-LABEL: func.func @constant_bounds
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 3)
// CHECK-DAG: affine.apply #[[OUTER]](%[[I]])
// CHECK-DAG: affine.apply #[[INNER]](%[[J]])
func.func @constant_bounds(%buf: memref<16x16xf32>) {
  affine.for %i = 4 to 12 step 2 {
    affine.for %j = 0 to 10 step 4 {
      %v = affine.load %buf[%i, %j] : memref<16x16xf32>
      affine.store %v, %buf[%i, %j] : memref<16x16xf32>
    }
  } {affine_opt_label = "band"}
  return
}

// -----

// Loops that are already normalized keep their induction variables.

// CHECK-LABEL: func.func @already_normalized
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 8)
// CHECK-NOT: affine.apply
// CHECK: affine.load %{{.*}}[%[[I]], %[[J]]]
func.func @already_normalized(%buf: memref<4x8xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %buf[%i, %j] : memref<4x8xf32>
      affine.store %v, %buf[%i, %j] : memref<4x8xf32>
    }
  } {affine_opt_label = "band"}
  return
}

// -----

// The lower bound reads a symbol. The trip count is still constant, and the
// rebuilt induction variable keeps the symbol as an operand.

// CHECK-DAG: #[[OUTER:.*]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
// CHECK-LABEL: func.func @symbolic_lower_bound
// CHECK-SAME: %[[BUF:[^:]*]]: memref<?x8xf32>, %[[N:.*]]: index
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 8)
// CHECK: affine.apply #[[OUTER]](%[[I]])[%[[N]]]
func.func @symbolic_lower_bound(%buf: memref<?x8xf32>, %n: index) {
  affine.for %i = %n to affine_map<()[s0] -> (s0 + 8)>()[%n] step 2 {
    affine.for %j = 0 to 8 {
      %v = affine.load %buf[%i, %j] : memref<?x8xf32>
      affine.store %v, %buf[%i, %j] : memref<?x8xf32>
    }
  } {affine_opt_label = "band"}
  return
}

// -----

// A lower bound with more than one result cannot be normalized, so the band
// is left as it is.

// CHECK-LABEL: func.func @max_lower_bound
// CHECK-NOT: affine.parallel
// CHECK: affine.for %{{.*}} = max
func.func @max_lower_bound(%buf: memref<16x8xf32>, %n: index) {
  affine.for %i = max affine_map<()[s0] -> (2, s0)>()[%n] to 10 {
    affine.for %j = 0 to 8 {
      %v = affine.load %buf[%i, %j] : memref<16x8xf32>
      affine.store %v, %buf[%i, %j] : memref<16x8xf32>
    }
  } {affine_opt_label = "band"}
  return
}

// -----

// The herd size must be a constant.

// CHECK-LABEL: func.func @dynamic_trip_count
// CHECK-NOT: affine.parallel
func.func @dynamic_trip_count(%buf: memref<?x8xf32>, %n: index) {
  // expected-error @+1 {{upper bound map isn't a single constant}}
  affine.for %i = 0 to %n {
    affine.for %j = 0 to 8 {
      %v = affine.load %buf[%i, %j] : memref<?x8xf32>
      affine.store %v, %buf[%i, %j] : memref<?x8xf32>
    }
  } {affine_opt_label = "band"}
  return
}
