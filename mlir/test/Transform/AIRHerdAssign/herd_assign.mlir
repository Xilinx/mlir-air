//===- herd_assign.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-herd-assign -verify-diagnostics | FileCheck %s

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

// -----

// A trip count that the step does not divide is rounded up.
// CHECK-LABEL: func.func @ceil_trip_count
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (3, 8) {
func.func @ceil_trip_count(%m: memref<16x8xi32>) {
  %c = arith.constant 1 : i32
  affine.for %i = 0 to 10 step 4 {
    affine.for %j = 0 to 8 {
      affine.store %c, %m[%i, %j] : memref<16x8xi32>
    }
  } {affine_opt_label = "x"}
  return
}

// -----

// Loops that are already normalized keep their induction variables.
// CHECK-LABEL: func.func @already_normalized
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 8) {
// CHECK-NOT: affine.apply
// CHECK:   affine.store %{{.*}}, %{{.*}}[%[[I]], %[[J]]]
func.func @already_normalized(%m: memref<4x8xi32>) {
  %c = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      affine.store %c, %m[%i, %j] : memref<4x8xi32>
    }
  } {affine_opt_label = "x"}
  return
}

// -----

// A lower bound with more than one result cannot be normalized, so the band
// is left as it is.
// CHECK-LABEL: func.func @max_lower_bound
// CHECK-NOT: affine.parallel
// CHECK: affine.for %{{.*}} = max
func.func @max_lower_bound(%m: memref<16x8xi32>, %n: index) {
  %c = arith.constant 1 : i32
  affine.for %i = max affine_map<()[s0] -> (2, s0)>()[%n] to 10 {
    affine.for %j = 0 to 8 {
      affine.store %c, %m[%i, %j] : memref<16x8xi32>
    }
  } {affine_opt_label = "x"}
  return
}

// -----

// The herd size must be a constant.
// CHECK-LABEL: func.func @dynamic_trip_count
// CHECK-NOT: affine.parallel
func.func @dynamic_trip_count(%m: memref<?x8xi32>, %n: index) {
  %c = arith.constant 1 : i32
  // expected-error @+1 {{upper bound map isn't a single constant}}
  affine.for %i = 0 to %n {
    affine.for %j = 0 to 8 {
      affine.store %c, %m[%i, %j] : memref<?x8xi32>
    }
  } {affine_opt_label = "x"}
  return
}
