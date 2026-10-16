//===- unswitch.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// A select on a flag defined outside the loop: the loop runs in two copies
// under scf.if, and the copy for a clear flag no longer computes the value
// it would have selected.
// CHECK-LABEL: @flag
// CHECK: scf.if %arg1 {
// CHECK: scf.for
// CHECK: math.tanh
// CHECK-NOT: arith.select
// CHECK: } else {
// CHECK: scf.for
// CHECK-NOT: math.tanh
// CHECK-NOT: arith.select
// CHECK: vector.transfer_write
func.func @flag(%m: memref<4x32xf32>, %act: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pad = arith.constant 0.0 : f32
  %b = vector.broadcast %act : i1 to vector<32xi1>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %m[%i, %c0], %pad {in_bounds = [true]} : memref<4x32xf32>, vector<32xf32>
    %t = math.tanh %v : vector<32xf32>
    %r = arith.select %b, %t, %v : vector<32xi1>, vector<32xf32>
    vector.transfer_write %r, %m[%i, %c0] {in_bounds = [true]} : vector<32xf32>, memref<4x32xf32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %u = transform.air.unswitch_loop %l : (!transform.any_op) -> !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.yield
  }
}

// -----

// A select whose condition changes in the loop is left alone.
// CHECK-LABEL: @varying
// CHECK-NOT: scf.if
// CHECK: arith.select
func.func @varying(%m: memref<4x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pad = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %m[%i, %c0], %pad {in_bounds = [true]} : memref<4x32xf32>, vector<32xf32>
    %c = arith.cmpi eq, %i, %c0 : index
    %b = vector.broadcast %c : i1 to vector<32xi1>
    %t = math.tanh %v : vector<32xf32>
    %r = arith.select %b, %t, %v : vector<32xi1>, vector<32xf32>
    vector.transfer_write %r, %m[%i, %c0] {in_bounds = [true]} : vector<32xf32>, memref<4x32xf32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %u = transform.air.unswitch_loop %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
