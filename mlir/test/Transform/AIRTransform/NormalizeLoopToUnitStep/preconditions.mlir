//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// A non-zero lower bound and an iter_arg.
// CHECK-LABEL: @lb_and_iter_args
// CHECK: %[[R:.*]] = scf.for %[[J:.*]] = %c0{{.*}} to %c15{{.*}} step %c1{{.*}} iter_args(%[[A:.*]] = %arg1) -> (index)
// CHECK: %[[I:.*]] = affine.apply #{{.*}}(%[[J]])
// CHECK: arith.addi %[[A]], %[[I]]
// CHECK: return %[[R]]
func.func @lb_and_iter_args(%t: index, %init: index) -> index {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c4s = arith.constant 4 : index
  %r = scf.for %i = %c4 to %c64 step %c4s iter_args(%acc = %init) -> (index) {
    %a = arith.addi %acc, %i : index
    scf.yield %a : index
  }
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %n = transform.air.normalize_loop_to_unit_step %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A loop over i32 is rejected before anything is rewritten.
func.func @i32_loop(%init: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c8 = arith.constant 8 : i32
  %c64 = arith.constant 64 : i32
  %r = scf.for %i = %c0 to %c64 step %c8 iter_args(%acc = %init) -> (i32) : i32 {
    %a = arith.addi %acc, %i : i32
    scf.yield %a : i32
  }
  return %r : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expects a signed scf.for over index values}}
    %n = transform.air.normalize_loop_to_unit_step %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// So is an unsigned loop.
func.func @unsigned_loop(%lb: index) {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  scf.for unsigned %i = %lb to %c64 step %c8 {
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expects a signed scf.for over index values}}
    %n = transform.air.normalize_loop_to_unit_step %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
