//===- error_moves_nothing.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -transform-interpreter | FileCheck %s

// One target has operand 1, the other not: the op fails before moving
// anything, so the first target's producer stays before the store.
// CHECK-LABEL: @nothing_moves
// CHECK: arith.negf
// CHECK: memref.store
// CHECK: arith.addf
func.func @nothing_moves(%a: f32, %m: memref<f32>) -> f32 {
  %p = arith.negf %a : f32
  memref.store %a, %m[] : memref<f32>
  %r = arith.addf %a, %p : f32
  %s = math.absf %r : f32
  return %s : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.sequence %arg1 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      %t = transform.structured.match ops{["arith.addf", "math.absf"]} in %root : (!transform.any_op) -> !transform.any_op
      transform.air.sink_operand_producers %t operand_indices = [1] : !transform.any_op
    }
    transform.yield
  }
}
