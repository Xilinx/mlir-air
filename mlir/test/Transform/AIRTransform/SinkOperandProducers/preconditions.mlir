//===- preconditions.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -transform-interpreter -verify-diagnostics -split-input-file

// An operand index the target does not have.
func.func @out_of_range(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %t = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{operand 2 out of range}}
    transform.air.sink_operand_producers %t operand_indices = [2] : !transform.any_op
    transform.yield
  }
}

