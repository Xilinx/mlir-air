//===- float_transform.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_cast_pair

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Basic hoisting of extf/truncf pair (bf16 <-> f32)
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_extf_truncf"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %extf1 = transform.structured.match ops{["arith.extf"]} in %loop1 : (!transform.any_op) -> !transform.any_op
  %truncf1 = transform.structured.match ops{["arith.truncf"]} in %loop1 : (!transform.any_op) -> !transform.any_op
  %new_loop1 = transform.air.hoist_cast_pair %extf1, %truncf1, %loop1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 2: Hoisting extf/truncf with shape_cast (f16 <-> f32)
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_extf_truncf_with_shape_cast"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %extf2 = transform.structured.match ops{["arith.extf"]} in %loop2 : (!transform.any_op) -> !transform.any_op
  %truncf2 = transform.structured.match ops{["arith.truncf"]} in %loop2 : (!transform.any_op) -> !transform.any_op
  %new_loop2 = transform.air.hoist_cast_pair %extf2, %truncf2, %loop2 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
