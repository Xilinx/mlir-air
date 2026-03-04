//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.normalize_for_bounds

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Basic affine.apply multiplication on loop induction variable
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "basic_multiply"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %new_loop1 = transform.air.normalize_for_bounds %loop1 : (!transform.any_op) -> !transform.any_op

  // Test case 2: Affine.apply with addition on loop induction variable
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "basic_add"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %new_loop2 = transform.air.normalize_for_bounds %loop2 : (!transform.any_op) -> !transform.any_op

  // Test case 3: Multiple affine.apply ops on the same loop induction variable
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "multiple_affine_apply"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %new_loop3 = transform.air.normalize_for_bounds %loop3 : (!transform.any_op) -> !transform.any_op

  // Test case 4: Nested loops with affine.apply on each level
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "nested_loops"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loops4 = transform.structured.match ops{["scf.for"]} in %func4 : (!transform.any_op) -> !transform.any_op
  %outer_loop4, %inner_loop4 = transform.split_handle %loops4 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %new_outer_loop4 = transform.air.normalize_for_bounds %outer_loop4 : (!transform.any_op) -> !transform.any_op
  %new_inner_loop4 = transform.air.normalize_for_bounds %inner_loop4 : (!transform.any_op) -> !transform.any_op

  // Test case 5: Affine.apply with combined multiplication and addition
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "mul_and_add"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop5 = transform.structured.match ops{["scf.for"]} in %func5 : (!transform.any_op) -> !transform.any_op
  %new_loop5 = transform.air.normalize_for_bounds %loop5 : (!transform.any_op) -> !transform.any_op

  // Test case 6: Loop with no affine.apply (should remain unchanged)
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "no_affine_apply"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop6 = transform.structured.match ops{["scf.for"]} in %func6 : (!transform.any_op) -> !transform.any_op
  %new_loop6 = transform.air.normalize_for_bounds %loop6 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
