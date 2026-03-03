//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.flatten_for_iter_args

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Basic flattening with single vector iter_arg
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_single_vector_iter_arg"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %new_loop1 = transform.air.flatten_for_iter_args %loop1 : (!transform.any_op) -> !transform.any_op

  // Test case 2: Flattening with multiple vector iter_args
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_multiple_vector_iter_args"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %new_loop2 = transform.air.flatten_for_iter_args %loop2 : (!transform.any_op) -> !transform.any_op

  // Test case 3: Mixed scalar and vector iter_args
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_mixed_iter_args"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %new_loop3 = transform.air.flatten_for_iter_args %loop3 : (!transform.any_op) -> !transform.any_op

  // Test case 4: Different vector shapes and types
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_different_vector_types"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop4 = transform.structured.match ops{["scf.for"]} in %func4 : (!transform.any_op) -> !transform.any_op
  %new_loop4 = transform.air.flatten_for_iter_args %loop4 : (!transform.any_op) -> !transform.any_op

  // Test case 5: Already flattened vector (1D)
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "already_flat_vector"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop5 = transform.structured.match ops{["scf.for"]} in %func5 : (!transform.any_op) -> !transform.any_op
  %new_loop5 = transform.air.flatten_for_iter_args %loop5 : (!transform.any_op) -> !transform.any_op

  // Test case 6: Nested loop with vector iter_args
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_nested_loops"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %outer_loop = transform.structured.match ops{["scf.for"]} in %func6 : (!transform.any_op) -> !transform.any_op
  // Split to get outer and inner loops
  %loop6_outer, %loop6_inner = transform.split_handle %outer_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  // Flatten outer loop first
  %new_loop6_outer = transform.air.flatten_for_iter_args %loop6_outer : (!transform.any_op) -> !transform.any_op
  // Flatten inner loop
  %new_loop6_inner = transform.air.flatten_for_iter_args %loop6_inner : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
