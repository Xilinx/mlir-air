//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.flatten_for_iter_args

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic flattening with single vector iter_arg
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_single_vector_iter_arg"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %new_loop1 = transform.air.flatten_for_iter_args %loop1

  // Test case 2: Flattening with multiple vector iter_args
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_multiple_vector_iter_args"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %new_loop2 = transform.air.flatten_for_iter_args %loop2

  // Test case 3: Mixed scalar and vector iter_args
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_mixed_iter_args"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %new_loop3 = transform.air.flatten_for_iter_args %loop3

  // Test case 4: Different vector shapes and types
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_different_vector_types"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop4 = transform.structured.match ops{["scf.for"]} in %func4 : (!pdl.operation) -> !pdl.operation
  %new_loop4 = transform.air.flatten_for_iter_args %loop4

  // Test case 5: Already flattened vector (1D)
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "already_flat_vector"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop5 = transform.structured.match ops{["scf.for"]} in %func5 : (!pdl.operation) -> !pdl.operation
  %new_loop5 = transform.air.flatten_for_iter_args %loop5

  // Test case 6: Nested loop with vector iter_args
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "flatten_nested_loops"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %outer_loop = transform.structured.match ops{["scf.for"]} in %func6 : (!pdl.operation) -> !pdl.operation
  // Split to get outer and inner loops
  %loop6_outer, %loop6_inner = transform.split_handle %outer_loop : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  // Flatten outer loop first
  %new_loop6_outer = transform.air.flatten_for_iter_args %loop6_outer
  // Flatten inner loop
  %new_loop6_inner = transform.air.flatten_for_iter_args %loop6_inner
}
