//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.normalize_for_bounds

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic affine.apply multiplication on loop induction variable
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "basic_multiply"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %new_loop1 = transform.air.normalize_for_bounds %loop1

  // Test case 2: Affine.apply with addition on loop induction variable
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "basic_add"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %new_loop2 = transform.air.normalize_for_bounds %loop2

  // Test case 3: Multiple affine.apply ops on the same loop induction variable
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "multiple_affine_apply"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %new_loop3 = transform.air.normalize_for_bounds %loop3

  // Test case 4: Nested loops with affine.apply on each level
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "nested_loops"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loops4 = transform.structured.match ops{["scf.for"]} in %func4 : (!pdl.operation) -> !pdl.operation
  %outer_loop4, %inner_loop4 = transform.split_handle %loops4 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %new_outer_loop4 = transform.air.normalize_for_bounds %outer_loop4
  %new_inner_loop4 = transform.air.normalize_for_bounds %inner_loop4

  // Test case 5: Affine.apply with combined multiplication and addition
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "mul_and_add"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop5 = transform.structured.match ops{["scf.for"]} in %func5 : (!pdl.operation) -> !pdl.operation
  %new_loop5 = transform.air.normalize_for_bounds %loop5

  // Test case 6: Loop with no affine.apply (should remain unchanged)
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "no_affine_apply"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop6 = transform.structured.match ops{["scf.for"]} in %func6 : (!pdl.operation) -> !pdl.operation
  %new_loop6 = transform.air.normalize_for_bounds %loop6
}
