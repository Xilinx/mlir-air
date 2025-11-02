//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.eliminate_redundant_vector_transfers

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic redundant transfer_read elimination
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "eliminate_simple_redundant_read"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func1 = transform.air.eliminate_redundant_vector_transfers %func1

  // Test case 2: Reads with write in between
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_read_after_write"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func2 = transform.air.eliminate_redundant_vector_transfers %func2

  // Test case 3: Reads with different indices
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_different_indices"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func3 = transform.air.eliminate_redundant_vector_transfers %func3

  // Test case 4: Reads with different result types
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_different_types"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func4 = transform.air.eliminate_redundant_vector_transfers %func4

  // Test case 5: Multiple redundant reads
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "eliminate_multiple_redundant_reads"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func5 = transform.air.eliminate_redundant_vector_transfers %func5

  // Test case 6: Redundant reads in a loop pattern
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "eliminate_in_loop_pattern"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func6 = transform.air.eliminate_redundant_vector_transfers %func6

  // Test case 7: Redundant reads with affine indices
  %func7 = transform.structured.match ops{["func.func"]} attributes{sym_name = "eliminate_with_affine_indices"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func7 = transform.air.eliminate_redundant_vector_transfers %func7

  // Test case 8: Mixed case
  %func8 = transform.structured.match ops{["func.func"]} attributes{sym_name = "mixed_redundant_and_unique"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func8 = transform.air.eliminate_redundant_vector_transfers %func8

  // Test case 9: Redundant reads with 2D vectors
  %func9 = transform.structured.match ops{["func.func"]} attributes{sym_name = "eliminate_2d_vector_reads"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func9 = transform.air.eliminate_redundant_vector_transfers %func9

  // Test case 10: No redundancy
  %func10 = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_all_unique_reads"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %new_func10 = transform.air.eliminate_redundant_vector_transfers %func10
}
