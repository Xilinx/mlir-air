//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_loop_invariant_transfers

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic hoisting
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_loop_invariant"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %read1 = transform.structured.match ops{["vector.transfer_read"]} in %loop1 : (!pdl.operation) -> !pdl.operation
  %write1 = transform.structured.match ops{["vector.transfer_write"]} in %loop1 : (!pdl.operation) -> !pdl.operation
  %new_loop1 = transform.air.hoist_loop_invariant_transfers %read1, %write1, %loop1

  // Test case 2: Hoisting with affine indices
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_affine_indices"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %read2 = transform.structured.match ops{["vector.transfer_read"]} in %loop2 : (!pdl.operation) -> !pdl.operation
  %write2 = transform.structured.match ops{["vector.transfer_write"]} in %loop2 : (!pdl.operation) -> !pdl.operation
  %new_loop2 = transform.air.hoist_loop_invariant_transfers %read2, %write2, %loop2

  // Test case 3: Hoisting two pairs from the same loop (tests handle chaining)
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_two_pairs_from_same_loop"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %all_reads = transform.structured.match ops{["vector.transfer_read"]} in %loop3 : (!pdl.operation) -> !pdl.operation
  %all_writes = transform.structured.match ops{["vector.transfer_write"]} in %loop3 : (!pdl.operation) -> !pdl.operation
  
  // Split to get individual read/write operations
  %read3_1, %read3_2 = transform.split_handle %all_reads : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %write3_1, %write3_2 = transform.split_handle %all_writes : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  
  // Hoist first pair
  %loop3_updated = transform.air.hoist_loop_invariant_transfers %read3_1, %write3_1, %loop3
  
  // Hoist second pair - using the updated loop handle
  %loop3_final = transform.air.hoist_loop_invariant_transfers %read3_2, %write3_2, %loop3_updated
}
