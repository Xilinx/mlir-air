//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_loop_invariant_transfers

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Basic hoisting
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_loop_invariant"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %read1 = transform.structured.match ops{["vector.transfer_read"]} in %loop1 : (!transform.any_op) -> !transform.any_op
  %write1 = transform.structured.match ops{["vector.transfer_write"]} in %loop1 : (!transform.any_op) -> !transform.any_op
  %new_loop1 = transform.air.hoist_loop_invariant_transfers %read1, %write1, %loop1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 2: Hoisting with affine indices
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_affine_indices"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %read2 = transform.structured.match ops{["vector.transfer_read"]} in %loop2 : (!transform.any_op) -> !transform.any_op
  %write2 = transform.structured.match ops{["vector.transfer_write"]} in %loop2 : (!transform.any_op) -> !transform.any_op
  %new_loop2 = transform.air.hoist_loop_invariant_transfers %read2, %write2, %loop2 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 3: Hoisting from inner loop where memref is allocated in outer loop
  // Tests that we don't incorrectly clone allocations defined outside the hoisted loop
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_outer_alloc"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %all_loops3 = transform.structured.match ops{["scf.for"]} in %func3 : (!transform.any_op) -> !transform.any_op
  // Split to get outer and inner loops
  %inner_loop3, %outer_loop3 = transform.split_handle %all_loops3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %read3 = transform.structured.match ops{["vector.transfer_read"]} in %inner_loop3 : (!transform.any_op) -> !transform.any_op
  %write3 = transform.structured.match ops{["vector.transfer_write"]} in %inner_loop3 : (!transform.any_op) -> !transform.any_op
  %new_loop3 = transform.air.hoist_loop_invariant_transfers %read3, %write3, %inner_loop3 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
  
  // Test case 4: Hoisting two pairs from the same loop (tests handle chaining)
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_two_pairs_from_same_loop"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop4 = transform.structured.match ops{["scf.for"]} in %func4 : (!transform.any_op) -> !transform.any_op
  %all_reads = transform.structured.match ops{["vector.transfer_read"]} in %loop4 : (!transform.any_op) -> !transform.any_op
  %all_writes = transform.structured.match ops{["vector.transfer_write"]} in %loop4 : (!transform.any_op) -> !transform.any_op
  
  // Split to get individual read/write operations
  %read4_1, %read4_2 = transform.split_handle %all_reads : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %write4_1, %write4_2 = transform.split_handle %all_writes : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  // Hoist first pair
  %loop4_updated = transform.air.hoist_loop_invariant_transfers %read4_1, %write4_1, %loop4 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
  
  // Hoist second pair - using the updated loop handle
  %loop4_final = transform.air.hoist_loop_invariant_transfers %read4_2, %write4_2, %loop4_updated : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
