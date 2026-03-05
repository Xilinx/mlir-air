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
  // Test case 1: Auto-discover and hoist a single invariant read/write pair
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_loop_invariant"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %new_loop1 = transform.air.hoist_loop_invariant_transfers %func1, %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 2: Auto-discover with affine indices (tests areEquivalentIndices)
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_affine_indices"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %new_loop2 = transform.air.hoist_loop_invariant_transfers %func2, %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 3: Auto-discover from inner loop with outer alloc as scope
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_outer_alloc"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %all_loops3 = transform.structured.match ops{["scf.for"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %inner_loop3, %outer_loop3 = transform.split_handle %all_loops3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %new_loop3 = transform.air.hoist_loop_invariant_transfers %func3, %inner_loop3 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 4: Auto-discover two pairs from the same loop (tests iterative hoisting)
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_two_pairs_from_same_loop"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop4 = transform.structured.match ops{["scf.for"]} in %func4 : (!transform.any_op) -> !transform.any_op
  // Auto-discover hoists both pairs in a single call
  %loop4_final = transform.air.hoist_loop_invariant_transfers %func4, %loop4 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
