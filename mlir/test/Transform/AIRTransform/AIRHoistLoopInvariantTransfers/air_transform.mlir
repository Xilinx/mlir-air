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
  transform.debug.emit_remark_at %loop4_final, "hoisted loop" : !transform.any_op

  // Test case 5: Hoisted writes keep their order in the loop body
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_three_pairs_keeps_write_order"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop5 = transform.structured.match ops{["scf.for"]} in %func5 : (!transform.any_op) -> !transform.any_op
  %loop5_final = transform.air.hoist_loop_invariant_transfers %func5, %loop5 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 6: Overlapping pairs stay in the loop
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_overlapping_pairs"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop6 = transform.structured.match ops{["scf.for"]} in %func6 : (!transform.any_op) -> !transform.any_op
  %loop6_final = transform.air.hoist_loop_invariant_transfers %func6, %loop6 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 7: Read and write compute the same index separately
  %func7 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_equivalent_indices"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop7 = transform.structured.match ops{["scf.for"]} in %func7 : (!transform.any_op) -> !transform.any_op
  %loop7_final = transform.air.hoist_loop_invariant_transfers %func7, %loop7 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 8: Only the given loop is rewritten
  %func8 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_only_target_loop"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loops8 = transform.structured.match ops{["scf.for"]} in %func8 : (!transform.any_op) -> !transform.any_op
  %inner8, %outer8 = transform.split_handle %loops8 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %loop8_final = transform.air.hoist_loop_invariant_transfers %func8, %inner8 : (!transform.any_op, !transform.any_op) -> !transform.any_op
  transform.debug.emit_remark_at %inner8, "input handle" : !transform.any_op

  // Test case 9: Pairs that cannot become an iter_arg
  %f_keep_masked_pair = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_masked_pair"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %l_keep_masked_pair = transform.structured.match ops{["scf.for"]} in %f_keep_masked_pair : (!transform.any_op) -> !transform.any_op
  %r_keep_masked_pair = transform.air.hoist_loop_invariant_transfers %f_keep_masked_pair, %l_keep_masked_pair : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %f_keep_view_base = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_view_base"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %l_keep_view_base = transform.structured.match ops{["scf.for"]} in %f_keep_view_base : (!transform.any_op) -> !transform.any_op
  %r_keep_view_base = transform.air.hoist_loop_invariant_transfers %f_keep_view_base, %l_keep_view_base : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %f_keep_write_before_read = transform.structured.match ops{["func.func"]} attributes{sym_name = "keep_write_before_read"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %l_keep_write_before_read = transform.structured.match ops{["scf.for"]} in %f_keep_write_before_read : (!transform.any_op) -> !transform.any_op
  %r_keep_write_before_read = transform.air.hoist_loop_invariant_transfers %f_keep_write_before_read, %l_keep_write_before_read : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
