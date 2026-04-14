// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Transform script for cascade-based bf16 matrix-vector multiplication.
//
// C[M] = A[M,K] @ B[K]   (bf16 inputs, bf16 output)
//
// Tiling strategy:
//   1. Tile M by HERD_COLS*TILE_M for launch-level parallelism
//   2. Tile M by TILE_M for per-column (herd) parallelism
//   3. tile_reduction_using_forall on K for cascade stages
//   4. Tile K by K_TILE for inner vectorization loop
//   5. Promote to L2 (memory_space=1) and L1 (memory_space=2)
//   6. forall_with_reduce_to_parallel for cascade semantics
//   7. Bufferize
//
// Default tile sizes (parameterized via run.py):
//   HERD_COLS=8, TILE_M=2, N_CASCADE=4, K_TILE=64

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %matvec = transform.structured.match ops{["linalg.matvec"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // === Phase 1: Tile M for launch (segment-level) parallelism ===
    // tile_sizes [HERD_COLS*TILE_M, 0]: tiles M dimension, leaves K untouched.
    // TILE_M=32 so cascade buffer = 32 bf16 = 512 bits = AIE2P cascade width.
    %matvec_1, %forall_launch = transform.structured.tile_using_forall %matvec tile_sizes [256, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_1, %fused_launch = transform.structured.fuse_into_containing_op %fill into %forall_launch : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad and promote to L2 (shared memory).
    %padded, %pad, %__ = transform.structured.pad %matvec_1 {
      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
      padding_dimensions=[0, 1, 2],
      nofold_flags=[1, 1, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!transform.any_op) -> !transform.any_op

    %padded_lhs = transform.get_producer_of_operand %padded[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_lhs_buffer, %padded_lhs_new = transform.structured.bufferize_to_allocation %padded_lhs
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_rhs_buffer, %padded_rhs_new = transform.structured.bufferize_to_allocation %padded_rhs
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_result = transform.get_producer_of_operand %padded[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Canonicalize
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func1 : !transform.any_op

    // === Phase 2: Tile M for per-column (herd) parallelism ===
    %tiled_ops = transform.structured.match ops{["linalg.fill", "linalg.matvec"]} in %fused_launch : (!transform.any_op) -> !transform.any_op
    %tiled_fill, %tiled_matvec = transform.split_handle %tiled_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // TILE_M=32: each column processes 32 output rows.
    // 32 bf16 = 512 bits = exact AIE2P cascade width.
    %matvec_2, %forall_herd = transform.structured.tile_using_forall %tiled_matvec tile_sizes [32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_2, %fused_herd = transform.structured.fuse_into_containing_op %tiled_fill into %forall_herd : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad for L1 promotion
    %padded_2, %pad_2, %_ = transform.structured.pad %matvec_2 {
      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
      padding_dimensions=[0, 1, 2],
      nofold_flags=[0, 0, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_2_dps = transform.structured.rewrite_in_destination_passing_style %pad_2 : (!transform.any_op) -> !transform.any_op

    // Canonicalize
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // === Phase 3: Tile K reduction for cascade ===
    // Split K dimension into N_CASCADE threads (cascade stages).
    %reduce_fill, %matvec_3, %reduce_comb, %reduce_forall = transform.structured.tile_reduction_using_forall %padded_2 by num_threads = [0, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %fused_fill_3, %fused_reduce = transform.structured.fuse_into_containing_op %reduce_fill into %reduce_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile remaining K for inner loop.
    // K_TILE=32: matches the native AIE2P bf16 vector width. After vectorization
    // + lower_contraction{dot}, this produces arith.mulf + vector.reduction<add>
    // on vector<32xbf16>, which convert-vector-to-aievec fully supports.
    // (vector<64xbf16> would crash Peano — no GlobalISel legalization.)
    %tiled_k, %k_loop = transform.structured.tile_using_for %matvec_3 tile_sizes [0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Canonicalize
    %func3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func3 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func3 : !transform.any_op

    // Pad inner tile and promote to L1.
    // Note: no M=1 tiling here — the linalg.matvec stays as-is (scalar per row).
    // Peano handles vectorization at the LLVM level. Tiling M=1 would create
    // per-element subview writes that break the cascade dependency chain.
    %padded_k, %pad_k, %___ = transform.structured.pad %tiled_k {
      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
      padding_dimensions=[0, 1, 2],
      nofold_flags=[1, 1, 0],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_k_dps = transform.structured.rewrite_in_destination_passing_style %pad_k : (!transform.any_op) -> !transform.any_op

    %padded_k_lhs = transform.get_producer_of_operand %padded_k[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_k_lhs_buffer, %padded_k_lhs_new = transform.structured.bufferize_to_allocation %padded_k_lhs
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_k_rhs = transform.get_producer_of_operand %padded_k[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_k_rhs_buffer, %padded_k_rhs_new = transform.structured.bufferize_to_allocation %padded_k_rhs
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // === Phase 4: Convert reduce forall to parallel (cascade semantics) ===
    %inner_parallel = transform.air.forall_with_reduce_to_parallel %reduce_forall : (!transform.any_op) -> (!transform.any_op)

    // Canonicalize
    %func4 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func4 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func4 : !transform.any_op
    transform.apply_patterns to %func4 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // Bufferize cascade and partial result buffers
    %fill_ops = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %herd_cascade_fill, %herd_reduce_fill = transform.split_handle %fill_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %cascade_local_buffer, %cascade_local_new = transform.structured.bufferize_to_allocation %herd_cascade_fill
      {memory_space = 2, bufferize_destination_only} : !transform.any_op
    %result_local_buffer, %result_local_new = transform.structured.bufferize_to_allocation %herd_reduce_fill
      {memory_space = 2, bufferize_destination_only} : !transform.any_op

    // Canonicalize
    %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func5 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func5 : !transform.any_op
    transform.apply_patterns to %func5 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // === Phase 5: Bufferize ===
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    // Final cleanup
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    transform.apply_patterns to %func6 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
