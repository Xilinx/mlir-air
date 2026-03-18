// Vectorization transform for f32 matmul with bf16 on AIE2 (NPU1).
// Uses 4x8x4 matmul shape (native bf16, no BFP16 emulation).
// Tiles truncf_op and block_matmul for vectorization, vectorizes herds,
// casts vector.contract accumulator types, and hoists transfers.
//
// Adapted from programming_examples/matrix_multiplication/bf16/run.py.
// The compute herd (herd2) has truncf_op + block_matmul.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

    // Match 2 truncf_ops + 1 block_matmul
    %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %truncf_a_g, %truncf_b_g, %matmul = transform.split_handle %all_generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile truncf_ops to [1,1,0,0] -> vector<8x8> for aievec
    %tiled_truncf_a, %truncf_a_loops:2 =
      transform.structured.tile_using_for %truncf_a_g tile_sizes [1, 1, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_truncf_b, %truncf_b_loops:2 =
      transform.structured.tile_using_for %truncf_b_g tile_sizes [1, 1, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile block_matmul for vectorization [2,2,1,0,0,0] then unroll

    %inner_most_matmul, %vec_loops:3 =
      transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
      transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !transform.any_op
    transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !transform.any_op

    %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_most_fills, %vec_fill_loops:2 =
      transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Vectorize all herds
    %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op

    %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

    %func1_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op

    // Re-vectorize after cleanup, then hoist transfers
    %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
    %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
    %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Cast vector.contract accumulator types (bf16->f32 for matmul)
    %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

    %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
    %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
    %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

    // Hoist extf/truncf pairs from the innermost loop.
    // The compute herd has truncf_op (f32->bf16) + block_matmul (bf16->f32 cast).
    // After vectorization, there are 4 extf and 4 truncf from the matmul contracts,
    // plus additional truncf from truncf_op. We hoist the 4 matmul pairs.
    %fors_to_hoist = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
    %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %all_extf = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
    %all_truncf = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op

    // Skip extf/truncf hoisting for now -- the truncf_op adds extra
    // cast ops that change the count. The matmul will still vectorize
    // correctly; hoisting is a performance optimization.

    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %func_fold_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_folded_3 = transform.air.fold_unit_extent_dims %func_fold_3 : (!transform.any_op) -> !transform.any_op
  transform.yield
  }
}
