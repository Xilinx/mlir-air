// Transform Script for 128x256 Matmul with BF16 Output (Triton Ver4, Vectorized)
//
// Adapted from test/xrt/48_triton_matmul_ver4_strix_4x4_bf16_output/transform_aie2p.mlir
// for 128x256 output tile (M_TILE=128, N_TILE=256, K=256).
//
// Target configuration: 4x2 AIE core array (Strix)
// Data types: BF16 inputs, F32 accumulation, BF16 output
//
// After packing [8,8,8] with C outer_perm [1,0]:
//   packed shape = [N/8, M/8, K/8, 8, 8, 8] = [32, 16, 32, 8, 8, 8]
// Phase 5 forall [8, 8, 0] → herd 4x2 = 8 cores
//
// Memory Hierarchy:
//   L3 (DDR) -> L2 (Shared Memory, memory_space=1) -> L1 (AIE Local, memory_space=2)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    //==========================================================================
    // PHASE 1: TILE L3->L2 MEMORY COPIES
    // Convert memref.copy to linalg.copy and tile for streaming data movement.
    //==========================================================================

        %func10 = transform.structured.match ops{["func.func"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %func10_updated = transform.air.convert_memref_copy_to_linalg_copy %func10 : (!transform.any_op) -> !transform.any_op
        %copies = transform.structured.match ops{["linalg.copy"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %copy1, %copy2 = transform.split_handle %copies : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_copy1, %tile_copy_loop1 =
          transform.structured.tile_using_for %copy1 tile_sizes [0, 64]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %tile_copy_loop1 "copy_a_loop" : !transform.any_op
        %tiled_copy2, %tile_copy_loop2 =
          transform.structured.tile_using_for %copy2 tile_sizes [64]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %tile_copy_loop2 "copy_b_loop" : !transform.any_op

    //==========================================================================
    // PHASE 2: FUSE TRUNCF AND PREPARE MATMUL
    // Fuse the output truncation into matmul and promote output buffer to L2.
    //==========================================================================

        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!transform.any_op) -> !transform.any_op

        %matmul_to_fuse = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %truncf_generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fused_generic = transform.air.fuse_truncf_linalg %truncf_generic, %matmul_to_fuse : (!transform.any_op, !transform.any_op) -> !transform.any_op
        %fused_matmul = transform.structured.specialize %fused_generic : (!transform.any_op) -> !transform.any_op

        %result_l2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
            {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !transform.any_op


    //==========================================================================
    // PHASE 3: PACK MATMUL FOR VECTORIZED COMPUTATION
    // Pack sizes [8, 8, 8] for M, N, K dimensions.
    //==========================================================================

        %packed = transform.structured.pack %fused_matmul packed_sizes = [8, 8, 8]
          : (!transform.any_op) -> (!transform.any_op)

        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %pack_c
            {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    //==========================================================================
    // PHASE 4: TILE REDUCTION AND FUSE PACK OPERATIONS
    // Tile K dimension with factor 8 and fuse packs into K-loop.
    //==========================================================================

        %tiled_reduction, %outer_for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 8]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %outer_for_loop "k_reduction_loop" : !transform.any_op

        %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %pack_a into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %pack_b into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    //==========================================================================
    // PHASE 5: TILE FOR MULTI-CORE PARALLELISM
    // For 128x256 tile: packed dims [32, 16], forall [8, 8, 0] → 4x2 herd = 8 cores.
    //==========================================================================

        %matmul_1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [8, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op
        transform.annotate %tiled_matmul_1 "matmul_compute" : !transform.any_op

        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_2 : !transform.any_op

    //==========================================================================
    // PHASE 6: PROMOTE INPUTS TO L1 AND TILE PROLOGUE/EPILOGUE
    //==========================================================================

        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Prologue: fill → generalize → interchange → tile_using_forall
        %fill_op = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generic_fill_op = transform.structured.generalize %fill_op
            : (!transform.any_op) -> !transform.any_op
        transform.annotate %generic_fill_op "init_fill" : !transform.any_op
        %interchanged_fill_op = transform.structured.interchange %generic_fill_op
          iterator_interchange = [1, 0, 2, 3]
          : (!transform.any_op) -> !transform.any_op
        %prologue_tiled_fill, %prologue_forall =
          transform.structured.tile_using_forall %interchanged_fill_op tile_sizes [8, 8]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %prologue_forall "prologue_forall" : !transform.any_op

    // Epilogue: unpack → tile_using_forall [64, 64]
        %unpack_op = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %epilogue_tiled_unpack, %epilogue_forall =
          transform.structured.tile_using_forall %unpack_op tile_sizes [64, 64]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %epilogue_forall "epilogue_forall" : !transform.any_op

        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_3 : !transform.any_op

    //==========================================================================
    // PHASE 7: BUFFERIZATION AND MEMORY OPTIMIZATION
    //==========================================================================

        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

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
        %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated : (!transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 8: FUSE LOOPS FOR L2 PINGPONG BUFFERING
    //==========================================================================

        %for_loop_copy_1 = transform.structured.match ops{["scf.for"]} attributes{copy_a_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        %for_loop_copy_2 = transform.structured.match ops{["scf.for"]} attributes{copy_b_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        %main_for_loop = transform.structured.match ops{["scf.for"]} attributes{k_reduction_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        %main_for_loop_norm = transform.air.normalize_for_bounds %main_for_loop : (!transform.any_op) -> !transform.any_op
        transform.apply_cse to %func_op_updated_1 : !transform.any_op
        %fused_for_loop_2 = transform.loop.fuse_sibling %for_loop_copy_2 into %main_for_loop_norm
          : (!transform.any_op, !transform.any_op) -> !transform.any_op
        %fused_for_loop_1 = transform.loop.fuse_sibling %for_loop_copy_1 into %fused_for_loop_2
          : (!transform.any_op, !transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 9: TILE FOR VECTORIZATION
    //==========================================================================

        %generic1 = transform.structured.match ops{["linalg.generic"]} attributes{init_fill} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generic2 = transform.structured.match ops{["linalg.generic"]} attributes{matmul_compute} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %generic2 tile_sizes [2, 2, 1, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
          transform.structured.tile_using_for %inner_most_generics tile_sizes [1, 1, 0, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !transform.any_op
        transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !transform.any_op

        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic1 tile_sizes [1, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    //==========================================================================
    // PHASE 10: CONVERT TO AIE HERDS AND VECTORIZE
    //==========================================================================

        %forall1 = transform.structured.match ops{["scf.forall"]} attributes{prologue_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall2 = transform.structured.match ops{["scf.forall"]} attributes{compute_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall3 = transform.structured.match ops{["scf.forall"]} attributes{epilogue_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel1 = transform.loop.forall_to_parallel %forall1  : (!transform.any_op) -> !transform.any_op
        %herd1 = transform.air.par_to_herd %parallel1 : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd1 "prologue_herd" : !transform.any_op
        %parallel2 = transform.loop.forall_to_parallel %forall2  : (!transform.any_op) -> !transform.any_op
        %herd2 = transform.air.par_to_herd %parallel2 : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd2 "compute_herd" : !transform.any_op
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!transform.any_op) -> !transform.any_op
        %herd3 = transform.air.par_to_herd %parallel3 : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd3 "epilogue_herd" : !transform.any_op

        %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

        %func7_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func7_rematch : (!transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 11: HOIST LOOP-INVARIANT VECTOR TRANSFERS
    //==========================================================================

        %herd2_1 = transform.structured.match ops{["air.herd"]} attributes{compute_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
        %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

        %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 12: HOIST EXTF/TRUNCF CAST PAIRS FOR BF16 OUTPUT
    //==========================================================================

        %fors_to_hoist_ptrs = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
        %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist_ptrs {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %all_extf_loop = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
        %extf_bf16_1, %extf_bf16_2, %extf_bf16_3, %extf_bf16_4 = transform.split_handle %all_extf_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %truncf_1, %truncf_2, %truncf_3, %truncf_4 = transform.split_handle %all_truncf_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %for1_1_hoisted_1 = transform.air.hoist_cast_pair %extf_bf16_1, %truncf_1, %innermost_for1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
        %all_extf_loop_2 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop_2 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
        %extf_bf16_2_new, %e2_5, %e2_6 = transform.split_handle %all_extf_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %truncf_2_1, %truncf_2_2, %truncf_2_3 = transform.split_handle %all_truncf_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %for1_1_hoisted_2 = transform.air.hoist_cast_pair %extf_bf16_2_new, %truncf_2_1, %for1_1_hoisted_1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
        %all_extf_loop_3 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop_3 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
        %extf_bf16_3_new, %e3_7 = transform.split_handle %all_extf_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %truncf_3_1, %truncf_3_2 = transform.split_handle %all_truncf_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %for1_1_hoisted_3 = transform.air.hoist_cast_pair %extf_bf16_3_new, %truncf_3_1, %for1_1_hoisted_2 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
        %all_extf_loop_4 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop_4 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
        %for1_1_hoisted_final = transform.air.hoist_cast_pair %all_extf_loop_4, %all_truncf_loop_4, %for1_1_hoisted_3 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 13: FINAL LOOP OPTIMIZATIONS
    //==========================================================================

        %innermost_for_updated_4 = transform.air.flatten_for_iter_args %for1_1_hoisted_final : (!transform.any_op) -> !transform.any_op
        %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

        %func9 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func9 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
