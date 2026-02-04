// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Matmul with BF16 Output (Triton Ver4, Vectorized)
// 
// This script transforms a matmul IR into a tiled, packed, bufferized, and
// hardware-friendly form suitable for AIE execution.
//
// Target configuration: 8x4 AIE core array (Strix)
// Data types: BF16 inputs, F32 accumulation, BF16 output
//
// Memory Hierarchy:
//   L3 (DDR) -> L2 (Shared Memory, memory_space=1) -> L1 (AIE Local, memory_space=2)
////////////////////////////////////////////////////////////////////////////////

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):

    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

    //==========================================================================
    // PHASE 1: TILE L3->L2 MEMORY COPIES
    // Convert memref.copy to linalg.copy and tile for streaming data movement.
    //==========================================================================
    
    // Step 1: Convert memref.copy ops to linalg.copy and tile them.
    // This transforms the A and B matrix copies from L3 to L2 into tileable loops.
        %func10 = transform.structured.match ops{["func.func"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %func10_updated = transform.air.convert_memref_copy_to_linalg_copy %func10
        %copies = transform.structured.match ops{["linalg.copy"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %copy1, %copy2 = transform.split_handle %copies : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_copy1, %tile_copy_loop1 =
          transform.structured.tile_using_for %copy1 tile_sizes [0, 64]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_copy2, %tile_copy_loop2 =
          transform.structured.tile_using_for %copy2 tile_sizes [64]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    //==========================================================================
    // PHASE 2: FUSE TRUNCF AND PREPARE MATMUL
    // Fuse the output truncation into matmul and promote output buffer to L2.
    //==========================================================================

    // Step 2: Match the fill and matmul operations.
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation

    // Step 3: Fuse the truncf linalg.generic into the matmul.
    // This produces BF16 output directly from the F32 accumulation.
        %matmul_to_fuse = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %truncf_generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fused_generic = transform.air.fuse_truncf_linalg %truncf_generic, %matmul_to_fuse
        %fused_matmul = transform.structured.specialize %fused_generic : (!pdl.operation) -> !pdl.operation

    // Step 4: Promote the result buffer (C matrix) to L2 shared memory.
    // memory_space = 1 corresponds to L2 (shared memory).
        %result_l2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
            {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !pdl.operation
        

    //==========================================================================
    // PHASE 3: PACK MATMUL FOR VECTORIZED COMPUTATION
    // Apply data tiling (packing) to enable efficient vectorized computation.
    //==========================================================================

    // Step 5: Pack matmul with tile sizes [8, 8, 8] for M, N, K dimensions.
    // This transforms linalg.matmul into linalg.generic with packed layout
    // optimized for AIE vector unit utilization.
        %packed = transform.structured.pack %fused_matmul packed_sizes = [8, 8, 8]
          : (!pdl.operation) -> (!pdl.operation)

    // Step 6: Transpose A matrix pack for correct memory layout.
    // Outer permutation [1, 0] swaps the outer tile dimensions.
        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 7: Transpose B matrix pack for correct memory layout.
    // Both outer_perm and inner_perm [1, 0] transpose outer and inner tile dimensions.
        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 8: Transpose C matrix pack/unpack for correct memory layout.
        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 9: Promote the output pack operation to L1 local memory.
    // memory_space = 2 corresponds to L1 (AIE local memory).
        %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %pack_c
            {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !pdl.operation

    //==========================================================================
    // PHASE 4: TILE REDUCTION AND FUSE PACK OPERATIONS
    // Tile the K dimension and fuse data movement into compute loops.
    //==========================================================================

    // Step 10: Tile the reduction (K) dimension with factor 8.
    // This enables streaming of A and B tiles along the K dimension.
        %tiled_reduction, %outer_for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 8]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 11: Fuse pack operations for A and B into the outer K-loop.
    // This moves data packing inside the loop for better locality and pipelining.
        %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %pack_a into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %pack_b into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

    //==========================================================================
    // PHASE 5: TILE FOR MULTI-CORE PARALLELISM
    // Create parallel loops for mapping to 8x4 AIE core array.
    //==========================================================================

    // Step 12: Tile matmul using scf.forall with tile sizes [8, 8, 0].
    // This introduces parallelism across M and N dimensions for multi-core execution.
        %matmul_1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [8, 8, 0] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 13: Fuse pack operations into the inner parallel loop.
    // This ensures each core has its own data packing for independent execution.
        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 14: Canonicalization and CSE after tiling.
        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_2 : !pdl.operation

    //==========================================================================
    // PHASE 6: PROMOTE INPUTS TO L1 AND TILE PROLOGUE/EPILOGUE
    // Move input data to L1, create tiled fill (prologue) and unpack (epilogue).
    //==========================================================================

    // Step 15: Promote input operands (A and B tiles) to L1 local memory.
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 16: Create tiled prologue (fill operation).
    // Generalize fill to generic, interchange dimensions, then tile with forall.
        %fill_op = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %generic_fill_op = transform.structured.generalize %fill_op
            : (!pdl.operation) -> !pdl.operation
        %interchanged_fill_op = transform.structured.interchange %generic_fill_op 
          iterator_interchange = [1, 0, 2, 3]
          : (!pdl.operation) -> !pdl.operation
        %prologue_tiled_fill, %prologue_forall =
          transform.structured.tile_using_forall %interchanged_fill_op tile_sizes [8, 8]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 17: Create tiled epilogue (unpack operation).
    // Tile sizes [64, 64] match the L2 tile dimensions.
        %unpack_op = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %epilogue_tiled_unpack, %epilogue_forall =
          transform.structured.tile_using_forall %unpack_op tile_sizes [64, 64]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 18: Canonicalization and CSE after buffer promotion.
        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_3 : !pdl.operation

    //==========================================================================
    // PHASE 7: BUFFERIZATION AND MEMORY OPTIMIZATION
    // Convert tensors to memrefs and optimize memory operations.
    //==========================================================================

    // Step 19: One-shot bufferization of the function.
    // Converts all remaining tensors to memrefs for hardware execution.
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

    // Step 20: AIR-specific cleanup and memory optimization.
    // Removes uninitialized copies and eliminates redundant cascade memcpy patterns.
        %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func6 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func6 : !pdl.operation
        transform.apply_patterns to %func6 {
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        %func_op_updated = transform.air.remove_uninitialized_copy %func6
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated

    //==========================================================================
    // PHASE 8: FUSE LOOPS FOR L2 PINGPONG BUFFERING
    // Fuse L3->L2 copy loops with main compute loop for double buffering.
    //==========================================================================

    // Step 21: Fuse L3->L2 copy loops with the main K-reduction loop.
    // This exposes L2 pingpong buffering opportunity by interleaving data transfer.
        %for_loops = transform.structured.match ops{["scf.for"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %for_loop_copy_1, %for_loop_copy_2, %main_for_loop = transform.split_handle %for_loops : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %main_for_loop_norm = transform.air.normalize_for_bounds %main_for_loop
        transform.apply_cse to %func_op_updated_1 : !pdl.operation
        %fused_for_loop_2 = transform.loop.fuse_sibling %for_loop_copy_2 into %main_for_loop_norm 
          : (!pdl.operation, !pdl.operation) -> !pdl.operation
        %fused_for_loop_1 = transform.loop.fuse_sibling %for_loop_copy_1 into %fused_for_loop_2 
          : (!pdl.operation, !pdl.operation) -> !pdl.operation

    //==========================================================================
    // PHASE 9: TILE FOR VECTORIZATION
    // Final tiling to enable efficient vectorized execution on AIE vector units.
    //==========================================================================

    // Step 22: Tile linalg.generic (matmul) for vectorization.
    // Tile sizes [2, 2, 1, 0, 0, 0] create register blocking for M and N.
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2 = transform.split_handle %linalg_generics : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %generic2 tile_sizes [2, 2, 1, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)   

    // Step 23: Further tile and unroll innermost loops for full vectorization.
    // Completely unrolls the innermost M and N loops for register allocation.
        %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
          transform.structured.tile_using_for %inner_most_generics tile_sizes [1, 1, 0, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)  
        transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !pdl.operation
        transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !pdl.operation  

    // Step 24: Tile linalg.generic (fill) for vectorized initialization.
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic1 tile_sizes [1, 1]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)   

    //==========================================================================
    // PHASE 10: CONVERT TO AIE HERDS AND VECTORIZE
    // Map parallel loops to AIE cores (herds) and apply vectorization.
    //==========================================================================

    // Step 25: Convert scf.forall loops to AIE herd operations.
    // Each forall becomes an air.herd representing multi-core execution.
        %foralls = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %forall1, %forall2, %forall3 = transform.split_handle %foralls : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %parallel1 = transform.loop.forall_to_parallel %forall1  : (!pdl.operation) -> !pdl.operation
        %herd1 = transform.air.par_to_herd %parallel1
        %parallel2 = transform.loop.forall_to_parallel %forall2  : (!pdl.operation) -> !pdl.operation
        %herd2 = transform.air.par_to_herd %parallel2
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!pdl.operation) -> !pdl.operation
        %herd3 = transform.air.par_to_herd %parallel3

    // Step 26: Apply vectorization to AIE herds.
    // Converts scalar operations to vector operations for AIE vector units.
        %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %vectorized_herds = transform.air.herd_vectorize %herds

    // Step 27: Canonicalization after vectorization.
        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func7 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !pdl.operation
                
    // Step 28: Eliminate redundant vector.transfer_read operations.
        %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func7

    //==========================================================================
    // PHASE 11: HOIST LOOP-INVARIANT VECTOR TRANSFERS
    // Move vector reads/writes out of innermost loops for register reuse.
    //==========================================================================

    // Step 29: Identify herds and vector operations for hoisting.
    // The matmul herd (herd2) contains the accumulator reads/writes to optimize.
        %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %herds_1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %all_reads_in_herd2 = transform.structured.match ops{["vector.transfer_read"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        %all_writes_in_herd2 = transform.structured.match ops{["vector.transfer_write"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        
    // Step 30: Identify the innermost K-loop for hoisting.
        %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        
    // Step 31: Split handles to get individual read/write operations.
    // After unrolling, there are 8 reads (4 for A tiles, 4 for C accumulators)
    // and 4 writes (for C accumulators) due to 2x2 unrolling.
        %read0, %read1, %read2, %read3, %read4, %read5, %read6, %read7 = transform.split_handle %all_reads_in_herd2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        %write0, %write1, %write2, %write3 = transform.split_handle %all_writes_in_herd2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        
    // Step 32: Cast vector types for correct accumulation precision.
    // Ensures vector.contract uses F32 for accumulation (BF16 inputs -> F32 output).
        %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]}

    // Step 33: Hoist accumulator read/write pairs from innermost K-loop.
    // Moves C matrix tile loads/stores outside the loop for register reuse.
    // Each of the 4 pairs corresponds to a position in the 2x2 unrolled tile.
        %innermost_for_updated = transform.air.hoist_loop_invariant_transfers %read2, %write0, %innermost_for
        %innermost_for_updated_1 = transform.air.hoist_loop_invariant_transfers %read4, %write1, %innermost_for_updated
        %innermost_for_updated_2 = transform.air.hoist_loop_invariant_transfers %read6, %write2, %innermost_for_updated_1
        %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %read7, %write3, %innermost_for_updated_2

    //==========================================================================
    // PHASE 12: HOIST EXTF/TRUNCF CAST PAIRS FOR BF16 OUTPUT
    // Move BF16<->F32 conversions out of innermost loop for efficiency.
    //==========================================================================

    // Step 34: Match extf/truncf operations in the innermost loop.
    // These handle BF16 accumulator -> F32 compute -> BF16 store conversions.
        %fors_to_hoist_ptrs = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist_ptrs {overflow_result = 1}: (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        %all_extf_loop = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!pdl.operation) -> !pdl.operation
        %all_truncf_loop = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!pdl.operation) -> !pdl.operation

    // Step 35: Hoist extf/truncf pairs iteratively.
    // There are 4 pairs corresponding to the 4 vector.contract results.
    // Each pair is hoisted one at a time, re-matching after each hoist.
        
        // Split to get individual operations (4 extf, 4 truncf)
        %extf_bf16_1, %extf_bf16_2, %extf_bf16_3, %extf_bf16_4 = transform.split_handle %all_extf_loop : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        %truncf_1, %truncf_2, %truncf_3, %truncf_4 = transform.split_handle %all_truncf_loop : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        
        // Hoist first extf/truncf pair
        %for1_1_hoisted_1 = transform.air.hoist_cast_pair %extf_bf16_1, %truncf_1, %innermost_for1
        
        // Re-match and hoist second pair
        %all_extf_loop_2 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_1 : (!pdl.operation) -> !pdl.operation
        %all_truncf_loop_2 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_1 : (!pdl.operation) -> !pdl.operation
        %extf_bf16_2_new, %e2_5, %e2_6 = transform.split_handle %all_extf_loop_2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %truncf_2_1, %truncf_2_2, %truncf_2_3 = transform.split_handle %all_truncf_loop_2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %for1_1_hoisted_2 = transform.air.hoist_cast_pair %extf_bf16_2_new, %truncf_2_1, %for1_1_hoisted_1
        
        // Re-match and hoist third pair
        %all_extf_loop_3 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_2 : (!pdl.operation) -> !pdl.operation
        %all_truncf_loop_3 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_2 : (!pdl.operation) -> !pdl.operation
        %extf_bf16_3_new, %e3_7 = transform.split_handle %all_extf_loop_3 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %truncf_3_1, %truncf_3_2 = transform.split_handle %all_truncf_loop_3 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %for1_1_hoisted_3 = transform.air.hoist_cast_pair %extf_bf16_3_new, %truncf_3_1, %for1_1_hoisted_2
        
        // Re-match and hoist fourth pair
        %all_extf_loop_4 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_3 : (!pdl.operation) -> !pdl.operation
        %all_truncf_loop_4 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_3 : (!pdl.operation) -> !pdl.operation
        %for1_1_hoisted_final = transform.air.hoist_cast_pair %all_extf_loop_4, %all_truncf_loop_4, %for1_1_hoisted_3

    //==========================================================================
    // PHASE 13: FINAL LOOP OPTIMIZATIONS
    // Flatten iteration arguments and hoist pointer computations.
    //==========================================================================

    // Step 36: Flatten loop iteration arguments.
    // Simplifies the loop structure by flattening iter_args.
        %innermost_for_updated_4 = transform.air.flatten_for_iter_args %for1_1_hoisted_final
        %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4

    // Step 37: Final canonicalization pass.
    // Cleans up the final IR for AIR/AIE lowering.
        %func9 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func9 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !pdl.operation

    }
}
