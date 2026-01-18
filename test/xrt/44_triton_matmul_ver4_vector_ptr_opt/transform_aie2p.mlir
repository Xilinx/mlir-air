// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Matmul (Triton Ver4, Vectorized): Step-by-Step Annotated
// This script transforms a matmul IR into a tiled, packed, bufferized, and
// hardware-friendly form suitable for AIE execution. Each step is annotated
// with its purpose, assumptions, and relation to the IR.
//
// Target configuration: 8x4 AIE core array (Strix)
// Data types: BF16 inputs, F32 accumulation
////////////////////////////////////////////////////////////////////////////////

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):

    // Main transformation sequence begins.
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

    //==========================================================================
    // PHASE 1: TILE L3->L2 MEMORY COPIES
    // Purpose: Tile the memref copy ops that move data from L3 (DDR) to L2 (shared memory).
    //==========================================================================
    
    // Step 1: Convert memref.copy to linalg.copy and tile for L3->L2 data movement.
    // Purpose: Transforms memref copies into tileable linalg operations for streaming data.
    // Assumption: The IR contains memref.copy ops for A and B matrices.
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
    // PHASE 2: MATCH AND PREPARE CORE OPERATIONS
    // Purpose: Identify fill and matmul operations, promote output to L2.
    //==========================================================================

    // Step 2: Match the fill and matmul ops.
    // Assumption: The IR contains linalg.fill and linalg.matmul ops representing 
    // initialization and main computation.
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation

    // Step 3: Promote the result buffer (C matrix) to L2 shared memory.
    // Purpose: Allocate output buffer in L2 for accumulation before writing back to L3.
    // memory_space = 1 corresponds to L2 (shared memory).
        %result_l2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
            {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !pdl.operation

    //==========================================================================
    // PHASE 3: PACK MATMUL FOR VECTORIZED COMPUTATION
    // Purpose: Apply data tiling (packing) to enable efficient vectorized computation.
    //==========================================================================

    // Step 4: Pack matmul with tile sizes [8, 8, 8].
    // Purpose: Transforms linalg.matmul into linalg.generic with packed layout.
    // Assumption: Pack sizes [8, 8, 8] correspond to M, N, K tile dimensions for 
    // efficient AIE vector unit utilization.
        %packed = transform.structured.pack %matmul packed_sizes = [8, 8, 8]
          : (!pdl.operation) -> (!pdl.operation)

    // Step 5: Transpose A matrix for packed layout.
    // Purpose: Ensures A operand has correct memory layout for vectorized access.
    // Outer permutation [1, 0] swaps the outer tile dimensions.
        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 6: Transpose B matrix for packed layout.
    // Purpose: Ensures B operand has correct memory layout for vectorized access.
    // Both outer_perm and inner_perm [1, 0] transpose outer and inner tile dimensions.
        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 7: Transpose C matrix for packed layout.
    // Purpose: Ensures C operand has correct memory layout matching A and B.
    // Outer permutation [1, 0] aligns output tile dimensions.
        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 8: Promote the output pack operation to L1 local memory.
    // Purpose: Allocate L1 buffer for C matrix tiles during computation.
    // memory_space = 2 corresponds to L1 (AIE local memory).
        %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %pack_c
            {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !pdl.operation

    //==========================================================================
    // PHASE 4: TILE REDUCTION AND FUSE PACK OPERATIONS
    // Purpose: Tile the K dimension and fuse data movement into compute loops.
    //==========================================================================

    // Step 9: Tile the reduction (K) dimension.
    // Purpose: Enables streaming of A and B tiles along K dimension.
    // Tile size [0, 0, 8] tiles only the K dimension with factor 8.
        %tiled_reduction, %outer_for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 8]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 10: Fuse pack operations for A and B into the outer K-loop.
    // Purpose: Moves data packing inside the loop for better locality and pipelining.
        %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %pack_a into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %pack_b into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

    //==========================================================================
    // PHASE 5: TILE FOR MULTI-CORE PARALLELISM
    // Purpose: Create parallel loops for mapping to 8x4 AIE core array.
    //==========================================================================

    // Step 11: Tile matmul using scf.forall with tile size [8, 8, 0].
    // Purpose: Introduces parallelism across M and N dimensions for multi-core execution.
    // Tile sizes [8, 8, 0] create 8x8 tiles for each AIE core to process.
        %matmul_1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [8, 8, 0] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 12: Fuse pack operations into the inner parallel loop.
    // Purpose: Ensures each core has its own data packing for independent execution.
        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 13: Canonicalization and CSE after tiling.
    // Purpose: Cleans up IR, merges redundant ops, and prepares for further transforms.
        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_2 : !pdl.operation

    //==========================================================================
    // PHASE 6: PROMOTE INPUTS TO L1 AND TILE PROLOGUE/EPILOGUE
    // Purpose: Move input data to L1, create tiled fill (prologue) and unpack (epilogue).
    //==========================================================================

    // Step 14: Promote input operands (A and B tiles) to L1 local memory.
    // Purpose: Allocates L1 buffers for fast access during computation.
    // memory_space = 2 corresponds to L1 (AIE local memory).
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 15: Create tiled prologue (fill operation).
    // Purpose: Initializes output buffers in parallel across cores.
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

    // Step 16: Create tiled epilogue (unpack operation).
    // Purpose: Unpacks and writes results back to L2 in parallel across cores.
    // Tile sizes [64, 64] match the L2 tile dimensions.
        %unpack_op = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %epilogue_tiled_unpack, %epilogue_forall =
          transform.structured.tile_using_forall %unpack_op tile_sizes [64, 64]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 17: Canonicalization and CSE after buffer promotion.
    // Purpose: Merges redundant allocs/copies and simplifies the IR.
        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_3 : !pdl.operation

    //==========================================================================
    // PHASE 7: BUFFERIZATION AND AIR CLEANUP
    // Purpose: Convert tensors to memrefs and optimize memory operations.
    //==========================================================================

    // Step 18: One-shot bufferization of the function.
    // Purpose: Converts all remaining tensors to memrefs for hardware execution.
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

    // Step 19: AIR-specific cleanup and memory optimization.
    // Purpose: Removes uninitialized copies and eliminates redundant cascade memcpy patterns.
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
    // Purpose: Fuse L3->L2 copy loops with main compute loop for double buffering.
    //==========================================================================

    // Step 20: Fuse L3->L2 copy loops with the main K-reduction loop.
    // Purpose: Expose L2 pingpong buffering opportunity by interleaving L3->L2 data transfer with L2->L1.
        %for_loops = transform.structured.match ops{["scf.for"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %for_loop_copy_1, %for_loop_copy_2, %main_for_loop = transform.split_handle %for_loops : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %main_for_loop_norm = transform.air.normalize_for_bounds %main_for_loop // Fold affine apply into for loop bound
        transform.apply_cse to %func_op_updated_1 : !pdl.operation // Ensure loop bounds use shared cst ssa values
        %fused_for_loop_2 = transform.loop.fuse_sibling %for_loop_copy_2 into %main_for_loop_norm 
          : (!pdl.operation, !pdl.operation) -> !pdl.operation
        %fused_for_loop_1 = transform.loop.fuse_sibling %for_loop_copy_1 into %fused_for_loop_2 
          : (!pdl.operation, !pdl.operation) -> !pdl.operation

    //==========================================================================
    // PHASE 9: TILE FOR VECTORIZATION
    // Purpose: Final tiling to enable efficient vectorized execution on AIE vector units.
    //==========================================================================

    // Step 21: Tile linalg.generic (matmul) for vectorization.
    // Purpose: Creates inner loops with sizes suitable for vector register usage.
    // Tile sizes [2, 2, 1, 0, 0, 0] unroll M and N by 2 for register blocking.
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2 = transform.split_handle %linalg_generics : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %generic2 tile_sizes [2, 2, 1, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)   

    // Step 22: Further tile and unroll innermost loops for full vectorization.
    // Purpose: Completely unrolls the innermost M and N loops for register allocation.
        %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
          transform.structured.tile_using_for %inner_most_generics tile_sizes [1, 1, 0, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)  
        transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !pdl.operation
        transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !pdl.operation  

    // Step 23: Tile linalg.generic (fill) for vectorized initialization.
    // Purpose: Creates vector-sized tiles for efficient zero-initialization.
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic1 tile_sizes [1, 1]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)   

    //==========================================================================
    // PHASE 10: CONVERT TO AIE HERDS AND VECTORIZE
    // Purpose: Map parallel loops to AIE cores (herds) and apply vectorization.
    //==========================================================================

    // Step 24: Convert scf.forall loops to AIE herd operations.
    // Purpose: Maps parallel work to the 8x4 AIE core array.
    // Each forall becomes an air.herd representing multi-core execution.
        %foralls = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %forall1, %forall2, %forall3 = transform.split_handle %foralls : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %parallel1 = transform.loop.forall_to_parallel %forall1  : (!pdl.operation) -> !pdl.operation
        %herd1 = transform.air.par_to_herd %parallel1
        %parallel2 = transform.loop.forall_to_parallel %forall2  : (!pdl.operation) -> !pdl.operation
        %herd2 = transform.air.par_to_herd %parallel2
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!pdl.operation) -> !pdl.operation
        %herd3 = transform.air.par_to_herd %parallel3

    // Step 25: Apply vectorization to AIE herds.
    // Purpose: Converts scalar operations to vector operations for AIE vector units.
        %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %vectorized_herds = transform.air.herd_vectorize %herds

    // Step 26: Canonicalization after vectorization.
    // Purpose: Simplifies vector operations and folds unit extent dimensions.
        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func7 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !pdl.operation
                
    // Step 27: Eliminate redundant vector.transfer_read operations.
    // Purpose: Removes duplicate memory reads for better performance.
        %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func7

    //==========================================================================
    // PHASE 11: HOIST LOOP-INVARIANT VECTOR TRANSFERS
    // Purpose: Move vector reads/writes out of innermost loops for register reuse.
    //==========================================================================

    // Step 28: Match herds and prepare for hoisting optimization.
    // Purpose: Identifies herds and their vector operations for register optimization.
        %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %herds_1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %all_reads_in_herd2 = transform.structured.match ops{["vector.transfer_read"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        %all_writes_in_herd2 = transform.structured.match ops{["vector.transfer_write"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        
    // Step 29: Identify the innermost loop for hoisting.
    // Purpose: The innermost K-loop contains accumulator reads/writes that can be hoisted.
        %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
        %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        
    // Step 30: Split handles to get individual read/write operations.
    // Purpose: Identifies the 4 read-write pairs for C matrix accumulator tiles.
    // The 8 reads include: 4 for A tiles, 4 for C accumulator tiles.
    // The 4 writes are for C accumulator tiles.
        %read0, %read1, %read2, %read3, %read4, %read5, %read6, %read7 = transform.split_handle %all_reads_in_herd2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        %write0, %write1, %write2, %write3 = transform.split_handle %all_writes_in_herd2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        
    // Step 31: Cast vector types for correct accumulation precision.
    // Purpose: Ensures vector.contract uses F32 for accumulation (BF16 inputs -> F32 output).
        %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]}
        
    // Step 32: Hoist accumulator read/write pairs from innermost loop.
    // Purpose: Moves C matrix tile loads/stores outside the K-loop for register reuse.
    // Each pair accesses different C tile positions: [i,j], [i+1,j], [i,j+1], [i+1,j+1].
        // Pair 1: reads[2] and writes[0] - C tile at position [arg27, arg26]
        %innermost_for_updated = transform.air.hoist_loop_invariant_transfers %read2, %write0, %innermost_for
        // Pair 2: reads[4] and writes[1] - C tile at position [arg27+1, arg26]
        %innermost_for_updated_1 = transform.air.hoist_loop_invariant_transfers %read4, %write1, %innermost_for_updated
        // Pair 3: reads[6] and writes[2] - C tile at position [arg27, arg26+1]
        %innermost_for_updated_2 = transform.air.hoist_loop_invariant_transfers %read6, %write2, %innermost_for_updated_1
        // Pair 4: reads[7] and writes[3] - C tile at position [arg27+1, arg26+1]
        %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %read7, %write3, %innermost_for_updated_2

    // Step 33: Flatten loop iteration arguments and hoist vector transfer pointers.
    // Purpose: Simplifies loop structure and moves pointer computations out of loops.
        %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3
        %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4

    // Step 34: Final canonicalization pass.
    // Purpose: Cleans up the final IR for AIR/AIE lowering.
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
