// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Matmul (Triton Ver3, Vectorized): Step-by-Step Annotated
// This script transforms a matmul IR into a tiled, packed, bufferized, and
// hardware-friendly form suitable for AIE execution. Each step is annotated
// with its purpose, assumptions, and relation to the IR.
////////////////////////////////////////////////////////////////////////////////

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):

    // Main transformation sequence begins.
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

    // Step 1: Match the fill and matmul ops.
    // Assumption: The IR contains linalg.fill and linalg.matmul ops representing initialization and main computation.
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation

    // Step 2: Bufferize fill result to shared (L2) memory allocation.
    // Purpose: Allocates the result buffer in memory space 1 (shared/L2), required for AIR/AIE memory hierarchy.
    // Assumption: The result of the fill op will be written to L2/shared memory.
        %buffer_res_shared, %new_fill = transform.structured.bufferize_to_allocation %fill
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 3: Tile matmul using scf.forall with tile size [64, 64].
    // Purpose: Introduces parallelism and prepares for mapping to AIE columns.
    // Assumption: The problem size is a multiple of 64, or padding will be handled later.
        %matmul_1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %tiled_matmul_1, %forall_1 =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [64, 64] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 4: Run canonicalization and CSE.
    // Purpose: Cleans up the IR after tiling, merges redundant ops, and prepares for further transforms.
    // Assumption: Canonicalization will simplify the IR and remove dead code.
        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_2 : !pdl.operation

    // Step 5: Fuse fill operation into the forall loop.
    // Purpose: Ensures initialization is fused with computation for efficiency.
    // Assumption: The fill op is a direct consumer in the loop.
        %fused_fill_1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fill_consumer = transform.get_consumers_of_result %fused_fill_1[0] : (!pdl.operation) -> (!pdl.operation)
        %fused_fill_2, %fused_loop_2 = transform.structured.fuse_into_containing_op %fused_fill_1 into %fill_consumer : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 6: Pack by applying data tiling; linalg.matmul becomes linalg.generic.
    // Purpose: Prepares data for vectorized computation and memory layout optimization.
    // Assumption: Packing sizes are chosen for hardware efficiency.
        %packed = transform.structured.pack %tiled_matmul_1 packed_sizes = [4, 4, 8]
          : (!pdl.operation) -> (!pdl.operation)

    // Step 7: Transpose A matrix for packed layout.
    // Purpose: Ensures correct memory layout for A operand.
    // Assumption: Outer permutation [1, 0] is correct for hardware mapping.
        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 8: Transpose B matrix for packed layout.
    // Purpose: Ensures correct memory layout for B operand.
    // Assumption: Outer and inner permutations [1, 0] are correct for hardware mapping.
        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 9: Transpose C matrix for packed layout.
    // Purpose: Ensures correct memory layout for C operand.
    // Assumption: Outer permutation [1, 0] is correct for hardware mapping.
        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 10: Bufferize result to local memory allocation (AIE local, memory_space=2).
    // Purpose: Moves result buffer to fast local memory for efficient AIE execution.
    // Assumption: The result fits in local memory and can be promoted.
        %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 11: Tile the reduction loop.
    // Purpose: Enables vectorized reduction and efficient computation.
    // Assumption: Tile size [0, 0, 4] is chosen for hardware efficiency.
        %tiled_reduction, %for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 4]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 12: Fuse pack ops into the for loop.
    // Purpose: Ensures packed data is available within the reduction loop.
    // Assumption: Packing ops are direct consumers in the loop.
        %fused_pack_a, %e1 = transform.structured.fuse_into_containing_op %pack_a into %for_loop
          : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_pack_b, %e2 = transform.structured.fuse_into_containing_op %pack_b into %for_loop
          : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 13: Promote the inputs to local memory (AIE local, memory_space=2).
    // Purpose: Moves input operands to fast local memory for efficient AIE execution.
    // Assumption: The operands are suitable for promotion and local memory is available.
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_pack_a
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_pack_b
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 14: Run canonicalization and CSE again.
    // Purpose: Cleans up after bufferization and promotion, merges redundant allocs/copies.
    // Assumption: Canonicalization will further simplify the IR.
        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_3 : !pdl.operation

    // Step 15: One-shot bufferization of the function.
    // Purpose: Converts all tensors to memrefs, finalizes bufferization for AIR/AIE lowering.
    // Assumption: The function is now in DPS form and ready for bufferization.
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

    // Step 16: Final canonicalization and AIR-specific cleanup.
    // Purpose: Removes redundant memcpy ops, eliminates cascade memcpy patterns, and canonicalizes.
    // Assumption: AIR passes will further optimize memory ops for hardware.
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

    // Step 17: Tile linalg.generics for vectorization.
    // Purpose: Final tiling to enable vectorized execution on AIE hardware.
    // Assumption: Tile sizes [1, 1, 1, 0, 0, 0] are chosen for hardware vectorization.
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [1, 1, 1, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)     

    // Step 18: Tile linalg.fills for vectorized write.
    // Purpose: Enables vectorized write for initialization.
    // Assumption: Tile sizes [1, 1] are chosen for hardware vectorization.
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %linalg_fills tile_sizes [1, 1]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)   

    // Step 19: AIR Constructs Mapping
    // Purpose: Convert high-level parallel constructs to AIE-specific operations for hardware execution.
    // Convert parallel loops to AIE herd operations for multi-core execution
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallel

    // Convert memory copies to DMA operations for efficient data movement
        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
        
    // Apply vectorization to optimize for AIE vector units
        %vectorized_herd = transform.air.herd_vectorize %herd
    }
}
