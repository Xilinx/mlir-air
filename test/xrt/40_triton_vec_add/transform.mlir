// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Vector Addition: Step-by-Step Annotated
// This script transforms a simple elementwise vector addition IR into a tiled,
// bufferized, and hardware-friendly form suitable for AIE execution.
// Each step is annotated with its purpose, assumptions, and relation to the IR.
////////////////////////////////////////////////////////////////////////////////

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):

    // Main transformation sequence begins.
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

    // Step 1: Match the main elementwise op (linalg.generic).
    // Assumption: The IR contains a linalg.generic op representing the elementwise add.
    // This is the main computation to be transformed.
        %add = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation

    // Step 2: Flatten elementwise op.
    // Purpose: Converts multi-dimensional elementwise ops into a 1D form for easier tiling/vectorization.
    // Assumption: The op is elementwise and can be flattened without changing semantics.
        %add_flattened = transform.structured.flatten_elementwise %add
        : (!pdl.operation) -> !pdl.operation

    // Step 3: Bufferize result to shared (L2) memory allocation.
    // Purpose: Allocates the result buffer in memory space 1 (shared/L2), required for AIR/AIE memory hierarchy.
    // Assumption: The result of the elementwise op will be written to L2/shared memory.
        %add_res_shared, %new_add = transform.structured.bufferize_to_allocation %add_flattened
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 4: Tile the computation using scf.forall with tile size 64.
    // Purpose: Introduces parallelism and prepares for mapping to AIE columns.
    // Assumption: The problem size is a multiple of 64, or padding will be handled later.
        %add_1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %tiled_add_1, %forall_add_1 =
          transform.structured.tile_using_forall %add_1 tile_sizes [64] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 5: Run canonicalization and CSE.
    // Purpose: Cleans up the IR after tiling, merges redundant ops, and prepares for further transforms.
    // Assumption: Canonicalization will simplify the IR and remove dead code.
        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_2 : !pdl.operation

    // Step 6: Match the (possibly tiled) linalg.generic for further transformation.
        %add_2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation

    // Step 7: Pad the operation.
    // Purpose: Ensures that the computation is aligned to tile sizes, handles boundary conditions.
    // Assumption: Padding values/types are correct for the op; nofold_flags prevent folding of padding.
        %padded_add, %pad_add, %__ = transform.structured.pad %add_2 {
            padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
            padding_dimensions=[0, 1, 2],
            nofold_flags=[1, 1, 1],
            copy_back_op="linalg.copy"
        } : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)

    // Step 8: Rewrite in destination-passing style (DPS).
    // Purpose: Converts the op to DPS, which is required for bufferization and explicit memory management.
    // Assumption: The op supports DPS conversion.
        %pad_dps_add = transform.structured.rewrite_in_destination_passing_style %pad_add : (!pdl.operation) -> !pdl.operation

    // Step 9: Promote the operands to local memory (AIE local, memory_space=2).
    // Purpose: Moves input operands to fast local memory for efficient AIE execution.
    // Assumption: The operands are suitable for promotion and local memory is available.
        %padded_add_lhs = transform.get_producer_of_operand %padded_add[0] : (!pdl.operation) -> (!pdl.operation)
        %padded_add_lhs_buffer, %padded_add_lhs_new = transform.structured.bufferize_to_allocation %padded_add_lhs
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %padded_add_rhs = transform.get_producer_of_operand %padded_add[1] : (!pdl.operation) -> (!pdl.operation)
        %padded_add_rhs_buffer, %padded_add_rhs_new = transform.structured.bufferize_to_allocation %padded_add_rhs
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 10: Promote the result to local memory (AIE local, memory_space=2).
    // Purpose: Ensures the result buffer is also in local memory for fast access.
    // Assumption: The result fits in local memory and can be promoted.
        %padded_add_result = transform.get_producer_of_operand %padded_add[2] : (!pdl.operation) -> (!pdl.operation)
        %padded_add_result_buffer, %padded_add_result_new = transform.structured.bufferize_to_allocation %padded_add_result
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

    // Step 11: Run canonicalization and CSE again.
    // Purpose: Cleans up after bufferization and promotion, merges redundant allocs/copies.
    // Assumption: Canonicalization will further simplify the IR.
        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_3 : !pdl.operation

    // Step 12: One-shot bufferization of the function.
    // Purpose: Converts all tensors to memrefs, finalizes bufferization for AIR/AIE lowering.
    // Assumption: The function is now in DPS form and ready for bufferization.
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

    // Step 13: Final canonicalization and AIR-specific cleanup.
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
        %linalg_copies = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %memref_copies = transform.structured.linalg_copy_to_memref %linalg_copies : (!pdl.operation) -> !pdl.operation
        %func_op_updated = transform.air.remove_uninitialized_copy %func6
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated

    // Step 14: Tile linalg.add for vectorization (tile size 16).
    // Purpose: Final tiling to enable vectorized execution on AIE hardware.
    // Assumption: The innermost dimension is a multiple of 16, or padding has handled the remainder. Vec size 16 for @llvm.aie2.add.accfloat(<8 x i64> %acc1, <8 x i64> %acc2).
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    // Step 15: AIR Constructs Mapping
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
