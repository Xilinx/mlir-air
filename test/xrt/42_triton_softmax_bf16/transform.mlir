// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton Softmax Tiling Recipe Transform Script
//===----------------------------------------------------------------------===//
// This transform script implements a comprehensive tiling and optimization
// strategy for softmax operations targeting AIE (AI Engine) hardware.
//
// SOFTMAX DECOMPOSITION OVERVIEW:
// The softmax function: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// is decomposed into the following computational stages:
// 1. Find maximum value across reduction dimension: max_val = max(input)
// 2. Subtract maximum from input: shifted = input - max_val  
// 3. Compute exponential: exp_vals = exp(shifted)
// 4. Sum exponentials: sum_exp = sum(exp_vals)
// 5. Divide by sum: output = exp_vals / sum_exp
//
// MEMORY HIERARCHY STRATEGY:
// - Memory space 0: Default/global memory (DDR)
// - Memory space 1: L2 memory (shared across cores)
// - Memory space 2: L1 memory (per-core local memory)
//
// The recipe assumes:
// 1. Input operations are in linalg dialect form
// 2. The softmax computation is decomposed into multiple linalg.generic ops
// 3. Memory hierarchy optimization is needed (L1/L2 memory spaces)
// 4. Operations can be fused for better performance
// 5. Vectorization is required to utilize AIE vector units efficiently
//===----------------------------------------------------------------------===//

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        // PURPOSE: Prepare the IR for subsequent transformations by applying
        // standard optimization patterns that simplify operations and remove
        // redundancies. This creates a clean foundation for tiling and fusion.
        
        // Match the function containing all softmax operations
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        
        // Apply comprehensive canonicalization patterns:
        transform.apply_patterns to %func0 {
            // Simplify tiling-related patterns (e.g., empty tensor operations)
            transform.apply_patterns.linalg.tiling_canonicalization
            // Optimize SCF for loops (e.g., loop bounds, step simplification)
            transform.apply_patterns.scf.for_loop_canonicalization
            // General MLIR canonicalization (constant folding, dead code elimination)
            transform.apply_patterns.canonicalization
            // CRITICAL: Remove unit dimensions and simplify tensor shapes
            // This is essential for AIE hardware which has specific shape constraints
            // and enables more efficient tiling patterns in subsequent phases
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !pdl.operation
        
        // Apply Common Subexpression Elimination to remove duplicate computations
        transform.apply_cse to %func0 : !pdl.operation

        //===================================================================
        // PHASE 2: Operation Preparation and Handle Splitting
        //===================================================================
        // PURPOSE: Split operation handles to enable individual manipulation of each
        // softmax computation stage.
        //
        // SOFTMAX OPERATION MAPPING:
        // - fill1, fill2: Initialize accumulator buffers (for max and sum reductions)
        // - generic1: Type extension (bf16 -> f32 for computation precision)
        // - reduce1: Maximum reduction across softmax dimension
        // - generic2: Broadcast maximum value
        // - generic3: Subtract maximum from input (x - max)
        // - generic4: Exponential computation (exp(x - max))
        // - reduce2: Sum reduction of exponentials
        // - generic5: Broadcast sum value
        // - generic6: Division (exp_vals / sum_exp)
        // - generic7: Type truncation (f32 -> bf16 for output)
        
        // Transpose linalg.reduce operations to ensure reduction at innermost dimension, 
        // mappable to vectorized AIE intrinsics.
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %transformed_reduces = transform.air.transpose_reduce %reduces
        
        // Clean up IR after reduction transformation to prepare for fusion
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func1 : !pdl.operation

        // Split operation handles for individual manipulation
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1, %fill2 = transform.split_handle %fill : (!pdl.operation<"linalg.fill">) -> (!pdl.operation<"linalg.fill">, !pdl.operation<"linalg.fill">)
        %generic = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2, %generic3, %generic4, %generic5, %generic6, %generic7 = transform.split_handle %generic : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        %transposed_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %reduce1, %reduce2 = transform.split_handle %transposed_reduces : (!pdl.operation<"linalg.reduce">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        
        //===================================================================
        // PHASE 3: Initial Tiling and Fusion Strategy
        //===================================================================
        // Assumption: generic7 is the final output operation that should drive
        // the tiling strategy. Memory space 1 represents L2 memory.

        // Bufferize the final operation to L2 memory (memory_space = 1)
        %generic7_output_buf, %new_generic7 = transform.structured.bufferize_to_allocation %generic7
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Tile the final operation with tile size [1] - assumes batch dimension tiling
        %tiled_generic_7, %forall_7 =
        transform.structured.tile_using_forall %generic7 tile_sizes [1]  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse all preceding operations into the tiled loop nest
        // Assumption: Operations can be fused in reverse order (generic6 -> generic1, reduce2 -> reduce1)
        // to create a producer-consumer fusion chain
        %tiled_generic_6, %4 = transform.structured.fuse_into_containing_op %generic6 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_5, %5 = transform.structured.fuse_into_containing_op %generic5 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_reduce_2, %7 = transform.structured.fuse_into_containing_op %reduce2 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_4, %6 = transform.structured.fuse_into_containing_op %generic4 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_3, %8 = transform.structured.fuse_into_containing_op %generic3 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_2, %9 = transform.structured.fuse_into_containing_op %generic2 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_reduce_1, %10 = transform.structured.fuse_into_containing_op %reduce1 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_1, %11 = transform.structured.fuse_into_containing_op %generic1 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_fills, %12 = transform.structured.fuse_into_containing_op %fill into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 4: Post-Fusion Canonicalization
        //===================================================================
        // Clean up the IR after fusion to remove redundant operations
        
        // Run canonicalization after fusion
        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func2 : !pdl.operation
        
        //===================================================================
        // PHASE 5: L1 Memory Allocation Strategy
        //===================================================================
        // Assumption: After fusion, we need to allocate intermediate buffers
        // in L1 memory (memory_space = 2) for efficient data movement.
        // This phase targets specific operations that benefit from L1 caching.
        
        // Allocate fill operations to L1 memory
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Re-split the fused generic operations for individual L1 allocation
        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %tiled_generic1, %tiled_generic2, %tiled_generic3, %tiled_generic4, %tiled_generic5, %tiled_generic6, %tiled_generic7 = transform.split_handle %generics2 : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        %reduces2 = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %tiled_reduce1, %tiled_reduce2 = transform.split_handle %reduces2 : (!pdl.operation<"linalg.reduce">) -> (!pdl.operation<"linalg.reduce">, !pdl.operation<"linalg.reduce">)

        %fused_reduce1 = transform.air.fuse_multi_op_linalg %tiled_generic1, %tiled_reduce1

        %op0 = transform.get_operand %fused_reduce1[0]
            : (!pdl.operation) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        %generics3 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %tiled_generic2_1, %tiled_reduce2_2, %tiled_generic2_3, %tiled_generic2_4, %tiled_generic2_5, %tiled_generic2_6, %tiled_generic2_7, %tiled_generic2_8 = transform.split_handle %generics3 : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        %tiled_generic1_out1_buffer, %tiled_generic1_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %tiled_generic3_out1_buffer, %tiled_generic3_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_3
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %tiled_generic4_out1_buffer, %tiled_generic4_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_4
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %tiled_generic5_out1_buffer, %tiled_generic5_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_5
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %tiled_generic6_out1_buffer, %tiled_generic6_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_6
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %tiled_generic7_out1_buffer, %tiled_generic7_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_7
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %tiled_generic8_out1_buffer, %tiled_generic8_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_8
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // //===================================================================
        // PHASE 6: Final Canonicalization and Bufferization
        //===================================================================
        // Clean up the IR after L1 allocation and prepare for final bufferization
        
        // Run canonicalization after L1 memory allocation
        %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func5 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func5 : !pdl.operation
        
        //===================================================================
        // PHASE 7: Complete Bufferization
        //===================================================================
        // Assumption: All tensor operations need to be converted to memref
        // operations for execution on AIE hardware. One-shot bufferization
        // handles the remaining tensor-to-memref conversions.
        
        // Apply one-shot bufferization to convert remaining tensors to memrefs
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

        //===================================================================
        // PHASE 8: Post-Bufferization Cleanup and Optimization
        //===================================================================
        // Assumption: Bufferization may introduce redundant memory operations
        // that need to be eliminated for optimal performance.
        
        // Run canonicalization to remove redundant memcpy (with linalg.generic form) ops created, 
        // which can be deleted by canonicalizer. We have to run it again because the memrefs are 
        // unified in CSE pass, so we can truly remove redundant memcpy.
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
        
        // Remove uninitialized copy operations that may have been introduced
        %func_op_updated = transform.air.remove_uninitialized_copy %func6
        
        //===================================================================
        // PHASE 9: Match vector ops against a supported AIE vector intrinsic
        //===================================================================

        // Reduce: 16-lane vector intrinsic
        %linalg_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_reduce1, %vec_loops_reduce1:1 =
          transform.structured.tile_using_for %linalg_reduces tile_sizes [0, 16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %linalg_generic1, %linalg_reduce2, %linalg_generic2, %linalg_generic3, %linalg_generic4, %linalg_generic5, %linalg_generic6, %linalg_generic7 = transform.split_handle %linalg_generics : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        %linalg_generic2_specialized = transform.structured.specialize %linalg_generic2 : (!pdl.operation) -> !pdl.operation
        %linalg_generic3_specialized = transform.structured.specialize %linalg_generic3 : (!pdl.operation) -> !pdl.operation
        %linalg_generic4_specialized = transform.structured.specialize %linalg_generic4 : (!pdl.operation) -> !pdl.operation
        %linalg_generic5_specialized = transform.structured.specialize %linalg_generic5 : (!pdl.operation) -> !pdl.operation
        %linalg_generic6_specialized = transform.structured.specialize %linalg_generic6 : (!pdl.operation) -> !pdl.operation

        // Bcast: 16-lane vector intrinsic
        %linalg_broadcasts = transform.structured.match ops{["linalg.broadcast"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_bcasts, %vec_loops_bcasts:1 =
          transform.structured.tile_using_for %linalg_broadcasts tile_sizes [0, 16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        
        // Div: scalar
        %linalg_divs = transform.structured.match ops{["linalg.div"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %linalg_divs_loops = transform.structured.convert_to_loops %linalg_divs : (!pdl.operation) -> !pdl.operation

        // Extf: 16-lane vector intrinsic
        %inner_most_extfs, %vec_loops_1:1 =
          transform.structured.tile_using_for %linalg_generic1 tile_sizes [0, 16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        
        // Sub: 16-lane vector intrinsic
        %linalg_subs = transform.structured.match ops{["linalg.sub"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_subs, %vec_loops_subs:1 =
          transform.structured.tile_using_for %linalg_subs tile_sizes [0, 16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        
        // Fill: scalar (only one single element)
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_fills = transform.structured.convert_to_loops %linalg_fills : (!pdl.operation) -> !pdl.operation
        
        // Exp: 16-lane vector intrinsic
        %linalg_exps = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_exps, %vec_loops_exps:1 =
          transform.structured.tile_using_for %linalg_exps tile_sizes [0, 16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        
        // Truncf: 16-lane vector intrinsic
        %inner_most_truncfs, %vec_loops_2:1 =
          transform.structured.tile_using_for %linalg_generic7 tile_sizes [0, 16]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // The remaining generic op (max. reduction): 32-lane vector intrinsic
        %inner_most_generic, %vec_loops_generic:1 =
          transform.structured.tile_using_for %linalg_reduce2 tile_sizes [0, 32]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        // Convert parallel loops to AIE herd operations for multi-core execution
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallel

        %extern_func_param = transform.param.constant "extern_func.o" -> !transform.any_param
        transform.annotate %herd "link_with" = %extern_func_param : !pdl.operation, !transform.any_param

        // Convert memory copies to DMA operations for efficient data movement
        %linalg_copies_in_herd = transform.structured.match ops{["linalg.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %memref_copies_in_herd = transform.structured.match ops{["memref.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %memref_copies_from_linalg_copies = transform.structured.linalg_copy_to_memref %linalg_copies_in_herd : (!pdl.operation) -> !pdl.operation
        %all_copies = transform.merge_handles %memref_copies_in_herd, %memref_copies_from_linalg_copies { deduplicate } : !pdl.operation
        %dmas_from_copies = transform.air.copy_to_dma %all_copies
        
        // Apply vectorization to optimize for AIE vector units
        %vectorized_herd = transform.air.herd_vectorize %herd

        // Cast vector reduce (max) to use bf16 (to map to AIE vectorized reduction intrinsic)
        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16}

        // Cast vector exp to use bf16 (to map to AIE vectorized exp intrinsic)
        %vector_exps_in_herd = transform.structured.match ops{["math.exp"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result11 = transform.air.vector_type_cast %vector_exps_in_herd {target_element_type = bf16}

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func7 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        } : !pdl.operation
        transform.apply_cse to %func7 : !pdl.operation

        // // Convert size-1 vectors to scalars (downstream compiler cannot handle size-1 vectors)
        // %vectorized_herd_scalar = transform.air.convert_size1_vector_to_scalar %vectorized_herd
    }
}
