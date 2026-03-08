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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        // PURPOSE: Prepare the IR for subsequent transformations by applying
        // standard optimization patterns that simplify operations and remove
        // redundancies. This creates a clean foundation for tiling and fusion.
        
        // Match the function containing all softmax operations
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        
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
        } : !transform.any_op
        
        // Apply Common Subexpression Elimination to remove duplicate computations
        transform.apply_cse to %func0 : !transform.any_op

        //===================================================================
        // PHASE 2: Operation Preparation via Data-Flow Navigation
        //===================================================================
        // PURPOSE: Identify each softmax computation stage by its semantic
        // identity rather than by fragile positional indexing. We use
        // linalg.reduce as natural anchor ops and navigate the data-flow
        // graph to find each operation by its role in the computation.
        //
        // SOFTMAX DATA-FLOW CHAIN:
        // input -> extf -> reduce_max -> broadcast_max -> sub -> exp
        //                                                        |
        //       output <- truncf <- div <- broadcast_sum <- reduce_sum
        
        // Transpose linalg.reduce operations to ensure reduction at innermost dimension, 
        // mappable to vectorized AIE intrinsics.
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %transformed_reduces = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op
        
        // Clean up IR after reduction transformation to prepare for fusion
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func1 : !transform.any_op

        // Data-flow navigation from linalg.reduce anchors (already named ops,
        // no specialize needed). Navigate the softmax data-flow graph to identify
        // each operation by its role rather than by fragile positional indexing.
        //
        // The two linalg.reduce ops are the natural anchors: reduce_max and
        // reduce_sum. From these, we walk the producer/consumer chain:
        //   extf -> reduce_max -> broadcast_max -> sub -> exp
        //                                                  |
        //         truncf <- div <- broadcast_sum <- reduce_sum

        // Match the two linalg.reduce ops
        %transposed_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %reduce_max, %reduce_sum = transform.split_handle %transposed_reduces : (!transform.any_op<"linalg.reduce">) -> (!transform.any_op, !transform.any_op)

        // Data-flow navigation from reduce_max: walk upstream to find extf,
        // and downstream to find broadcast_max -> sub -> exp.
        // Note: after transpose_reduce + canonicalization, the reduce results
        // feed directly into broadcast generics (no tensor.expand_shape in between).
        %extf_op = transform.get_producer_of_operand %reduce_max[0]
            : (!transform.any_op) -> !transform.any_op
        %broadcast_max = transform.get_consumers_of_result %reduce_max[0]
            : (!transform.any_op) -> !transform.any_op
        %sub_op = transform.get_consumers_of_result %broadcast_max[0]
            : (!transform.any_op) -> !transform.any_op
        %exp_op = transform.get_consumers_of_result %sub_op[0]
            : (!transform.any_op) -> !transform.any_op

        // Data-flow navigation from reduce_sum: walk downstream to find
        // broadcast_sum -> div -> truncf
        %broadcast_sum = transform.get_consumers_of_result %reduce_sum[0]
            : (!transform.any_op) -> !transform.any_op
        %div_op = transform.get_consumers_of_result %broadcast_sum[0]
            : (!transform.any_op) -> !transform.any_op
        %truncf_op = transform.get_consumers_of_result %div_op[0]
            : (!transform.any_op) -> !transform.any_op

        // Match fill operations
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 3: Initial Tiling and Fusion Strategy
        //===================================================================
        // truncf_op is the final output operation that drives the tiling strategy.

        // Bufferize the final operation to L2 memory (memory_space = 1)
        %truncf_output_buf, %new_truncf = transform.structured.bufferize_to_allocation %truncf_op
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Tile the final operation with tile size [1] - batch dimension tiling
        %tiled_truncf, %forall_7 =
        transform.structured.tile_using_forall %truncf_op tile_sizes [1]  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Fuse all preceding operations into the tiled loop nest
        // in reverse data-flow order (semantic, not positional)
        %tiled_div, %4 = transform.structured.fuse_into_containing_op %div_op into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_bcast_sum, %5 = transform.structured.fuse_into_containing_op %broadcast_sum into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_reduce_sum, %7 = transform.structured.fuse_into_containing_op %reduce_sum into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_exp, %6 = transform.structured.fuse_into_containing_op %exp_op into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_sub, %8 = transform.structured.fuse_into_containing_op %sub_op into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_bcast_max, %9 = transform.structured.fuse_into_containing_op %broadcast_max into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_reduce_max, %10 = transform.structured.fuse_into_containing_op %reduce_max into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %tiled_extf, %11 = transform.structured.fuse_into_containing_op %extf_op into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_fills, %12 = transform.structured.fuse_into_containing_op %fill into %forall_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        //===================================================================
        // PHASE 4: Post-Fusion Canonicalization
        //===================================================================
        // Clean up the IR after fusion to remove redundant operations
        
        // Run canonicalization after fusion
        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func2 : !transform.any_op
        
        //===================================================================
        // PHASE 5: L1 Memory Allocation Strategy
        //===================================================================
        // Assumption: After fusion, we need to allocate intermediate buffers
        // in L1 memory (memory_space = 2) for efficient data movement.
        // This phase targets specific operations that benefit from L1 caching.
        
        // Allocate fill operations to L1 memory
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Re-split the fused generic operations for individual L1 allocation
        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %tiled_generic1, %tiled_generic2, %tiled_generic3, %tiled_generic4, %tiled_generic5, %tiled_generic6, %tiled_generic7 = transform.split_handle %generics2 : (!transform.any_op<"linalg.generic">) -> (!transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">)
        %reduces2 = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %tiled_reduce1, %tiled_reduce2 = transform.split_handle %reduces2 : (!transform.any_op<"linalg.reduce">) -> (!transform.any_op<"linalg.reduce">, !transform.any_op<"linalg.reduce">)

        %fused_reduce1 = transform.air.fuse_multi_op_linalg %tiled_generic1, %tiled_reduce1 : (!transform.any_op, !transform.any_op) -> !transform.any_op

        %op0 = transform.get_operand %fused_reduce1[0]
            : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        %generics3 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %tiled_generic2_1, %tiled_reduce2_2, %tiled_generic2_3, %tiled_generic2_4, %tiled_generic2_5, %tiled_generic2_6, %tiled_generic2_7, %tiled_generic2_8 = transform.split_handle %generics3 : (!transform.any_op<"linalg.generic">) -> (!transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">)
        %tiled_generic1_out1_buffer, %tiled_generic1_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %tiled_generic3_out1_buffer, %tiled_generic3_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_3
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %tiled_generic4_out1_buffer, %tiled_generic4_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_4
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %tiled_generic5_out1_buffer, %tiled_generic5_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_5
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %tiled_generic6_out1_buffer, %tiled_generic6_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_6
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %tiled_generic7_out1_buffer, %tiled_generic7_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_7
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %tiled_generic8_out1_buffer, %tiled_generic8_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2_8
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // //===================================================================
        // PHASE 6: Final Canonicalization and Bufferization
        //===================================================================
        // Clean up the IR after L1 allocation and prepare for final bufferization
        
        // Run canonicalization after L1 memory allocation
        %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func5 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func5 : !transform.any_op
        
        //===================================================================
        // PHASE 7: Complete Bufferization
        //===================================================================
        // Assumption: All tensor operations need to be converted to memref
        // operations for execution on AIE hardware. One-shot bufferization
        // handles the remaining tensor-to-memref conversions.
        
        // Apply one-shot bufferization to convert remaining tensors to memrefs
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 8: Post-Bufferization Cleanup and Optimization
        //===================================================================
        // Assumption: Bufferization may introduce redundant memory operations
        // that need to be eliminated for optimal performance.
        
        // Run canonicalization to remove redundant memcpy (with linalg.generic form) ops created, 
        // which can be deleted by canonicalizer. We have to run it again because the memrefs are 
        // unified in CSE pass, so we can truly remove redundant memcpy.
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
        
        // Remove uninitialized copy operations that may have been introduced
        %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
        
        //===================================================================
        // PHASE 9: Match vector ops against a supported AIE vector intrinsic
        //===================================================================

        // Reduce: 16-lane vector intrinsic
        %linalg_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_reduce1, %vec_loops_reduce1:1 =
          transform.structured.tile_using_for %linalg_reduces tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %linalg_generic1, %linalg_reduce2, %linalg_generic2, %linalg_generic3, %linalg_generic4, %linalg_generic5, %linalg_generic6, %linalg_generic7 = transform.split_handle %linalg_generics : (!transform.any_op<"linalg.generic">) -> (!transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">)
        %linalg_generic2_specialized = transform.structured.specialize %linalg_generic2 : (!transform.any_op) -> !transform.any_op
        %linalg_generic3_specialized = transform.structured.specialize %linalg_generic3 : (!transform.any_op) -> !transform.any_op
        %linalg_generic4_specialized = transform.structured.specialize %linalg_generic4 : (!transform.any_op) -> !transform.any_op
        %linalg_generic5_specialized = transform.structured.specialize %linalg_generic5 : (!transform.any_op) -> !transform.any_op
        %linalg_generic6_specialized = transform.structured.specialize %linalg_generic6 : (!transform.any_op) -> !transform.any_op

        // Bcast: 16-lane vector intrinsic
        %linalg_broadcasts = transform.structured.match ops{["linalg.broadcast"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_bcasts, %vec_loops_bcasts:1 =
          transform.structured.tile_using_for %linalg_broadcasts tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        
        // Div: scalar
        %linalg_divs = transform.structured.match ops{["linalg.div"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %linalg_divs_loops = transform.structured.convert_to_loops %linalg_divs : (!transform.any_op) -> !transform.any_op

        // Extf: 16-lane vector intrinsic
        %inner_most_extfs, %vec_loops_1:1 =
          transform.structured.tile_using_for %linalg_generic1 tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        
        // Sub: 16-lane vector intrinsic
        %linalg_subs = transform.structured.match ops{["linalg.sub"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_subs, %vec_loops_subs:1 =
          transform.structured.tile_using_for %linalg_subs tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        
        // Fill: scalar (only one single element)
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills = transform.structured.convert_to_loops %linalg_fills : (!transform.any_op) -> !transform.any_op
        
        // Exp: 16-lane vector intrinsic
        %linalg_exps = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_exps, %vec_loops_exps:1 =
          transform.structured.tile_using_for %linalg_exps tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        
        // Truncf: 16-lane vector intrinsic
        %inner_most_truncfs, %vec_loops_2:1 =
          transform.structured.tile_using_for %linalg_generic7 tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // The remaining generic op (max. reduction): 32-lane vector intrinsic
        %inner_most_generic, %vec_loops_generic:1 =
          transform.structured.tile_using_for %linalg_reduce2 tile_sizes [0, 32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        // Convert parallel loops to AIE herd operations for multi-core execution
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!transform.any_op) -> !transform.any_op
        %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

        %extern_func_param = transform.param.constant "extern_func.o" -> !transform.any_param
        transform.annotate %herd "link_with" = %extern_func_param : !transform.any_op, !transform.any_param

        // Convert memory copies to DMA operations for efficient data movement
        %linalg_copies_in_herd = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %memref_copies_in_herd = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %memref_copies_from_linalg_copies = transform.structured.linalg_copy_to_memref %linalg_copies_in_herd : (!transform.any_op) -> !transform.any_op
        %all_copies = transform.merge_handles %memref_copies_in_herd, %memref_copies_from_linalg_copies { deduplicate } : !transform.any_op
        %dmas_from_copies = transform.air.copy_to_dma %all_copies : (!transform.any_op) -> !transform.any_op
        
        // Apply vectorization to optimize for AIE vector units
        %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

        // Cast vector reduce (max) to use bf16 (to map to AIE vectorized reduction intrinsic)
        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        // Cast vector exp to use bf16 (to map to AIE vectorized exp intrinsic)
        %vector_exps_in_herd = transform.structured.match ops{["math.exp"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_exps_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func7_transformed = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7_transformed {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        } : !transform.any_op
        transform.apply_cse to %func7_transformed : !transform.any_op
    transform.yield
  }
}
