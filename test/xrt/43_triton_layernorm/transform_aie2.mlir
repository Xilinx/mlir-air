// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton LayerNorm Tiling Recipe Transform Script
//===----------------------------------------------------------------------===//
// This transform script implements a comprehensive tiling and optimization
// strategy for layer normalization operations targeting AIE (AI Engine) hardware.
// The recipe assumes:
// 1. Input operations are in linalg dialect form
// 2. The layernorm computation is decomposed into multiple linalg operations
// 3. Memory hierarchy optimization is needed (L1/L2 memory spaces)
// 4. Operations can be fused for better performance
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        // Assumption: The input IR contains linalg operations that can benefit
        // from standard canonicalization patterns to simplify the computation
        // before applying more complex transformations.
        
        // Run canonicalization
        // Apply initial canonicalization patterns to clean up the IR
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func0 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            // CRITICAL: fold_unit_extent_dims_via_reshapes is essential.
            // This pattern removes unit dimensions and simplifies tensor shapes, which is
            // crucial for subsequent tiling and bufferization passes.
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_cse to %func0 : !transform.any_op

        //===================================================================
        // PHASE 2: Elementwise Fusion and Reduction Transformation
        //===================================================================
        // This phase prepares the layernorm computation by fusing elementwise
        // operations and transforming reduction operations for optimal execution.
        
        // Step 1: Fuse elementwise linalg operations
        // Combines compatible elementwise operations (e.g., add, mul, div) to reduce
        // intermediate memory traffic and create larger computational kernels
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fused_func = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
        
        // Step 2: Transform and generalize linalg.reduce operations
        // The layernorm computation contains reduction operations (e.g., sum, mean)
        // that need special handling:
        // Transpose linalg.reduce operations. linalg.reduce ops preserved as
        // data-flow anchors (generalize deferred to before vectorization).
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %transformed_reduces = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op

        // Canonicalization after fusion and transformation
        transform.apply_patterns to %fused_func {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %fused_func : !transform.any_op

        // Data-flow navigation from linalg.reduce anchors.
        // Chain: generic1 -> reduce1 -> generic2 -> reduce2 -> output_generic
        %transposed_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %reduce1, %reduce2 = transform.split_handle %transposed_reduces : (!transform.any_op<"linalg.reduce">) -> (!transform.any_op, !transform.any_op)

        %generic1 = transform.get_producer_of_operand %reduce1[0]
            : (!transform.any_op) -> !transform.any_op
        %generic2 = transform.get_producer_of_operand %reduce2[0]
            : (!transform.any_op) -> !transform.any_op

        // Find output generic via function output anchor
        %materialize = transform.structured.match ops{["bufferization.materialize_in_destination"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %output_generic = transform.get_producer_of_operand %materialize[0]
            : (!transform.any_op) -> !transform.any_op

        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 3: Batch-Level Tiling and Producer-Consumer Fusion
        //===================================================================
        %output_buf, %new_output = transform.structured.bufferize_to_allocation %output_generic
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %tiled_output, %forall_4 =
        transform.structured.tile_using_forall %output_generic tile_sizes [1]  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Backward fusion in reverse data-flow order
        %fused_reduce2, %4 = transform.structured.fuse_into_containing_op %reduce2 into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_generic2, %5 = transform.structured.fuse_into_containing_op %generic2 into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_reduce1, %8 = transform.structured.fuse_into_containing_op %reduce1 into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_generic1, %6 = transform.structured.fuse_into_containing_op %generic1 into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_fill, %7 = transform.structured.fuse_into_containing_op %fill into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        
        // //===================================================================
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
        // PHASE 5: L2 Memory Allocation for Intermediate Buffers
        //===================================================================
        // This phase allocates intermediate buffers in L2 memory (memory_space = 2)
        // to stage data between operations within the fused computation kernel.
        // L2 serves as an intermediate cache layer between L1 (output) and main memory.
        
        // Step 1: Allocate fill operation outputs to L2 memory
        // The fill operation initializes reduction accumulators. Placing these in L2
        // enables efficient access by subsequent reduction operations within the kernel.
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Step 2: Re-match operations for individual L2 allocation.
        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %tiled_generic1, %tiled_generic2 = transform.split_handle %generics2 : (!transform.any_op<"linalg.generic">) -> (!transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">)
        %reduces2 = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %tiled_reduce_a, %tiled_reduce_b = transform.split_handle %reduces2 : (!transform.any_op<"linalg.reduce">) -> (!transform.any_op<"linalg.reduce">, !transform.any_op<"linalg.reduce">)

        // Step 3: Promote input tensor to L2 memory
        %op0 = transform.get_operand %tiled_generic1[0]
            : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        // Step 4: Allocate intermediate outputs to L2 memory
        %gen1_in_buffer, %gen1_in_new = transform.structured.bufferize_to_allocation %tiled_generic1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %red1_in_buffer, %red1_in_new = transform.structured.bufferize_to_allocation %tiled_reduce_a
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %gen2_in_buffer, %gen2_in_new = transform.structured.bufferize_to_allocation %tiled_generic2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %red2_in_buffer, %red2_in_new = transform.structured.bufferize_to_allocation %tiled_reduce_b
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op


        //===================================================================
        // PHASE 6: Final Canonicalization and Bufferization
        //===================================================================
        // Clean up the IR after L2 allocation and prepare for final bufferization
        
        // Run canonicalization after L2 memory allocation
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
        // PHASE 9: Vectorization Preparation
        //===================================================================
        // This phase prepares operations for vectorization by tiling to match
        // AIE vector lane widths and optimizing mathematical operations.
        
        // Delayed generalize: convert remaining linalg.reduce ops to linalg.generic
        // BEFORE vectorization. Deferred from Phase 2 to preserve data-flow anchors.
        %remaining_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generalized_reduces = transform.structured.generalize %remaining_reduces : (!transform.any_op) -> !transform.any_op

        // Step 1: Tile for 16-lane vector operations
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Step 2: Optimize division by sqrt to reciprocal sqrt
        // LayerNorm contains 1/sqrt(variance) operations. AIE has native rsqrt
        // instructions that are more efficient than divf(1.0, sqrt(x)). This
        // transformation converts divf-on-sqrt patterns to rsqrt operations.
        %func_op_updated_1 = transform.air.convert_divf_sqrt_to_rsqrt %func_op_updated : (!transform.any_op) -> !transform.any_op
        

        //===================================================================
        // PHASE 10: AIE Hardware Mapping and Vectorization
        //===================================================================
        // This phase maps high-level constructs to AIE-specific operations,
        // applies vectorization, and prepares operations for AIE vector intrinsics.
        
        // Step 1: Create AIE herd for multi-core execution
        // Convert the parallel scf.forall loops (from PHASE 3) to AIE herd operations.
        // A herd represents a collection of AIE cores executing in parallel, enabling
        // multi-core execution of the tiled batch iterations.
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!transform.any_op) -> !transform.any_op
        %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

        // Step 2: Link external functions for specialized operations
        // Annotate the herd with external object file containing optimized
        // implementations for operations that may not have direct AIE intrinsics.
        %extern_func_param = transform.param.constant "extern_func.o" -> !transform.any_param
        transform.annotate %herd "link_with" = %extern_func_param : !transform.any_op, !transform.any_param

        // Step 3: Convert memory copies to DMA operations
        // AIE uses dedicated DMA engines for efficient data movement. Convert
        // explicit memory copy operations to DMA operations that can be executed
        // asynchronously on DMA hardware, overlapping with computation.
        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd : (!transform.any_op) -> !transform.any_op          
        
        // Step 4: Apply AIE-specific vectorization
        // Transform scalar operations within the herd to vector operations that
        // map to AIE vector instructions. This vectorization is AIE-aware and
        // considers the 16-lane vector width established in PHASE 9.
        %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

        // Step 5: Clean up vector shapes
        // Remove leading dimensions of size 1 from vector types (e.g., vector<1x16xf32>
        // -> vector<16xf32>). This simplification is necessary for proper matching to
        // AIE vector intrinsics which expect canonical vector shapes.
        %func4 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func4 {
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        } : !transform.any_op

        // Step 6: Insert broadcasts for unary vector operations
        // AIE vector unary operations (like rsqrt) may require explicit broadcasts
        // to replicate scalar values across vector lanes. This transformation inserts
        // necessary broadcast operations before math.rsqrt calls.
        %vectorized_herd_updated = transform.air.broadcast_before_unary %func4 {op_name = "math.rsqrt"} : (!transform.any_op) -> !transform.any_op

        // Step 7: Type cast operations to bf16 for AIE intrinsics
        // AIE provides optimized bf16 vector intrinsics. Cast vector operations
        // to bf16 to enable matching to these hardware-accelerated instructions.
        
        // Cast vector reductions (e.g., sum, max) to bf16
        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd_updated : (!transform.any_op) -> !transform.any_op
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        // Cast vector multiplications to bf16
        %vector_muls_in_herd = transform.structured.match ops{["arith.mulf"]} in %vectorized_herd_updated : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_muls_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        // Cast math.rsqrt operations to bf16
        %math_rsqrts_in_herd = transform.structured.match ops{["math.rsqrt"]} in %vectorized_herd_updated : (!transform.any_op) -> !transform.any_op
        %result12 = transform.air.vector_type_cast %math_rsqrts_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        // Step 8: Final cleanup and vector lowering
        // Apply final canonicalization passes and lower multi_reduction operations
        // using the "innerreduction" strategy, which generates code suitable for
        // AIE's reduction intrinsics.
        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func7_transformed = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7_transformed {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        } : !transform.any_op
        transform.apply_cse to %func7_transformed : !transform.any_op
    transform.yield
  }
}
