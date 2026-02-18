// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
        // PHASE 2: Operation Fusion and Preparation
        //===================================================================
        // PURPOSE: Fuse compatible operations and prepare IR for tiling by
        // transforming reduce operations for efficient AIE execution.
        //
        // OPERATION FLOW:
        // 1. Fuse elementwise operations to reduce intermediate memory traffic
        //    and create larger computational kernels suitable for vectorization
        // 2. Transform reduction operations to ensure reduction dimension is
        //    innermost (required for mapping to vectorized AIE intrinsics)
        // 3. Generalize reductions to linalg.generic form for uniform handling
        
        // Fuse elementwise linalg operations
        // Combines compatible elementwise operations (e.g., add, mul, div) to reduce
        // intermediate memory traffic and create larger computational kernels
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fused_func = transform.air.fuse_elementwise_linalg %func1
        
        // Transpose linalg.reduce operations to ensure reduction at innermost dimension, 
        // mappable to vectorized AIE intrinsics
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %fused_func  : (!pdl.operation) -> !pdl.operation
        %transformed_reduces = transform.air.transpose_reduce %reduces
        %generalized_reduces = transform.structured.generalize %transformed_reduces  : (!pdl.operation) -> !pdl.operation
        
        // Clean up IR after reduction transformation to prepare for fusion
        // %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %fused_func {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %fused_func : !pdl.operation

        // Split operation handles for individual manipulation
        // After fusion, we have 5 linalg.generic operations representing the
        // fused softmax computation stages
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1, %fill2 = transform.split_handle %fill : (!pdl.operation<"linalg.fill">) -> (!pdl.operation<"linalg.fill">, !pdl.operation<"linalg.fill">)
        %generic = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2, %generic3, %generic4, %generic5 = transform.split_handle %generic : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        
        // Further fuse pairs of generic operations to optimize data locality
        %fused_generic1 = transform.air.fuse_multi_op_linalg %generic1, %generic2
        %fused_generic2 = transform.air.fuse_multi_op_linalg %generic3, %generic4

        //===================================================================
        // PHASE 3: Tiling and Producer-Consumer Fusion
        //===================================================================
        // STRATEGY: Use the final output operation (generic5) to drive tiling,
        // then fuse all producer operations into the tiled loop.
        // Memory space 1 represents L2 memory.

        // Bufferize the final operation to L2 memory (memory_space = 1)
        %generic5_output_buf, %new_generic5 = transform.structured.bufferize_to_allocation %generic5
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Tile the final operation with tile size [1] for batch dimension
        %tiled_generic_5, %forall_5 =
        transform.structured.tile_using_forall %generic5 tile_sizes [1]  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse producer operations into the tiled loop in reverse dependency order
        // This creates a producer-consumer fusion chain where each operation is
        // computed within the same iteration as its consumers
        %tiled_fused_generic_2, %4 = transform.structured.fuse_into_containing_op %fused_generic2 into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_fused_generic_1, %5 = transform.structured.fuse_into_containing_op %fused_generic1 into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_fill, %7 = transform.structured.fuse_into_containing_op %fill into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

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
        // PURPOSE: Allocate intermediate buffers in L1 memory (memory_space = 2)
        // for low-latency access by compute cores. L1 memory is local to each
        // compute tile and provides the fastest access for frequently used data.
        
        // Allocate fill operations to L1 memory for reduction accumulation
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Split generic operations after tiling for individual L1 buffer allocation
        // Each tiled generic operation will have its output allocated in L1
        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %tiled_generic1, %tiled_generic2, %tiled_generic3 = transform.split_handle %generics2 : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)

        // Promote the first input operand to L1 memory
        %op0 = transform.get_operand %tiled_generic1[0]
            : (!pdl.operation) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        // Allocate output buffers in L1 for each tiled generic operation
        %gen1_in_buffer, %gen1_in_new = transform.structured.bufferize_to_allocation %tiled_generic1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %gen2_in_buffer, %gen2_in_new = transform.structured.bufferize_to_allocation %tiled_generic2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %gen3_in_buffer, %gen3_in_new = transform.structured.bufferize_to_allocation %tiled_generic3
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        //===================================================================
        // PHASE 6: Pre-Bufferization Canonicalization
        //===================================================================
        // PURPOSE: Clean up the IR after L1 allocation and prepare for complete
        // bufferization by removing redundant operations and simplifying patterns
        
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
        // PURPOSE: Eliminate redundant memory operations introduced during bufferization.
        // The canonicalizer removes redundant memcpy operations (represented as linalg.generic).
        // CSE unifies memrefs first, enabling the canonicalizer to identify and remove duplicates.
        
        // Run canonicalization to remove redundant memcpy operations
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
        // PHASE 9: Prepare Operations for AIE Vector Intrinsics
        //===================================================================
        // PURPOSE: Convert operations to forms that can be mapped to AIE vector
        // intrinsics or scalar operations as appropriate.

        // Tile generic operations for vectorization with tile size 32 (AIE2P vector width)
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [0, 32]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        // Convert parallel loops to AIE herd operations for multi-core execution
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallel

        // No external function linking required for aie2p
        // Unlike aie2, rsqrt lowering in aie2p does not require an external aie_api
        // implementation. The aie2p architecture provides native support for rsqrt
        // operations through direct hardware intrinsics.

        // Convert memory copies to DMA operations
        // AIE uses dedicated DMA engines for efficient data movement. Convert
        // explicit memory copy operations to DMA operations that can be executed
        // asynchronously on DMA hardware, overlapping with computation.
        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
        
        // Apply vectorization to optimize for AIE vector units
        %vectorized_herd = transform.air.herd_vectorize %herd

        // Cast vector reduce to use bf16 (to map to AIE vectorized reduction intrinsic)
        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16}

        // Cast vector exp to use bf16 (to map to AIE vectorized exp intrinsic)
        %vector_exps_in_herd = transform.structured.match ops{["math.exp"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result11 = transform.air.vector_type_cast %vector_exps_in_herd {target_element_type = bf16}

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation

        // Convert size-1 vectors to scalars (downstream compiler cannot handle size-1 vectors)
        %func7_transformed = transform.air.convert_size1_vector_to_scalar %func7
        transform.apply_patterns to %func7_transformed {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        } : !pdl.operation
        transform.apply_cse to %func7_transformed : !pdl.operation
    }
}
