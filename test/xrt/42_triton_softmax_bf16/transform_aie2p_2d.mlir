// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton Softmax Tiling Recipe Transform Script (2D Herd Version)
//===----------------------------------------------------------------------===//
// This is the 2D version of the transform script for 4×4 herd (16 cores).
// Key difference from 1D version: uses tile_sizes [1, 1] for 2D parallelism.
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
//===----------------------------------------------------------------------===//

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        
        transform.apply_patterns to %func0 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !pdl.operation
        
        transform.apply_cse to %func0 : !pdl.operation

        //===================================================================
        // PHASE 2: Operation Fusion and Preparation
        //===================================================================
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fused_func = transform.air.fuse_elementwise_linalg %func1
        
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %fused_func  : (!pdl.operation) -> !pdl.operation
        %transformed_reduces = transform.air.transpose_reduce %reduces
        %generalized_reduces = transform.structured.generalize %transformed_reduces  : (!pdl.operation) -> !pdl.operation
        
        transform.apply_patterns to %fused_func {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %fused_func : !pdl.operation

        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1, %fill2 = transform.split_handle %fill : (!pdl.operation<"linalg.fill">) -> (!pdl.operation<"linalg.fill">, !pdl.operation<"linalg.fill">)
        %generic = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2, %generic3, %generic4, %generic5 = transform.split_handle %generic : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        
        %fused_generic1 = transform.air.fuse_multi_op_linalg %generic1, %generic2
        %fused_generic2 = transform.air.fuse_multi_op_linalg %generic3, %generic4

        //===================================================================
        // PHASE 3: Tiling and Producer-Consumer Fusion (2D VERSION)
        //===================================================================
        // KEY CHANGE: Use tile_sizes [1, 1] for 2D parallelism (4×4 herd)
        // This creates an scf.forall with 4×4 iteration space

        %generic5_output_buf, %new_generic5 = transform.structured.bufferize_to_allocation %generic5
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // 2D tiling: [1, 1] creates 4×4 iteration space from 4×4×1024 tensor
        %tiled_generic_5, %forall_5 =
        transform.structured.tile_using_forall %generic5 tile_sizes [1, 1]  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        %tiled_fused_generic_2, %4 = transform.structured.fuse_into_containing_op %fused_generic2 into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_fused_generic_1, %5 = transform.structured.fuse_into_containing_op %fused_generic1 into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_fill, %7 = transform.structured.fuse_into_containing_op %fill into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 4: Post-Fusion Canonicalization
        //===================================================================
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
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %tiled_generic1, %tiled_generic2, %tiled_generic3 = transform.split_handle %generics2 : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)

        %op0 = transform.get_operand %tiled_generic1[0]
            : (!pdl.operation) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        %gen1_in_buffer, %gen1_in_new = transform.structured.bufferize_to_allocation %tiled_generic1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %gen2_in_buffer, %gen2_in_new = transform.structured.bufferize_to_allocation %tiled_generic2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %gen3_in_buffer, %gen3_in_new = transform.structured.bufferize_to_allocation %tiled_generic3
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        //===================================================================
        // PHASE 6: Pre-Bufferization Canonicalization
        //===================================================================
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
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

        //===================================================================
        // PHASE 8: Post-Bufferization Cleanup and Optimization
        //===================================================================
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

        //===================================================================
        // PHASE 9: Prepare Operations for AIE Vector Intrinsics
        //===================================================================
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [0, 0, 32]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallel

        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
        
        %vectorized_herd = transform.air.herd_vectorize %herd

        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16}

        %vector_exps_in_herd = transform.structured.match ops{["math.exp"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result11 = transform.air.vector_type_cast %vector_exps_in_herd {target_element_type = bf16}

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation

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
