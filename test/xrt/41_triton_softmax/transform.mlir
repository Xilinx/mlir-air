// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton Softmax Tiling Recipe Transform Script
//===----------------------------------------------------------------------===//
// This transform script implements a comprehensive tiling and optimization
// strategy for softmax operations targeting AIE (AI Engine) hardware.
// The recipe assumes:
// 1. Input operations are in linalg dialect form
// 2. The softmax computation is decomposed into multiple linalg.generic ops
// 3. Memory hierarchy optimization is needed (L1/L2 memory spaces)
// 4. Operations can be fused for better performance
//===----------------------------------------------------------------------===//

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        // Assumption: The input IR contains linalg operations that can benefit
        // from standard canonicalization patterns to simplify the computation
        // before applying more complex transformations.
        
        // Run canonicalization
        // Apply initial canonicalization patterns to clean up the IR
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func0 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            // CRITICAL: fold_unit_extent_dims_via_reshapes is essential.
            // This pattern removes unit dimensions and simplifies tensor shapes, which is
            // crucial for subsequent tiling and bufferization passes.
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !pdl.operation
        transform.apply_cse to %func0 : !pdl.operation

        //===================================================================
        // PHASE 2: Operation Preparation and Handle Splitting
        //===================================================================
        // Assumption: The softmax computation contains linalg.reduce operations
        // that need to be transformed and generalized for uniform handling.
        
        // Transform and convert linalg.reduce operations for consistent processing
        // 1. First apply transpose_reduce transformation to optimize reduction patterns
        // 2. Then generalize the transformed operations to linalg.generic for uniform handling
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %transformed_reduces = transform.air.transpose_reduce %reduces
        %generalized_reduces = transform.structured.generalize %transformed_reduces  : (!pdl.operation) -> !pdl.operation
        
        // Run canonicalization after transformation and generalization
        // This additional canonicalization stage cleans up the IR after the transpose_reduce
        // transformation and generalization, ensuring optimal patterns before fusion
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func1 : !pdl.operation

        // Split operation handles for individual manipulation
        // Assumption: There are exactly 2 fill operations and 7 generic operations
        // representing the decomposed softmax computation stages
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1, %fill2 = transform.split_handle %fill : (!pdl.operation<"linalg.fill">) -> (!pdl.operation<"linalg.fill">, !pdl.operation<"linalg.fill">)
        %generic = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2, %generic3, %generic4, %generic5, %generic6, %generic7 = transform.split_handle %generic : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        
        //===================================================================
        // PHASE 3: Initial Tiling and Fusion Strategy
        //===================================================================
        // Assumption: generic7 is the final output operation that should drive
        // the tiling strategy.
        
        // Bufferize the final operation to L2 memory (memory_space = 1)
        // Memory space mapping: 0=L3(DDR), 1=L2(Tile), 2=L1(Core)
        %generic7_output_buf, %new_generic7 = transform.structured.bufferize_to_allocation %generic7
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Tile the final operation with tile size [1] - assumes batch dimension tiling
        %tiled_generic_7, %forall_7 =
        transform.structured.tile_using_forall %generic7 tile_sizes [1]  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse all preceding operations into the tiled loop nest
        // Assumption: Operations can be fused in reverse order (generic6 -> generic1)
        // to create a producer-consumer fusion chain
        %tiled_generic_6, %1 = transform.structured.fuse_into_containing_op %generic6 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_5, %2 = transform.structured.fuse_into_containing_op %generic5 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_4, %3 = transform.structured.fuse_into_containing_op %generic4 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_3, %4 = transform.structured.fuse_into_containing_op %generic3 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_2, %5 = transform.structured.fuse_into_containing_op %generic2 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_generic_1, %6 = transform.structured.fuse_into_containing_op %generic1 into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_fills, %7 = transform.structured.fuse_into_containing_op %fill into %forall_7 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

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
        // in L1 memory (memory_space = 2) for computation in AIE cores.
        // Memory space mapping: 0=L3(DDR), 1=L2(Tile), 2=L1(Core)
        
        // Allocate fill operations to L1 memory
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Re-split the fused generic operations for individual L1 allocation
        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %tiled_generic1, %tiled_generic2, %tiled_generic3, %tiled_generic4, %tiled_generic5, %tiled_generic6, %tiled_generic7 = transform.split_handle %generics2 : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)

        // Allocate input producer to L1 memory for efficient data access
        %padded_gen1_in = transform.get_producer_of_operand %tiled_generic1[0] : (!pdl.operation) -> (!pdl.operation)
        
        %padded_gen1_in_buffer, %padded_gen1_in_new = transform.structured.bufferize_to_allocation %padded_gen1_in
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Allocate intermediate computation results to L1 memory
        // Assumption: These operations produce intermediate results that need
        // to be cached in L1 for subsequent operations in the softmax pipeline
        %padded_gen2_out1_buffer, %padded_gen2_out1_new = transform.structured.bufferize_to_allocation %tiled_generic2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %padded_gen3_out1_buffer, %padded_gen3_out1_new = transform.structured.bufferize_to_allocation %tiled_generic3
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %tiled_generic4_buffer, %tiled_generic4_new = transform.structured.bufferize_to_allocation %tiled_generic4
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %padded_gen6_out1_buffer, %padded_gen6_out1_new = transform.structured.bufferize_to_allocation %tiled_generic6
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %padded_gen7_out1_buffer, %padded_gen7_out1_new = transform.structured.bufferize_to_allocation %tiled_generic7
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation


        //===================================================================
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
        // PHASE 9: Library Call Optimization
        //===================================================================
        // Assumption: The softmax computation contains math.exp operations
        // that can be optimized by replacing with vectorized library calls
        // for better performance on AIE hardware.
        
        // Convert math.exp operations to optimized library calls
        // Assumption: exp_vec16_f32 is a vectorized exponential function
        // that operates on 16 f32 elements and is available in extern_func.o
        %math_exp = transform.structured.match ops{["math.exp"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %math_exp_linalg = transform.get_parent_op %math_exp { op_name = "linalg.generic" } : (!pdl.operation) -> !pdl.operation
        %call = transform.air.linalg_to_library_call %math_exp_linalg { function_name = "exp_vec16_f32", link_with = "extern_func.o" } : (!pdl.operation) -> !pdl.operation

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        // Convert parallel loops to AIE herd operations for multi-core execution
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallel

        // Convert memory copies to DMA operations for efficient data movement
        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
    }
}
