// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %vecmat = transform.structured.match ops{["linalg.vecmat"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        // Tiling to generate air.launch
        %vecmat_1, %forall = transform.structured.tile_using_forall %vecmat tile_sizes [64, 0] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fill_1, %fused_for_all = transform.structured.fuse_into_containing_op %fill into %forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        
        // Pad operation.
        %padded, %pad, %__ = transform.structured.pad %vecmat_1 {
            padding_values=[0 : i32, 0 : i32, 0 : i32],
            padding_dimensions=[0, 1, 2],
            nofold_flags=[1, 1, 1],
            copy_back_op="linalg.copy"
        } : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!pdl.operation) -> !pdl.operation

        // Promote the operands to shared memory.
        %padded_lhs = transform.get_producer_of_operand %padded[0] : (!pdl.operation) -> (!pdl.operation)
        %padded_lhs_buffer, %padded_lhs_new = transform.structured.bufferize_to_allocation %padded_lhs
            {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %padded_rhs = transform.get_producer_of_operand %padded[1] : (!pdl.operation) -> (!pdl.operation)
        %padded_rhs_buffer, %padded_rhs_new = transform.structured.bufferize_to_allocation %padded_rhs
            {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Promote the result to shared memrory
        %padded_result = transform.get_producer_of_operand %padded[2] : (!pdl.operation) -> (!pdl.operation)
        %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
            {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Run canonicalization.
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func1 : !pdl.operation

        // Find the matmul and fill again
        %tiled_ops = transform.structured.match ops{["linalg.fill", "linalg.vecmat"]} in %fused_for_all : (!pdl.operation) -> !pdl.operation
        %tiled_fill_op, %tiled_padded_matmul = transform.split_handle %tiled_ops : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Second level tile to forall with tile_sizes [2].
        %tiled_matmul_1, %forall_1 =
        transform.structured.tile_using_forall %tiled_padded_matmul tile_sizes [32, 0]  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_fill_2, %fused_for_all_2 = transform.structured.fuse_into_containing_op %tiled_fill_op into %forall_1 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Pad operation.
        %padded_1, %pad_1, %_ = transform.structured.pad %tiled_matmul_1 {
            padding_values=[0 : i32, 0 : i32, 0 : i32],
            padding_dimensions=[0, 1, 2],
            nofold_flags=[0, 0, 1],
            copy_back_op="linalg.copy"
        } : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %pad_1_dps = transform.structured.rewrite_in_destination_passing_style %pad_1 : (!pdl.operation) -> !pdl.operation

        // Run canonicalization.
        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func2 : !pdl.operation

        // Tile reduction dimension into 4 threads (cascade stages).            
        %reduce_fill, %vecmat_6, %reduce_comb, %reduce_forall = transform.structured.tile_reduction_using_forall %padded_1 by num_threads = [0, 4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        %fused_fill_3, %fused_reduce_for_all_3 = transform.structured.fuse_into_containing_op %reduce_fill into %reduce_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        // Tile the remaining reduction using for.
        %tiled_reduction, %loop =
        transform.structured.tile_using_for %vecmat_6 tile_sizes [0, 32]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Run canonicalization to remove redundant tensor.extract_slice.
        %func3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func3 : !pdl.operation

        // Pad operation.
        %padded_reduction, %pad_reduction, %___ = transform.structured.pad %tiled_reduction {
            padding_values=[0 : i32, 0 : i32, 0 : i32],
            padding_dimensions=[0, 1, 2],
            nofold_flags=[1, 1, 0],
            copy_back_op="linalg.copy"
        } : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
        %pad_2_dps = transform.structured.rewrite_in_destination_passing_style %pad_reduction : (!pdl.operation) -> !pdl.operation

        // Promote to local memory
        %padded_reduction_lhs = transform.get_producer_of_operand %padded_reduction[0] : (!pdl.operation) -> (!pdl.operation)
        %padded_reduction_lhs_buffer, %padded_reduction_lhs_new = transform.structured.bufferize_to_allocation %padded_reduction_lhs
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %padded_reduction_rhs = transform.get_producer_of_operand %padded_reduction[1] : (!pdl.operation) -> (!pdl.operation)
        %padded_reduction_rhs_buffer, %padded_reduction_rhs_new = transform.structured.bufferize_to_allocation %padded_reduction_rhs
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %inner_parallel = transform.air.forall_with_reduce_to_parallel %reduce_forall : (!pdl.operation) -> (!pdl.operation)

        // Run canonicalization to remove redundant memcpy (with linalg.generic form) ops created, which can be deleted by canonicalizer. We have to run it again because the memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
        %func4 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func4 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func4 : !pdl.operation
        transform.apply_patterns to %func4 {
            transform.apply_patterns.canonicalization
        } : !pdl.operation

        // Bufferize the cascade buffer and partial result buffer.
        %fill_ops = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %herd_cascade_fill, %herd_reduce_fill = transform.split_handle %fill_ops : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
        %cascade_local_buffer, %cascade_local_new = transform.structured.bufferize_to_allocation %herd_cascade_fill
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %padded_result_local_buffer, %padded_result_local_new = transform.structured.bufferize_to_allocation %herd_reduce_fill
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Run canonicalization.
        %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func5 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func5 : !pdl.operation
        transform.apply_patterns to %func5 {
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        
        // Bufferize
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_op {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_op : !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

        // Run canonicalization to remove redundant memcpy (with linalg.generic form) ops created, which can be deleted by canonicalizer. We have to run it again because the memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
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
    }
}
