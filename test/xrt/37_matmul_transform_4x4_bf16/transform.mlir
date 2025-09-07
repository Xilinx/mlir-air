// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_copy : benefit(1) {
        %args = pdl.operands
        %results = pdl.types
        %op = pdl.operation "memref.copy"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
        pdl.rewrite %op with "transform.dialect"
    }
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation

        // First level tile to forall.
        %first_level_tiled_matmul, %outer_forall =
        transform.structured.tile_using_forall %matmul tile_sizes [256, 256]  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse fill operation into the forall loop.
        %fused_fill, %1 = transform.structured.fuse_into_containing_op %fill into %outer_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // First level pack the matmul.
        %first_level_tiled_transposed_l2_packed_matmul = transform.structured.pack %first_level_tiled_matmul packed_sizes = [64, 64, 32]
        : (!pdl.operation) -> (!pdl.operation)

        %lhs_transposed_l2_pack_op = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[0] : (!pdl.operation) -> (!pdl.operation)
        %first_level_tiled_l2_packed_matmul, %lhs_l2_pack, %lhs_unpack =
        transform.structured.pack_transpose %lhs_transposed_l2_pack_op with_compute_op(%first_level_tiled_transposed_l2_packed_matmul)
        outer_perm = [0, 1] inner_perm = [0, 1] : (!pdl.operation, !pdl.operation)
        -> (!pdl.operation, !pdl.operation, !pdl.operation)

        %rhs_transposed_l2_pack_op = transform.get_producer_of_operand %first_level_tiled_l2_packed_matmul[1] : (!pdl.operation) -> (!pdl.operation)
        %first_level_tiled_l2_packed_matmul_lhs_transposed, %rhs_l2_pack, %rhs_unpack =
        transform.structured.pack_transpose %rhs_transposed_l2_pack_op with_compute_op(%first_level_tiled_l2_packed_matmul)
        outer_perm = [1, 0] inner_perm = [1, 0] : (!pdl.operation, !pdl.operation)
        -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Run canonicalization
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func1 : !pdl.operation

        // Promote the fused fill to shared memory
        %result_l2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
            {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !pdl.operation

        // Second level pack the matmul.
        %generic_op = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %l1_packed = transform.structured.pack %generic_op packed_sizes = [0, 0, 0, 4, 4, 8]
          : (!pdl.operation) -> (!pdl.operation)

        // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
        %l1_packed_lhs = transform.get_producer_of_operand %l1_packed[0]
          : (!pdl.operation) -> (!pdl.operation)
        %lhs_l1_packed_matmul, %lhs_l1_pack_op, %lhs_l1_unpack_op =
          transform.structured.pack_transpose %l1_packed_lhs with_compute_op(%l1_packed)
          outer_perm = [0, 1, 3, 2] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
        %l1_packed_rhs = transform.get_producer_of_operand %lhs_l1_packed_matmul[1]
          : (!pdl.operation) -> (!pdl.operation)
        %operands_l1_packed_matmul, %rhs_l1_pack_op, %rhs_l1_unpack_op =
          transform.structured.pack_transpose %l1_packed_rhs with_compute_op(%lhs_l1_packed_matmul)
          outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
        %l1_packed_output = transform.get_consumers_of_result %operands_l1_packed_matmul[0]
          : (!pdl.operation) -> (!pdl.operation)
        %l1_packed_matmul, %output_l1_pack_op, %output_l1_unpack_op =
          transform.structured.pack_transpose %l1_packed_output with_compute_op(%operands_l1_packed_matmul)
          outer_perm = [0, 1, 3, 2] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Promote the result to local memory
        %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %output_l1_pack_op
            {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !pdl.operation

        // First level for loop.
        %first_level_tiled_reduction_matmul, %outer_for_loop =
          transform.structured.tile_using_for %l1_packed_matmul tile_sizes [0, 0, 1]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse the pack operations in the outer for loop.
        %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %lhs_l1_pack_op into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %rhs_l1_pack_op into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_lhs_l2_pack, %4 = transform.structured.fuse_into_containing_op %lhs_l2_pack into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l2_pack, %5 = transform.structured.fuse_into_containing_op %rhs_l2_pack into %outer_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Promote the lhs to shared memory
        %lhs_l2_pack_buffer, %lhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l2_pack
          {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !pdl.operation

        // Promote the rhs to shared memory
        %rhs_l2_pack_buffer, %rhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l2_pack
          {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !pdl.operation

        // Run canonicalization
        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func2 : !pdl.operation

        // Second level tile to forall with tile_sizes.
        %second_level_tiled_matmul, %inner_forall =
          transform.structured.tile_using_forall %first_level_tiled_reduction_matmul tile_sizes [1, 1, 0, 0, 0, 0]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse the pack operations in inner forall loop.
        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Second level for loop.
        %generic_op1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %second_level_tiled_reduction_matmul, %inner_for_loop =
          transform.structured.tile_using_for %generic_op1 tile_sizes [0, 0, 0, 0, 0, 4]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse the pack operations in inner for loop.
        %fused_lhs_l1_pack3, %8 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack2 into %inner_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_rhs_l1_pack3, %9 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack2 into %inner_for_loop : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Promote the LHS to local memory.
        %lhs_l1_pack_buffer, %lhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack3
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Promote the RHS to local memory.
        %rhs_l1_pack_buffer, %rhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack3
          {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !pdl.operation

        // Run canonicalization
        %func3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func3 : !pdl.operation

        // Hoist static alloc out of the loops
        %func8 = transform.structured.match ops{["func.func"]} in %arg1
          : (!pdl.operation) -> !pdl.operation
        transform.air.hoist_static_alloc %func8 : (!pdl.operation) -> ()

        // Peel the for loop
        %for_op = transform.structured.match ops{["scf.for"]} in %arg1 : (!pdl.operation) -> !transform.op<"scf.for">

        // Find the producer operation (fill), and tile using for_all, as the prologue.
        %fill_op = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %prologue_tiled_fill, %prologue_forall =
          transform.structured.tile_using_forall %fill_op tile_sizes [1, 1]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Find the consumer operation (unpack), and tile using for_all, as the epilogue.
        %unpack_ops = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %l1_to_l2_unpack, %l2_to_l3_unpack = transform.split_handle %unpack_ops : (!pdl.operation<"linalg.unpack">) -> (!pdl.operation<"linalg.unpack">, !pdl.operation<"linalg.unpack">)
        %epilogue_tiled_unpack, %epilogue_forall =
          transform.structured.tile_using_forall %l1_to_l2_unpack tile_sizes [1, 1]
            : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Run canonicalization
        %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func5 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func5 : !pdl.operation
        
        // Bufferize
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
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
        %func_op_updated = transform.air.remove_uninitialized_memref_copy %func6
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated

        // Tile linalg.generics for vectorization
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:6 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [1, 1, 1, 1, 1, 1, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)     

        // Tile linalg.fills for vectorized write
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_fills, %vec_fill_loops:4 =
          transform.structured.tile_using_for %linalg_fills tile_sizes [1, 1, 1, 1]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation) 
    }
}
