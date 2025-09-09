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

        // Bufferize result to shared (L2) memory allocation
        %buffer_res_shared, %new_fill = transform.structured.bufferize_to_allocation %fill
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Find the copy operations to tile using for.
        %func_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.air.convert_memref_copy_to_linalg_copy %func_1
        %copies = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %copy_1, %copy_2 = transform.split_handle %copies : (!pdl.operation<"linalg.copy">) -> (!pdl.operation<"linalg.copy">, !pdl.operation<"linalg.copy">)
        %tiled_copy_1, %tiled_copy_for_loop_1 =
          transform.structured.tile_using_for %copy_1 tile_sizes [0, 256]
          : (!pdl.operation) -> (!pdl.operation, !transform.op<"scf.for">)
        %tiled_copy_2, %tiled_copy_for_loop_2 =
          transform.structured.tile_using_for %copy_2 tile_sizes [256, 0]
          : (!pdl.operation) -> (!pdl.operation, !transform.op<"scf.for">)

        // Second level tile to forall with tile_sizes.
        %matmul_1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %tiled_matmul_1, %forall_1 =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [64, 64] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Run canonicalization
        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_2 : !pdl.operation

        // Fuse fill operation into the forall loop.
        %fused_fill_1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fill_consumer = transform.get_consumers_of_result %fused_fill_1[0] : (!pdl.operation) -> (!pdl.operation)
        %fused_fill_2, %fused_loop_2 = transform.structured.fuse_into_containing_op %fused_fill_1 into %fill_consumer : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
        %packed = transform.structured.pack %tiled_matmul_1 packed_sizes = [4, 4, 8]
          : (!pdl.operation) -> (!pdl.operation)

        // Transpose A matrix.
        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Transpose B matrix.
        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Transpose C matrix.
        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!pdl.operation) -> (!pdl.operation)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!pdl.operation, !pdl.operation)
          -> (!pdl.operation, !pdl.operation, !pdl.operation)

        // Bufferize result to local memory allocation
        %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Tile the reduction loop.
        %tiled_reduction, %for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 4]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse pack ops into the for loop.
        %fused_pack_a, %e1 = transform.structured.fuse_into_containing_op %pack_a into %for_loop
          : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_pack_b, %e2 = transform.structured.fuse_into_containing_op %pack_b into %for_loop
          : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Promote the inputs to local memory.
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_pack_a
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_pack_b
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Run canonicalization
        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func_3 : !pdl.operation

        
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
        %func_op_updated = transform.air.remove_uninitialized_copy %func6
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated

        // Tile linalg.generics for vectorization
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [1, 1, 1, 0, 0, 0]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)     

        // Tile linalg.fills for vectorized write
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %linalg_fills tile_sizes [1, 1]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)         
    }
}
