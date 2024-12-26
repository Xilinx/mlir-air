
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):

transform.sequence %arg0 : !transform.any_op failures(propagate) {
^bb1(%variant_op: !transform.any_op):
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.generic"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    // First level tile to forall.
    %tiled_matmul, %forall =
      transform.structured.tile_using_forall %matmul num_threads [] tile_sizes [32, 32, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the forall loop.
    %fused_fill, %fused_loop = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_matmul {
      padding_values=[0.000000e+00 : bf16, 0.000000e+00 : bf16, 0.000000e+00 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 1],
      nofold_flags=[1, 1, 1],
      copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!transform.any_op) -> !transform.any_op

    // Promote the operands to shared memory.
    %padded_lhs = transform.get_producer_of_operand %padded[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_lhs_buffer, %padded_lhs_new = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_rhs_buffer, %padded_rhs_new = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote the result to shared memrory
    %padded_result = transform.get_producer_of_operand %padded[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Find the copy operations to tile using for.
    %copy_1 = transform.get_producer_of_operand %padded[0] : (!transform.any_op) -> (!transform.any_op)
    %copy_2 = transform.get_producer_of_operand %padded[1] : (!transform.any_op) -> (!transform.any_op)
    %tiled_copy_1, %tiled_copy_for_loop_1 =
      transform.structured.tile_using_for %copy_1 tile_sizes [0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    %tiled_copy_2, %tiled_copy_for_loop_2 =
      transform.structured.tile_using_for %copy_2 tile_sizes [64, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

    // Second level tile to forall with tile_sizes.
    %tiled_matmul_1, %forall_1 =
      transform.structured.tile_using_forall %padded tile_sizes [16, 16]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Fuse fill operation into the forall loop.
    %fused_fill_1 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill_consumer = transform.get_consumers_of_result %fused_fill_1[0] : (!transform.any_op) -> (!transform.any_op)
    %fused_fill_2, %fused_loop_2 = transform.structured.fuse_into_containing_op %fused_fill_1 into %fill_consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !pdl.operation)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
    // %packed = transform.structured.pack %tiled_matmul_1 packed_sizes = [4, 4, 8]
      // : (!transform.any_op) -> (!transform.any_op)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
    // %packed = transform.structured.pack %tiled_matmul_1 packed_sizes = [64, 64, 64]
    //   : (!transform.any_op) -> (!transform.any_op)

    %func1 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func1 : !transform.any_op

    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    // %pack_producer_a = transform.get_producer_of_operand %packed[0]
    //   : (!transform.any_op) -> (!transform.any_op)
    // %packed_a, %pack_a, %empty_unpack_a =
    //   transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
    //   outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
    //   -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    // %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
    //   : (!transform.any_op) -> (!transform.any_op)
    // %packed_b, %pack_b, %empty_unpack_b =
    //   transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
    //   outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
    //   -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    // %unpack = transform.get_consumers_of_result %packed_b[0]
    //  : (!transform.any_op) -> (!transform.any_op)
    // %packed_c, %pack_c, %unpack_c =
    //   transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
    //   outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
    //   -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Bufferize result to local memory allocation
    // %pack_a = transform.get_producer_of_operand %packed[0] : (!transform.any_op) -> (!transform.any_op)
    // %pack_b = transform.get_producer_of_operand %packed[1] : (!transform.any_op) -> (!transform.any_op)
    // %pack_c = transform.get_producer_of_operand %packed[2] : (!transform.any_op) -> (!transform.any_op)
    // %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
    //   {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // First level for loop.
    // %tiled_reduction, %for_loop =
    //   transform.structured.tile_using_for %packed_c [0, 0, 8]
    //   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // %fused_pack_a, %e1 = transform.structured.fuse_into_containing_op %pack_a into %for_loop
    //   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // %fused_pack_b, %e2 = transform.structured.fuse_into_containing_op %pack_b into %for_loop
    //   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // // Promote the inputs to local memory.
    // %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_pack_a
    //   {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    // %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_pack_b
    //   {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Clean up.
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    //%variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
