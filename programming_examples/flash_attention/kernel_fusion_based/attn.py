# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from math import cos, sin, sqrt, exp
import numpy as np

import air
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.dialects import scf, affine, arith

range_ = for_


@module_builder
def build_module(
    lk=12288,
    lkp=96,
    lq=512,
    lqp=128,
    dk=64,
    dv=64,
    num_q_tiles=4,
    num_cascade_stages=4,
    num_heads=12,
):
    """Build the attention module using Python bindings

    Args:
        lk: Total sequence length for K/V matrices (default: 12288)
        lkp: Chunk size for K/V processing per AIE tile (default: 96)
        lq: Total sequence length for Q matrix (default: 512)
        lqp: Chunk size for Q processing per launch iteration (default: 128)
        dk: Key dimension (default: 64)
        dv: Value dimension (default: 64)
        num_q_tiles: Number of tiles to partition Q chunk (lqp) into (default: 4)
        num_cascade_stages: Number of cascade pipeline stages (default: 4)
        num_heads: Number of attention heads (default: 12)
    """
    # Validate divisibility requirements
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert (
        lk % (lkp * num_cascade_stages) == 0
    ), f"lk ({lk}) must be divisible by lkp * num_cascade_stages ({lkp * num_cascade_stages})"
    assert (
        num_heads % 2 == 0
    ), f"num_heads ({num_heads}) must be divisible by 2 (segment unroll constraint)"

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    # Architecture-specific matrix multiplication dimensions
    mmul_mkn = [8, 8, 8]  # For aie2p
    mmul_m, mmul_k, mmul_n = mmul_mkn

    # Hardware constraint: max 2 heads per segment unroll
    num_heads_per_unroll = 2
    num_head_groups = num_heads // num_heads_per_unroll

    # Derived parameters
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // num_cascade_stages
    num_lq_iters = lq // lqp  # Number of launch iterations for lq dimension
    tile_size_q = lqp // num_q_tiles  # Tile size within each lqp chunk

    # Memory spaces: L1 = 2 : i32, L2 = 1 : i32
    l1_space = IntegerAttr.get(i32, 2)  # L1 uses memory space 2
    l2_space = IntegerAttr.get(i32, 1)  # L2 uses memory space 1

    # L1 MemRefTypes (memory space 2 : i32) - used in herd bodies
    memref_lqp_dv_l1 = MemRefType.get([tile_size_q, dk], bf16, memory_space=l1_space)
    memref_lqp_l1 = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)
    memref_lqp_lkp_l1 = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    memref_dv_lkp_l1 = MemRefType.get([dk, lkp], bf16, memory_space=l1_space)

    # L2 MemRefTypes (memory space 1 : i32) - segment allocations
    memref_lqp_dk_l2 = MemRefType.get([tile_size_q, dk], bf16, memory_space=l2_space)
    memref_dk_lkp_l2 = MemRefType.get([dk, lkp], bf16, memory_space=l2_space)
    memref_lkp_dv_l2 = MemRefType.get([lkp, dk], bf16, memory_space=l2_space)
    memref_output_lqp_dv_l2 = MemRefType.get(
        [lqp, dk], bf16, memory_space=l2_space
    )  # Per-iteration output buffer

    # L3 MemRefTypes (no memory space annotation = default L3) - with head dimension
    memref_input_q_lq_dk = MemRefType.get([num_heads, lq, dk], bf16)
    memref_output_lq_dv = MemRefType.get([num_heads, lq, dk], bf16)
    memref_input_k_dk_lk = MemRefType.get([num_heads, dk, lk], bf16)
    memref_input_v_lk_dv = MemRefType.get([num_heads, lk, dk], bf16)
    memref_input_m_lq_lk = MemRefType.get([num_heads, lq, lk], bf16)

    # Helper function to create external function declarations
    def external_func(name, inputs, outputs=None, link_with=None, visibility="private"):
        if outputs is None:
            outputs = []
        func_type = FunctionType.get(inputs, outputs)
        func = FuncOp(name=name, type=func_type, visibility=visibility)
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    # External function declarations
    external_func("zero_fill_gp_bf16", [memref_lqp_dv_l1], link_with="attn.o")
    external_func("zero_fill_sp_bf16", [memref_lqp_l1], link_with="attn.o")
    external_func("zero_fill_g_bf16", [memref_lqp_lkp_l1], link_with="attn.o")
    external_func("neg_inf_fill_up_bf16", [memref_lqp_l1], link_with="attn.o")
    external_func(
        "matmul_a_b_bf16",
        [memref_lqp_dv_l1, memref_dv_lkp_l1, memref_lqp_lkp_l1],
        link_with="attn.o",
    )
    external_func(
        "matmul_g_b_bf16",
        [memref_lqp_lkp_l1, memref_dv_lkp_l1, memref_lqp_dv_l1],
        link_with="attn.o",
    )
    external_func("max_g_bf16", [memref_lqp_lkp_l1, memref_lqp_l1], link_with="attn.o")
    external_func(
        "maximum_up_u_bf16", [memref_lqp_l1, memref_lqp_l1], link_with="attn.o"
    )
    external_func(
        "exp_g_minus_u", [memref_lqp_l1, memref_lqp_lkp_l1], link_with="attn.o"
    )
    external_func(
        "exp_up_minus_u",
        [memref_lqp_l1, memref_lqp_l1, memref_lqp_l1],
        link_with="attn.o",
    )
    external_func("mul_r_gp", [memref_lqp_l1, memref_lqp_dv_l1], link_with="attn.o")
    external_func("sum_g", [memref_lqp_lkp_l1, memref_lqp_l1], link_with="attn.o")
    external_func(
        "accum_sp_r_s",
        [memref_lqp_l1, memref_lqp_l1, memref_lqp_l1],
        link_with="attn.o",
    )
    external_func(
        "vector_copy_32elems", [i32, memref_lqp_l1, memref_lqp_l1], link_with="attn.o"
    )
    external_func("div_gp_sp", [memref_lqp_l1, memref_lqp_dv_l1], link_with="attn.o")
    external_func(
        "vector_copy_swizzle_elems",
        [i32, memref_lqp_lkp_l1, memref_lqp_lkp_l1],
        link_with="attn.o",
    )
    external_func(
        "vector_copy_unswizzle_elems",
        [i32, memref_lqp_lkp_l1, memref_lqp_lkp_l1],
        link_with="attn.o",
    )
    external_func("add_gp_g", [memref_lqp_dv_l1, memref_lqp_dv_l1], link_with="attn.o")

    # Channel declarations - use num_heads_per_unroll (2) for segment unroll
    Channel("L3ToL2Chan1", size=[num_heads_per_unroll, num_cascade_stages])
    Channel("L3ToL2Chan2", size=[num_heads_per_unroll, num_cascade_stages])
    chan_l2_to_l1_1 = Channel(
        "L2ToL1Chan1",
        size=[num_q_tiles, 1],
        broadcast_shape=[num_q_tiles, num_cascade_stages],
    )
    chan_l2_to_l1_1.attributes["channel_type"] = StringAttr.get("dma_packet")
    chan_l2_to_l1_2 = Channel(
        "L2ToL1Chan2",
        size=[1, num_cascade_stages],
        broadcast_shape=[num_q_tiles, num_cascade_stages],
    )
    chan_l2_to_l1_2.attributes["channel_type"] = StringAttr.get("dma_packet")
    chan_l2_to_l1_3 = Channel(
        "L2ToL1Chan3",
        size=[1, num_cascade_stages],
        broadcast_shape=[num_q_tiles, num_cascade_stages],
    )
    Channel("L1ToL2Chan1", size=[num_q_tiles, 1])
    Channel("L2ToL3Chan1", size=[num_heads_per_unroll])
    chan_cascade = Channel("cascade", size=[num_q_tiles, num_cascade_stages - 1])
    chan_cascade.attributes["channel_type"] = StringAttr.get("cascade")

    # Main attention function
    @FuncOp.from_py_func(
        memref_input_q_lq_dk,
        memref_input_k_dk_lk,
        memref_input_v_lk_dv,
        memref_input_m_lq_lk,
        memref_output_lq_dv,
    )
    def attention_bf16(arg0, arg1, arg2, arg3, arg4):
        c_num_lq_iters = ConstantOp(index_type, num_lq_iters)
        c_num_head_groups = ConstantOp(index_type, num_head_groups)

        # Launch iterates over (lq_iters, head_groups) - [4, 6] for 12 heads
        @launch(
            operands=[arg0, arg1, arg2, arg4], sizes=[c_num_lq_iters, c_num_head_groups]
        )
        def launch_body(arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12):
            # arg5 = lq iteration index, arg6 = head group index (0..5 for 12 heads)
            c0 = ConstantOp(index_type, 0)
            c1 = ConstantOp(index_type, 1)

            # Compute actual head indices from head group
            # head_base = arg6 * 2 (for head groups 0,1,2,3,4,5 -> heads 0-1, 2-3, 4-5, 6-7, 8-9, 10-11)
            affine_map_head_base = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(num_heads_per_unroll),
                    )
                ],
            )
            head_base = affine_apply(affine_map_head_base, [arg6])
            affine_map_add_one = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_add(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(1)
                    )
                ],
            )
            head_1 = affine_apply(affine_map_add_one, [head_base])

            # Affine map for Q tile partitioning within lqp chunk
            affine_map_tileq = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_size_q)
                    )
                ],
            )
            # Affine map for launch offset: arg5 * lqp * dk
            affine_map_launch_offset = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(lqp * dk)
                    )
                ],
            )
            # Affine map for Q head offset: head * lq * dk + launch_offset
            affine_map_q_head_offset = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_add(
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(lq * dk)
                        ),
                        AffineSymbolExpr.get(1),
                    )
                ],
            )
            # Affine map for K head offset with column: head * dk * lk + col_offset
            affine_map_head_col = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_add(
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(dk * lk)
                        ),
                        AffineSymbolExpr.get(1),
                    )
                ],
            )
            # Affine map for V head offset: head * lk * dv
            affine_map_v_head_offset = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(lk * dv)
                    )
                ],
            )

            # scf.parallel for L3 to L2 Q matrix transfers for both heads in group
            par_1 = scf.ForallOp(
                lower_bounds=[0], upper_bounds=[num_cascade_stages], steps=[1]
            )
            with InsertionPoint(par_1.body):
                tile_offset = affine_apply(
                    affine_map_tileq, [par_1.induction_variables[0]]
                )
                launch_offset = affine_apply(affine_map_launch_offset, [arg5])
                # Head 0 in group (head_base)
                q_head0_off = affine_apply(
                    affine_map_q_head_offset, [head_base, launch_offset]
                )
                ChannelPut(
                    "L3ToL2Chan1",
                    arg9,
                    indices=[c0, par_1.induction_variables[0]],
                    offsets=[tile_offset, q_head0_off],
                    sizes=[tile_size_q, dk],
                    strides=[dk, 1],
                )
                # Head 1 in group (head_base + 1)
                q_head1_off = affine_apply(
                    affine_map_q_head_offset, [head_1, launch_offset]
                )
                ChannelPut(
                    "L3ToL2Chan1",
                    arg9,
                    indices=[c1, par_1.induction_variables[0]],
                    offsets=[tile_offset, q_head1_off],
                    sizes=[tile_size_q, dk],
                    strides=[dk, 1],
                )
                scf.InParallelOp()

            # L3 to L2 channel puts for K matrix - both heads in group
            for i in range(num_cascade_stages):
                col_off = ConstantOp(index_type, i * lkp)
                # Head 0 in group
                k_head0_off = affine_apply(affine_map_head_col, [head_base, col_off])
                ChannelPut(
                    "L3ToL2Chan1",
                    arg10,
                    indices=[c0, i],
                    offsets=[0, 0, k_head0_off],
                    sizes=[chunks_per_stage, dk, lkp],
                    strides=[lkp * num_cascade_stages, lk, 1],
                )
                # Head 1 in group
                k_head1_off = affine_apply(affine_map_head_col, [head_1, col_off])
                ChannelPut(
                    "L3ToL2Chan1",
                    arg10,
                    indices=[c1, i],
                    offsets=[0, 0, k_head1_off],
                    sizes=[chunks_per_stage, dk, lkp],
                    strides=[lkp * num_cascade_stages, lk, 1],
                )

            # L3 to L2 channel puts for V matrix - both heads in group
            for i in range(num_cascade_stages):
                # Head 0 in group
                v_head0_off = affine_apply(affine_map_v_head_offset, [head_base])
                ChannelPut(
                    "L3ToL2Chan2",
                    arg11,
                    indices=[c0, i],
                    offsets=[0, i * lkp, v_head0_off],
                    sizes=[chunks_per_stage, lkp, dv],
                    strides=[lkp * num_cascade_stages * dv, dv, 1],
                )
                # Head 1 in group
                v_head1_off = affine_apply(affine_map_v_head_offset, [head_1])
                ChannelPut(
                    "L3ToL2Chan2",
                    arg11,
                    indices=[c1, i],
                    offsets=[0, i * lkp, v_head1_off],
                    sizes=[chunks_per_stage, lkp, dv],
                    strides=[lkp * num_cascade_stages * dv, dv, 1],
                )

            # Segment unrolls over 2 heads (hardware constraint)
            c_num_heads_unroll = ConstantOp(index_type, num_heads_per_unroll)
            c_dummy_size = ConstantOp(index_type, 1)

            @segment(
                name="attention_seg",
                operands=[],
                sizes=[c_num_heads_unroll, c_dummy_size],
            )
            def segment_body(head_idx, dummy_idx, head_size, dummy_size):
                # L2 allocations
                alloc = AllocOp(memref_lqp_dk_l2, [], [])
                alloc_col1 = AllocOp(memref_lqp_dk_l2, [], [])
                alloc_col2 = AllocOp(memref_lqp_dk_l2, [], [])
                alloc_col3 = AllocOp(memref_lqp_dk_l2, [], [])
                alloc_2 = AllocOp(memref_dk_lkp_l2, [], [])
                alloc_21 = AllocOp(memref_dk_lkp_l2, [], [])
                alloc_22 = AllocOp(memref_dk_lkp_l2, [], [])
                alloc_23 = AllocOp(memref_dk_lkp_l2, [], [])
                alloc_3 = AllocOp(memref_lkp_dv_l2, [], [])
                alloc_31 = AllocOp(memref_lkp_dv_l2, [], [])
                alloc_32 = AllocOp(memref_lkp_dv_l2, [], [])
                alloc_33 = AllocOp(memref_lkp_dv_l2, [], [])
                alloc_5 = AllocOp(memref_output_lqp_dv_l2, [], [])
                up = AllocOp(memref_lqp_l1, [], [])
                sp = AllocOp(memref_lqp_l1, [], [])
                Gp = AllocOp(memref_lqp_dv_l1, [], [])
                alloc_6 = AllocOp(memref_lqp_dv_l1, [], [])

                c_num_q_tiles = ConstantOp(index_type, num_q_tiles)
                c_num_cascade = ConstantOp(index_type, num_cascade_stages)
                c0_seg = ConstantOp(index_type, 0)
                c1_seg = ConstantOp(index_type, 1)
                c2_seg = ConstantOp(index_type, 2)
                c3_seg = ConstantOp(index_type, 3)

                # L3 to L2 channel gets for Q matrix - use head_idx
                ChannelGet("L3ToL2Chan1", alloc.result, indices=[head_idx, c0_seg])
                ChannelGet("L3ToL2Chan1", alloc_col1.result, indices=[head_idx, c1_seg])
                ChannelGet("L3ToL2Chan1", alloc_col2.result, indices=[head_idx, c2_seg])
                ChannelGet("L3ToL2Chan1", alloc_col3.result, indices=[head_idx, c3_seg])

                # L2 to L1 channel puts for Q matrix
                ChannelPut(
                    "L2ToL1Chan1",
                    alloc.result,
                    indices=[c0_seg, c0_seg],
                    offsets=[0, 0, 0, 0],
                    sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                    strides=[mmul_k, dk * mmul_k, dk, 1],
                )
                ChannelPut(
                    "L2ToL1Chan1",
                    alloc_col1.result,
                    indices=[c1_seg, c0_seg],
                    offsets=[0, 0, 0, 0],
                    sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                    strides=[mmul_k, dk * mmul_k, dk, 1],
                )
                ChannelPut(
                    "L2ToL1Chan1",
                    alloc_col2.result,
                    indices=[c2_seg, c0_seg],
                    offsets=[0, 0, 0, 0],
                    sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                    strides=[mmul_k, dk * mmul_k, dk, 1],
                )
                ChannelPut(
                    "L2ToL1Chan1",
                    alloc_col3.result,
                    indices=[c3_seg, c0_seg],
                    offsets=[0, 0, 0, 0],
                    sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                    strides=[mmul_k, dk * mmul_k, dk, 1],
                )

                # First herd - initialization
                @herd(
                    name="herd_0",
                    sizes=[c_num_q_tiles, c_num_cascade],
                    operands=[alloc_6, up, sp, Gp],
                    link_with="attn.o",
                )
                def herd_body_init(
                    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29
                ):
                    ChannelGet("L2ToL1Chan1", arg26, indices=[arg22, arg23])
                    CallOp([], "zero_fill_gp_bf16", [arg29])
                    CallOp([], "zero_fill_sp_bf16", [arg28])
                    CallOp([], "neg_inf_fill_up_bf16", [arg27])

                # Main loop over lk chunks
                for arg21 in range_(0, chunks_per_stage, 1):
                    # Channel gets for K and V - use head_idx
                    ChannelGet(
                        "L3ToL2Chan1", alloc_2.result, indices=[head_idx, c0_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan2", alloc_3.result, indices=[head_idx, c0_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan1", alloc_21.result, indices=[head_idx, c1_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan2", alloc_31.result, indices=[head_idx, c1_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan1", alloc_22.result, indices=[head_idx, c2_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan2", alloc_32.result, indices=[head_idx, c2_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan1", alloc_23.result, indices=[head_idx, c3_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan2", alloc_33.result, indices=[head_idx, c3_seg]
                    )

                    # Channel puts for K matrix to L1
                    ChannelPut(
                        "L2ToL1Chan2",
                        alloc_2.result,
                        indices=[c0_seg, c0_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[lkp // mmul_n, dk // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, lkp * mmul_n, lkp, 1],
                    )
                    ChannelPut(
                        "L2ToL1Chan2",
                        alloc_21.result,
                        indices=[c0_seg, c1_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[lkp // mmul_n, dk // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, lkp * mmul_n, lkp, 1],
                    )
                    ChannelPut(
                        "L2ToL1Chan2",
                        alloc_22.result,
                        indices=[c0_seg, c2_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[lkp // mmul_n, dk // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, lkp * mmul_n, lkp, 1],
                    )
                    ChannelPut(
                        "L2ToL1Chan2",
                        alloc_23.result,
                        indices=[c0_seg, c3_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[lkp // mmul_n, dk // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, lkp * mmul_n, lkp, 1],
                    )

                    # Channel puts for V matrix to L1
                    ChannelPut(
                        "L2ToL1Chan3",
                        alloc_3.result,
                        indices=[c0_seg, c0_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[dv // mmul_n, lkp // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, dv * mmul_n, dv, 1],
                    )
                    ChannelPut(
                        "L2ToL1Chan3",
                        alloc_31.result,
                        indices=[c0_seg, c1_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[dv // mmul_n, lkp // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, dv * mmul_n, dv, 1],
                    )
                    ChannelPut(
                        "L2ToL1Chan3",
                        alloc_32.result,
                        indices=[c0_seg, c2_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[dv // mmul_n, lkp // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, dv * mmul_n, dv, 1],
                    )
                    ChannelPut(
                        "L2ToL1Chan3",
                        alloc_33.result,
                        indices=[c0_seg, c3_seg],
                        offsets=[0, 0, 0, 0],
                        sizes=[dv // mmul_n, lkp // mmul_k, mmul_k, mmul_n],
                        strides=[mmul_n, dv * mmul_n, dv, 1],
                    )

                    # Second herd - computation inside loop
                    @herd(
                        name="herd_0",
                        sizes=[c_num_q_tiles, c_num_cascade],
                        operands=[alloc_6, up, sp, Gp],
                        link_with="attn.o",
                    )
                    def herd_body_compute(
                        arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29
                    ):
                        u_l1 = AllocOp(memref_lqp_l1, [], [])
                        s_l1 = AllocOp(memref_lqp_l1, [], [])
                        r_l1 = AllocOp(memref_lqp_l1, [], [])
                        alloc_56 = AllocOp(memref_dv_lkp_l1, [], [])
                        alloc_57 = AllocOp(memref_dv_lkp_l1, [], [])
                        G_l1 = AllocOp(memref_lqp_lkp_l1, [], [])

                        CallOp([], "zero_fill_g_bf16", [G_l1.result])
                        ChannelGet(
                            "L2ToL1Chan2", alloc_56.result, indices=[arg22, arg23]
                        )
                        CallOp(
                            [], "matmul_a_b_bf16", [arg26, alloc_56.result, G_l1.result]
                        )
                        DeallocOp(alloc_56)

                        c0_i32 = ConstantOp(i32, 0)
                        CallOp([], "max_g_bf16", [G_l1.result, u_l1.result])
                        CallOp([], "maximum_up_u_bf16", [arg27, u_l1.result])
                        CallOp([], "exp_g_minus_u", [u_l1.result, G_l1.result])
                        CallOp([], "exp_up_minus_u", [arg27, u_l1.result, r_l1.result])
                        CallOp([], "mul_r_gp", [r_l1.result, arg29])
                        ChannelGet(
                            "L2ToL1Chan3", alloc_57.result, indices=[arg22, arg23]
                        )
                        CallOp(
                            [], "matmul_g_b_bf16", [G_l1.result, alloc_57.result, arg29]
                        )
                        DeallocOp(alloc_57)
                        CallOp([], "sum_g", [G_l1.result, s_l1.result])
                        CallOp([], "accum_sp_r_s", [arg28, r_l1.result, s_l1.result])
                        CallOp([], "vector_copy_32elems", [c0_i32, s_l1.result, arg28])
                        CallOp([], "vector_copy_32elems", [c0_i32, u_l1.result, arg27])

                        DeallocOp(u_l1)
                        DeallocOp(s_l1)
                        DeallocOp(r_l1)
                        DeallocOp(G_l1)

                    yield_([])

                # Third herd - final processing with cascade and affine.if
                @herd(
                    name="herd_0",
                    sizes=[c_num_q_tiles, c_num_cascade],
                    operands=[alloc_6, up, sp, Gp],
                    link_with="attn.o",
                )
                def herd_body_final(
                    arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29
                ):
                    c1_h = ConstantOp(index_type, 1)
                    r_l1 = AllocOp(memref_lqp_l1, [], [])

                    # affine.if for last cascade stage
                    affine_set_last = IntegerSet.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(-num_cascade_stages + 1),
                            ),
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_add(
                                AffineConstantExpr.get(num_q_tiles - 1),
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0), AffineConstantExpr.get(-1)
                                ),
                            ),
                        ],
                        [True, False, False],
                    )

                    affine_if_last = affine.AffineIfOp(
                        affine_set_last, cond_operands=[arg22, arg23], has_else=True
                    )
                    with InsertionPoint(affine_if_last.then_block):
                        subi = arith.SubIOp(arg23, c1_h)
                        ChannelPut("cascade", arg29, indices=[arg22, subi])
                        ChannelPut("cascade", arg27, indices=[arg22, subi])
                        ChannelPut("cascade", arg28, indices=[arg22, subi])
                        affine.AffineYieldOp([])

                    with InsertionPoint(affine_if_last.else_block):
                        affine_set_middle = IntegerSet.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(1), AffineConstantExpr.get(-1)
                                ),
                                AffineExpr.get_add(
                                    AffineConstantExpr.get(num_cascade_stages - 2),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(-1),
                                    ),
                                ),
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_add(
                                    AffineConstantExpr.get(num_q_tiles - 1),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(-1),
                                    ),
                                ),
                            ],
                            [False, False, False, False],
                        )

                        affine_if_middle = affine.AffineIfOp(
                            affine_set_middle,
                            cond_operands=[arg22, arg23],
                            has_else=True,
                        )
                        with InsertionPoint(affine_if_middle.then_block):
                            Gp_cascade = AllocOp(memref_lqp_dv_l1, [], [])
                            up_cascade = AllocOp(memref_lqp_l1, [], [])
                            sp_cascade = AllocOp(memref_lqp_l1, [], [])
                            ChannelGet(
                                "cascade", Gp_cascade.result, indices=[arg22, arg23]
                            )
                            ChannelGet(
                                "cascade", up_cascade.result, indices=[arg22, arg23]
                            )
                            ChannelGet(
                                "cascade", sp_cascade.result, indices=[arg22, arg23]
                            )
                            CallOp([], "maximum_up_u_bf16", [up_cascade.result, arg27])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_cascade.result, arg27, r_l1.result],
                            )
                            CallOp([], "mul_r_gp", [r_l1.result, Gp_cascade.result])
                            CallOp([], "add_gp_g", [arg29, Gp_cascade.result])
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [arg28, r_l1.result, sp_cascade.result],
                            )
                            subi = arith.SubIOp(arg23, c1_h)
                            ChannelPut(
                                "cascade", Gp_cascade.result, indices=[arg22, subi]
                            )
                            ChannelPut(
                                "cascade", up_cascade.result, indices=[arg22, subi]
                            )
                            ChannelPut(
                                "cascade", sp_cascade.result, indices=[arg22, subi]
                            )
                            affine.AffineYieldOp([])

                        with InsertionPoint(affine_if_middle.else_block):
                            Gp_cascade = AllocOp(memref_lqp_dv_l1, [], [])
                            up_cascade = AllocOp(memref_lqp_l1, [], [])
                            sp_cascade = AllocOp(memref_lqp_l1, [], [])
                            ChannelGet(
                                "cascade", Gp_cascade.result, indices=[arg22, arg23]
                            )
                            ChannelGet(
                                "cascade", up_cascade.result, indices=[arg22, arg23]
                            )
                            ChannelGet(
                                "cascade", sp_cascade.result, indices=[arg22, arg23]
                            )
                            CallOp([], "maximum_up_u_bf16", [up_cascade.result, arg27])
                            CallOp(
                                [],
                                "exp_up_minus_u",
                                [up_cascade.result, arg27, r_l1.result],
                            )
                            CallOp([], "mul_r_gp", [r_l1.result, Gp_cascade.result])
                            CallOp([], "add_gp_g", [arg29, Gp_cascade.result])
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [arg28, r_l1.result, sp_cascade.result],
                            )
                            CallOp(
                                [], "div_gp_sp", [sp_cascade.result, Gp_cascade.result]
                            )
                            ChannelPut(
                                "L1ToL2Chan1",
                                Gp_cascade.result,
                                indices=[arg22, 0],
                                offsets=[0, 0, 0, 0],
                                sizes=[
                                    tile_size_q // mmul_n,
                                    mmul_m,
                                    dv // mmul_m,
                                    mmul_n,
                                ],
                                strides=[
                                    mmul_m * mmul_n,
                                    mmul_n,
                                    tile_size_q * mmul_n,
                                    1,
                                ],
                            )
                            affine.AffineYieldOp([])
                        affine.AffineYieldOp([])

                # Parallel gather results from L1 to L2
                affine_map_tileq_seg = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_size_q)
                        )
                    ],
                )
                par_final = scf.ForallOp(
                    lower_bounds=[0], upper_bounds=[c_num_q_tiles], steps=[1]
                )
                with InsertionPoint(par_final.body):
                    apply_final = affine_apply(
                        affine_map_tileq_seg, [par_final.induction_variables[0]]
                    )
                    ChannelGet(
                        "L1ToL2Chan1",
                        alloc_5.result,
                        indices=[par_final.induction_variables[0], 0],
                        offsets=[apply_final, 0],
                        sizes=[tile_size_q, dv],
                        strides=[dv, 1],
                    )
                    scf.InParallelOp()

                # L2 to L3 transfer - use head_idx
                ChannelPut("L2ToL3Chan1", alloc_5.result, indices=[head_idx])

            # Output channel gets for both heads in group
            launch_offset_out = affine_apply(affine_map_launch_offset, [arg5])
            out_head0_off = affine_apply(
                affine_map_q_head_offset, [head_base, launch_offset_out]
            )
            out_head1_off = affine_apply(
                affine_map_q_head_offset, [head_1, launch_offset_out]
            )
            ChannelGet(
                "L2ToL3Chan1",
                arg12,
                indices=[c0],
                offsets=[0, out_head0_off],
                sizes=[lqp, dk],
                strides=[dk, 1],
            )
            ChannelGet(
                "L2ToL3Chan1",
                arg12,
                indices=[c1],
                offsets=[0, out_head1_off],
                sizes=[lqp, dk],
                strides=[dk, 1],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="attn.py")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--mlir-file",
        type=str,
        default=None,
        help="Path to external MLIR file to compile (instead of generating)",
    )
    parser.add_argument(
        "--lk", type=int, default=12288, help="Total sequence length for K/V matrices"
    )
    parser.add_argument(
        "--lkp", type=int, default=96, help="Chunk size for K/V processing"
    )
    parser.add_argument(
        "--lq", type=int, default=512, help="Total sequence length for Q matrix"
    )
    parser.add_argument(
        "--lqp",
        type=int,
        default=128,
        help="Chunk size for Q processing per launch iteration",
    )
    parser.add_argument("--dk", type=int, default=64, help="Key dimension")
    parser.add_argument("--dv", type=int, default=64, help="Value dimension")
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="run",
        choices=["run", "compile"],
        help="Compilation mode: run (default, compile + test), compile (generate binary only)",
    )
    args = parser.parse_args()

    lk, lkp, lq, lqp, dk, dv = args.lk, args.lkp, args.lq, args.lqp, args.dk, args.dv
    num_heads = args.num_heads

    if args.mlir_file:
        with open(args.mlir_file, "r") as f:
            mlir_source = f.read()
        with Context() as ctx, Location.unknown():
            registry = DialectRegistry()
            air.dialects.air.register_dialect(registry)
            ctx.append_dialect_registry(registry)
            ctx.load_all_available_dialects()
            mlir_module = Module.parse(mlir_source)
        print(f"Loaded MLIR module from: {args.mlir_file}")
    else:
        mlir_module = build_module(
            lk=lk,
            lkp=lkp,
            lq=lq,
            lqp=lqp,
            dk=dk,
            dv=dv,
            num_q_tiles=4,
            num_cascade_stages=4,
            num_heads=num_heads,
        )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner, type_mapper
    from air.backend.xrt import XRTBackend
    from air.extras import types as extrasT
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = VM_ACC_DATATYPE = OUTPUT_DATATYPE = bfloat16

    input_q = np.zeros((num_heads, lq, dk), dtype=INPUT_DATATYPE)
    input_k = np.zeros((num_heads, dk, lk), dtype=INPUT_DATATYPE)
    input_v = np.zeros((num_heads, lk, dv), dtype=INPUT_DATATYPE)
    input_m = np.zeros((num_heads, lq, lk), dtype=INPUT_DATATYPE)

    for h in range(num_heads):
        input_q[h] = (
            np.arange(0, lq * dk, dtype=INPUT_DATATYPE).reshape(lq, dk) / (lq * dk) * 2
        ).astype(INPUT_DATATYPE)
        input_k[h] = (
            np.arange(0, dk * lk, dtype=INPUT_DATATYPE).reshape(dk, lk) / (dk * lk) * 2
        ).astype(INPUT_DATATYPE)
        input_v[h] = (
            np.arange(0, lk * dv, dtype=INPUT_DATATYPE).reshape(lk, dv) / (lk * dv) * 2
        ).astype(INPUT_DATATYPE)

    input_q_scaled = (input_q / sqrt(dk)).astype(INPUT_DATATYPE)

    lazy_attn_output = np.zeros((num_heads, lq, dv), dtype=OUTPUT_DATATYPE)
    for h in range(num_heads):
        A = input_q_scaled[h]
        Gp = np.zeros((lq, dv), dtype=VM_ACC_DATATYPE)
        up = np.full((lq, 1), -np.inf, dtype=VM_ACC_DATATYPE)
        sp = np.zeros((lq, 1), dtype=VM_ACC_DATATYPE)
        for j in range(0, lk // lkp):
            G = input_m[h, :, j * lkp : (j + 1) * lkp]
            B = input_k[h, :, j * lkp : (j + 1) * lkp]
            G = A @ B + G
            G = G.astype(VM_ACC_DATATYPE)
            u = np.max(G, axis=-1, keepdims=True).astype(VM_ACC_DATATYPE)
            u = np.maximum(u, up)
            G = np.exp(G - u)
            G = G.astype(VM_ACC_DATATYPE)
            B = input_v[h, j * lkp : (j + 1) * lkp, :]
            r = np.exp(up - u).astype(VM_ACC_DATATYPE)
            Gp = Gp * r
            Gp = G @ B + Gp
            Gp = Gp.astype(VM_ACC_DATATYPE)
            s = np.sum(G, axis=-1, keepdims=True).astype(VM_ACC_DATATYPE)
            s += sp * r
            sp, up = s, u
        lazy_attn_output[h] = (Gp / sp).astype(OUTPUT_DATATYPE)

    runner = XRTRunner(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=[1, 1],
        output_format="elf",
        instance_name="attention_bf16",
    )

    if args.compile_mode == "run":
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_q_scaled, input_k, input_v, input_m],
                expected_outputs=[lazy_attn_output],
                rtol=1e-1,
            )
        )
    elif args.compile_mode == "compile":
        backend = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            verbose=args.verbose,
            runtime_loop_tiling_sizes=[1, 1],
            output_format="elf",
            instance_name="attention_bf16",
        )
        module_function = backend.compile(mlir_module)
        print(f"Compilation complete. Generated elf binary")
