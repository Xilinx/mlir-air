# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import os
from math import cos, sin, sqrt, exp
import numpy as np

import air
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, CollapseShapeOp, DeallocOp, load, store
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
    num_kv_heads=None,
    causal=False,
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
        num_heads: Number of Q attention heads (default: 12)
        num_kv_heads: Number of K/V heads (default: num_heads for MHA, < num_heads for GQA)
        causal: Enable causal masking (default: False)
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads  # MHA: every Q head has its own KV head

    # Validate divisibility requirements
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert (
        lk % (lkp * num_cascade_stages) == 0
    ), f"lk ({lk}) must be divisible by lkp * num_cascade_stages ({lkp * num_cascade_stages})"
    # Shared buffers: Q and K reuse the same L2 buffer (sized lkp×dk).
    # Only valid when tile_size_q <= lkp so Q tile fits in the K buffer.
    tile_size_q_check = lqp // num_q_tiles
    enable_shared_buffers = lkp == dk and tile_size_q_check <= lkp
    if causal:
        assert lq == lk, f"Causal masking requires lq == lk, got lq={lq}, lk={lk}"
        assert lkp == dk, (
            f"Causal masking requires lkp == dk (enable_shared_buffers) for "
            f"the prefix+suffix BD collapse to produce infinite-loop DMAs "
            f"(no PDI reset between iterations). Got lkp={lkp}, dk={dk}."
        )
        tile_size_q = lqp // num_q_tiles
        assert (
            tile_size_q == lkp
        ), f"Causal masking requires tile_size_q == lkp, got {tile_size_q} vs {lkp}"
    assert (
        num_heads % 2 == 0
    ), f"num_heads ({num_heads}) must be divisible by 2 (segment unroll constraint)"
    assert num_kv_heads > 0, "num_kv_heads must be positive"
    assert (
        num_heads % num_kv_heads == 0
    ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    gqa_group_size = num_heads // num_kv_heads

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
    num_lq_iters = lq // lqp  # Total Q iterations
    # Q iteration at launch level for both causal and non-causal.
    # Keeping Q at launch level avoids DMA task ordering conflicts: when Q
    # iterates on-device, Q and K share the same compute-tile S2MM channel,
    # and getRepeatCounts groups them into sequential tasks [Q×N, K×M]
    # instead of interleaved [Q, K×M, Q, K×M, ...], causing deadlock.
    # For causal masking, the launch Q index is threaded through to the herd
    # body for the block index computation.
    launch_lq_iters = num_lq_iters
    device_lq_iters = 1
    tile_size_q = lqp // num_q_tiles  # Tile size within each lqp chunk

    # Memory spaces: L1 = 2 : i32, L2 = 1 : i32
    l1_space = IntegerAttr.get(i32, 2)  # L1 uses memory space 2
    l2_space = IntegerAttr.get(i32, 1)  # L2 uses memory space 1

    # L1 MemRefTypes (memory space 2 : i32) - used in herd bodies
    memref_lqp_dv_l1 = MemRefType.get([tile_size_q, dk], bf16, memory_space=l1_space)
    memref_lqp_l1 = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)
    memref_lqp_lkp_l1 = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    memref_dv_lkp_l1 = MemRefType.get([lkp, dk], bf16, memory_space=l1_space)
    memref_g_shared_l1 = MemRefType.get([tile_size_q, lkp], bf16, memory_space=l1_space)

    # L2 MemRefTypes (memory space 1 : i32) - segment allocations
    memref_lqp_dk_l2 = MemRefType.get([tile_size_q, dk], bf16, memory_space=l2_space)
    memref_dk_lkp_l2 = MemRefType.get([lkp, dk], bf16, memory_space=l2_space)
    memref_lkp_dv_l2 = MemRefType.get([lkp, dk], bf16, memory_space=l2_space)
    memref_output_lqp_dv_l2 = MemRefType.get(
        [lqp, dk], bf16, memory_space=l2_space
    )  # Per-iteration output buffer

    # L3 MemRefTypes (no memory space annotation = default L3) - with head dimension
    memref_input_q_lq_dk = MemRefType.get([num_heads, lq, dk], bf16)
    memref_output_lq_dv = MemRefType.get([num_heads, lq, dk], bf16)
    memref_input_k_dk_lk = MemRefType.get([num_kv_heads, lk, dk], bf16)
    memref_input_v_lk_dv = MemRefType.get([num_kv_heads, lk, dk], bf16)
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
    external_func("copy_tile", [memref_dv_lkp_l1, memref_lqp_dv_l1], link_with="attn.o")
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
    # Local i32 buffer for passing block indices to apply_causal_mask
    # (IRON pattern: unconditional i32 stores, kernel handles conditionals)
    # counter[0]=q_block, counter[1]=boot_flag, counter[2]=head_iter
    memref_2xi32_l1 = MemRefType.get([3], i32, memory_space=l1_space)
    if causal:
        external_func(
            "apply_causal_mask",
            [memref_lqp_lkp_l1, i32, i32],
            link_with="attn.o",
        )

    # Channel declarations - use num_heads_per_unroll (2) for segment unroll
    Channel("L3ToL2Chan1", size=[num_heads_per_unroll, num_cascade_stages])
    Channel("L3ToL2Chan2", size=[num_heads_per_unroll, num_cascade_stages])
    chan_l2_to_l1_2 = Channel(
        "L2ToL1Chan2",
        size=[1, num_cascade_stages],
        broadcast_shape=[num_q_tiles, num_cascade_stages],
    )
    chan_l2_to_l1_2.attributes["channel_type"] = StringAttr.get("dma_packet")
    if not enable_shared_buffers:
        chan_l2_to_l1_1 = Channel(
            "L2ToL1Chan1",
            size=[num_q_tiles, 1],
            broadcast_shape=[num_q_tiles, num_cascade_stages],
        )
        chan_l2_to_l1_1.attributes["channel_type"] = StringAttr.get("dma_packet")
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
        c_launch_lq = ConstantOp(index_type, launch_lq_iters)
        c_num_head_groups = ConstantOp(index_type, num_head_groups)

        # Non-causal: launch iterates Q blocks at host level (no BD chain limit)
        # Causal: launch size 1, Q iteration inside herd (device-local q_block)
        @launch(
            operands=[arg0, arg1, arg2, arg4], sizes=[c_launch_lq, c_num_head_groups]
        )
        def launch_body(arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12):
            # arg5 = Q iteration index (0..launch_lq_iters-1), arg6 = head group
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

            # GQA: compute KV head indices from Q head indices
            # kv_head = q_head // gqa_group_size
            if gqa_group_size == 1:
                # MHA: kv_head == q_head
                kv_head_base = head_base
                kv_head_1 = head_1
            else:
                affine_map_kv_head = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_floor_div(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(gqa_group_size),
                        )
                    ],
                )
                kv_head_base = affine_apply(affine_map_kv_head, [head_base])
                kv_head_1 = affine_apply(affine_map_kv_head, [head_1])

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
            # Affine map for K head offset: head * lk * dk + row_offset * dk
            # K stored as [num_kv_heads, lk, dk] (row-major, matching IRON)
            affine_map_head_row = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_add(
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(lk * dk)
                        ),
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(1), AffineConstantExpr.get(dk)
                        ),
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

            # Combined Q/K/V/output DMA loop — one iteration per q_iter
            # Must be a single loop so Q, K, V, and output are interleaved in
            # the correct order matching the segment's consumption pattern.
            c_device_lq_iters = ConstantOp(index_type, device_lq_iters)
            for lq_it in range_(c0, c_device_lq_iters, c1):
                # Combine launch Q index (arg5) + device Q index (lq_it)
                # Non-causal: arg5 varies, lq_it=0. Causal: arg5=0, lq_it varies.
                q_iter_global = arith.AddIOp(arg5, lq_it)

                # (A) Q: L3→L2 for this q_iter
                par_1 = scf.ForallOp(
                    lower_bounds=[0], upper_bounds=[num_cascade_stages], steps=[1]
                )
                with InsertionPoint(par_1.body):
                    tile_offset = affine_apply(
                        affine_map_tileq, [par_1.induction_variables[0]]
                    )
                    launch_offset = affine_apply(
                        affine_map_launch_offset, [q_iter_global.result]
                    )
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

                # (B) K: L3→L2 for this q_iter (same K data re-sent each iter)
                for i in range(num_cascade_stages):
                    row_off = ConstantOp(index_type, i * chunks_per_stage * lkp)
                    k_head0_off = affine_apply(
                        affine_map_head_row, [kv_head_base, row_off]
                    )
                    ChannelPut(
                        "L3ToL2Chan1",
                        arg10,
                        indices=[c0, i],
                        offsets=[0, 0, k_head0_off],
                        sizes=[chunks_per_stage, lkp, dk],
                        strides=[lkp * dk, dk, 1],
                    )
                    k_head1_off = affine_apply(
                        affine_map_head_row, [kv_head_1, row_off]
                    )
                    ChannelPut(
                        "L3ToL2Chan1",
                        arg10,
                        indices=[c1, i],
                        offsets=[0, 0, k_head1_off],
                        sizes=[chunks_per_stage, lkp, dk],
                        strides=[lkp * dk, dk, 1],
                    )

                # (C) V: L3→L2 for this q_iter (same V data re-sent each iter)
                for i in range(num_cascade_stages):
                    v_head0_off = affine_apply(affine_map_v_head_offset, [kv_head_base])
                    ChannelPut(
                        "L3ToL2Chan2",
                        arg11,
                        indices=[c0, i],
                        offsets=[0, i * chunks_per_stage * lkp, v_head0_off],
                        sizes=[chunks_per_stage, lkp, dv],
                        strides=[lkp * dv, dv, 1],
                    )
                    v_head1_off = affine_apply(affine_map_v_head_offset, [kv_head_1])
                    ChannelPut(
                        "L3ToL2Chan2",
                        arg11,
                        indices=[c1, i],
                        offsets=[0, i * chunks_per_stage * lkp, v_head1_off],
                        sizes=[chunks_per_stage, lkp, dv],
                        strides=[lkp * dv, dv, 1],
                    )

                # (D) Output: L2→L3 for this q_iter
                launch_offset_out = affine_apply(
                    affine_map_launch_offset, [q_iter_global.result]
                )
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

                yield_([])

            # Segment unrolls over 2 heads (hardware constraint)
            c_num_heads_unroll = ConstantOp(index_type, num_heads_per_unroll)
            c_dummy_size = ConstantOp(index_type, 1)

            seg_operands = []

            @segment(
                name="attention_seg",
                operands=seg_operands,
                sizes=[c_num_heads_unroll, c_dummy_size],
            )
            def segment_body(*seg_args):
                head_idx, dummy_idx, head_size, dummy_size = seg_args[:4]
                # L2 allocations
                if enable_shared_buffers:
                    alloc = alloc_col1 = alloc_col2 = alloc_col3 = None
                else:
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
                if enable_shared_buffers:
                    G_shared = AllocOp(memref_g_shared_l1, [], [])
                    QK_shared = AllocOp(memref_dv_lkp_l1, [], [])
                else:
                    G_shared = None
                    QK_shared = None
                # Local counter for causal block index tracking (IRON pattern).
                # Passed as memref operand (NOT scalar) → no RTP, no herd lock.
                causal_counter = AllocOp(memref_2xi32_l1, [], []) if causal else None

                c_num_q_tiles = ConstantOp(index_type, num_q_tiles)
                c_num_cascade = ConstantOp(index_type, num_cascade_stages)
                c0_seg = ConstantOp(index_type, 0)
                c1_seg = ConstantOp(index_type, 1)
                c2_seg = ConstantOp(index_type, 2)
                c3_seg = ConstantOp(index_type, 3)

                # Q/K/V/output DMA loop over lq_iters (Q iteration moved from launch to device)
                q_l2_bufs = (
                    [alloc_2, alloc_21, alloc_22, alloc_23]
                    if enable_shared_buffers
                    else [alloc, alloc_col1, alloc_col2, alloc_col3]
                )
                q_chan = "L2ToL1Chan2" if enable_shared_buffers else "L2ToL1Chan1"
                q_idx = lambda col: (
                    [c0_seg, col] if enable_shared_buffers else [col, c0_seg]
                )

                c_device_lq_seg = ConstantOp(index_type, device_lq_iters)
                for lq_it_seg in range_(c0_seg, c_device_lq_seg, c1_seg):
                    # (A) Q: L3→L2 gets for this q_iter's 4 tiles
                    ChannelGet(
                        "L3ToL2Chan1", q_l2_bufs[0].result, indices=[head_idx, c0_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan1", q_l2_bufs[1].result, indices=[head_idx, c1_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan1", q_l2_bufs[2].result, indices=[head_idx, c2_seg]
                    )
                    ChannelGet(
                        "L3ToL2Chan1", q_l2_bufs[3].result, indices=[head_idx, c3_seg]
                    )

                    # (B) Q: L2→L1 puts for this q_iter's 4 tiles
                    ChannelPut(
                        q_chan,
                        q_l2_bufs[0].result,
                        indices=q_idx(c0_seg),
                        offsets=[0, 0, 0, 0],
                        sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                        strides=[mmul_k, dk * mmul_k, dk, 1],
                    )
                    ChannelPut(
                        q_chan,
                        q_l2_bufs[1].result,
                        indices=q_idx(c1_seg),
                        offsets=[0, 0, 0, 0],
                        sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                        strides=[mmul_k, dk * mmul_k, dk, 1],
                    )
                    ChannelPut(
                        q_chan,
                        q_l2_bufs[2].result,
                        indices=q_idx(c2_seg),
                        offsets=[0, 0, 0, 0],
                        sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                        strides=[mmul_k, dk * mmul_k, dk, 1],
                    )
                    ChannelPut(
                        q_chan,
                        q_l2_bufs[3].result,
                        indices=q_idx(c3_seg),
                        offsets=[0, 0, 0, 0],
                        sizes=[dk // mmul_k, tile_size_q // mmul_m, mmul_m, mmul_k],
                        strides=[mmul_k, dk * mmul_k, dk, 1],
                    )

                    # (C) K/V streaming: L3→L2 + L2→L1 (inner loop)
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
                            sizes=[lkp // mmul_n, dk // mmul_k, mmul_n, mmul_k],
                            strides=[mmul_n * dk, mmul_k, dk, 1],
                        )
                        ChannelPut(
                            "L2ToL1Chan2",
                            alloc_21.result,
                            indices=[c0_seg, c1_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[lkp // mmul_n, dk // mmul_k, mmul_n, mmul_k],
                            strides=[mmul_n * dk, mmul_k, dk, 1],
                        )
                        ChannelPut(
                            "L2ToL1Chan2",
                            alloc_22.result,
                            indices=[c0_seg, c2_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[lkp // mmul_n, dk // mmul_k, mmul_n, mmul_k],
                            strides=[mmul_n * dk, mmul_k, dk, 1],
                        )
                        ChannelPut(
                            "L2ToL1Chan2",
                            alloc_23.result,
                            indices=[c0_seg, c3_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[lkp // mmul_n, dk // mmul_k, mmul_n, mmul_k],
                            strides=[mmul_n * dk, mmul_k, dk, 1],
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

                        yield_([])

                    # (D) Output: L1→L2 gather for this q_iter
                    affine_map_tileq_seg = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(tile_size_q),
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

                    # (E) Output: L2→L3 transfer for this q_iter
                    ChannelPut("L2ToL3Chan1", alloc_5.result, indices=[head_idx])

                    yield_([])

                # Unified herd: init + compute loop + cascade merge + output
                unified_operands = (
                    [alloc_6, up, sp, Gp, G_shared, QK_shared]
                    if enable_shared_buffers
                    else [alloc_6, up, sp, Gp]
                )
                # Causal: pass counter as memref operand (no RTP/lock)
                if causal:
                    unified_operands = unified_operands + [causal_counter]

                @herd(
                    name="herd_0",
                    sizes=[c_num_q_tiles, c_num_cascade],
                    operands=unified_operands,
                    link_with="attn.o",
                )
                def unified_herd_body(*args):
                    arg22, arg23, arg24, arg25 = args[0], args[1], args[2], args[3]
                    if enable_shared_buffers:
                        arg26, arg27, arg28, arg29, arg30, arg31 = args[4:10]
                        counter_buf = args[10] if causal else None
                    else:
                        arg26, arg27, arg28, arg29 = args[4:8]
                        arg30 = arg31 = None
                        counter_buf = args[8] if causal else None

                    if causal:
                        # IRON-style local counter. With lkp==dk (shared
                        # buffers), DMAs are infinite loops → no PDI reset
                        # → core loops continuously → counter persists.
                        # counter[0] = q_block_global
                        # counter[1] = boot flag (CDO initializes to 0)
                        # counter[2] = head iteration counter
                        c0_ctr = ConstantOp(index_type, 0)
                        c1_ctr = ConstantOp(index_type, 1)
                        boot_flag = load(counter_buf, [c1_ctr])
                        c0_i32_ctr = ConstantOp(i32, 0)
                        is_first = arith.CmpIOp(
                            arith.CmpIPredicate.eq, boot_flag, c0_i32_ctr
                        )
                        if_first = scf.IfOp(is_first)
                        with InsertionPoint(if_first.then_block):
                            q_init = arith.IndexCastOp(i32, arg22)
                            store(q_init, counter_buf, [c0_ctr])
                            c1_i32_f = ConstantOp(i32, 1)
                            store(c1_i32_f, counter_buf, [c1_ctr])
                            scf.YieldOp([])

                    # === OUTER Q ITERATION LOOP (device-side) ===
                    c_lq_iters_herd = ConstantOp(index_type, device_lq_iters)
                    c0_q = ConstantOp(index_type, 0)
                    c1_q = ConstantOp(index_type, 1)

                    for q_iter in range_(c0_q, c_lq_iters_herd, c1_q):

                        # === INIT PHASE ===
                        if enable_shared_buffers:
                            ChannelGet("L2ToL1Chan2", arg31, indices=[arg22, arg23])
                            CallOp([], "copy_tile", [arg31, arg26])
                        else:
                            ChannelGet("L2ToL1Chan1", arg26, indices=[arg22, arg23])
                        CallOp([], "zero_fill_gp_bf16", [arg29])
                        CallOp([], "zero_fill_sp_bf16", [arg28])
                        CallOp([], "neg_inf_fill_up_bf16", [arg27])

                        # === COMPUTE LOOP (on-device) ===
                        c_chunks = ConstantOp(index_type, chunks_per_stage)
                        c0_loop = ConstantOp(index_type, 0)
                        c1_loop = ConstantOp(index_type, 1)

                        for chunk_idx in range_(c0_loop, c_chunks, c1_loop):
                            if enable_shared_buffers:
                                G_l1 = CollapseShapeOp(
                                    memref_lqp_lkp_l1, arg30, [[0, 1]]
                                )
                            else:
                                G_alloc = AllocOp(memref_g_shared_l1, [], [])
                                G_l1 = CollapseShapeOp(
                                    memref_lqp_lkp_l1, G_alloc.result, [[0, 1]]
                                )

                            CallOp([], "zero_fill_g_bf16", [G_l1])

                            if enable_shared_buffers:
                                ChannelGet("L2ToL1Chan2", arg31, indices=[arg22, arg23])
                                CallOp([], "matmul_a_b_bf16", [arg26, arg31, G_l1])
                            else:
                                QK_alloc = AllocOp(memref_dv_lkp_l1, [], [])
                                ChannelGet(
                                    "L2ToL1Chan2",
                                    QK_alloc.result,
                                    indices=[arg22, arg23],
                                )
                                CallOp(
                                    [],
                                    "matmul_a_b_bf16",
                                    [arg26, QK_alloc.result, G_l1],
                                )

                            alloc_57 = AllocOp(memref_dv_lkp_l1, [], [])
                            ChannelGet(
                                "L2ToL1Chan3", alloc_57.result, indices=[arg22, arg23]
                            )

                            if causal:
                                # Local counter gives q_block_global.
                                # No RTP/herd lock — counter loaded from
                                # local L1 buffer (IRON pattern).
                                c_cps = ConstantOp(index_type, chunks_per_stage)
                                kv_block = arith.AddIOp(
                                    arith.MulIOp(arg23, c_cps).result, chunk_idx
                                )
                                kv_i32 = arith.IndexCastOp(i32, kv_block.result)
                                c0_ctr_use = ConstantOp(index_type, 0)
                                q_i32 = load(counter_buf, [c0_ctr_use])
                                CallOp([], "apply_causal_mask", [G_l1, q_i32, kv_i32])

                                c0_i32 = ConstantOp(i32, 0)
                                u_l1 = AllocOp(memref_lqp_l1, [], [])
                                s_l1 = AllocOp(memref_lqp_l1, [], [])
                                r_l1 = AllocOp(memref_lqp_l1, [], [])

                                CallOp([], "max_g_bf16", [G_l1, u_l1.result])
                                CallOp([], "maximum_up_u_bf16", [arg27, u_l1.result])
                                CallOp([], "exp_g_minus_u", [u_l1.result, G_l1])
                                CallOp(
                                    [],
                                    "exp_up_minus_u",
                                    [arg27, u_l1.result, r_l1.result],
                                )
                                CallOp([], "mul_r_gp", [r_l1.result, arg29])
                                CallOp(
                                    [],
                                    "matmul_g_b_bf16",
                                    [G_l1, alloc_57.result, arg29],
                                )
                                CallOp([], "sum_g", [G_l1, s_l1.result])
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [arg28, r_l1.result, s_l1.result],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32, s_l1.result, arg28],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32, u_l1.result, arg27],
                                )

                                DeallocOp(u_l1)
                                DeallocOp(s_l1)
                                DeallocOp(r_l1)
                            else:
                                c0_i32 = ConstantOp(i32, 0)
                                u_l1 = AllocOp(memref_lqp_l1, [], [])
                                s_l1 = AllocOp(memref_lqp_l1, [], [])
                                r_l1 = AllocOp(memref_lqp_l1, [], [])

                                CallOp([], "max_g_bf16", [G_l1, u_l1.result])
                                CallOp([], "maximum_up_u_bf16", [arg27, u_l1.result])
                                CallOp([], "exp_g_minus_u", [u_l1.result, G_l1])
                                CallOp(
                                    [],
                                    "exp_up_minus_u",
                                    [arg27, u_l1.result, r_l1.result],
                                )
                                CallOp([], "mul_r_gp", [r_l1.result, arg29])
                                CallOp(
                                    [],
                                    "matmul_g_b_bf16",
                                    [G_l1, alloc_57.result, arg29],
                                )
                                CallOp([], "sum_g", [G_l1, s_l1.result])
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [arg28, r_l1.result, s_l1.result],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32, s_l1.result, arg28],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32, u_l1.result, arg27],
                                )

                                DeallocOp(u_l1)
                                DeallocOp(s_l1)
                                DeallocOp(r_l1)

                            DeallocOp(alloc_57)

                            if not enable_shared_buffers:
                                DeallocOp(QK_alloc)
                                DeallocOp(G_alloc)
                            yield_([])

                        # === CASCADE MERGE ===
                        c1_h = ConstantOp(index_type, 1)
                        r_l1_c = AllocOp(memref_lqp_l1, [], [])

                        def get_gp_cascade():
                            if enable_shared_buffers:
                                return arg30
                            else:
                                return AllocOp(memref_lqp_dv_l1, [], []).result

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
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(-1),
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
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(-1),
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
                                Gp_cascade = get_gp_cascade()
                                up_cascade = AllocOp(memref_lqp_l1, [], [])
                                sp_cascade = AllocOp(memref_lqp_l1, [], [])
                                ChannelGet(
                                    "cascade", Gp_cascade, indices=[arg22, arg23]
                                )
                                ChannelGet(
                                    "cascade", up_cascade.result, indices=[arg22, arg23]
                                )
                                ChannelGet(
                                    "cascade", sp_cascade.result, indices=[arg22, arg23]
                                )
                                up_B_saved = AllocOp(memref_lqp_l1, [], [])
                                c0_i32_m = ConstantOp(i32, 0)
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32_m, arg27, up_B_saved.result],
                                )
                                CallOp(
                                    [], "maximum_up_u_bf16", [up_cascade.result, arg27]
                                )
                                CallOp(
                                    [],
                                    "exp_up_minus_u",
                                    [up_cascade.result, arg27, r_l1_c.result],
                                )
                                r_B = AllocOp(memref_lqp_l1, [], [])
                                CallOp(
                                    [],
                                    "exp_up_minus_u",
                                    [up_B_saved.result, arg27, r_B.result],
                                )
                                CallOp([], "mul_r_gp", [r_l1_c.result, Gp_cascade])
                                CallOp([], "mul_r_gp", [r_B.result, arg29])
                                CallOp([], "add_gp_g", [arg29, Gp_cascade])
                                sp_temp = AllocOp(memref_lqp_l1, [], [])
                                CallOp([], "zero_fill_sp_bf16", [sp_temp.result])
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [sp_cascade.result, r_l1_c.result, sp_temp.result],
                                )
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [arg28, r_B.result, sp_temp.result],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32_m, sp_temp.result, sp_cascade.result],
                                )
                                subi2 = arith.SubIOp(arg23, c1_h)
                                ChannelPut(
                                    "cascade", Gp_cascade, indices=[arg22, subi2]
                                )
                                ChannelPut("cascade", arg27, indices=[arg22, subi2])
                                ChannelPut(
                                    "cascade", sp_cascade.result, indices=[arg22, subi2]
                                )
                                DeallocOp(up_B_saved)
                                DeallocOp(r_B)
                                DeallocOp(sp_temp)
                                affine.AffineYieldOp([])

                            with InsertionPoint(affine_if_middle.else_block):
                                Gp_cascade2 = get_gp_cascade()
                                up_cascade2 = AllocOp(memref_lqp_l1, [], [])
                                sp_cascade2 = AllocOp(memref_lqp_l1, [], [])
                                ChannelGet(
                                    "cascade", Gp_cascade2, indices=[arg22, arg23]
                                )
                                ChannelGet(
                                    "cascade",
                                    up_cascade2.result,
                                    indices=[arg22, arg23],
                                )
                                ChannelGet(
                                    "cascade",
                                    sp_cascade2.result,
                                    indices=[arg22, arg23],
                                )
                                up_B_saved2 = AllocOp(memref_lqp_l1, [], [])
                                c0_i32_f = ConstantOp(i32, 0)
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32_f, arg27, up_B_saved2.result],
                                )
                                CallOp(
                                    [], "maximum_up_u_bf16", [up_cascade2.result, arg27]
                                )
                                CallOp(
                                    [],
                                    "exp_up_minus_u",
                                    [up_cascade2.result, arg27, r_l1_c.result],
                                )
                                r_B2 = AllocOp(memref_lqp_l1, [], [])
                                CallOp(
                                    [],
                                    "exp_up_minus_u",
                                    [up_B_saved2.result, arg27, r_B2.result],
                                )
                                CallOp([], "mul_r_gp", [r_l1_c.result, Gp_cascade2])
                                CallOp([], "mul_r_gp", [r_B2.result, arg29])
                                CallOp([], "add_gp_g", [arg29, Gp_cascade2])
                                sp_temp2 = AllocOp(memref_lqp_l1, [], [])
                                CallOp([], "zero_fill_sp_bf16", [sp_temp2.result])
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [
                                        sp_cascade2.result,
                                        r_l1_c.result,
                                        sp_temp2.result,
                                    ],
                                )
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [arg28, r_B2.result, sp_temp2.result],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [c0_i32_f, sp_temp2.result, sp_cascade2.result],
                                )
                                CallOp(
                                    [], "div_gp_sp", [sp_cascade2.result, Gp_cascade2]
                                )
                                DeallocOp(up_B_saved2)
                                DeallocOp(r_B2)
                                DeallocOp(sp_temp2)
                                ChannelPut(
                                    "L1ToL2Chan1",
                                    Gp_cascade2,
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

                        # Increment counters. The launch is 2D
                        # [launch_lq_iters, num_head_groups], so the
                        # while-true loop runs once per (Q_iter, head_group).
                        # Only advance q_block after all head groups are done.
                        if causal:
                            c0_ci = ConstantOp(index_type, 0)
                            c2_ci = ConstantOp(index_type, 2)
                            c1_i32_ci = ConstantOp(i32, 1)
                            # Increment head counter
                            head_cur = load(counter_buf, [c2_ci])
                            head_next = arith.AddIOp(head_cur, c1_i32_ci)
                            total_heads_i32 = ConstantOp(i32, num_head_groups)
                            wrapped = arith.CmpIOp(
                                arith.CmpIPredicate.sge,
                                head_next,
                                total_heads_i32,
                            )
                            if_wrap = scf.IfOp(wrapped)
                            with InsertionPoint(if_wrap.then_block):
                                # All heads done: increment q_block, reset head
                                q_cur = load(counter_buf, [c0_ci])
                                c_nqt_i32 = ConstantOp(i32, num_q_tiles)
                                q_next = arith.AddIOp(q_cur, c_nqt_i32)
                                store(q_next, counter_buf, [c0_ci])
                                c0_i32_ci = ConstantOp(i32, 0)
                                store(c0_i32_ci, counter_buf, [c2_ci])
                                scf.YieldOp([])
                            if_wrap_else = scf.IfOp(
                                arith.CmpIOp(
                                    arith.CmpIPredicate.slt,
                                    head_next,
                                    total_heads_i32,
                                )
                            )
                            with InsertionPoint(if_wrap_else.then_block):
                                store(head_next, counter_buf, [c2_ci])
                                scf.YieldOp([])

                        yield_([])  # end of q_iter loop

            # Output channel gets are inside the combined Q/K/V/output loop above


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
        "--num-heads", type=int, default=12, help="Number of Q attention heads"
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of K/V heads (default: num_heads for MHA, set < num_heads for GQA)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="run",
        choices=["run", "compile"],
        help="Compilation mode: run (default, compile + test), compile (generate binary only)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal masking (autoregressive attention)",
    )
    args = parser.parse_args()

    lk, lkp, lq, lqp, dk, dv = args.lk, args.lkp, args.lq, args.lqp, args.dk, args.dv
    causal = args.causal
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads

    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

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
            num_kv_heads=num_kv_heads,
            causal=causal,
        )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner, type_mapper
    from air.backend.xrt import XRTBackend
    from air.extras import types as extrasT
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16
    VM_ACC_DATATYPE = np.float32

    gqa_group_size = num_heads // num_kv_heads

    rng = np.random.default_rng(42)
    # Use small positive values to stay within BFP16 matmul precision range
    input_q = (rng.uniform(0, 1, (num_heads, lq, dk)) * 0.5 + 0.5).astype(
        INPUT_DATATYPE
    )
    input_k = (rng.uniform(0, 1, (num_kv_heads, lk, dk)) * 0.5 + 0.5).astype(
        INPUT_DATATYPE
    )
    input_v = (rng.uniform(0, 1, (num_kv_heads, lk, dv)) * 0.5 + 0.5).astype(
        INPUT_DATATYPE
    )
    input_m = np.zeros((num_heads, lq, lk), dtype=INPUT_DATATYPE)

    input_q_scaled = (input_q / sqrt(dk)).astype(INPUT_DATATYPE)

    num_cascade_stages_ref = 4
    num_chunks_ref = lk // lkp
    chunks_per_stage_ref = num_chunks_ref // num_cascade_stages_ref

    # bf16 lowest (0xff7f) ≈ -3.39e38, used instead of -inf to avoid NaN on AIE2P
    bf16_lowest = np.float32(
        np.frombuffer(np.array([0xFF7F], dtype=np.uint16).tobytes(), dtype=bfloat16)[0]
    )

    def flash_attn_per_stage(A, kv_h, stage, mask_h):
        """Run flash attention on contiguous K chunks for one cascade stage."""
        Gp = np.zeros((lq, dv), dtype=VM_ACC_DATATYPE)
        up = np.full((lq, 1), bf16_lowest, dtype=VM_ACC_DATATYPE)
        sp = np.zeros((lq, 1), dtype=VM_ACC_DATATYPE)
        for ci in range(chunks_per_stage_ref):
            j = stage * chunks_per_stage_ref + ci
            G = mask_h[:, j * lkp : (j + 1) * lkp]
            B = input_k[kv_h, j * lkp : (j + 1) * lkp, :].T
            G = A @ B + G
            if causal:
                # Apply causal mask: mask positions where kv_col > q_row
                # Pre-fill mask: bf16_lowest for masked positions.
                # The matmul adds scores, producing bf16_lowest+score≈bf16_lowest.
                kv_cols = np.arange(j * lkp, (j + 1) * lkp)
                q_rows = np.arange(lq)[:, np.newaxis]
                G = np.where(kv_cols > q_rows, bf16_lowest, G)
            G = G.astype(VM_ACC_DATATYPE)
            u = np.max(G, axis=-1, keepdims=True).astype(VM_ACC_DATATYPE)
            # Clamp u to bf16_lowest (matching hardware max_g_bf16 which uses
            # bf16 lowest as initial value, so fully-masked rows get bf16_lowest)
            u = np.maximum(u, bf16_lowest)
            u = np.maximum(u, up)
            G = np.exp(G - u)
            G = G.astype(VM_ACC_DATATYPE)
            B = input_v[kv_h, j * lkp : (j + 1) * lkp, :]
            r = np.exp(up - u).astype(VM_ACC_DATATYPE)
            Gp = Gp * r
            Gp = G @ B + Gp
            Gp = Gp.astype(VM_ACC_DATATYPE)
            s = np.sum(G, axis=-1, keepdims=True).astype(VM_ACC_DATATYPE)
            s += sp * r
            sp, up = s, u
        return Gp, up, sp

    def cascade_merge(Gp_A, up_A, sp_A, Gp_B, up_B, sp_B):
        """Merge two partial flash attention results (corrected algorithm)."""
        new_max = np.maximum(up_A, up_B)
        r_A = np.exp(up_A - new_max).astype(VM_ACC_DATATYPE)
        r_B = np.exp(up_B - new_max).astype(VM_ACC_DATATYPE)
        Gp_merged = (Gp_A * r_A + Gp_B * r_B).astype(VM_ACC_DATATYPE)
        sp_merged = (sp_A * r_A + sp_B * r_B).astype(VM_ACC_DATATYPE)
        return Gp_merged, new_max, sp_merged

    lazy_attn_output = np.zeros((num_heads, lq, dv), dtype=OUTPUT_DATATYPE)
    for h in range(num_heads):
        kv_h = h // gqa_group_size
        A = input_q_scaled[h]
        stage_results = []
        for stage in range(num_cascade_stages_ref):
            Gp_s, up_s, sp_s = flash_attn_per_stage(A, kv_h, stage, input_m[h])
            stage_results.append((Gp_s, up_s, sp_s))
        # Cascade merge: stage 3 -> 2 -> 1 -> 0
        Gp_acc, up_acc, sp_acc = stage_results[num_cascade_stages_ref - 1]
        for stage in range(num_cascade_stages_ref - 2, -1, -1):
            Gp_local, up_local, sp_local = stage_results[stage]
            Gp_acc, up_acc, sp_acc = cascade_merge(
                Gp_acc, up_acc, sp_acc, Gp_local, up_local, sp_local
            )
        if os.environ.get("SKIP_DIV"):
            lazy_attn_output[h] = Gp_acc.astype(OUTPUT_DATATYPE)
        else:
            lazy_attn_output[h] = (Gp_acc / sp_acc).astype(OUTPUT_DATATYPE)

    enable_shared_buffers_main = lkp == dk and lqp // 4 <= lkp
    # Causal requires enable_shared_buffers (lkp==dk), which already sets
    # omit_while_true_loop=False. The while-true loop lets the core maintain
    # a local counter across launch iterations without PDI reset.
    omit_loop = not enable_shared_buffers_main
    runner = XRTRunner(
        omit_while_true_loop=omit_loop,
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
                atol=0.5,
                rtol=0.2,
            )
        )
    elif args.compile_mode == "compile":
        backend = XRTBackend(
            omit_while_true_loop=omit_loop,
            omit_pingpong="all",
            verbose=args.verbose,
            runtime_loop_tiling_sizes=[1, 1],
            output_format="elf",
            instance_name="attention_bf16",
        )
        module_function = backend.compile(mlir_module)
        print(f"Compilation complete. Generated elf binary")
