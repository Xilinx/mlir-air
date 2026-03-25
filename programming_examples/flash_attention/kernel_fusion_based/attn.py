# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention with memtile-relayed dataflow — selective Q capture.

All data (Q, K, V) routes through memtile for L3→L2→L1 transfer.
Per-stage QKIn/QK2L1 and VIn/V2L1 channels handle the relay.
Q tiles are selectively captured: each tile receives all NQ Q sends
but only copies the one matching its tx. Cascade merge follows the
cascade-after pattern.

Multi-head support via 3D channels with segment unroll:
  - num_heads_per_unroll=2 heads are processed per segment unroll
  - Segment sizes=[num_heads_per_unroll, 1], each segment instance handles
    one head index
  - 3D channels have head dimension as first index
  - Cascade channels remain 2D (shared within each segment instance)

Design parameters:
  lk=512, lkp=64, lq=512, lqp=256, dk=64, dv=64
  num_q_tiles=4, num_cascade_stages=4, num_heads=2
  Non-causal only.
  Shared-buffer mode (lkp == dk).

DMA channel strategy (2 S2MM + 2 MM2S per compute tile):
  S2MM 0: QK channel (Q selective capture, then K chunks)
  S2MM 1: V (per-stage via memtile)
  MM2S 0: Cascade or output
  MM2S 1: Cascade

Channel layout:
  QKIn_s/QK2L1_s: per-stage memtile relay with horizontal broadcast
  VIn_s/V2L1_s: per-stage memtile relay with horizontal broadcast
  cascade_gp/cascade_up/cascade_sp: 2D cascade channels (per-segment)
  Gp2L2/GpOut: output from ty=0 tiles
"""

import argparse
from math import sqrt

import numpy as np

import air
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.air import channel
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, CollapseShapeOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_ as scf_range, yield_
from air.dialects import scf, affine, arith


@module_builder
def build_module(
    lk=512,
    lkp=64,
    lq=512,
    lqp=256,
    dk=64,
    dv=64,
    num_q_tiles=4,
    num_cascade_stages=4,
    num_heads=2,
    num_kv_heads=None,
    causal=False,
):
    """Build flash attention module with selective Q capture pattern.

    Args:
        lk: Total K/V sequence length (default: 512)
        lkp: K/V chunk size per tile (default: 64)
        lq: Total Q sequence length (default: 512)
        lqp: Q chunk size per launch iteration (default: 256)
        dk: Key dimension (default: 64)
        dv: Value dimension (default: 64)
        num_q_tiles: Number of tiles to partition Q chunk into (default: 4)
        num_cascade_stages: Number of cascade pipeline stages (default: 4)
        num_heads: Number of attention heads (default: 2)
    """
    # Validate
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert lqp % num_q_tiles == 0, (
        f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    )
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert lk % (lkp * num_cascade_stages) == 0, (
        f"lk ({lk}) must be divisible by lkp * num_cascade_stages "
        f"({lkp * num_cascade_stages})"
    )
    assert lkp == dk, "L3-to-L1 mode requires lkp == dk (shared buffers)"
    if causal:
        assert lq == lk, f"Causal masking requires lq == lk, got lq={lq}, lk={lk}"
        assert lqp // num_q_tiles == lkp, (
            f"Causal masking requires tile_size_q == lkp, got "
            f"tile_size_q={lqp // num_q_tiles}, lkp={lkp}"
        )

    # Multi-head / GQA parameters
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_kv_heads > 0, f"num_kv_heads must be positive, got {num_kv_heads}"
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by "
        f"num_kv_heads ({num_kv_heads})"
    )
    gqa_group_size = num_heads // num_kv_heads

    num_heads_per_unroll = 2
    assert num_heads % num_heads_per_unroll == 0, (
        f"num_heads ({num_heads}) must be divisible by "
        f"num_heads_per_unroll ({num_heads_per_unroll})"
    )
    num_head_groups = num_heads // num_heads_per_unroll

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    M = 8  # mmul_m = mmul_k = mmul_n

    # Derived parameters
    num_lq_iters = lq // lqp
    tile_size_q = lqp // num_q_tiles
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // num_cascade_stages
    lk_per_stage = lkp * chunks_per_stage

    NQ = num_q_tiles
    NS = num_cascade_stages

    # Memory spaces
    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # L1 MemRefTypes
    q_l1_t = MemRefType.get([tile_size_q, dk], bf16, memory_space=l1_space)
    k_l1_t = MemRefType.get([lkp, dk], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv], bf16, memory_space=l1_space)
    g_l1_2d = MemRefType.get([tile_size_q, lkp], bf16, memory_space=l1_space)
    g_l1_1d = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    gp_l1_t = MemRefType.get([tile_size_q, dv], bf16, memory_space=l1_space)
    up_l1_t = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)

    # L2 MemRefTypes
    qk_l2_t = MemRefType.get([lkp, dk], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, dv], bf16, memory_space=l2_space)
    gp_l2_t = MemRefType.get([lqp, dv], bf16, memory_space=l2_space)

    # L3 MemRefTypes (3D with head dimension)
    q_l3_t = MemRefType.get([num_heads, lq, dk], bf16)
    k_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    v_l3_t = MemRefType.get([num_kv_heads, lk, dv], bf16)
    gp_l3_t = MemRefType.get([num_heads, lq, dv], bf16)

    # External function declarations
    def external_func(name, inputs, outputs=None, link_with=None,
                      visibility="private"):
        if outputs is None:
            outputs = []
        func_type = FunctionType.get(inputs, outputs)
        func = FuncOp(name=name, type=func_type, visibility=visibility)
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    external_func("zero_fill_g_bf16", [g_l1_1d], link_with="attn.o")
    external_func("zero_fill_gp_bf16", [gp_l1_t], link_with="attn.o")
    external_func("zero_fill_sp_bf16", [up_l1_t], link_with="attn.o")
    external_func("neg_inf_fill_up_bf16", [up_l1_t], link_with="attn.o")
    external_func(
        "matmul_a_b_bf16",
        [q_l1_t, k_l1_t, g_l1_1d],
        link_with="attn.o",
    )
    external_func(
        "matmul_g_b_bf16",
        [g_l1_1d, v_l1_t, gp_l1_t],
        link_with="attn.o",
    )
    external_func(
        "fused_softmax",
        [g_l1_1d, up_l1_t, up_l1_t, up_l1_t],
        link_with="attn.o",
    )
    external_func(
        "maximum_up_u_bf16", [up_l1_t, up_l1_t], link_with="attn.o"
    )
    external_func(
        "exp_up_minus_u",
        [up_l1_t, up_l1_t, up_l1_t],
        link_with="attn.o",
    )
    external_func("mul_r_gp", [up_l1_t, gp_l1_t], link_with="attn.o")
    external_func(
        "accum_sp_r_s",
        [up_l1_t, up_l1_t, up_l1_t],
        link_with="attn.o",
    )
    external_func(
        "vector_copy_32elems", [i32, up_l1_t, up_l1_t], link_with="attn.o"
    )
    external_func("copy_tile", [k_l1_t, q_l1_t], link_with="attn.o")
    external_func("div_gp_sp", [up_l1_t, gp_l1_t], link_with="attn.o")
    external_func("add_gp_g", [gp_l1_t, gp_l1_t], link_with="attn.o")
    if causal:
        external_func(
            "apply_causal_mask", [g_l1_2d, i32, i32], link_with="attn.o"
        )

    # ----------------------------------------------------------------
    # Channel declarations (3D with head dimension for multi-head)
    # ----------------------------------------------------------------

    # QK: per-stage through memtile (3D with head dimension)
    # L3→memtile via QKIn_s, memtile→L1 via QK2L1_s with broadcast
    for s in range(NS):
        Channel(
            f"QK2L1_{s}",
            size=[num_heads_per_unroll, 1, 1],
            broadcast_shape=[num_heads_per_unroll, 1, NQ],
        )
        Channel(f"QKIn_{s}", size=[num_heads_per_unroll])

    # V: per-stage through memtile (3D with head dimension)
    for s in range(NS):
        Channel(
            f"V2L1_{s}",
            size=[num_heads_per_unroll, 1, 1],
            broadcast_shape=[num_heads_per_unroll, 1, NQ],
        )
        Channel(f"VIn_{s}", size=[num_heads_per_unroll])

    # Cascade: 2D per-segment (shared within each segment instance)
    channel("cascade_gp", size=[NQ, NS - 1], channel_type="cascade")
    channel("cascade_up", size=[NQ, NS - 1], channel_type="cascade")
    channel("cascade_sp", size=[NQ, NS - 1], channel_type="cascade")

    # Output: L1-to-L2 gather, then L2-to-L3
    Channel("Gp2L2", size=[NQ, 1])
    Channel("GpOut", size=[num_heads_per_unroll])

    # ----------------------------------------------------------------
    # Main attention function
    # ----------------------------------------------------------------
    @FuncOp.from_py_func(q_l3_t, k_l3_t, v_l3_t, gp_l3_t)
    def attention_bf16(q_in, k_in, v_in, gp_out):
        c1 = ConstantOp(index_type, 1)
        c_lq_iters = ConstantOp(index_type, num_lq_iters)
        c_num_head_groups = ConstantOp(index_type, num_head_groups)

        @launch(
            operands=[q_in, k_in, v_in, gp_out],
            sizes=[c_lq_iters, c_num_head_groups],
        )
        def launch_body(lx, ly, lsx, lsy, q, k, v, gp):

            # Compute Q offset from launch iteration index
            affine_map_q_launch = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(lqp * dk),
                )],
            )
            q_launch_off = affine_apply(affine_map_q_launch, [lx])

            # Output launch offset (output dim is dv, not dk)
            affine_map_out_launch = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(lqp * dv),
                )],
            )
            out_launch_off = affine_apply(affine_map_out_launch, [lx])

            # Compute head base from head group index (ly)
            # head_base = ly * num_heads_per_unroll
            affine_map_head_base = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(num_heads_per_unroll),
                )],
            )
            head_base = affine_apply(affine_map_head_base, [ly])

            # Offset maps for one head's worth of Q/K/V/output data
            affine_map_head_q = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(lq * dk),
                )],
            )
            affine_map_head_k = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(lk * dk),
                )],
            )
            affine_map_head_v = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(lk * dv),
                )],
            )
            affine_map_head_out = AffineMap.get(
                0, 1,
                [AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(lq * dv),
                )],
            )

            # s0 + s1
            affine_map_add = AffineMap.get(
                0, 2,
                [AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineSymbolExpr.get(1),
                )],
            )

            # head_1 = head_base + 1
            affine_map_plus1 = AffineMap.get(
                0, 1,
                [AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(1),
                )],
            )

            # ----------------------------------------------------------
            # For each head in the unroll group, send Q/K/V and get output
            # ----------------------------------------------------------
            # GQA: compute KV head index from Q head index
            # kv_head = q_head // gqa_group_size
            if gqa_group_size > 1:
                affine_map_kv_head = AffineMap.get(
                    0, 1,
                    [AffineExpr.get_floor_div(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(gqa_group_size),
                    )],
                )

            for head_local in range(num_heads_per_unroll):
                if head_local == 0:
                    head_idx = head_base
                else:
                    head_idx = affine_apply(affine_map_plus1, [head_base])

                # KV head index: same as Q head for MHA, floor-div for GQA
                if gqa_group_size == 1:
                    kv_head_idx = head_idx
                else:
                    kv_head_idx = affine_apply(
                        affine_map_kv_head, [head_idx],
                    )

                head_q_off = affine_apply(affine_map_head_q, [head_idx])
                head_k_off = affine_apply(affine_map_head_k, [kv_head_idx])
                head_v_off = affine_apply(affine_map_head_v, [kv_head_idx])
                head_out_off = affine_apply(affine_map_head_out, [head_idx])

                head_offset_idx = ConstantOp(index_type, head_local)

                # Combined Q offset = head_q_off + q_launch_off
                q_combined = affine_apply(affine_map_add, [head_q_off, q_launch_off])
                # Combined output offset (uses dv stride, not dk)
                out_combined = affine_apply(affine_map_add, [head_out_off, out_launch_off])

                # Q puts: flat transfer per stage to QKIn (memtile relay)
                for stage in range(NS):
                    ChannelPut(
                        f"QKIn_{stage}", q,
                        indices=[head_offset_idx],
                        offsets=[0, q_combined],
                        sizes=[NQ * tile_size_q, dk],
                        strides=[dk, 1],
                    )

                # K puts: flat transfer per stage to QKIn (memtile relay)
                for stage in range(NS):
                    k_stage_off_val = stage * lk_per_stage * dk
                    k_combined = affine_apply(
                        affine_map_add,
                        [head_k_off, ConstantOp(index_type, k_stage_off_val)],
                    )
                    ChannelPut(
                        f"QKIn_{stage}", k,
                        indices=[head_offset_idx],
                        offsets=[0, k_combined],
                        sizes=[lk_per_stage, dk],
                        strides=[dk, 1],
                    )

                # V puts: combined offset = head_v_off + stage_off
                for stage in range(NS):
                    v_stage_off_val = stage * lk_per_stage * dv
                    v_combined = affine_apply(
                        affine_map_add,
                        [head_v_off, ConstantOp(index_type, v_stage_off_val)],
                    )
                    ChannelPut(
                        f"VIn_{stage}", v,
                        indices=[head_offset_idx],
                        offsets=[0, 0, v_combined],
                        sizes=[chunks_per_stage, lkp, dv],
                        strides=[lkp * dv, dv, 1],
                    )

                # Output get: combined offset = head_out_off + out_launch_off
                ChannelGet(
                    "GpOut", gp,
                    indices=[head_offset_idx],
                    offsets=[out_combined],
                    sizes=[lqp * dv],
                    strides=[1],
                )

            # ----------------------------------------------------------
            # Segment: unrolled over heads
            # ----------------------------------------------------------
            c_num_heads_unroll = ConstantOp(index_type, num_heads_per_unroll)
            c1_seg = ConstantOp(index_type, 1)

            @segment(
                name="attn_seg",
                operands=[],
                sizes=[c_num_heads_unroll, c1_seg],
            )
            def segment_body(seg_x, seg_y, seg_sx, seg_sy):
                # L2 allocations for QK and V (per-stage) and output
                qk_l2_bufs = [
                    AllocOp(qk_l2_t, [], []) for _ in range(NS)
                ]
                v_l2_bufs = [
                    AllocOp(v_l2_t, [], []) for _ in range(NS)
                ]
                gp_l2 = AllocOp(gp_l2_t, [], [])

                # L1 allocations passed to herd
                q_saved = AllocOp(q_l1_t, [], [])
                qk_buf = AllocOp(k_l1_t, [], [])
                v_l1 = AllocOp(v_l1_t, [], [])
                g_l1 = AllocOp(g_l1_2d, [], [])
                gp_l1 = AllocOp(gp_l1_t, [], [])
                up_l1 = AllocOp(up_l1_t, [], [])
                sp_l1 = AllocOp(up_l1_t, [], [])
                if causal:
                    ctr_t = MemRefType.get([3], i32, memory_space=l1_space)
                    causal_ctr = AllocOp(ctr_t, [], [])

                c_nq = ConstantOp(index_type, NQ)
                c_ns = ConstantOp(index_type, NS)
                c0_seg = ConstantOp(index_type, 0)
                c_chunks_s = ConstantOp(index_type, chunks_per_stage)

                # QK streaming: L3-to-L2-to-L1 per stage
                # Q: NQ tiles, then K: chunks_per_stage chunks
                for stage in range(NS):
                    for qt_iter in scf_range(0, c_nq, 1):
                        ChannelGet(
                            f"QKIn_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"QK2L1_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dk // M, lkp // M, M, M],
                            strides=[M, dk * M, dk, 1],
                        )
                        yield_([])
                    for chunk_iter in scf_range(0, c_chunks_s, 1):
                        ChannelGet(
                            f"QKIn_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"QK2L1_{stage}",
                            qk_l2_bufs[stage].result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dk // M, lkp // M, M, M],
                            strides=[M, dk * M, dk, 1],
                        )
                        yield_([])

                # V streaming: L3-to-L2-to-L1 per stage
                for stage in range(NS):
                    for chunk_iter in scf_range(0, c_chunks_s, 1):
                        ChannelGet(
                            f"VIn_{stage}", v_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"V2L1_{stage}", v_l2_bufs[stage].result,
                            indices=[seg_x, c0_seg, c0_seg],  # [head, stage_dim=0, col_dim=0]
                            offsets=[0, 0, 0, 0],
                            sizes=[dv // M, lkp // M, M, M],
                            strides=[M, dv * M, dv, 1],
                        )
                        yield_([])

                # Output gather from ty=0 tiles
                affine_map_col = AffineMap.get(
                    0, 1,
                    [AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(tile_size_q),
                    )],
                )
                par_out = scf.ForallOp(
                    lower_bounds=[0], upper_bounds=[NQ], steps=[1]
                )
                with InsertionPoint(par_out.body):
                    apply_off = affine_apply(
                        affine_map_col,
                        [par_out.induction_variables[0]],
                    )
                    ChannelGet(
                        "Gp2L2", gp_l2.result,
                        indices=[par_out.induction_variables[0], 0],
                        offsets=[apply_off, 0],
                        sizes=[tile_size_q, dv],
                        strides=[dv, 1],
                    )
                    scf.InParallelOp()

                # Output: L2-to-L3
                ChannelPut("GpOut", gp_l2.result, indices=[seg_x])

                # ----------------------------------------------------------
                # Herd: [NQ, NS] — pass seg_x as operand
                # ----------------------------------------------------------
                herd_operands = [
                    q_saved, qk_buf, v_l1, g_l1, gp_l1, up_l1, sp_l1,
                    seg_x,
                ]
                if causal:
                    herd_operands.append(causal_ctr)

                @herd(
                    name="herd_0",
                    sizes=[c_nq, c_ns],
                    operands=herd_operands,
                    link_with="attn.o",
                )
                def herd_body(tx, ty, hsx, hsy,
                              q, qk, v, g, gp, up_buf, sp_buf,
                              h_seg_x, *extra_args):
                    counter_buf = extra_args[0] if causal else None
                    # Precompute affine sets for per-stage V dispatch
                    s0 = AffineSymbolExpr.get(0)
                    s1 = AffineSymbolExpr.get(1)
                    c_ns_m1 = AffineConstantExpr.get(NS - 1)
                    stage_sets = []
                    for s in range(NS):
                        cs = AffineConstantExpr.get(s)
                        stage_sets.append(
                            IntegerSet.get(
                                0, 2,
                                [s0, s1 - cs],
                                [False, True],
                            )
                        )

                    # === INIT PHASE (FIRST — before any channel ops) ===
                    CallOp([], "zero_fill_gp_bf16", [gp])
                    CallOp([], "zero_fill_sp_bf16", [sp_buf])
                    CallOp([], "neg_inf_fill_up_bf16", [up_buf])

                    # === CAUSAL COUNTER INIT ===
                    if causal:
                        c0_ctr = ConstantOp(index_type, 0)
                        c1_ctr = ConstantOp(index_type, 1)
                        c2_ctr = ConstantOp(index_type, 2)
                        boot_flag = load(counter_buf, [c1_ctr])
                        is_first = arith.CmpIOp(
                            arith.CmpIPredicate.eq,
                            boot_flag, ConstantOp(i32, 0),
                        )
                        if_first = scf.IfOp(is_first)
                        with InsertionPoint(if_first.then_block):
                            store(ConstantOp(i32, 0),
                                  counter_buf, [c0_ctr])
                            store(ConstantOp(i32, 1),
                                  counter_buf, [c1_ctr])
                            store(ConstantOp(i32, 0),
                                  counter_buf, [c2_ctr])
                            scf.YieldOp([])

                    # === Q SELECTIVE CAPTURE ===
                    # Receive all NQ Q tiles, but only copy the one
                    # matching this tile's tx index.
                    # Stage-gated get from per-stage QK2L1_s channels.
                    for qt in range(NQ):
                        for s in range(NS):
                            if_qk_q = affine.AffineIfOp(
                                stage_sets[s],
                                cond_operands=[tx, ty],
                            )
                            with InsertionPoint(if_qk_q.then_block):
                                ChannelGet(
                                    f"QK2L1_{s}", qk,
                                    indices=[h_seg_x, ty, tx],
                                )
                                affine.AffineYieldOp([])
                        cmp = arith.CmpIOp(
                            arith.CmpIPredicate.eq,
                            arith.IndexCastOp(i32, tx),
                            arith.ConstantOp(i32, qt),
                        )
                        if_cap = scf.IfOp(cmp)
                        with InsertionPoint(if_cap.then_block):
                            CallOp([], "copy_tile", [qk, q])
                            scf.YieldOp([])

                    # === K CHUNK LOOP ===
                    c_chunks_h = ConstantOp(index_type, chunks_per_stage)
                    for chunk_iter in scf_range(0, c_chunks_h, 1):
                        # 1. Zero fill G (FIRST)
                        g1d = CollapseShapeOp(g_l1_1d, g, [[0, 1]])
                        CallOp([], "zero_fill_g_bf16", [g1d])

                        # 2. K get (SECOND) — stage-gated per-stage channels
                        for s in range(NS):
                            if_qk_k = affine.AffineIfOp(
                                stage_sets[s],
                                cond_operands=[tx, ty],
                            )
                            with InsertionPoint(if_qk_k.then_block):
                                ChannelGet(
                                    f"QK2L1_{s}", qk,
                                    indices=[h_seg_x, ty, tx],
                                )
                                affine.AffineYieldOp([])

                        # 3. V get via affine.if per stage (THIRD)
                        #    — 3D index with head dim
                        for s in range(NS):
                            if_v = affine.AffineIfOp(
                                stage_sets[s],
                                cond_operands=[tx, ty],
                            )
                            with InsertionPoint(if_v.then_block):
                                ChannelGet(
                                    f"V2L1_{s}", v,
                                    indices=[h_seg_x, ty, tx],
                                )
                                affine.AffineYieldOp([])

                        # 4. Matmul Q @ K^T (FOURTH)
                        CallOp([], "matmul_a_b_bf16", [q, qk, g1d])

                        # 4b. Apply causal mask (after matmul, before softmax)
                        if causal:
                            c_cps_i32 = ConstantOp(i32, chunks_per_stage)
                            ty_i32 = arith.IndexCastOp(i32, ty).result
                            chunk_i32 = arith.IndexCastOp(
                                i32, chunk_iter,
                            ).result
                            kv_base = arith.MulIOp(ty_i32, c_cps_i32)
                            kv_block = arith.AddIOp(
                                kv_base.result, chunk_i32,
                            )
                            q_base = load(counter_buf, [c0_ctr])
                            tx_i32 = arith.IndexCastOp(i32, tx).result
                            q_block = arith.AddIOp(q_base, tx_i32)
                            CallOp(
                                [], "apply_causal_mask",
                                [g, q_block.result, kv_block.result],
                            )

                        # 5. Softmax + accumulate
                        s_tmp = AllocOp(up_l1_t, [], [])
                        r_tmp = AllocOp(up_l1_t, [], [])
                        CallOp(
                            [], "fused_softmax",
                            [g1d, up_buf, s_tmp.result, r_tmp.result],
                        )
                        CallOp([], "mul_r_gp", [r_tmp.result, gp])
                        CallOp([], "matmul_g_b_bf16", [g1d, v, gp])
                        c0_i32 = ConstantOp(i32, 0)
                        CallOp(
                            [], "accum_sp_r_s",
                            [sp_buf, r_tmp.result, s_tmp.result],
                        )
                        CallOp(
                            [], "vector_copy_32elems",
                            [c0_i32, s_tmp.result, sp_buf],
                        )
                        DeallocOp(s_tmp)
                        DeallocOp(r_tmp)
                        yield_([])

                    # === CASCADE MERGE (last/middle/first) ===
                    # Exactly matching step_test.py ordering.
                    set_first_stage = IntegerSet.get(
                        0, 2, [s0, s1 - c_ns_m1], [False, True]
                    )
                    set_middle_stage = IntegerSet.get(
                        0, 2,
                        [
                            AffineExpr.get_add(
                                s1, AffineConstantExpr.get(-1)
                            ),
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NS - 2),
                                AffineExpr.get_mul(
                                    s1, AffineConstantExpr.get(-1)
                                ),
                            ),
                            s0,
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NQ - 1),
                                AffineExpr.get_mul(
                                    s0, AffineConstantExpr.get(-1)
                                ),
                            ),
                        ],
                        [False, False, False, False],
                    )
                    c1_h = ConstantOp(index_type, 1)

                    # Last stage (ty == NS-1): send cascade down
                    if_last = affine.AffineIfOp(
                        set_first_stage, cond_operands=[tx, ty],
                        has_else=True,
                    )
                    with InsertionPoint(if_last.then_block):
                        subi_l = arith.SubIOp(ty, c1_h)
                        ChannelPut(
                            "cascade_gp", gp, indices=[tx, subi_l]
                        )
                        ChannelPut(
                            "cascade_up", up_buf, indices=[tx, subi_l]
                        )
                        ChannelPut(
                            "cascade_sp", sp_buf, indices=[tx, subi_l]
                        )
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_last.else_block):
                        # Middle stages: 1 <= ty <= NS-2
                        if_mid = affine.AffineIfOp(
                            set_middle_stage,
                            cond_operands=[tx, ty],
                            has_else=True,
                        )
                        with InsertionPoint(if_mid.then_block):
                            gp_c = AllocOp(gp_l1_t, [], [])
                            up_c = AllocOp(up_l1_t, [], [])
                            sp_c = AllocOp(up_l1_t, [], [])
                            ChannelGet(
                                "cascade_gp", gp_c.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_up", up_c.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_sp", sp_c.result,
                                indices=[tx, ty],
                            )
                            up_s = AllocOp(up_l1_t, [], [])
                            c0m = ConstantOp(i32, 0)
                            CallOp(
                                [], "vector_copy_32elems",
                                [c0m, up_buf, up_s.result],
                            )
                            CallOp(
                                [], "maximum_up_u_bf16",
                                [up_c.result, up_buf],
                            )
                            rc = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u",
                                [up_c.result, up_buf, rc.result],
                            )
                            rl = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u",
                                [up_s.result, up_buf, rl.result],
                            )
                            CallOp(
                                [], "mul_r_gp", [rc.result, gp_c.result]
                            )
                            CallOp([], "mul_r_gp", [rl.result, gp])
                            CallOp(
                                [], "add_gp_g", [gp, gp_c.result]
                            )
                            st = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "zero_fill_sp_bf16", [st.result]
                            )
                            CallOp(
                                [], "accum_sp_r_s",
                                [sp_c.result, rc.result, st.result],
                            )
                            CallOp(
                                [], "accum_sp_r_s",
                                [sp_buf, rl.result, st.result],
                            )
                            CallOp(
                                [], "vector_copy_32elems",
                                [c0m, st.result, sp_c.result],
                            )
                            subi_m = arith.SubIOp(ty, c1_h)
                            ChannelPut(
                                "cascade_gp", gp_c.result,
                                indices=[tx, subi_m],
                            )
                            ChannelPut(
                                "cascade_up", up_buf,
                                indices=[tx, subi_m],
                            )
                            ChannelPut(
                                "cascade_sp", sp_c.result,
                                indices=[tx, subi_m],
                            )
                            DeallocOp(gp_c)
                            DeallocOp(up_c)
                            DeallocOp(sp_c)
                            DeallocOp(up_s)
                            DeallocOp(rc)
                            DeallocOp(rl)
                            DeallocOp(st)
                            affine.AffineYieldOp([])

                        with InsertionPoint(if_mid.else_block):
                            # First stage (ty == 0): cascade in, merge,
                            # div, output
                            gp_c2 = AllocOp(gp_l1_t, [], [])
                            up_c2 = AllocOp(up_l1_t, [], [])
                            sp_c2 = AllocOp(up_l1_t, [], [])
                            ChannelGet(
                                "cascade_gp", gp_c2.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_up", up_c2.result,
                                indices=[tx, ty],
                            )
                            ChannelGet(
                                "cascade_sp", sp_c2.result,
                                indices=[tx, ty],
                            )
                            up_s2 = AllocOp(up_l1_t, [], [])
                            c0f = ConstantOp(i32, 0)
                            CallOp(
                                [], "vector_copy_32elems",
                                [c0f, up_buf, up_s2.result],
                            )
                            CallOp(
                                [], "maximum_up_u_bf16",
                                [up_c2.result, up_buf],
                            )
                            rc2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u",
                                [up_c2.result, up_buf, rc2.result],
                            )
                            rl2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u",
                                [up_s2.result, up_buf, rl2.result],
                            )
                            CallOp(
                                [], "mul_r_gp",
                                [rc2.result, gp_c2.result],
                            )
                            CallOp(
                                [], "mul_r_gp", [rl2.result, gp]
                            )
                            CallOp(
                                [], "add_gp_g", [gp, gp_c2.result]
                            )
                            st2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "zero_fill_sp_bf16", [st2.result]
                            )
                            CallOp(
                                [], "accum_sp_r_s",
                                [sp_c2.result, rc2.result, st2.result],
                            )
                            CallOp(
                                [], "accum_sp_r_s",
                                [sp_buf, rl2.result, st2.result],
                            )
                            CallOp(
                                [], "vector_copy_32elems",
                                [c0f, st2.result, sp_c2.result],
                            )
                            CallOp(
                                [], "div_gp_sp",
                                [sp_c2.result, gp_c2.result],
                            )
                            c0_out = ConstantOp(index_type, 0)
                            ChannelPut(
                                "Gp2L2", gp_c2.result,
                                indices=[tx, c0_out],
                                offsets=[0, 0, 0, 0],
                                sizes=[
                                    tile_size_q // M,
                                    M,
                                    dv // M,
                                    M,
                                ],
                                strides=[
                                    M * M,
                                    M,
                                    tile_size_q * M,
                                    1,
                                ],
                            )
                            DeallocOp(gp_c2)
                            DeallocOp(up_c2)
                            DeallocOp(sp_c2)
                            DeallocOp(up_s2)
                            DeallocOp(rc2)
                            DeallocOp(rl2)
                            DeallocOp(st2)
                            affine.AffineYieldOp([])
                        affine.AffineYieldOp([])

                    # === CAUSAL COUNTER INCREMENT ===
                    if causal:
                        head_cur = load(counter_buf, [c2_ctr])
                        c1_i32_inc = ConstantOp(i32, 1)
                        head_next = arith.AddIOp(head_cur, c1_i32_inc)
                        total_hg = ConstantOp(i32, num_head_groups)
                        wrapped = arith.CmpIOp(
                            arith.CmpIPredicate.sge,
                            head_next.result, total_hg,
                        )
                        if_wrap = scf.IfOp(wrapped)
                        with InsertionPoint(if_wrap.then_block):
                            q_cur = load(counter_buf, [c0_ctr])
                            c_nq_i32 = ConstantOp(i32, NQ)
                            q_next = arith.AddIOp(q_cur, c_nq_i32)
                            store(q_next.result,
                                  counter_buf, [c0_ctr])
                            store(ConstantOp(i32, 0),
                                  counter_buf, [c2_ctr])
                            scf.YieldOp([])
                        not_wrapped = arith.CmpIOp(
                            arith.CmpIPredicate.slt,
                            head_next.result, total_hg,
                        )
                        if_no_wrap = scf.IfOp(not_wrapped)
                        with InsertionPoint(if_no_wrap.then_block):
                            store(head_next.result,
                                  counter_buf, [c2_ctr])
                            scf.YieldOp([])

                # Deallocs for segment-level buffers
                DeallocOp(q_saved)
                DeallocOp(qk_buf)
                DeallocOp(v_l1)
                DeallocOp(g_l1)
                DeallocOp(gp_l1)
                DeallocOp(up_l1)
                DeallocOp(sp_l1)
                for stage in range(NS):
                    DeallocOp(v_l2_bufs[stage])
                DeallocOp(gp_l2)
                if causal:
                    DeallocOp(causal_ctr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="attn_l3l1.py",
        description="Flash attention with L3-to-L1 direct Q/K — "
                    "selective capture",
    )
    parser.add_argument(
        "-p", "--print-module-only", action="store_true",
        help="Print MLIR module and exit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--lk", type=int, default=512,
        help="Total K/V sequence length (default: 512)",
    )
    parser.add_argument(
        "--lq", type=int, default=512,
        help="Total Q sequence length (default: 512)",
    )
    parser.add_argument(
        "--lqp", type=int, default=256,
        help="Q chunk size per launch iteration (default: 256)",
    )
    parser.add_argument(
        "--lkp", type=int, default=64,
        help="K/V chunk size per tile (default: 64)",
    )
    parser.add_argument(
        "--num-cascade-stages", type=int, default=4,
        help="Number of cascade pipeline stages (default: 4)",
    )
    parser.add_argument(
        "--num-heads", type=int, default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=None,
        help="Number of KV heads (default: num_heads for MHA, "
             "< num_heads for GQA)",
    )
    parser.add_argument(
        "--compile-mode", type=str, default="compile-and-run",
        choices=["compile-only", "compile-and-run"],
        help="Compilation mode (default: compile-and-run)",
    )
    parser.add_argument(
        "--output-format", type=str, default="elf",
        choices=["xclbin", "elf"],
        help="Output format (default: elf)",
    )
    parser.add_argument(
        "--causal", action="store_true",
        help="Enable causal masking (autoregressive attention)",
    )
    args = parser.parse_args()

    lk = args.lk
    lkp = args.lkp
    lq = args.lq
    lqp = args.lqp
    dk = 64
    dv = 64
    num_cascade_stages = args.num_cascade_stages
    num_q_tiles = 4
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads
    causal = args.causal
    gqa_group_size = num_heads // num_kv_heads

    mlir_module = build_module(
        lk=lk,
        lkp=lkp,
        lq=lq,
        lqp=lqp,
        dk=dk,
        dv=dv,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=num_cascade_stages,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner
    from air.backend.xrt import XRTBackend
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16
    rng = np.random.default_rng(42)
    val_range = 3.0
    input_q = rng.uniform(
        0, val_range, (num_heads, lq, dk)
    ).astype(INPUT_DATATYPE)
    input_k = rng.uniform(
        0, val_range, (num_kv_heads, lk, dk)
    ).astype(INPUT_DATATYPE)
    input_v = rng.uniform(
        0, val_range, (num_kv_heads, lk, dv)
    ).astype(INPUT_DATATYPE)

    inv_sqrt_dk = 1.0 / sqrt(dk)
    sdpa_output = np.zeros(
        (num_heads, lq, dv), dtype=OUTPUT_DATATYPE
    )
    for h in range(num_heads):
        kv_h = h // gqa_group_size
        Qf = input_q[h].astype(np.float32)
        Kf = input_k[kv_h].astype(np.float32)
        Vf = input_v[kv_h].astype(np.float32)
        scores = Qf @ Kf.T * inv_sqrt_dk
        if causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        P = np.exp(scores - mx)
        P = P / np.sum(P, axis=-1, keepdims=True)
        sdpa_output[h] = (P @ Vf).astype(OUTPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        import filelock
        backend = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            verbose=args.verbose,
            runtime_loop_tiling_sizes=[1, 1],
            output_format=args.output_format,
            instance_name="attention_bf16",
            target_device="npu2",
        )
        artifact = backend.compile(mlir_module)
        print("Compiled. Running on device...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)
            results = invoker(
                input_q, input_k, input_v,
                np.zeros((num_heads, lq, dv), dtype=INPUT_DATATYPE),
            )
        npu = results[3].reshape(num_heads, lq, dv).astype(np.float32)
        ref = sdpa_output.astype(np.float32)
        backend.unload()
        # Per-head correlation
        all_pass = True
        for h in range(num_heads):
            c_h = float(
                np.corrcoef(
                    npu[h].flatten(), ref[h].flatten()
                )[0, 1]
            )
            status = "PASS" if c_h > 0.99 else "FAIL"
            print(f"Head {h} correlation: {c_h:.6f} [{status}]")
            if c_h <= 0.99:
                all_pass = False
        c = float(np.corrcoef(npu.flatten(), ref.flatten())[0, 1])
        print(f"Overall correlation: {c:.6f}")
        print(f"{'PASS!' if all_pass else 'FAIL'}")
        exit(0 if all_pass else 1)
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            verbose=args.verbose,
            runtime_loop_tiling_sizes=[1, 1],
            output_format=args.output_format,
            instance_name="attention_bf16",
            target_device="npu2",
        )
        module_function = backend.compile(mlir_module)
        print("Compilation complete.")
