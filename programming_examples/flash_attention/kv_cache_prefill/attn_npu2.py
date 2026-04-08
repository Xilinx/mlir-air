# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused Flash Attention + KV Cache Write-Back (Single Launch).

Single-launch design that fuses flash attention and KV cache write-back
into one AIE program.  When RoPE is enabled, Q and K are pre-RoPE'd by
the host before being sent to the NPU -- the NPU performs attention on
already-rotated data.  Both K and V data are written back to an
interleaved KV cache buffer in L3 during attention computation.

TODO: Replace host-side RoPE with on-chip rope_sincos kernel that
computes sin/cos directly on AIE2P without needing a LUT.

DMA channel strategy (2 S2MM + 2 MM2S per compute tile):
  S2MM 0: QK channel (Q and K via L2 relay)
  S2MM 1: V (per-stage via memtile)
  MM2S 0: CacheWB (K+V write-back, tx=0 only) or cascade/output
  MM2S 1: Gp2L2 output gather (ty=0 only)

Channel layout:
  QKIn_s/QK2L1_s: per-stage memtile relay with horizontal broadcast
  VIn_s/V2L1_s: per-stage memtile relay with horizontal broadcast
  cascade_gp/cascade_up/cascade_sp: 2D cascade channels (per-segment)
  Gp2L2/GpOut: output from ty=0 tiles
  CacheWB: K+V cache write-back (tx=0 tiles, single channel for both K
           and V via shared kwb_buf staging buffer)

KV Cache Layout
===============
The KV cache is stored as a single flat buffer with K and V chunks
interleaved per-chunk.  This layout is chosen because:
  1. It allows a single DMA channel (CacheWB) to write both K and V,
     avoiding shim S2MM channel exhaustion.
  2. The DMA uses a single BD with a staging buffer (kwb_buf), sending K
     then V for each chunk iteration.  The interleaved layout means
     consecutive DMA transfers write to consecutive L3 offsets.
  3. The scf.for loop in the launch body enables the compiler to fold
     multiple chunk transfers into a single higher-dimensional shim BD,
     preventing BD exhaustion at large sequence lengths.

Layout (per KV head):
    [K_chunk0, V_chunk0, K_chunk1, V_chunk1, ..., K_chunkN, V_chunkN]

Where each chunk is [lkp, dk_tile] = [64, 64] bf16 elements (8 KB).

Full buffer shape (logical):
    [num_kv_heads, num_chunks, 2, lkp, dk_tile]
     |              |          |   |    |
     |              |          |   |    head dimension tile (= lkp)
     |              |          |   chunk rows (= lkp)
     |              |          0=K (RoPE'd), 1=V (raw)
     |              lk / lkp chunks per head
     KV head index

Physical flat size: num_kv_heads * num_chunks * 2 * lkp * dk_tile bf16.

K data: RoPE-rotated key data, un-tiled from [M,M] blocked L1 format to
        row-major [lkp, dk_tile] during DMA write-back.
V data: Raw value data (no RoPE), un-tiled from [M,M] blocked L1 format
        to row-major [lkp, dv_tile] during DMA write-back.

Limitation: the interleaved layout currently requires dk == dv == lkp
(i.e., dk_chunks == dv_chunks == 1).  This is satisfied by the target
model (head_dim=64, lkp=64).  Supporting dk > lkp would require
extending the interleaving pattern to handle multiple dk_tile transfers
per K chunk.

For decode (future consumers): to read K or V separately from this
interleaved buffer, use a DMA stride of 2*lkp*dk_tile between
consecutive K (or V) chunks.  For example, to read all K chunks for
head h:
    base_offset = h * num_chunks * 2 * lkp * dk_tile
    K_chunk[i] at offset: base_offset + i * 2 * lkp * dk_tile
    V_chunk[i] at offset: base_offset + i * 2 * lkp * dk_tile + lkp*dk_tile
"""

import argparse
import os
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
    enable_k_writeback=True,
    enable_v_writeback=True,
):
    """Build flash attention + KV cache module (RoPE applied on host).

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
        num_kv_heads: Number of key/value heads for grouped-query attention
            (GQA). If None, defaults to num_heads (standard MHA).
        causal: Whether to enable causal (autoregressive) masking.
    """
    # Validate
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert lk % (lkp * num_cascade_stages) == 0, (
        f"lk ({lk}) must be divisible by lkp * num_cascade_stages "
        f"({lkp * num_cascade_stages})"
    )
    dk_tile = lkp
    assert dk % dk_tile == 0, f"dk ({dk}) must be divisible by dk_tile/lkp ({dk_tile})"
    dk_chunks = dk // dk_tile
    dv_tile = lkp
    assert dv % dv_tile == 0, f"dv ({dv}) must be divisible by dv_tile/lkp ({dv_tile})"
    dv_chunks = dv // dv_tile
    enable_cache_writeback = enable_k_writeback or enable_v_writeback
    if enable_cache_writeback:
        assert dk == lkp and dv == lkp, (
            f"Interleaved KV cache write-back requires dk == dv == lkp, "
            f"got dk={dk}, dv={dv}, lkp={lkp}. "
            f"Use --no-k-writeback --no-v-writeback to disable."
        )
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

    # L1 MemRefTypes (Q and K use dk_tile, not full dk)
    q_l1_t = MemRefType.get([tile_size_q, dk_tile], bf16, memory_space=l1_space)
    k_l1_t = MemRefType.get([lkp, dk_tile], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l1_space)
    g_l1_2d = MemRefType.get([tile_size_q, lkp], bf16, memory_space=l1_space)
    g_l1_1d = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    gp_l1_t = MemRefType.get([tile_size_q, dv_tile], bf16, memory_space=l1_space)
    up_l1_t = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)

    # L2 MemRefTypes (QK relay uses dk_tile)
    qk_l2_t = MemRefType.get([lkp, dk_tile], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l2_space)
    gp_l2_t = MemRefType.get([lqp, dv_tile], bf16, memory_space=l2_space)

    # L3 MemRefTypes (3D with head dimension)
    q_l3_t = MemRefType.get([num_heads, lq, dk], bf16)
    k_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    # V and output L3 use transposed layout for contiguous dv_tile access:
    # [heads * dv_chunks, seq, dv_tile] instead of [heads, seq, dv]
    v_l3_t = MemRefType.get([num_kv_heads * dv_chunks, lk, dv_tile], bf16)
    gp_l3_t = MemRefType.get([num_heads * dv_chunks, lq, dv_tile], bf16)

    # KV cache L3 types
    if enable_k_writeback or enable_v_writeback:
        # Interleaved KV cache: [num_kv_heads, num_chunks, 2, lkp, dk_tile]
        # K at index 0, V at index 1 within each chunk pair
        kv_cache_l3_t = MemRefType.get(
            [num_kv_heads * num_chunks * 2 * lkp * dk_tile], bf16
        )
        kv_chunk_stride = 2 * lkp * dk_tile  # stride between K and V in a pair
    # Legacy separate caches (when writeback disabled, used as placeholders)
    k_cache_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    v_cache_l3_t = MemRefType.get([num_kv_heads * dv_chunks, lk, dv_tile], bf16)

    # External function declarations
    def external_func(name, inputs, outputs=None, link_with=None, visibility="private"):
        if outputs is None:
            outputs = []
        func_type = FunctionType.get(inputs, outputs)
        func = FuncOp(name=name, type=func_type, visibility=visibility)
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    external_func("zero_fill_g_bf16", [g_l1_1d], link_with="attn_npu2.o")
    external_func("zero_fill_gp_bf16", [gp_l1_t], link_with="attn_npu2.o")
    external_func("zero_fill_sp_bf16", [up_l1_t], link_with="attn_npu2.o")
    external_func("neg_inf_fill_up_bf16", [up_l1_t], link_with="attn_npu2.o")
    external_func(
        "matmul_a_b_bf16",
        [q_l1_t, k_l1_t, g_l1_1d],
        link_with="attn_npu2.o",
    )
    external_func(
        "matmul_g_b_bf16",
        [g_l1_1d, v_l1_t, gp_l1_t],
        link_with="attn_npu2.o",
    )
    external_func(
        "fused_softmax",
        [g_l1_1d, up_l1_t, up_l1_t, up_l1_t],
        link_with="attn_npu2.o",
    )
    external_func("maximum_up_u_bf16", [up_l1_t, up_l1_t], link_with="attn_npu2.o")
    external_func(
        "exp_up_minus_u",
        [up_l1_t, up_l1_t, up_l1_t],
        link_with="attn_npu2.o",
    )
    external_func("mul_r_gp", [up_l1_t, gp_l1_t], link_with="attn_npu2.o")
    external_func(
        "accum_sp_r_s",
        [up_l1_t, up_l1_t, up_l1_t],
        link_with="attn_npu2.o",
    )
    external_func(
        "vector_copy_32elems", [i32, up_l1_t, up_l1_t], link_with="attn_npu2.o"
    )
    external_func("copy_tile", [k_l1_t, q_l1_t], link_with="attn_npu2.o")
    external_func("div_gp_sp", [up_l1_t, gp_l1_t], link_with="attn_npu2.o")
    external_func("add_gp_g", [gp_l1_t, gp_l1_t], link_with="attn_npu2.o")
    if causal:
        external_func("apply_causal_mask", [g_l1_2d, i32, i32], link_with="attn_npu2.o")

    # ----------------------------------------------------------------
    # Channel declarations
    # ----------------------------------------------------------------

    # QK: per-stage through memtile (3D with head dimension)
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

    # KV cache write-back: single channel, tx=0 tiles send K then V per chunk
    # into interleaved KV cache buffer.
    if enable_cache_writeback:
        Channel("CacheWB", size=[num_heads_per_unroll, NS, 1])

    # ----------------------------------------------------------------
    # Main function: fused RoPE + attention + KV cache
    # ----------------------------------------------------------------
    func_args = [q_l3_t, k_l3_t, v_l3_t, gp_l3_t]
    if enable_cache_writeback:
        func_args.append(kv_cache_l3_t)
    else:
        func_args.extend([k_cache_l3_t, v_cache_l3_t])

    @FuncOp.from_py_func(*func_args)
    def attention_bf16(*func_params):
        if enable_cache_writeback:
            q_in, k_in, v_in, gp_out, kv_cache = func_params
        else:
            q_in, k_in, v_in, gp_out, k_cache, v_cache = func_params
            kv_cache = None
        c1 = ConstantOp(index_type, 1)
        c_lq_iters = ConstantOp(index_type, num_lq_iters)
        c_num_head_groups = ConstantOp(index_type, num_head_groups)

        if dv_chunks > 1:
            c_dv_chunks = ConstantOp(index_type, dv_chunks)
            launch_sizes = [c_lq_iters, c_num_head_groups, c_dv_chunks]
        else:
            launch_sizes = [c_lq_iters, c_num_head_groups]

        if enable_cache_writeback:
            launch_operands = [q_in, k_in, v_in, gp_out, kv_cache]
        else:
            launch_operands = [q_in, k_in, v_in, gp_out, k_cache, v_cache]

        @launch(
            operands=launch_operands,
            sizes=launch_sizes,
        )
        def launch_body(*launch_args):
            if enable_cache_writeback:
                if dv_chunks > 1:
                    lx, ly, lz, lsx, lsy, lsz, q, k, v, gp, kv_cache_arg = launch_args
                else:
                    lx, ly, lsx, lsy, q, k, v, gp, kv_cache_arg = launch_args
                    lz = ConstantOp(index_type, 0)
            else:
                if dv_chunks > 1:
                    lx, ly, lz, lsx, lsy, lsz, q, k, v, gp, kcache, vcache = launch_args
                else:
                    lx, ly, lsx, lsy, q, k, v, gp, kcache, vcache = launch_args
                    lz = ConstantOp(index_type, 0)

            # Compute Q offset from launch iteration index
            affine_map_q_launch = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lqp * dk),
                    )
                ],
            )
            q_launch_off = affine_apply(affine_map_q_launch, [lx])

            # Output launch offset (transposed layout uses dv_tile, not dv)
            affine_map_out_launch = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lqp * dv_tile),
                    )
                ],
            )
            out_launch_off = affine_apply(affine_map_out_launch, [lx])

            # Compute head base from head group index (ly)
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
            head_base = affine_apply(affine_map_head_base, [ly])

            # Offset maps for one head's worth of Q/K/V/output data
            affine_map_head_q = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lq * dk),
                    )
                ],
            )
            affine_map_head_k = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(lk * dk),
                    )
                ],
            )
            affine_map_head_v_dv = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(dv_chunks),
                            ),
                            AffineSymbolExpr.get(1),
                        ),
                        AffineConstantExpr.get(lk * dv_tile),
                    )
                ],
            )
            affine_map_head_out_dv = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(dv_chunks),
                            ),
                            AffineSymbolExpr.get(1),
                        ),
                        AffineConstantExpr.get(lq * dv_tile),
                    )
                ],
            )

            # s0 + s1
            affine_map_add = AffineMap.get(
                0,
                2,
                [
                    AffineExpr.get_add(
                        AffineSymbolExpr.get(0),
                        AffineSymbolExpr.get(1),
                    )
                ],
            )

            # head_1 = head_base + 1
            affine_map_plus1 = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_add(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(1),
                    )
                ],
            )

            # GQA head map
            if gqa_group_size > 1:
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

            for head_local in range(num_heads_per_unroll):
                if head_local == 0:
                    head_idx = head_base
                else:
                    head_idx = affine_apply(affine_map_plus1, [head_base])

                if gqa_group_size == 1:
                    kv_head_idx = head_idx
                else:
                    kv_head_idx = affine_apply(
                        affine_map_kv_head,
                        [head_idx],
                    )

                head_q_off = affine_apply(affine_map_head_q, [head_idx])
                head_k_off = affine_apply(affine_map_head_k, [kv_head_idx])
                head_v_off = affine_apply(affine_map_head_v_dv, [kv_head_idx, lz])
                head_out_off = affine_apply(affine_map_head_out_dv, [head_idx, lz])

                head_offset_idx = ConstantOp(index_type, head_local)

                q_combined = affine_apply(affine_map_add, [head_q_off, q_launch_off])
                out_combined = affine_apply(
                    affine_map_add, [head_out_off, out_launch_off]
                )

                # ----------------------------------------------------------
                # Q puts: bulk Q data to QKIn (pre-RoPE'd when enabled)
                # ----------------------------------------------------------
                for stage in range(NS):
                    ChannelPut(
                        f"QKIn_{stage}",
                        q,
                        indices=[head_offset_idx],
                        offsets=[0, q_combined],
                        sizes=[NQ, dk_chunks, tile_size_q, dk_tile],
                        strides=[tile_size_q * dk, dk_tile, dk, 1],
                    )

                    # ----------------------------------------------------------
                    # K puts: bulk K data to QKIn (pre-RoPE'd when enabled)
                    # ----------------------------------------------------------
                    k_stage_off_val = stage * lk_per_stage * dk
                    k_combined = affine_apply(
                        affine_map_add,
                        [head_k_off, ConstantOp(index_type, k_stage_off_val)],
                    )
                    ChannelPut(
                        f"QKIn_{stage}",
                        k,
                        indices=[head_offset_idx],
                        offsets=[0, k_combined],
                        sizes=[chunks_per_stage, dk_chunks, lkp, dk_tile],
                        strides=[lkp * dk, dk_tile, dk, 1],
                    )

                    # ----------------------------------------------------------
                    # V puts: bulk V data to VIn
                    # ----------------------------------------------------------
                    v_stage_off_val = stage * lk_per_stage * dv_tile
                    v_combined = affine_apply(
                        affine_map_add,
                        [head_v_off, ConstantOp(index_type, v_stage_off_val)],
                    )
                    ChannelPut(
                        f"VIn_{stage}",
                        v,
                        indices=[head_offset_idx],
                        offsets=[0, 0, v_combined],
                        sizes=[chunks_per_stage, lkp, dv_tile],
                        strides=[lkp * dv_tile, dv_tile, 1],
                    )

                # ----------------------------------------------------------
                # KV cache gets (L1→L3 via CacheWB channel)
                # Interleaved KV cache: [K_c0, V_c0, K_c1, V_c1, ...]
                # Each chunk pair occupies 2 * lkp * dk_tile elements.
                # The tile BD chain alternates BD0(K)→BD1(V), matching
                # this K,V,K,V get ordering.
                # ----------------------------------------------------------
                if enable_cache_writeback:
                    is_first_in_gqa_group = (
                        gqa_group_size == 1 or head_local % gqa_group_size == 0
                    )
                    if is_first_in_gqa_group:
                        # KV head offset into the flat kv_cache buffer
                        kv_head_off = affine_apply(
                            AffineMap.get(
                                0,
                                1,
                                [
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(
                                            num_chunks * 2 * lkp * dk_tile
                                        ),
                                    )
                                ],
                            ),
                            [kv_head_idx],
                        )
                        # Use scf.for to allow the compiler to fold
                        # consecutive CacheWB BDs into higher-dimensional
                        # shim DMA BDs, avoiding BD exhaustion at scale.
                        c_cps2 = ConstantOp(index_type, chunks_per_stage * 2)
                        affine_map_kv_off = AffineMap.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(lkp * dk_tile),
                                    ),
                                )
                            ],
                        )
                        for stage in range(NS):
                            stage_base_val = stage * chunks_per_stage * kv_chunk_stride
                            stage_base = affine_apply(
                                affine_map_add,
                                [kv_head_off, ConstantOp(index_type, stage_base_val)],
                            )
                            for chunk_kv_iter in scf_range(0, c_cps2, 1):
                                wb_off = affine_apply(
                                    affine_map_kv_off,
                                    [stage_base, chunk_kv_iter],
                                )
                                ChannelGet(
                                    "CacheWB",
                                    kv_cache_arg,
                                    indices=[
                                        head_offset_idx,
                                        ConstantOp(index_type, stage),
                                        ConstantOp(index_type, 0),
                                    ],
                                    offsets=[wb_off],
                                    sizes=[lkp * dk_tile],
                                    strides=[1],
                                )
                                yield_([])

                # ----------------------------------------------------------
                # Output get (after KV cache, matches segment ordering)
                # ----------------------------------------------------------
                ChannelGet(
                    "GpOut",
                    gp,
                    indices=[head_offset_idx],
                    offsets=[out_combined],
                    sizes=[lqp * dv_tile],
                    strides=[1],
                )

            # ----------------------------------------------------------
            # Segment: unrolled over heads
            # ----------------------------------------------------------
            c_num_heads_unroll = ConstantOp(index_type, num_heads_per_unroll)
            c1_seg = ConstantOp(index_type, 1)

            @segment(
                name="attn_seg",
                operands=[lx],
                sizes=[c_num_heads_unroll, c1_seg],
            )
            def segment_body(seg_x, seg_y, seg_sx, seg_sy, seg_lx):
                # L2 allocations for QK and V (per-stage) and output
                qk_l2_bufs = [AllocOp(qk_l2_t, [], []) for _ in range(NS)]
                v_l2_bufs = [AllocOp(v_l2_t, [], []) for _ in range(NS)]
                gp_l2 = AllocOp(gp_l2_t, [], [])

                # L1 allocations passed to herd
                q_saved_bufs = [AllocOp(q_l1_t, [], []) for _ in range(dk_chunks)]
                # K buffer: reused across chunks in the chunk loop.
                # Only one K buffer needed since K is processed one chunk
                # at a time (receive K, matmul, write-back).
                k_saved_bufs = [AllocOp(k_l1_t, [], [])]
                if enable_k_writeback:
                    kwb_l1 = AllocOp(k_l1_t, [], [])
                v_l1 = AllocOp(v_l1_t, [], [])
                g_l1 = AllocOp(g_l1_2d, [], [])
                gp_l1 = AllocOp(gp_l1_t, [], [])
                up_l1 = AllocOp(up_l1_t, [], [])
                sp_l1 = AllocOp(up_l1_t, [], [])
                if causal:
                    ctr_size = 4 if dv_chunks > 1 else 3
                    ctr_t = MemRefType.get([ctr_size], i32, memory_space=l1_space)
                    causal_ctr = AllocOp(ctr_t, [], [])

                c_nq = ConstantOp(index_type, NQ)
                c_ns = ConstantOp(index_type, NS)
                c0_seg = ConstantOp(index_type, 0)
                c_chunks_s = ConstantOp(index_type, chunks_per_stage)

                # ----------------------------------------------------------
                # Per-stage relay: Q tiles, then per-chunk K + K-WB + V.
                # ----------------------------------------------------------
                c_q_relay = ConstantOp(index_type, NQ * dk_chunks)
                c_dk_chunks = ConstantOp(index_type, dk_chunks)

                # Tiling transform for QK2L1 relay
                qk_relay_sizes = [dk_tile // M, lkp // M, M, M]
                qk_relay_strides = [M, dk_tile * M, dk_tile, 1]

                for stage in range(NS):
                    # === Q phase: Q tiles ===
                    for qt_iter in scf_range(0, c_q_relay, 1):
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
                            sizes=qk_relay_sizes,
                            strides=qk_relay_strides,
                        )
                        yield_([])

                    # === Per-chunk: K data, V ===
                    for chunk_iter in scf_range(0, c_chunks_s, 1):
                        # K data relay: QKIn → QK2L1
                        for k_iter in scf_range(0, c_dk_chunks, 1):
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
                                sizes=qk_relay_sizes,
                                strides=qk_relay_strides,
                            )
                            yield_([])

                        # V relay: VIn → V2L1
                        ChannelGet(
                            f"VIn_{stage}",
                            v_l2_bufs[stage].result,
                            indices=[seg_x],
                        )
                        ChannelPut(
                            f"V2L1_{stage}",
                            v_l2_bufs[stage].result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dv_tile // M, lkp // M, M, M],
                            strides=[M, dv_tile * M, dv_tile, 1],
                        )
                        yield_([])

                # Output gather from ty=0 tiles (after K write-back)
                affine_map_col = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_size_q),
                        )
                    ],
                )
                par_out = scf.ForallOp(lower_bounds=[0], upper_bounds=[NQ], steps=[1])
                with InsertionPoint(par_out.body):
                    apply_off = affine_apply(
                        affine_map_col,
                        [par_out.induction_variables[0]],
                    )
                    ChannelGet(
                        "Gp2L2",
                        gp_l2.result,
                        indices=[par_out.induction_variables[0], 0],
                        offsets=[apply_off, 0],
                        sizes=[tile_size_q, dv_tile],
                        strides=[dv_tile, 1],
                    )
                    scf.InParallelOp()

                # Output: L2-to-L3
                ChannelPut("GpOut", gp_l2.result, indices=[seg_x])

                # ----------------------------------------------------------
                # Herd: [NQ, NS]
                # ----------------------------------------------------------
                herd_operands = (
                    q_saved_bufs
                    + k_saved_bufs
                    + [
                        v_l1,
                        g_l1,
                        gp_l1,
                        up_l1,
                        sp_l1,
                        seg_x,
                    ]
                )
                if enable_k_writeback:
                    herd_operands.append(kwb_l1)
                if causal:
                    herd_operands.append(causal_ctr)

                @herd(
                    name="herd_0",
                    sizes=[c_nq, c_ns],
                    operands=herd_operands,
                    link_with="attn_npu2.o",
                )
                def herd_body(tx, ty, hsx, hsy, *all_args):
                    # Unpack: dk_chunks Q bufs, 1 K buf (reused per chunk),
                    #         v, g, gp, up, sp, seg_x, [kwb_buf], [causal_ctr]
                    q_bufs = list(all_args[:dk_chunks])
                    qk_tmp = all_args[dk_chunks]  # single K/QK temp buffer
                    base = dk_chunks + 1
                    v = all_args[base]
                    g = all_args[base + 1]
                    gp = all_args[base + 2]
                    up_buf = all_args[base + 3]
                    sp_buf = all_args[base + 4]
                    h_seg_x = all_args[base + 5]
                    next_idx = base + 6
                    kwb_buf = None
                    if enable_k_writeback:
                        kwb_buf = all_args[next_idx]
                        next_idx += 1
                    counter_buf = all_args[next_idx] if causal else None

                    # Precompute affine sets for per-stage dispatch
                    s0 = AffineSymbolExpr.get(0)
                    s1 = AffineSymbolExpr.get(1)
                    c_ns_m1 = AffineConstantExpr.get(NS - 1)
                    stage_sets = []
                    for s in range(NS):
                        cs = AffineConstantExpr.get(s)
                        stage_sets.append(
                            IntegerSet.get(
                                0,
                                2,
                                [s0, s1 - cs],
                                [False, True],
                            )
                        )

                    # === INIT PHASE ===
                    CallOp([], "zero_fill_gp_bf16", [gp])
                    CallOp([], "zero_fill_sp_bf16", [sp_buf])
                    CallOp([], "neg_inf_fill_up_bf16", [up_buf])

                    # === CAUSAL COUNTER INIT ===
                    if causal:
                        c0_ctr = ConstantOp(index_type, 0)
                        c1_ctr = ConstantOp(index_type, 1)
                        c2_ctr = ConstantOp(index_type, 2)
                        c3_ctr = ConstantOp(index_type, 3) if dv_chunks > 1 else None
                        boot_flag = load(counter_buf, [c1_ctr])
                        is_first = arith.CmpIOp(
                            arith.CmpIPredicate.eq,
                            boot_flag,
                            ConstantOp(i32, 0),
                        )
                        if_first = scf.IfOp(is_first)
                        with InsertionPoint(if_first.then_block):
                            store(ConstantOp(i32, 0), counter_buf, [c0_ctr])
                            store(ConstantOp(i32, 1), counter_buf, [c1_ctr])
                            store(ConstantOp(i32, 0), counter_buf, [c2_ctr])
                            if dv_chunks > 1:
                                store(ConstantOp(i32, 0), counter_buf, [c3_ctr])
                            scf.YieldOp([])

                    # === Q SELECTIVE CAPTURE ===
                    # Phase 1: Receive all Q data tiles, copy to saved bufs.
                    for qt in range(NQ):
                        for dk_c in range(dk_chunks):
                            # Receive Q tile → qk_tmp
                            for s in range(NS):
                                if_qk_q = affine.AffineIfOp(
                                    stage_sets[s],
                                    cond_operands=[tx, ty],
                                )
                                with InsertionPoint(if_qk_q.then_block):
                                    ChannelGet(
                                        f"QK2L1_{s}",
                                        qk_tmp,
                                        indices=[h_seg_x, ty, tx],
                                    )
                                    affine.AffineYieldOp([])
                            # Copy Q to saved buffer if tx==qt
                            cmp = arith.CmpIOp(
                                arith.CmpIPredicate.eq,
                                arith.IndexCastOp(i32, tx),
                                arith.ConstantOp(i32, qt),
                            )
                            if_cap = scf.IfOp(cmp)
                            with InsertionPoint(if_cap.then_block):
                                CallOp([], "copy_tile", [qk_tmp, q_bufs[dk_c]])
                                scf.YieldOp([])

                    # === CHUNK LOOP: K receive, matmul, V, softmax ===
                    # K tiles are received one chunk at a time (not bulk).
                    c_chunks_h = ConstantOp(index_type, chunks_per_stage)
                    for chunk_iter in scf_range(0, c_chunks_h, 1):
                        # 1. Zero fill G
                        g1d = CollapseShapeOp(g_l1_1d, g, [[0, 1]])
                        CallOp([], "zero_fill_g_bf16", [g1d])

                        # 2. Receive K tile(s) + matmul
                        for dk_c in range(dk_chunks):
                            # Receive K → qk_tmp
                            for s in range(NS):
                                if_qk_k = affine.AffineIfOp(
                                    stage_sets[s],
                                    cond_operands=[tx, ty],
                                )
                                with InsertionPoint(if_qk_k.then_block):
                                    ChannelGet(
                                        f"QK2L1_{s}",
                                        qk_tmp,
                                        indices=[h_seg_x, ty, tx],
                                    )
                                    affine.AffineYieldOp([])

                            # Matmul Q @ K → G
                            CallOp(
                                [],
                                "matmul_a_b_bf16",
                                [q_bufs[dk_c], qk_tmp, g1d],
                            )

                            # K write-back via CacheWB (tx=0 only)
                            # Copy K to staging buffer, then send via CacheWB.
                            # This is the first BD in the 2-BD rotation.
                            if enable_cache_writeback and kwb_buf is not None:
                                cmp_tx0 = arith.CmpIOp(
                                    arith.CmpIPredicate.eq,
                                    arith.IndexCastOp(i32, tx),
                                    arith.ConstantOp(i32, 0),
                                )
                                if_tx0 = scf.IfOp(cmp_tx0)
                                with InsertionPoint(if_tx0.then_block):
                                    CallOp([], "copy_tile", [qk_tmp, kwb_buf])
                                    for s in range(NS):
                                        if_kwb = affine.AffineIfOp(
                                            stage_sets[s],
                                            cond_operands=[tx, ty],
                                        )
                                        with InsertionPoint(if_kwb.then_block):
                                            ChannelPut(
                                                "CacheWB",
                                                kwb_buf,
                                                indices=[
                                                    h_seg_x,
                                                    ConstantOp(index_type, s),
                                                    tx,
                                                ],
                                                offsets=[0, 0, 0, 0],
                                                sizes=[
                                                    lkp // M,
                                                    M,
                                                    dk_tile // M,
                                                    M,
                                                ],
                                                strides=[
                                                    M * M,
                                                    M,
                                                    lkp * M,
                                                    1,
                                                ],
                                            )
                                            affine.AffineYieldOp([])
                                    scf.YieldOp([])

                        # 3. V get via affine.if per stage
                        for s in range(NS):
                            if_v = affine.AffineIfOp(
                                stage_sets[s],
                                cond_operands=[tx, ty],
                            )
                            with InsertionPoint(if_v.then_block):
                                ChannelGet(
                                    f"V2L1_{s}",
                                    v,
                                    indices=[h_seg_x, ty, tx],
                                )
                                affine.AffineYieldOp([])

                        # 4b. Apply causal mask
                        if causal:
                            c_cps_i32 = ConstantOp(i32, chunks_per_stage)
                            ty_i32 = arith.IndexCastOp(i32, ty).result
                            chunk_i32 = arith.IndexCastOp(
                                i32,
                                chunk_iter,
                            ).result
                            kv_base = arith.MulIOp(ty_i32, c_cps_i32)
                            kv_block = arith.AddIOp(
                                kv_base.result,
                                chunk_i32,
                            )
                            q_base = load(counter_buf, [c0_ctr])
                            tx_i32 = arith.IndexCastOp(i32, tx).result
                            q_block = arith.AddIOp(q_base, tx_i32)
                            CallOp(
                                [],
                                "apply_causal_mask",
                                [g, q_block.result, kv_block.result],
                            )

                        # 5. Softmax + accumulate
                        s_tmp = AllocOp(up_l1_t, [], [])
                        r_tmp = AllocOp(up_l1_t, [], [])
                        CallOp(
                            [],
                            "fused_softmax",
                            [g1d, up_buf, s_tmp.result, r_tmp.result],
                        )
                        CallOp([], "mul_r_gp", [r_tmp.result, gp])
                        CallOp([], "matmul_g_b_bf16", [g1d, v, gp])

                        # V write-back via CacheWB (tx=0 only)
                        # Copy V to kwb_buf staging buffer (same buffer as
                        # K writeback) to avoid DMA race with V2L1 receive.
                        # Both K and V use the same buffer → single BD →
                        # single lock → proper serialization.
                        if enable_cache_writeback and kwb_buf is not None:
                            cmp_tx0_v = arith.CmpIOp(
                                arith.CmpIPredicate.eq,
                                arith.IndexCastOp(i32, tx),
                                arith.ConstantOp(i32, 0),
                            )
                            if_tx0_v = scf.IfOp(cmp_tx0_v)
                            with InsertionPoint(if_tx0_v.then_block):
                                CallOp([], "copy_tile", [v, kwb_buf])
                                for s in range(NS):
                                    if_vwb = affine.AffineIfOp(
                                        stage_sets[s],
                                        cond_operands=[tx, ty],
                                    )
                                    with InsertionPoint(if_vwb.then_block):
                                        ChannelPut(
                                            "CacheWB",
                                            kwb_buf,
                                            indices=[
                                                h_seg_x,
                                                ConstantOp(index_type, s),
                                                tx,
                                            ],
                                            offsets=[0, 0, 0, 0],
                                            sizes=[
                                                lkp // M,
                                                M,
                                                dv_tile // M,
                                                M,
                                            ],
                                            strides=[
                                                M * M,
                                                M,
                                                lkp * M,
                                                1,
                                            ],
                                        )
                                        affine.AffineYieldOp([])
                                scf.YieldOp([])

                        c0_i32 = ConstantOp(i32, 0)
                        CallOp(
                            [],
                            "accum_sp_r_s",
                            [sp_buf, r_tmp.result, s_tmp.result],
                        )
                        CallOp(
                            [],
                            "vector_copy_32elems",
                            [c0_i32, s_tmp.result, sp_buf],
                        )
                        DeallocOp(s_tmp)
                        DeallocOp(r_tmp)
                        yield_([])

                    # === CASCADE MERGE ===
                    set_first_stage = IntegerSet.get(
                        0, 2, [s0, s1 - c_ns_m1], [False, True]
                    )
                    set_middle_stage = IntegerSet.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(s1, AffineConstantExpr.get(-1)),
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NS - 2),
                                AffineExpr.get_mul(s1, AffineConstantExpr.get(-1)),
                            ),
                            s0,
                            AffineExpr.get_add(
                                AffineConstantExpr.get(NQ - 1),
                                AffineExpr.get_mul(s0, AffineConstantExpr.get(-1)),
                            ),
                        ],
                        [False, False, False, False],
                    )
                    c1_h = ConstantOp(index_type, 1)

                    # Last stage (ty == NS-1): send cascade down
                    if_last = affine.AffineIfOp(
                        set_first_stage,
                        cond_operands=[tx, ty],
                        has_else=True,
                    )
                    with InsertionPoint(if_last.then_block):
                        subi_l = arith.SubIOp(ty, c1_h)
                        ChannelPut("cascade_gp", gp, indices=[tx, subi_l])
                        ChannelPut("cascade_up", up_buf, indices=[tx, subi_l])
                        ChannelPut("cascade_sp", sp_buf, indices=[tx, subi_l])
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_last.else_block):
                        if_mid = affine.AffineIfOp(
                            set_middle_stage,
                            cond_operands=[tx, ty],
                            has_else=True,
                        )
                        with InsertionPoint(if_mid.then_block):
                            gp_c = AllocOp(gp_l1_t, [], [])
                            up_c = AllocOp(up_l1_t, [], [])
                            sp_c = AllocOp(up_l1_t, [], [])
                            ChannelGet("cascade_gp", gp_c.result, indices=[tx, ty])
                            ChannelGet("cascade_up", up_c.result, indices=[tx, ty])
                            ChannelGet("cascade_sp", sp_c.result, indices=[tx, ty])
                            up_s = AllocOp(up_l1_t, [], [])
                            c0m = ConstantOp(i32, 0)
                            CallOp(
                                [], "vector_copy_32elems", [c0m, up_buf, up_s.result]
                            )
                            CallOp([], "maximum_up_u_bf16", [up_c.result, up_buf])
                            rc = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u", [up_c.result, up_buf, rc.result]
                            )
                            rl = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u", [up_s.result, up_buf, rl.result]
                            )
                            CallOp([], "mul_r_gp", [rc.result, gp_c.result])
                            CallOp([], "mul_r_gp", [rl.result, gp])
                            CallOp([], "add_gp_g", [gp, gp_c.result])
                            st = AllocOp(up_l1_t, [], [])
                            CallOp([], "zero_fill_sp_bf16", [st.result])
                            CallOp(
                                [], "accum_sp_r_s", [sp_c.result, rc.result, st.result]
                            )
                            CallOp([], "accum_sp_r_s", [sp_buf, rl.result, st.result])
                            CallOp(
                                [], "vector_copy_32elems", [c0m, st.result, sp_c.result]
                            )
                            subi_m = arith.SubIOp(ty, c1_h)
                            ChannelPut("cascade_gp", gp_c.result, indices=[tx, subi_m])
                            ChannelPut("cascade_up", up_buf, indices=[tx, subi_m])
                            ChannelPut("cascade_sp", sp_c.result, indices=[tx, subi_m])
                            DeallocOp(gp_c)
                            DeallocOp(up_c)
                            DeallocOp(sp_c)
                            DeallocOp(up_s)
                            DeallocOp(rc)
                            DeallocOp(rl)
                            DeallocOp(st)
                            affine.AffineYieldOp([])

                        with InsertionPoint(if_mid.else_block):
                            # First stage (ty == 0): cascade in, merge, div, output
                            gp_c2 = AllocOp(gp_l1_t, [], [])
                            up_c2 = AllocOp(up_l1_t, [], [])
                            sp_c2 = AllocOp(up_l1_t, [], [])
                            ChannelGet("cascade_gp", gp_c2.result, indices=[tx, ty])
                            ChannelGet("cascade_up", up_c2.result, indices=[tx, ty])
                            ChannelGet("cascade_sp", sp_c2.result, indices=[tx, ty])
                            up_s2 = AllocOp(up_l1_t, [], [])
                            c0f = ConstantOp(i32, 0)
                            CallOp(
                                [], "vector_copy_32elems", [c0f, up_buf, up_s2.result]
                            )
                            CallOp([], "maximum_up_u_bf16", [up_c2.result, up_buf])
                            rc2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u", [up_c2.result, up_buf, rc2.result]
                            )
                            rl2 = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [], "exp_up_minus_u", [up_s2.result, up_buf, rl2.result]
                            )
                            CallOp([], "mul_r_gp", [rc2.result, gp_c2.result])
                            CallOp([], "mul_r_gp", [rl2.result, gp])
                            CallOp([], "add_gp_g", [gp, gp_c2.result])
                            st2 = AllocOp(up_l1_t, [], [])
                            CallOp([], "zero_fill_sp_bf16", [st2.result])
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp_c2.result, rc2.result, st2.result],
                            )
                            CallOp([], "accum_sp_r_s", [sp_buf, rl2.result, st2.result])
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [c0f, st2.result, sp_c2.result],
                            )
                            CallOp([], "div_gp_sp", [sp_c2.result, gp_c2.result])
                            c0_out = ConstantOp(index_type, 0)
                            ChannelPut(
                                "Gp2L2",
                                gp_c2.result,
                                indices=[tx, c0_out],
                                offsets=[0, 0, 0, 0],
                                sizes=[
                                    tile_size_q // M,
                                    M,
                                    dv_tile // M,
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

                        def _emit_counter_increment():
                            head_cur = load(counter_buf, [c2_ctr])
                            c1_i32_inc = ConstantOp(i32, 1)
                            head_next = arith.AddIOp(head_cur, c1_i32_inc)
                            total_hg = ConstantOp(i32, num_head_groups)
                            wrapped = arith.CmpIOp(
                                arith.CmpIPredicate.sge,
                                head_next.result,
                                total_hg,
                            )
                            if_wrap = scf.IfOp(wrapped)
                            with InsertionPoint(if_wrap.then_block):
                                q_cur = load(counter_buf, [c0_ctr])
                                c_nq_i32 = ConstantOp(i32, NQ)
                                q_next = arith.AddIOp(q_cur, c_nq_i32)
                                store(q_next.result, counter_buf, [c0_ctr])
                                store(ConstantOp(i32, 0), counter_buf, [c2_ctr])
                                scf.YieldOp([])
                            not_wrapped = arith.CmpIOp(
                                arith.CmpIPredicate.slt,
                                head_next.result,
                                total_hg,
                            )
                            if_no_wrap = scf.IfOp(not_wrapped)
                            with InsertionPoint(if_no_wrap.then_block):
                                store(head_next.result, counter_buf, [c2_ctr])
                                scf.YieldOp([])

                        if dv_chunks > 1:
                            dv_iter_cur = load(counter_buf, [c3_ctr])
                            c_dv_last_i32 = ConstantOp(i32, dv_chunks - 1)
                            is_last_dv = arith.CmpIOp(
                                arith.CmpIPredicate.sge,
                                dv_iter_cur,
                                c_dv_last_i32,
                            )
                            if_last_dv = scf.IfOp(is_last_dv)
                            with InsertionPoint(if_last_dv.then_block):
                                _emit_counter_increment()
                                store(
                                    ConstantOp(i32, 0),
                                    counter_buf,
                                    [c3_ctr],
                                )
                                scf.YieldOp([])
                            not_last_dv = arith.CmpIOp(
                                arith.CmpIPredicate.slt,
                                dv_iter_cur,
                                c_dv_last_i32,
                            )
                            if_not_last = scf.IfOp(not_last_dv)
                            with InsertionPoint(if_not_last.then_block):
                                c1_i32_dv = ConstantOp(i32, 1)
                                dv_next = arith.AddIOp(dv_iter_cur, c1_i32_dv)
                                store(
                                    dv_next.result,
                                    counter_buf,
                                    [c3_ctr],
                                )
                                scf.YieldOp([])
                        else:
                            _emit_counter_increment()

                # Deallocs for segment-level buffers
                for q_buf in q_saved_bufs:
                    DeallocOp(q_buf)
                DeallocOp(k_saved_bufs[0])
                if enable_k_writeback:
                    DeallocOp(kwb_l1)
                DeallocOp(v_l1)
                DeallocOp(g_l1)
                DeallocOp(gp_l1)
                DeallocOp(up_l1)
                DeallocOp(sp_l1)
                for stage in range(NS):
                    DeallocOp(v_l2_bufs[stage])
                for stage in range(NS):
                    DeallocOp(qk_l2_bufs[stage])
                DeallocOp(gp_l2)
                if causal:
                    DeallocOp(causal_ctr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="attn_npu2.py",
        description="Fused RoPE + flash attention + KV cache write-back",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print MLIR module and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--lk",
        type=int,
        default=512,
        help="Total K/V sequence length (default: 512)",
    )
    parser.add_argument(
        "--lq",
        type=int,
        default=512,
        help="Total Q sequence length (default: 512)",
    )
    parser.add_argument(
        "--lqp",
        type=int,
        default=256,
        help="Q chunk size per launch iteration (default: 256)",
    )
    parser.add_argument(
        "--lkp",
        type=int,
        default=64,
        help="K/V chunk size per tile (default: 64)",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=64,
        help="Key dimension (default: 64). Must be divisible by lkp.",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=64,
        help="Value dimension (default: 64). Must be divisible by lkp.",
    )
    parser.add_argument(
        "--num-cascade-stages",
        type=int,
        default=4,
        help="Number of cascade pipeline stages (default: 4)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (default: num_heads for MHA, < num_heads for GQA)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="compile-and-run",
        choices=["compile-only", "compile-and-run"],
        help="Compilation mode (default: compile-and-run)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="elf",
        choices=["xclbin", "elf"],
        help="Output format (default: elf)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal masking (autoregressive attention)",
    )
    parser.add_argument(
        "--no-k-writeback",
        action="store_true",
        help="Disable K cache write-back (for debugging)",
    )
    parser.add_argument(
        "--no-v-writeback",
        action="store_true",
        help="Disable V cache write-back (for debugging)",
    )
    parser.add_argument(
        "--no-rope",
        action="store_true",
        help="Disable RoPE application (for debugging data flow)",
    )
    args = parser.parse_args()

    lk = args.lk
    lkp = args.lkp
    lq = args.lq
    lqp = args.lqp
    dk = args.dk
    dv = args.dv
    num_cascade_stages = args.num_cascade_stages
    num_q_tiles = 4
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads
    causal = args.causal
    enable_k_writeback = not args.no_k_writeback
    enable_v_writeback = not args.no_v_writeback
    enable_rope = not args.no_rope
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
        enable_k_writeback=enable_k_writeback,
        enable_v_writeback=enable_v_writeback,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner
    from air.backend.xrt import XRTBackend
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16
    rng = np.random.default_rng(42)
    val_range = 4.0
    input_q = rng.uniform(0, val_range, (num_heads, lq, dk)).astype(INPUT_DATATYPE)
    input_k = rng.uniform(0, val_range, (num_kv_heads, lk, dk)).astype(INPUT_DATATYPE)
    input_v_orig = rng.uniform(0, val_range, (num_kv_heads, lk, dv)).astype(
        INPUT_DATATYPE
    )
    # Transpose V to [num_kv_heads * dv_chunks, lk, dv_tile] for contiguous access
    dv_chunks_host = dv // lkp
    input_v = (
        input_v_orig.reshape(num_kv_heads, lk, dv_chunks_host, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_kv_heads * dv_chunks_host, lk, lkp)
        .copy()
    )

    # ================================================================
    # Generate RoPE LUT: interleaved [cos, sin, cos, sin, ...] per row
    # ================================================================
    THETA = 10000.0
    rope_seq_len = max(lq, lk)
    rope_lut_f32 = np.zeros((rope_seq_len, dk), dtype=np.float32)
    for r in range(rope_seq_len):
        for i in range(dk // 2):
            freq = 1.0 / (THETA ** (2.0 * i / dk))
            angle = r * freq
            rope_lut_f32[r, 2 * i] = np.cos(angle)
            rope_lut_f32[r, 2 * i + 1] = np.sin(angle)
    rope_lut_input = rope_lut_f32.astype(INPUT_DATATYPE)

    # ================================================================
    # Apply RoPE to Q and K for reference computation
    # ================================================================
    def apply_rope_ref(x, lut_slice):
        """Apply RoPE rotation. x: [seq, dk], lut_slice: [seq, dk]."""
        x_f = x.astype(np.float32)
        lut_f = lut_slice.astype(np.float32)
        x_even = x_f[:, 0::2]
        x_odd = x_f[:, 1::2]
        cos_v = lut_f[:, 0::2]
        sin_v = lut_f[:, 1::2]
        out = np.zeros_like(x_f)
        out[:, 0::2] = x_even * cos_v - x_odd * sin_v
        out[:, 1::2] = x_even * sin_v + x_odd * cos_v
        return out.astype(x.dtype)

    # RoPE'd Q and K (host-side pre-rotation when enabled)
    if enable_rope:
        q_roped = np.zeros_like(input_q)
        for h in range(num_heads):
            q_roped[h] = apply_rope_ref(input_q[h], rope_lut_input[:lq, :dk])
        k_roped = np.zeros_like(input_k)
        for kv_h in range(num_kv_heads):
            k_roped[kv_h] = apply_rope_ref(input_k[kv_h], rope_lut_input[:lk, :dk])
    else:
        q_roped = input_q.copy()
        k_roped = input_k.copy()

    # NPU receives pre-RoPE'd Q and K when RoPE is enabled
    npu_input_q = q_roped
    npu_input_k = k_roped

    # Reference attention using RoPE'd Q and K
    inv_sqrt_dk = 1.0 / sqrt(dk)
    sdpa_output = np.zeros((num_heads, lq, dv), dtype=OUTPUT_DATATYPE)
    for h in range(num_heads):
        kv_h = h // gqa_group_size
        Qf = q_roped[h].astype(np.float32)
        Kf = k_roped[kv_h].astype(np.float32)
        Vf = input_v_orig[kv_h].astype(np.float32)
        scores = Qf @ Kf.T * inv_sqrt_dk
        if causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        P = np.exp(scores - mx)
        P = P / np.sum(P, axis=-1, keepdims=True)
        sdpa_output[h] = (P @ Vf).astype(OUTPUT_DATATYPE)

    # Transpose expected output to match transposed L3 layout
    sdpa_output_transposed = (
        sdpa_output.reshape(num_heads, lq, dv_chunks_host, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_heads * dv_chunks_host, lq, lkp)
        .copy()
    )

    tiling = [1, 1, 1] if dv_chunks_host > 1 else [1, 1]
    backend_opts = dict(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=tiling,
        output_format=args.output_format,
        instance_name="attention_bf16",
        target_device="npu2",
    )

    # Build expected KV cache (interleaved: [K_c0, V_c0, K_c1, V_c1, ...])
    enable_cache_writeback = enable_k_writeback or enable_v_writeback
    if enable_cache_writeback:
        num_chunks_host = lk // lkp
        kv_cache_size = num_kv_heads * num_chunks_host * 2 * lkp * dk
        expected_kv_cache = np.zeros(kv_cache_size, dtype=INPUT_DATATYPE)
        for h in range(num_kv_heads):
            for c in range(num_chunks_host):
                k_off = h * num_chunks_host * 2 * lkp * dk + c * 2 * lkp * dk
                v_off = k_off + lkp * dk
                # K chunk: RoPE'd K data, row-major [lkp, dk]
                expected_kv_cache[k_off : k_off + lkp * dk] = k_roped[
                    h, c * lkp : (c + 1) * lkp, :
                ].flatten()
                # V chunk: raw V data in transposed tile format [lkp, dv_tile]
                # V was transposed to [num_kv_heads * dv_chunks, lk, dv_tile]
                # For dv_chunks=1, this is just [num_kv_heads, lk, dv_tile]
                expected_kv_cache[v_off : v_off + lkp * dk] = input_v[
                    h, c * lkp : (c + 1) * lkp, :
                ].flatten()
    else:
        expected_k_cache = k_roped.copy()

    if args.compile_mode == "compile-and-run":
        import filelock, tempfile

        backend = XRTBackend(**backend_opts)
        if enable_cache_writeback:
            kv_cache_placeholder = np.zeros(kv_cache_size, dtype=INPUT_DATATYPE)
            expected_outputs = [sdpa_output_transposed, kv_cache_placeholder]
        else:
            v_cache_placeholder = np.zeros_like(input_v)
            expected_outputs = [
                sdpa_output_transposed,
                k_roped.copy(),
                v_cache_placeholder,
            ]
        output_placeholders = [np.zeros(o.shape, o.dtype) for o in expected_outputs]
        # NPU receives pre-RoPE'd Q and K (RoPE applied on host when enabled)
        input_list = [npu_input_q, npu_input_k, input_v]
        num_inputs = len(input_list)
        expanded_inputs = input_list + output_placeholders

        compiled_module = backend.compile(mlir_module)
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            module_function = backend.load(compiled_module)
            actual_outputs = module_function(*expanded_inputs)
        backend.unload()

        # Remove input slots
        actual_outputs = list(actual_outputs[num_inputs:])

        failed = False

        # --- Output 0: Attention (SDPA) output ---
        attn_actual = (
            actual_outputs[0].reshape(sdpa_output_transposed.shape).astype(np.float32)
        )
        attn_expected = sdpa_output_transposed.astype(np.float32)
        attn_corr = float(
            np.corrcoef(attn_actual.flatten(), attn_expected.flatten())[0, 1]
        )
        attn_close = np.isclose(attn_actual, attn_expected, rtol=0.04, atol=0.15)
        attn_mismatches = int(np.sum(~attn_close))
        attn_total = attn_actual.size
        attn_pct = attn_mismatches / attn_total * 100
        print(
            f"Output 0 (attention): correlation={attn_corr:.4f}, "
            f"mismatches={attn_mismatches}/{attn_total} ({attn_pct:.2f}%)"
        )
        if attn_corr < 0.94:
            print(f"FAIL: Output 0 correlation {attn_corr:.4f} below 0.94")
            failed = True
        else:
            print(
                f"PASS: Output 0 correlation {attn_corr:.4f} >= 0.94 "
                "(BFP16 emulation tolerance)"
            )

        # --- Output 1: KV cache (interleaved [K_c0, V_c0, K_c1, V_c1, ...]) ---
        if enable_cache_writeback:
            kv_actual = actual_outputs[1].flatten()
            kv_mismatches = int(np.sum(kv_actual != expected_kv_cache))
            kv_total = kv_actual.size
            chunk_size = lkp * dk
            num_chunks_total = num_kv_heads * (lk // lkp)
            print(f"Output 1 (KV cache): mismatches={kv_mismatches}/{kv_total}")
            if kv_mismatches > 0:
                print(f"FAIL: KV cache has {kv_mismatches} mismatches")
                for h in range(num_kv_heads):
                    for c in range(lk // lkp):
                        k_off = h * (lk // lkp) * 2 * chunk_size + c * 2 * chunk_size
                        v_off = k_off + chunk_size
                        k_m = int(
                            np.sum(
                                kv_actual[k_off : k_off + chunk_size]
                                == expected_kv_cache[k_off : k_off + chunk_size]
                            )
                        )
                        v_m = int(
                            np.sum(
                                kv_actual[v_off : v_off + chunk_size]
                                == expected_kv_cache[v_off : v_off + chunk_size]
                            )
                        )
                        k_label = (
                            "EXACT" if k_m == chunk_size else f"{k_m}/{chunk_size}"
                        )
                        v_label = (
                            "EXACT" if v_m == chunk_size else f"{v_m}/{chunk_size}"
                        )
                        if k_m < chunk_size or v_m < chunk_size:
                            print(f"  h={h} c={c}: K={k_label}, V={v_label}")
                failed = True
            else:
                print("PASS: KV cache matches expected (K=RoPE'd K, V=raw V)")
        else:
            print("Output 1 (KV cache): SKIPPED (cache write-back disabled)")

        if failed:
            print("OVERALL: FAILED")
            exit(-1)
        else:
            print("OVERALL: PASSED")
            exit(0)
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(**backend_opts)
        module_function = backend.compile(mlir_module)
        print("Compilation complete.")
