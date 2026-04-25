# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode-phase flash attention on NPU2 (AIE2P).

Implements a drop-in replacement for decode_attention_cpu() using matvec-style
kernels (group_size too small for 8x8 tiled matmul).

Architecture:
  Herd: [num_kv_parallel, num_cascade_stages]  (e.g. [4,4] = 16 tiles)
    tx = KV head index within the current launch group
    ty = cascade stage (partitions the KV sequence)
  Launch iterations: num_kv_heads // num_kv_parallel

Per tile (tx, ty):
  - DMA pos [1] from L3 via QKV_dec (once)
  - DMA Q [group_size, dk] from L3 via QKV_dec (once)
  - Init output, max, sum
  - For each of chunks_per_stage K/V chunks:
      DMA K [lkp, dk] and V [lkp, dv] from L3
      compute_qk_scores  -> scores[group_size, lkp]
      apply_decode_mask  -> set scores to -inf for positions > current_pos
      online softmax update (max, exp, sum, rescale, PV accumulate)
  - Cascade merge (north-to-south: ty=NS-1 first, ty=0 last)
  - ty==0: decode_div_output, write result to L2, then segment DMA to L3

Default parameters:
  lk=2048, lkp=64, dk=64, dv=64, group_size=4,
  num_kv_heads=4, num_kv_parallel=4, num_cascade_stages=4

DMA channel budget per tile (2 S2MM + 2 MM2S):
  S2MM 0: pos (once), Q (once), then K and V chunks interleaved in loop
  MM2S 0: output L1->L2 (ty==0 only)
  Cascade put/get use cascade stream hardware (separate from DMA channels)
  Note: K and V share S2MM 0 via QKV_dec to avoid S2MM-1 infinite-BD-loop
  deadlock and to keep shim MM2S channels ≤ NKV*NS ≤ 16 total.
  Note: num_kv_heads must equal num_kv_parallel (kv_head_groups=1) to stay
  within the 16-BD-per-shim-column budget (2 MM2S × 4 BDs + 1 S2MM = 9 ≤ 16).

Design deviation from ticket T003:
  T003 prescribed two separate channels (QK_dec for Q+K on S2MM 0, V_dec for
  V on S2MM 1). The two-channel approach was implemented (commit be5a4ec7) but
  produced 30.86% output mismatch due to S2MM-1 BD-chain ordering deadlock.
  The single QKV_dec channel (this implementation) avoids the deadlock by
  routing all inputs through S2MM 0, with segment ChannelPuts Python-loop-
  unrolled to guarantee K₀,V₀,K₁,V₁,... ordering that matches the tile BD
  chain. This produces O(chunks_per_stage × NKV × NS) MLIR ops; for lk=2048
  with defaults this is 8×4×4=128 K+V ops (manageable). Users targeting
  lk > 4096 should expect proportionally larger MLIR modules.
  TODO: Investigate restoring S2MM-1 for V traffic (would allow larger lk
  with fewer unrolled ops) once the BD-chain ordering issue is resolved.

Output path: L1 -> L2 (herd, ty==0) -> L3 (segment, after herd)
"""

import argparse
from math import sqrt

import numpy as np

import air
from air.ir import *
from air.dialects.air import *
from air.dialects.air import channel
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_ as scf_range, yield_
from air.dialects import scf, affine, arith
from air.dialects.affine import apply as affine_apply


@module_builder
def build_module(
    lk=2048,
    lkp=64,
    dk=64,
    dv=64,
    group_size=4,
    num_kv_heads=4,
    num_kv_parallel=4,
    num_cascade_stages=4,
):
    """Build decode flash attention module.

    Args:
        lk: Maximum KV cache sequence length (default: 2048)
        lkp: K/V chunk size per cascade stage chunk (default: 64)
        dk: Key head dimension (default: 64)
        dv: Value head dimension (default: 64)
        group_size: Q heads per KV head (= num_heads // num_kv_heads, default: 4)
        num_kv_heads: Number of KV heads (default: 8)
        num_kv_parallel: KV heads processed in parallel / herd x-dim (default: 4)
        num_cascade_stages: Cascade depth / herd y-dim (NS, default: 4)
    """
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert (
        lk % (lkp * num_cascade_stages) == 0
    ), f"lk ({lk}) must be divisible by lkp*NS ({lkp * num_cascade_stages})"
    assert num_kv_heads % num_kv_parallel == 0, (
        f"num_kv_heads ({num_kv_heads}) must be divisible by "
        f"num_kv_parallel ({num_kv_parallel})"
    )

    num_heads = group_size * num_kv_heads
    kv_head_groups = num_kv_heads // num_kv_parallel
    assert kv_head_groups == 1, (
        f"Decode requires kv_head_groups==1 (num_kv_heads==num_kv_parallel) "
        f"to stay within the 16-BD-per-shim-column budget. "
        f"Got kv_head_groups={kv_head_groups}"
    )
    NS = num_cascade_stages
    NKV = num_kv_parallel
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // NS

    # Cascade alignment: 512-bit bus = 32 bfloat16 elements.
    max_sum_buf_size = max(group_size, 32)

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    # Memory spaces
    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # L1 MemRefTypes
    q_l1_t = MemRefType.get([group_size, dk], bf16, memory_space=l1_space)
    k_l1_t = MemRefType.get([lkp, dk], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv], bf16, memory_space=l1_space)
    scores_l1_t = MemRefType.get([group_size, lkp], bf16, memory_space=l1_space)
    out_l1_t = MemRefType.get([group_size, dv], bf16, memory_space=l1_space)
    ms_l1_t = MemRefType.get([max_sum_buf_size], bf16, memory_space=l1_space)
    pos_l1_t = MemRefType.get([1], i32, memory_space=l1_space)
    # L2 MemRefTypes
    # Output staging: [NKV, group_size, dv] — one row per tx (filled by ty==0 tiles)
    out_l2_t = MemRefType.get([NKV, group_size, dv], bf16, memory_space=l2_space)

    # L3 MemRefTypes
    q_l3_t = MemRefType.get([num_heads, dk], bf16)
    k_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    v_l3_t = MemRefType.get([num_kv_heads, lk, dv], bf16)
    pos_l3_t = MemRefType.get([1], i32)
    out_l3_t = MemRefType.get([num_heads, dv], bf16)

    # ------------------------------------------------------------------
    # External function declarations (link with decode kernel .o)
    # ------------------------------------------------------------------
    def external_func(name, inputs, link_with=None):
        func_type = FunctionType.get(inputs, [])
        func = FuncOp(name=name, type=func_type, visibility="private")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    kobj = "attn_decode_npu2.o"
    external_func("decode_zero_output", [out_l1_t], link_with=kobj)
    external_func("decode_zero_sum", [ms_l1_t], link_with=kobj)
    external_func("decode_neg_inf_max", [ms_l1_t], link_with=kobj)
    external_func(
        "compute_qk_scores_bf16", [k_l1_t, q_l1_t, scores_l1_t], link_with=kobj
    )
    external_func("apply_decode_mask", [scores_l1_t, i32, i32], link_with=kobj)
    external_func("decode_softmax_max", [scores_l1_t, ms_l1_t], link_with=kobj)
    external_func(
        "decode_softmax_exp",
        [scores_l1_t, ms_l1_t, ms_l1_t, ms_l1_t],
        link_with=kobj,
    )
    external_func("decode_softmax_sum", [scores_l1_t, ms_l1_t, ms_l1_t], link_with=kobj)
    external_func("decode_rescale_output", [ms_l1_t, out_l1_t], link_with=kobj)
    external_func(
        "compute_pv_output_bf16", [scores_l1_t, v_l1_t, out_l1_t], link_with=kobj
    )
    external_func("decode_div_output", [ms_l1_t, out_l1_t], link_with=kobj)
    external_func("decode_cascade_merge_max", [ms_l1_t, ms_l1_t], link_with=kobj)
    external_func("decode_compute_rescale", [ms_l1_t, ms_l1_t, ms_l1_t], link_with=kobj)
    external_func("decode_add_output", [out_l1_t, out_l1_t], link_with=kobj)
    external_func("decode_add_sum", [ms_l1_t, ms_l1_t], link_with=kobj)
    external_func("decode_rescale_sum", [ms_l1_t, ms_l1_t], link_with=kobj)
    external_func("decode_copy_max_sum", [ms_l1_t, ms_l1_t], link_with=kobj)

    # ------------------------------------------------------------------
    # DMA channels: QKV_dec routes pos, Q, then interleaved K+V chunks
    #               (→ S2MM 0 only; avoids S2MM 1 infinite-BD-loop bug
    #               and keeps NKV*NS ≤ 16 total shim MM2S channels),
    #               out_dec routes L1→L2 output (ty==0 tiles only)
    # ------------------------------------------------------------------
    channel("QKV_dec", size=[NKV, NS])
    channel("out_dec", size=[NKV, 1])

    # ------------------------------------------------------------------
    # Cascade channels (north-to-south: ty=NS-1 first, ty=0 last)
    # ------------------------------------------------------------------
    channel("cascade_out_dec", size=[NKV, NS - 1], channel_type="cascade")
    channel("cascade_max_dec", size=[NKV, NS - 1], channel_type="cascade")
    channel("cascade_sum_dec", size=[NKV, NS - 1], channel_type="cascade")

    # ------------------------------------------------------------------
    # Main decode attention function
    # ------------------------------------------------------------------
    @FuncOp.from_py_func(q_l3_t, k_l3_t, v_l3_t, pos_l3_t, out_l3_t)
    def decode_attention_bf16(q_in, k_in, v_in, pos_in, out):
        c_kv_groups = ConstantOp(index_type, kv_head_groups)

        @launch(
            operands=[q_in, k_in, v_in, pos_in, out],
            sizes=[c_kv_groups],
        )
        def launch_body(lx, lsx, q, k, v, pos, gp_out):

            @segment(
                name="decode_seg",
                operands=[lx, q, k, v, pos, gp_out],
            )
            def segment_body(lx_s, q_s, k_s, v_s, pos_s, out_s):
                # ----------------------------------------------------------
                # L2 output staging buffer: each tx tile writes to its row
                # ----------------------------------------------------------
                out_l2 = AllocOp(out_l2_t, [], [])

                # L1 per-tile scratch buffers (shared across tile instances)
                q_l1 = AllocOp(q_l1_t, [], [])
                k_l1 = AllocOp(k_l1_t, [], [])
                v_l1 = AllocOp(v_l1_t, [], [])
                scores_l1 = AllocOp(scores_l1_t, [], [])
                out_l1 = AllocOp(out_l1_t, [], [])
                max_l1 = AllocOp(ms_l1_t, [], [])
                sum_l1 = AllocOp(ms_l1_t, [], [])
                score_max_tmp = AllocOp(ms_l1_t, [], [])
                rescale_tmp = AllocOp(ms_l1_t, [], [])
                # Cascade receive buffers
                out_c = AllocOp(out_l1_t, [], [])
                max_c = AllocOp(ms_l1_t, [], [])
                sum_c = AllocOp(ms_l1_t, [], [])
                # For cascade merge: save old max to compute two rescales
                old_max_save = AllocOp(ms_l1_t, [], [])
                rescale_recv_c = AllocOp(ms_l1_t, [], [])
                rescale_local_c = AllocOp(ms_l1_t, [], [])
                pos_l1 = AllocOp(pos_l1_t, [], [])

                c_nkv = ConstantOp(index_type, NKV)
                c_ns = ConstantOp(index_type, NS)

                # ----------------------------------------------------------
                # Segment ChannelPut loops: feed pos, Q, K, V to herd via QKV_dec.
                #   QKV_dec → S2MM 0: pos (once), Q (once), then per chunk K then V.
                #   Single channel avoids the S2MM-1 infinite-BD-loop deadlock
                #   and keeps total shim MM2S ≤ NKV*NS ≤ 16.
                # ----------------------------------------------------------
                # Affine maps for per-tile (tx_i) offset calculation.
                # kv_h = lx_s * NKV + tx_i
                kv_head_map_seg = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(NKV),
                            ),
                            AffineSymbolExpr.get(1),
                        )
                    ],
                )
                # q_head_base = kv_h * group_size
                q_head_map_seg = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(group_size),
                        )
                    ],
                )
                # --- pos ChannelPuts (once per tile, sent first) ---
                for tx_i in range(NKV):
                    c_tx_i_pos = ConstantOp(index_type, tx_i)
                    for ty_i in range(NS):
                        c_ty_i_pos = ConstantOp(index_type, ty_i)
                        ChannelPut(
                            "QKV_dec",
                            pos_s,
                            offsets=[0],
                            sizes=[1],
                            strides=[1],
                            indices=[c_tx_i_pos, c_ty_i_pos],
                        )

                # --- Q ChannelPuts (once per tile) ---
                for tx_i in range(NKV):
                    c_tx_i = ConstantOp(index_type, tx_i)
                    kv_h_q = affine_apply(kv_head_map_seg, [lx_s, c_tx_i])
                    q_base_seg = affine_apply(q_head_map_seg, [kv_h_q])
                    for ty_i in range(NS):
                        c_ty_i = ConstantOp(index_type, ty_i)
                        ChannelPut(
                            "QKV_dec",
                            q_s,
                            offsets=[q_base_seg, 0],
                            sizes=[group_size, dk],
                            strides=[dk, 1],
                            indices=[c_tx_i, c_ty_i],
                        )

                # --- K and V ChannelPuts interleaved per chunk, per tile ---
                # K then V on QKV_dec (S2MM 0): avoids the V S2MM-1 deadlock.
                # Use Python range (not scf_range) so each chunk becomes a separate
                # DMA task, preserving K0,V0,K1,V1 ordering that matches the tile
                # S2MM BD chain (repeat_count = chunks_per_stage - 1).
                for chunk_i_val in range(chunks_per_stage):
                    for tx_i in range(NKV):
                        c_tx_i = ConstantOp(index_type, tx_i)
                        kv_h_kv = affine_apply(kv_head_map_seg, [lx_s, c_tx_i])
                        for ty_i in range(NS):
                            c_ty_i = ConstantOp(index_type, ty_i)
                            chunk_pos_val = (
                                ty_i * chunks_per_stage * lkp + chunk_i_val * lkp
                            )
                            c_chunk_pos = ConstantOp(index_type, chunk_pos_val)
                            ChannelPut(
                                "QKV_dec",
                                k_s,
                                offsets=[kv_h_kv, c_chunk_pos, 0],
                                sizes=[1, lkp, dk],
                                strides=[lk * dk, dk, 1],
                                indices=[c_tx_i, c_ty_i],
                            )
                            ChannelPut(
                                "QKV_dec",
                                v_s,
                                offsets=[kv_h_kv, c_chunk_pos, 0],
                                sizes=[1, lkp, dv],
                                strides=[lk * dv, dv, 1],
                                indices=[c_tx_i, c_ty_i],
                            )

                @herd(
                    name="decode_herd",
                    sizes=[c_nkv, c_ns],
                    operands=[
                        q_l1,
                        k_l1,
                        v_l1,
                        scores_l1,
                        out_l1,
                        max_l1,
                        sum_l1,
                        score_max_tmp,
                        rescale_tmp,
                        out_c,
                        max_c,
                        sum_c,
                        old_max_save,
                        rescale_recv_c,
                        rescale_local_c,
                        pos_l1,
                    ],
                    link_with="attn_decode_npu2.o",
                )
                def herd_body(
                    tx,
                    ty,
                    hsx,
                    hsy,
                    _q_l1,
                    _k_l1,
                    _v_l1,
                    _scores_l1,
                    _out_l1,
                    _max_l1,
                    _sum_l1,
                    _score_max_tmp,
                    _rescale_tmp,
                    _out_c,
                    _max_c,
                    _sum_c,
                    _old_max_save,
                    _rescale_recv_c,
                    _rescale_local_c,
                    _pos_l1,
                ):
                    # ------------------------------------------------
                    # Compute per-tile stage offset for causal mask.
                    # kv_h offset is handled in segment ChannelPuts.
                    # ty_pos_start = ty * chunks_per_stage * lkp
                    # ------------------------------------------------
                    # ty_pos_start = ty * chunks_per_stage * lkp
                    ty_pos_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(chunks_per_stage * lkp),
                            )
                        ],
                    )
                    ty_pos_start = affine_apply(ty_pos_map, [ty])

                    # ------------------------------------------------
                    # pos [1] L3->L1 via QKV_dec (sent first in segment)
                    # ------------------------------------------------
                    ChannelGet("QKV_dec", _pos_l1, indices=[tx, ty])
                    c0_idx = ConstantOp(index_type, 0)
                    pos_val = load(_pos_l1, [c0_idx])

                    # ------------------------------------------------
                    # Q [group_size, dk] L3->L1 via QKV_dec channel
                    # (segment ChannelPut sends correct slice per tile)
                    # ------------------------------------------------
                    ChannelGet("QKV_dec", _q_l1, indices=[tx, ty])

                    # ------------------------------------------------
                    # Initialize output, max, sum
                    # ------------------------------------------------
                    CallOp([], "decode_zero_output", [_out_l1])
                    CallOp([], "decode_neg_inf_max", [_max_l1])
                    CallOp([], "decode_zero_sum", [_sum_l1])

                    # ------------------------------------------------
                    # Chunk loop: process chunks_per_stage K/V chunks
                    # K then V both via QKV_dec (single S2MM 0 channel)
                    # ------------------------------------------------
                    c_chunks = ConstantOp(index_type, chunks_per_stage)

                    # chunk_pos_map: ty_pos_start + chunk_idx * lkp
                    chunk_pos_map = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(lkp),
                                ),
                            )
                        ],
                    )

                    for chunk_idx in scf_range(0, c_chunks, 1):
                        chunk_pos = affine_apply(
                            chunk_pos_map, [ty_pos_start, chunk_idx]
                        )

                        # K chunk via QKV_dec (S2MM 0)
                        ChannelGet("QKV_dec", _k_l1, indices=[tx, ty])

                        # V chunk via QKV_dec (S2MM 0, same channel as K)
                        ChannelGet("QKV_dec", _v_l1, indices=[tx, ty])

                        # QK scores: scores[h,i] = K[i,:].Q[h,:]
                        CallOp(
                            [],
                            "compute_qk_scores_bf16",
                            [_k_l1, _q_l1, _scores_l1],
                        )

                        # Causal mask for decode
                        chunk_pos_i32 = arith.IndexCastOp(i32, chunk_pos).result
                        CallOp(
                            [],
                            "apply_decode_mask",
                            [_scores_l1, chunk_pos_i32, pos_val],
                        )

                        # Online softmax:
                        #   score_max_tmp = row_max(scores)
                        CallOp(
                            [],
                            "decode_softmax_max",
                            [_scores_l1, _score_max_tmp],
                        )
                        #   rescale_tmp = exp(old_max - new_max)
                        #   scores = exp(scores - new_max)   [in-place]
                        #   max_l1 updated to max(max_l1, score_max_tmp)
                        CallOp(
                            [],
                            "decode_softmax_exp",
                            [_scores_l1, _max_l1, _score_max_tmp, _rescale_tmp],
                        )
                        #   Rescale accumulated output
                        CallOp(
                            [],
                            "decode_rescale_output",
                            [_rescale_tmp, _out_l1],
                        )
                        #   Accumulate P @ V
                        CallOp(
                            [],
                            "compute_pv_output_bf16",
                            [_scores_l1, _v_l1, _out_l1],
                        )
                        #   Update running sum
                        CallOp(
                            [],
                            "decode_softmax_sum",
                            [_scores_l1, _rescale_tmp, _sum_l1],
                        )

                        yield_([])

                    # ------------------------------------------------
                    # Cascade merge (north-to-south)
                    # ty = NS-1 (northernmost): just send cascade south
                    # ty = 1..NS-2 (middle): receive, merge, send
                    # ty = 0 (southernmost): receive, merge, div, output
                    # ------------------------------------------------
                    c1_idx = ConstantOp(index_type, 1)
                    ns_m1 = AffineConstantExpr.get(NS - 1)
                    s0 = AffineSymbolExpr.get(0)
                    s1 = AffineSymbolExpr.get(1)

                    set_first_stage = IntegerSet.get(
                        0, 2, [s0, s1 - ns_m1], [False, True]
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
                                AffineConstantExpr.get(NKV - 1),
                                AffineExpr.get_mul(s0, AffineConstantExpr.get(-1)),
                            ),
                        ],
                        [False, False, False, False],
                    )

                    # ty == NS-1: send cascade south
                    if_last = affine.AffineIfOp(
                        set_first_stage,
                        cond_operands=[tx, ty],
                        has_else=True,
                    )
                    with InsertionPoint(if_last.then_block):
                        subi_l = arith.SubIOp(ty, c1_idx)
                        ChannelPut("cascade_out_dec", _out_l1, indices=[tx, subi_l])
                        ChannelPut("cascade_max_dec", _max_l1, indices=[tx, subi_l])
                        ChannelPut("cascade_sum_dec", _sum_l1, indices=[tx, subi_l])
                        affine.AffineYieldOp([])

                    with InsertionPoint(if_last.else_block):
                        if_mid = affine.AffineIfOp(
                            set_middle_stage,
                            cond_operands=[tx, ty],
                            has_else=True,
                        )

                        # --- Middle stages: receive, merge, send ---
                        with InsertionPoint(if_mid.then_block):
                            ChannelGet("cascade_out_dec", _out_c, indices=[tx, ty])
                            ChannelGet("cascade_max_dec", _max_c, indices=[tx, ty])
                            ChannelGet("cascade_sum_dec", _sum_c, indices=[tx, ty])
                            _cascade_merge(
                                _out_l1,
                                _max_l1,
                                _sum_l1,
                                _out_c,
                                _max_c,
                                _sum_c,
                                _old_max_save,
                                _rescale_recv_c,
                                _rescale_local_c,
                            )
                            subi_m = arith.SubIOp(ty, c1_idx)
                            ChannelPut("cascade_out_dec", _out_l1, indices=[tx, subi_m])
                            ChannelPut("cascade_max_dec", _max_l1, indices=[tx, subi_m])
                            ChannelPut("cascade_sum_dec", _sum_l1, indices=[tx, subi_m])
                            affine.AffineYieldOp([])

                        # --- ty == 0: receive, merge, div, write output ---
                        with InsertionPoint(if_mid.else_block):
                            ChannelGet("cascade_out_dec", _out_c, indices=[tx, ty])
                            ChannelGet("cascade_max_dec", _max_c, indices=[tx, ty])
                            ChannelGet("cascade_sum_dec", _sum_c, indices=[tx, ty])
                            _cascade_merge(
                                _out_l1,
                                _max_l1,
                                _sum_l1,
                                _out_c,
                                _max_c,
                                _sum_c,
                                _old_max_save,
                                _rescale_recv_c,
                                _rescale_local_c,
                            )
                            # Final normalization
                            CallOp([], "decode_div_output", [_sum_l1, _out_l1])
                            # Send result L1 -> L2 via explicit out_dec channel
                            # (ty==0 only; segment ChannelGet receives into out_l2)
                            c0_h = ConstantOp(index_type, 0)
                            ChannelPut("out_dec", _out_l1, indices=[tx, c0_h])
                            affine.AffineYieldOp([])

                        affine.AffineYieldOp([])

                # --------------------------------------------------------
                # Receive herd output into L2 staging via out_dec channel.
                # Each ty==0 tile (tx=0..NKV-1) puts _out_l1 to out_dec[tx, 0].
                # The matching segment ChannelGets fill out_l2[tx, :, :].
                # --------------------------------------------------------
                c0_out = ConstantOp(index_type, 0)
                for tx_i in range(NKV):
                    c_tx_i_out = ConstantOp(index_type, tx_i)
                    ChannelGet(
                        "out_dec",
                        out_l2.result,
                        offsets=[tx_i, 0, 0],
                        sizes=[1, group_size, dv],
                        strides=[group_size * dv, dv, 1],
                        indices=[c_tx_i_out, c0_out],
                    )

                # --------------------------------------------------------
                # After herd: DMA output L2 -> L3
                # q_head_base_launch = lx_s * NKV * group_size
                # --------------------------------------------------------
                q_launch_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(NKV * group_size),
                        )
                    ],
                )
                q_launch_base = affine_apply(q_launch_map, [lx_s])

                dma_memcpy_nd(
                    out_s,
                    out_l2.result,
                    dst_offsets=[q_launch_base, 0],
                    dst_sizes=[NKV * group_size, dv],
                    dst_strides=[dv, 1],
                    src_offsets=[0, 0, 0],
                    src_sizes=[NKV, group_size, dv],
                    src_strides=[group_size * dv, dv, 1],
                )

                # Deallocate all scratch buffers
                DeallocOp(out_l2)
                DeallocOp(q_l1)
                DeallocOp(k_l1)
                DeallocOp(v_l1)
                DeallocOp(scores_l1)
                DeallocOp(out_l1)
                DeallocOp(max_l1)
                DeallocOp(sum_l1)
                DeallocOp(score_max_tmp)
                DeallocOp(rescale_tmp)
                DeallocOp(out_c)
                DeallocOp(max_c)
                DeallocOp(sum_c)
                DeallocOp(old_max_save)
                DeallocOp(rescale_recv_c)
                DeallocOp(rescale_local_c)
                DeallocOp(pos_l1)


def _cascade_merge(
    out_local,
    max_local,
    sum_local,
    out_recv,
    max_recv,
    sum_recv,
    old_max_save,
    rescale_recv,
    rescale_local,
):
    """Emit IR to merge two partial softmax results via cascade.

    Both partial results (local and received) are combined into the local
    buffers using the log-sum-exp identity:
        new_max   = max(max_local, max_recv)
        r_local   = exp(old_max_local - new_max)
        r_recv    = exp(max_recv    - new_max)
        out_local = r_local * out_local + r_recv * out_recv
        sum_local = r_local * sum_local + r_recv * sum_recv
        max_local = new_max
    """
    # Save old local max before updating
    CallOp([], "decode_copy_max_sum", [max_local, old_max_save])
    # max_local = max(max_local, max_recv)
    CallOp([], "decode_cascade_merge_max", [max_recv, max_local])
    # Compute rescale factors from old maxes to combined max
    CallOp([], "decode_compute_rescale", [max_recv, max_local, rescale_recv])
    CallOp([], "decode_compute_rescale", [old_max_save, max_local, rescale_local])
    # Rescale both output buffers and merge
    CallOp([], "decode_rescale_output", [rescale_recv, out_recv])
    CallOp([], "decode_rescale_output", [rescale_local, out_local])
    CallOp([], "decode_add_output", [out_recv, out_local])
    # Rescale both sum buffers and merge
    CallOp([], "decode_rescale_sum", [rescale_recv, sum_recv])
    CallOp([], "decode_rescale_sum", [rescale_local, sum_local])
    CallOp([], "decode_add_sum", [sum_recv, sum_local])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="attn_decode_npu2.py",
        description="Decode-phase flash attention on NPU2 (AIE2P)",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print MLIR module and exit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--lk",
        type=int,
        default=2048,
        help="Maximum KV cache sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lkp",
        type=int,
        default=64,
        help="K/V chunk size per cascade stage chunk (default: 64)",
    )
    parser.add_argument("--dk", type=int, default=64, help="Key head dim (default: 64)")
    parser.add_argument(
        "--dv", type=int, default=64, help="Value head dim (default: 64)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Q heads per KV head / GQA group size (default: 4)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=4,
        help="Number of KV heads (default: 4, must equal num-kv-parallel)",
    )
    parser.add_argument(
        "--num-kv-parallel",
        type=int,
        default=4,
        help="KV heads in parallel / herd x-dim (default: 4)",
    )
    parser.add_argument(
        "--num-cascade-stages",
        type=int,
        default=4,
        help="Cascade stages / herd y-dim (default: 4)",
    )
    parser.add_argument(
        "--current-pos",
        type=int,
        default=None,
        help="Current decode position (default: lk-1 = full cache)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="elf",
        choices=["elf"],
        help="Output format (must be 'elf'; decode uses air.launch iteration space "
        "which requires ELF format on NPU2)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="compile-and-run",
        choices=["compile-only", "compile-and-run"],
        help="Compilation mode (default: compile-and-run)",
    )
    args = parser.parse_args()

    lk = args.lk
    lkp = args.lkp
    dk = args.dk
    dv = args.dv
    group_size = args.group_size
    num_kv_heads = args.num_kv_heads
    num_kv_parallel = args.num_kv_parallel
    num_cascade_stages = args.num_cascade_stages
    num_heads = group_size * num_kv_heads
    current_pos = args.current_pos if args.current_pos is not None else lk - 1

    mlir_module = build_module(
        lk=lk,
        lkp=lkp,
        dk=dk,
        dv=dv,
        group_size=group_size,
        num_kv_heads=num_kv_heads,
        num_kv_parallel=num_kv_parallel,
        num_cascade_stages=num_cascade_stages,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner
    from air.backend.xrt import XRTBackend
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16
    rng = np.random.default_rng(42)
    val_range = 2.0

    # Inputs
    input_q = rng.uniform(-val_range, val_range, (num_heads, dk)).astype(INPUT_DATATYPE)
    input_k = rng.uniform(-val_range, val_range, (num_kv_heads, lk, dk)).astype(
        INPUT_DATATYPE
    )
    input_v = rng.uniform(-val_range, val_range, (num_kv_heads, lk, dv)).astype(
        INPUT_DATATYPE
    )
    input_pos = np.array([current_pos], dtype=np.int32)

    # NumPy reference: decode attention (causal, only positions 0..current_pos)
    inv_sqrt_dk = 1.0 / sqrt(dk)
    expected_out = np.zeros((num_heads, dv), dtype=OUTPUT_DATATYPE)
    for h in range(num_heads):
        kv_h = h // group_size
        q_h = input_q[h].astype(np.float32)
        K_valid = input_k[kv_h, : current_pos + 1, :].astype(np.float32)
        V_valid = input_v[kv_h, : current_pos + 1, :].astype(np.float32)
        scores = K_valid @ q_h * inv_sqrt_dk  # [current_pos+1]
        mx = scores.max()
        P = np.exp(scores - mx)
        P = P / P.sum()
        expected_out[h] = (P @ V_valid).astype(OUTPUT_DATATYPE)

    backend_opts = dict(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="decode_attention_bf16",
        target_device="npu2",
    )

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)

        # Test 1: specified current_pos (default: lk-1 = full cache, mask no-op).
        ret = runner.run_test(
            mlir_module,
            inputs=[input_q, input_k, input_v, input_pos],
            expected_outputs=[expected_out],
            atol=0.15,
            rtol=0.04,
            # 2.0% tolerance (4x prefill's 0.5%): matvec-based online softmax with
            # cascade merge accumulates more bf16 rounding steps than tiled matmul.
            max_mismatch_percentage=2.0,
            min_correlation=0.99,
        )
        if ret != 0:
            exit(ret)

        # Test 2: mid-sequence position to exercise causal masking code path.
        # apply_decode_mask sets scores[h,i]=-inf for (chunk_start+i) > current_pos.
        # Using lk//2-1 ensures positions in the upper half are masked to -inf,
        # validating the masking logic that is a no-op with the default lk-1 position.
        masked_pos = lk // 2 - 1
        if masked_pos != current_pos:
            masked_out = np.zeros((num_heads, dv), dtype=OUTPUT_DATATYPE)
            for h in range(num_heads):
                kv_h = h // group_size
                q_h = input_q[h].astype(np.float32)
                K_valid = input_k[kv_h, : masked_pos + 1, :].astype(np.float32)
                V_valid = input_v[kv_h, : masked_pos + 1, :].astype(np.float32)
                scores_ref = K_valid @ q_h * inv_sqrt_dk
                mx = scores_ref.max()
                P = np.exp(scores_ref - mx)
                P = P / P.sum()
                masked_out[h] = (P @ V_valid).astype(OUTPUT_DATATYPE)
            exit(
                runner.run_test(
                    mlir_module,
                    inputs=[
                        input_q,
                        input_k,
                        input_v,
                        np.array([masked_pos], dtype=np.int32),
                    ],
                    expected_outputs=[masked_out],
                    atol=0.15,
                    rtol=0.04,
                    max_mismatch_percentage=2.0,
                    min_correlation=0.99,
                )
            )
        exit(0)
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(**backend_opts)
        backend.compile(mlir_module)
        print("Compilation complete.")
