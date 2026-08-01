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

Supports multi-head (MHA), grouped-query (GQA), and causal masking.

Default design parameters:
  lk=512, lkp=64, lq=512, lqp=256, dk=64, dv=64
  num_q_tiles=4, num_cascade_stages=4, num_heads=2
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
from air.dialects.memref import (
    AllocOp,
    CastOp,
    CollapseShapeOp,
    DeallocOp,
    SubViewOp,
    load,
    store,
    subview,
)
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
    num_heads_per_unroll=2,
    causal_skip=True,
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
        num_kv_heads: Number of key/value heads for grouped-query attention
            (GQA). If None, defaults to num_heads (standard MHA).
        causal: Whether to enable causal (autoregressive) masking.
        num_heads_per_unroll: Heads processed per segment instance (default: 2).
            Acts as the physical-column multiplier — physical columns =
            num_heads_per_unroll * num_q_tiles (must be <= 8 on NPU2). Requires
            num_heads % num_heads_per_unroll == 0.
        causal_skip: Under causal masking, skip the matmul/softmax/PV for
            fully-future K-blocks (kv_block > q_block) instead of computing them
            and then masking to -inf. Numerically identical (a fully-masked block
            contributes exactly exp(-inf)=0), and it reuses the same block
            indices the mask already computes, so it is correct wherever the mask
            is. DMA gets/puts stay unconditional (channels balanced); a skipped
            block leaves the stage's neutral local so the cascade merge is an
            identity. Saves the wasted block-matmul (grows with sequence length).
            Default on; no effect when causal=False.
    """
    # TEMPORAL PROTOTYPE: no spatial cascade. Each herd tile loops over ALL K
    # blocks in-core (temporal online softmax) and writes its own finalized
    # output — matching the reference's single-dispatch reduction. Force num_cascade_stages
    # = 1 regardless of the caller (the cascade merge / channels are dropped).
    num_cascade_stages = 1
    # causal_skip is incompatible with the long single-buffer temporal K-loop:
    # it guards only the COMPUTE (scf.if) while the K/V gets stay unconditional,
    # which desyncs get<->compute over many chunks (wrong past ~8 K-blocks); a
    # real skip would also have to skip the STREAMING (the conditional-channel
    # deadlock wall). The mask alone is correct, and this prototype targets the
    # launch/cascade overhead, not the causal work-skip — so force it off.
    causal_skip = False
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
    # The seq-first L3 layout interleaves all kv-heads at column-stride dv,
    # so a single chunk's per-head DMA descriptor is straightforward; with
    # dv_chunks > 1 the per-chunk strides need additional validation that is
    # not yet covered by a test. Restrict to dv == lkp until that exists.
    assert dv_chunks == 1, (
        f"attn_npu2_seqfirst.py currently supports only dv == lkp "
        f"(dv_chunks == 1); got dv={dv}, lkp={lkp}. "
        f"Use attn_npu2.py for the dv_chunks > 1 / heads-first layout."
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

    # PHASE 1 TEMPORAL: the herd ROWS map to the gqa_group_size q-heads that
    # share one kv-head. K/V are broadcast across rows (same kv-head), Q differs
    # per row (different q-head). So num_heads_per_unroll now counts KV-HEADS per
    # dispatch (= segment instances), and the launch head axis iterates kv-head
    # groups. This fills all 32 tiles (was 8) and cuts the launch count 4x.
    R = gqa_group_size
    assert R <= 4, f"gqa_group_size ({R}) must be <= 4 (physical herd rows)"
    assert num_kv_heads % num_heads_per_unroll == 0, (
        f"num_kv_heads ({num_kv_heads}) must be divisible by "
        f"num_heads_per_unroll ({num_heads_per_unroll})"
    )
    assert num_heads_per_unroll * num_q_tiles <= 8, (
        f"physical columns = num_heads_per_unroll ({num_heads_per_unroll}) * "
        f"num_q_tiles ({num_q_tiles}) must be <= 8"
    )
    num_head_groups = num_kv_heads // num_heads_per_unroll

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
    # 2-tile Q buffer: a 2-way Q broadcast delivers a col-pair's 2 q-tiles to
    # both cols; each col subviews its half (tile lc) for the matmul.
    qpair_l1_t = MemRefType.get([2 * tile_size_q, dk_tile], bf16, memory_space=l1_space)
    # A subview of qpair (tile lc) has a strided layout with a (dynamic) offset;
    # copy_tile must accept that type. Same C ABI as a plain memref (the offset
    # rides in the descriptor), so it links against the same attn_npu2.o symbol.
    q_half_strided_t = MemRefType.get(
        [tile_size_q, dk_tile],
        bf16,
        layout=StridedLayoutAttr.get(
            ShapedType.get_dynamic_stride_or_offset(), [dk_tile, 1]
        ),
        memory_space=l1_space,
    )
    k_l1_t = MemRefType.get([lkp, dk_tile], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l1_space)
    g_l1_2d = MemRefType.get([tile_size_q, lkp], bf16, memory_space=l1_space)
    g_l1_1d = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    gp_l1_t = MemRefType.get([tile_size_q, dv_tile], bf16, memory_space=l1_space)
    up_l1_t = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)

    # L2 MemRefTypes (QK relay uses dk_tile)
    qk_l2_t = MemRefType.get([lkp, dk_tile], bf16, memory_space=l2_space)
    # Q relay buffer holds a whole row's NQ q-tiles ([lqp, dk_tile]); one QIn get
    # fills it, then NQ disjoint-offset Q2L1 puts scatter per-tile (partition).
    q_relay_l2_t = MemRefType.get([lqp, dk_tile], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l2_space)
    gp_l2_t = MemRefType.get([lqp, dv_tile], bf16, memory_space=l2_space)
    # Half-height output buffer: the reference splits a q-head's output gather across TWO
    # memtiles (first/second half of the sequence), 4 tiles each, to keep each
    # memtile's gather light enough to route alongside the K/V broadcast transit.
    gp_half_l2_t = MemRefType.get([lqp // 2, dv_tile], bf16, memory_space=l2_space)

    # L3 MemRefTypes — SEQ-FIRST layout (no head dimension in shape)
    # Q: [lq, num_heads * dk] — all heads interleaved per position
    # K: [lk, num_kv_heads * dk] — all KV heads interleaved per position
    # V: [lk, num_kv_heads * dv] — all KV heads interleaved per position
    # Output: [lq, num_heads * dv] — all heads interleaved per position
    q_l3_t = MemRefType.get([lq, num_heads * dk], bf16)
    k_l3_t = MemRefType.get([lk, num_kv_heads * dk], bf16)
    v_l3_t = MemRefType.get([lk, num_kv_heads * dv], bf16)
    gp_l3_t = MemRefType.get([lq, num_heads * dv], bf16)

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
    # copy_half_tile(qpair[2*tile_size_q, dk], q_buf[tile_size_q, dk], lc): extract
    # the lc-th tile from a 2-way Q broadcast (the reference's per-column q offset).
    external_func("copy_half_tile", [qpair_l1_t, q_l1_t, i32], link_with="attn_npu2.o")
    external_func("div_gp_sp", [up_l1_t, gp_l1_t], link_with="attn_npu2.o")
    external_func("add_gp_g", [gp_l1_t, gp_l1_t], link_with="attn_npu2.o")
    if causal:
        external_func("apply_causal_mask", [g_l1_2d, i32, i32], link_with="attn_npu2.o")

    # ----------------------------------------------------------------
    # Channel declarations (3D with head dimension for multi-head)
    # ----------------------------------------------------------------

    # Q is fully PARTITIONED (NOT broadcast) AND laid out the reference-style in COLUMN
    # BLOCKS: the 32 tiles are 4 col-blocks (b = tx//2) x 2 local-cols x 4 rows.
    # Each col-block b = one q-head of the GQA group; its 8 tiles do 8 distinct
    # q-seq-tiles (s = ty*2 + tx%2). ONE Q channel per block (Q2L1_b) feeds ONLY
    # that block's local 2 columns, so place-tiles distributes the 4 Q memtiles
    # across the array (like the reference's even-col Q memtiles) — ~8 BDs each (< 24) —
    # instead of all 32 Q BDs piling on one central memtile's MM2S 0.
    # size=[R, NQ] indexed raw [ty, tx] (mirrors the original per-stage QK2L1_s);
    # only block b's tiles (2b<=tx<=2b+1) consume it (affine-gated in the herd).
    # reference-faithful fanout: per block, per row, ONE 2-way broadcast flow to the
    # row's 2-col pair (bcast_shape [R,2]) — matches the reference's mem_tile DMA:i -> 2
    # cols. 4 flows/block (one per row) instead of 8 single-tile flows.
    for b in range(R):
        Channel(f"Q2L1_{b}", size=[R, 1], broadcast_shape=[R, 2])
    # QIn is PER-BLOCK (one endpoint per col-block) so each block's Q gets its own
    # shim->memtile flow to its own Q memtile (cols 0,2,4,6). A single-endpoint QIn
    # only creates ONE shim->memtile flow, so only one block's Q memtile is fed and
    # the others deadlock (their tiles never receive Q). One endpoint per block.
    Channel("QIn", size=[R])

    # K+V share ONE consolidated broadcast channel (KV2L1) to ALL R*NQ tiles
    # (rows AND cols) — mirrors the reference's single MT[3] 32-way K/V multicast. Only one
    # kv-head is live per dispatch (kv-head is the launch outer loop), so the
    # whole array shares this K/V. Packing K+V onto ONE S2MM leaves the tile's
    # other S2MM for Q (which is per-row here, a different q-head per row, so it
    # cannot broadcast-to-all like the original cascade design). Per temporal
    # chunk the producer puts K then V; the herd gets K then V in the same order.
    # size=[nhpu,1,1], broadcast_shape spans rows+cols; put [seg,0,0], get [seg,ty,tx].
    # PACKET-switched: a 32-way CIRCUIT multicast can deadlock when the auto-router
    # lays its streams to share physical switchbox channels with the Q/output flows
    # (lock-step circuit backpressure); packet flows time-multiplex instead of
    # holding a circuit, avoiding that. (the reference's IRON hand-routes circuits to avoid
    # conflicts; AIR's auto-router doesn't, so packet is the idiomatic fix.)
    Channel(
        "KV2L1",
        size=[num_heads_per_unroll, 1, 1],
        broadcast_shape=[num_heads_per_unroll, R, NQ],
    )
    Channel("KIn", size=[num_heads_per_unroll])
    Channel("VIn", size=[num_heads_per_unroll])

    # Output split per col-block into lo (rows 0,1 = seq-tiles 0..3, col 2b) and
    # hi (rows 2,3 = seq-tiles 4..7, col 2b+1) — the reference's 4-way split gather. Each
    # gathers 4 tiles into its own column-pinned memtile, then one GpOut half.
    for b in range(R):
        Channel(f"Gp2L2_{b}_lo", size=[R, NQ])
        Channel(f"Gp2L2_{b}_hi", size=[R, NQ])
    # GpOut per (block, half) endpoints so the 8 output streams SPREAD across
    # shim tiles (like the reference's per-memtile DMA:5 output) instead of funneling all 8
    # concurrent memtile sources into ONE shim channel (an 8-to-1 circuit merge
    # that deadlocks).
    Channel("GpOut", size=[R, 2])

    # ----------------------------------------------------------------
    # Main attention function
    # ----------------------------------------------------------------
    @FuncOp.from_py_func(q_l3_t, k_l3_t, v_l3_t, gp_l3_t)
    def attention_bf16(q_in, k_in, v_in, gp_out):
        c_num_head_groups = ConstantOp(index_type, num_head_groups)

        # CAUSAL DMA-TRIANGLE SKIP: the q-chunk (lx) axis is UNROLLED in-core
        # at build time, NOT a launch axis. Each round lx streams only its causal
        # K/V prefix ((lx+1)*NQ blocks) with a STATIC size and a matching static
        # in-core K-loop bound — mirroring the reference's host-unrolled per-round DMA
        # (seq_gen.hpp: kv_length/256 = round+1). A launch axis would reuse one
        # core ELF (fixed loop bound) and emit one uniform-size BD; unrolling gives
        # each round its own static BD (the triangle) with no runtime-varying fire
        # count -> no dynamic-BD deadlock. Launch iterates KV-HEAD GROUPS only, so
        # dispatches drop from num_lq_iters*num_head_groups to num_head_groups.
        @launch(
            operands=[q_in, k_in, v_in, gp_out],
            sizes=[c_num_head_groups],
        )
        def launch_body(ly, lsy, q, k, v, gp):

            # PHASE 1: build column offsets as affine maps in ly. The launch
            # head axis (ly) iterates KV-HEAD GROUPS of num_heads_per_unroll
            # kv-heads; kv_local / row_slot are python-int loop constants folded
            # into each map. _linmap(a, b) -> (ly * a + b).
            def _linmap(a, b):
                return AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(a),
                            ),
                            AffineConstantExpr.get(b),
                        )
                    ],
                )

            emb_dim_q = num_heads * dk
            emb_dim_k = num_kv_heads * dk
            emb_dim_v = num_kv_heads * dv

            # K/V causal triangle as a TEMPORAL scf.for: round lx streams its
            # causal prefix cps_lx = (lx+1)*NQ blocks. The transfer SIZE is
            # IV-dependent, so wrap-and-stride can't fold it into one constant
            # BD (a variant-size transfer isn't a strided access) -- it now
            # DECLINES and AIRUnrollScfForIntoBDChain unrolls the static-trip
            # loop into the same per-round constant-size BDs (8/16/24/32...).
            _c_nlq_kv = ConstantOp(index_type, num_lq_iters)
            # cps(lx) = (lx+1)*NQ
            _cps_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(1)
                        ),
                        AffineConstantExpr.get(NQ),
                    )
                ],
            )
            for lx_iv in scf_range(0, _c_nlq_kv, 1):
                cps_lx = affine_apply(_cps_map, [lx_iv])
                for kv_local in range(num_heads_per_unroll):
                    kv_offset_idx = ConstantOp(index_type, kv_local)
                    # kv_head = ly * num_heads_per_unroll + kv_local
                    head_k_off = affine_apply(
                        _linmap(num_heads_per_unroll * dk, kv_local * dk), [ly]
                    )
                    head_v_off = affine_apply(
                        _linmap(num_heads_per_unroll * dv, kv_local * dv), [ly]
                    )

                    # K put: causal prefix (cps_lx blocks from pos 0); bcast to rows.
                    ChannelPut(
                        "KIn",
                        k,
                        indices=[kv_offset_idx],
                        offsets=[0, head_k_off],
                        sizes=[cps_lx, dk_chunks, lkp, dk_tile],
                        strides=[lkp * emb_dim_k, dk_tile, emb_dim_k, 1],
                    )
                    # V put: causal prefix (cps_lx blocks); bcast to rows.
                    ChannelPut(
                        "VIn",
                        v,
                        indices=[kv_offset_idx],
                        offsets=[0, head_v_off],
                        sizes=[cps_lx, lkp, dv_tile],
                        strides=[lkp * emb_dim_v, emb_dim_v, 1],
                    )
                yield_([])

            # Q puts: RECTANGULAR in lx (row offset lx*lqp, constant [lqp, dk_tile]
            # size), so the round axis is a single scf.for that wrap-and-stride
            # folds to one strided BD per (row_slot) endpoint -- collapsing the
            # per-round QIn unroll. Separate from the K/V triangle above (which is
            # a different channel/dataflow and must stay unrolled).
            _c_nlq = ConstantOp(index_type, num_lq_iters)
            _lqp_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(lqp)
                    )
                ],
            )
            for lx_iv in scf_range(0, _c_nlq, 1):
                row0 = affine_apply(_lqp_map, [lx_iv])
                for kv_local in range(num_heads_per_unroll):
                    # Q puts: one per col-block (q-head), a SINGLE 2D [lqp, dk_tile]
                    # block at row lx*lqp. q_head = (ly*nhpu + kv_local)*gqa + slot.
                    for row_slot in range(R):
                        q_col_off = affine_apply(
                            _linmap(
                                num_heads_per_unroll * gqa_group_size * dk,
                                (kv_local * gqa_group_size + row_slot) * dk,
                            ),
                            [ly],
                        )
                        ChannelPut(
                            "QIn",
                            q,
                            indices=[ConstantOp(index_type, row_slot)],
                            offsets=[row0, q_col_off],
                            sizes=[lqp, dk_tile],
                            strides=[emb_dim_q, 1],
                        )
                yield_([])

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
                # L2 allocations for QK and V (per-stage) and output. The K/V
                # relay buffers feed the ONE consolidated 32-way broadcast; pin
                # them to a central ODD column (3, like the reference's MT[3]) so they sit
                # off the even Q/output memtiles and broadcast centrally.
                qk_l2_bufs = []
                for _ in range(NS):
                    _kb = AllocOp(qk_l2_t, [], [])
                    _kb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        i32, 3
                    )
                    qk_l2_bufs.append(_kb)
                v_l2_bufs = []
                for _ in range(NS):
                    _vb = AllocOp(v_l2_t, [], [])
                    _vb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        i32, 3
                    )
                    v_l2_bufs.append(_vb)
                # One Q relay buffer PER col-block (q-head), each PINNED to its
                # block's even memtile column (0,2,4,6 — the reference's Q memtile columns)
                # via air.memtile_col so place-tiles distributes them instead of
                # clustering all 4 on one central memtile (which piled 32 Q BDs on
                # one MM2S channel). Each memtile then holds ~8 Q BDs (< 24).
                q_relay_l2_bufs = []
                for b in range(R):
                    _qb = AllocOp(q_relay_l2_t, [], [])
                    _qb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        i32, 2 * b
                    )
                    # Keep the whole [lqp, dk] buffer on ONE pinned memtile: without
                    # no_split, air-split-l2-memref partitions it (consumers read
                    # disjoint per-tile slices) and re-tiles the shim read into NQ
                    # per-tile BDs -> shim BD overflow. The per-tile scatter still
                    # happens at the memtile->L1 Q2L1_b stage.
                    _qb.operation.attributes["air.no_split"] = UnitAttr.get()
                    q_relay_l2_bufs.append(_qb)
                # Output gather buffer per col-block on ODD columns, AVOIDING
                # col 3 (that is the dedicated K/V 32-way broadcast memtile, like
                # the reference's MT[3] — it must carry nothing else or its switchbox
                # overflows). Spread output across cols {1,5,7}; with 4 blocks one
                # column hosts two blocks' gathers.
                # Output split lo/hi across two memtiles per block: lo on col 2b
                # (co-located with Q), hi on col 2b+1. Each gathers 4 tiles.
                # lo co-located with Q on even col 2b; hi on odd col 2b+1.
                # (the reference also co-locates an output gather on its K/V memtile col3, so
                # this is reference-faithful.)
                # Output relay buffers DOUBLE-BUFFERED across rounds (n_ob), same
                # reason as gp_l1_pp: round lx's GpOut drain must not race round
                # lx+1's Gp2L2 gather into the same L2 buffer. gp_*_bufs[b][lx%n_ob].
                n_ob_l2 = 2 if num_lq_iters > 1 else 1
                gp_lo_bufs = []
                gp_hi_bufs = []
                for b in range(R):
                    _glos = []
                    _ghis = []
                    for _ in range(n_ob_l2):
                        _glo = AllocOp(gp_half_l2_t, [], [])
                        _glo.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            i32, 2 * b
                        )
                        _glo.operation.attributes["air.no_split"] = UnitAttr.get()
                        _glos.append(_glo)
                        _ghi = AllocOp(gp_half_l2_t, [], [])
                        _ghi.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            i32, 2 * b + 1
                        )
                        _ghi.operation.attributes["air.no_split"] = UnitAttr.get()
                        _ghis.append(_ghi)
                    gp_lo_bufs.append(_glos)
                    gp_hi_bufs.append(_ghis)

                # L1 allocations passed to herd
                q_saved_bufs = [AllocOp(q_l1_t, [], []) for _ in range(dk_chunks)]
                qpair_bufs = [AllocOp(qpair_l1_t, [], []) for _ in range(dk_chunks)]
                qk_buf = AllocOp(k_l1_t, [], [])
                v_l1 = AllocOp(v_l1_t, [], [])
                g_l1 = AllocOp(g_l1_2d, [], [])
                # Output accumulator is DOUBLE-BUFFERED across rounds (the reference's
                # o_ping/o_pong): round lx uses gp_l1_pp[lx%n_ob] so round lx+1's
                # zero_fill can't race round lx's Gp2L2 drain (single-buffer +
                # omit_pingpong gave a non-deterministic race that zeroed output).
                n_ob = 2 if num_lq_iters > 1 else 1
                gp_l1_pp = [AllocOp(gp_l1_t, [], []) for _ in range(n_ob)]
                up_l1 = AllocOp(up_l1_t, [], [])
                sp_l1 = AllocOp(up_l1_t, [], [])

                c_nq = ConstantOp(index_type, NQ)
                c_r = ConstantOp(index_type, R)
                c0_seg = ConstantOp(index_type, 0)

                # Per-round (lx) relay, Python-unrolled. Each round's K/V loop
                # uses its OWN qk_l2/v_l2 buffer (qk_l2_bufs[lx]) so the CONSTANT-
                # bound scf.for folds to ONE cyclic ping-pong BD per round on the
                # memtile — the causal triangle is the per-round STATIC trip count.
                for lx in range(num_lq_iters):
                    c_cps_lx = ConstantOp(index_type, (lx + 1) * NQ)
                    qk_l2 = qk_l2_bufs[0]
                    v_l2 = v_l2_bufs[0]
                    # Q relay for round lx: one QIn get of the q-head's whole
                    # lqp-row block into q_relay_l2, then R 2-way broadcasts (per
                    # row) to the col-pair; each col extracts its half in-kernel
                    # (copy_half_tile), mirroring the reference's q+col*lq*dh.
                    # offset[1](ty_i) = (2*ty_i)*(lkp//M)
                    _q2l1_off_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(2 * (lkp // M)),
                            )
                        ],
                    )
                    for b in range(R):
                        q_relay_l2 = q_relay_l2_bufs[b]
                        ChannelGet(
                            "QIn",
                            q_relay_l2.result,
                            indices=[ConstantOp(index_type, b)],
                        )
                        # R per-row 2-way broadcasts as a spatial scf.forall: the
                        # Q2L1_b bundle ROW index is the forall IV (a legal bundle
                        # index, unlike scf.for), col-pair via broadcast_shape.
                        # Makes the R-way row scatter explicit; AIR unrolls it to
                        # the same BDs (consistent with the Gp2L2 gather).
                        par_q = scf.ForallOp(
                            lower_bounds=[0], upper_bounds=[R], steps=[1]
                        )
                        with InsertionPoint(par_q.body):
                            ty_i = par_q.induction_variables[0]
                            ChannelPut(
                                f"Q2L1_{b}",
                                q_relay_l2.result,
                                indices=[ty_i, c0_seg],
                                offsets=[
                                    0,
                                    affine_apply(_q2l1_off_map, [ty_i]),
                                    0,
                                    0,
                                ],
                                sizes=[dk_tile // M, 2 * (lkp // M), M, M],
                                strides=[M, dk_tile * M, dk_tile, 1],
                            )
                            scf.InParallelOp()
                    # K+V relay for round lx: cps_lx blocks on ONE broadcast
                    # channel, K then V interleaved (herd's K-then-V gets stay in
                    # FIFO order). Packs both onto the tiles' one shared S2MM.
                    for chunk_iter in scf_range(0, c_cps_lx, 1):
                        ChannelGet("KIn", qk_l2.result, indices=[seg_x])
                        ChannelPut(
                            "KV2L1",
                            qk_l2.result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dk_tile // M, lkp // M, M, M],
                            strides=[M, dk_tile * M, dk_tile, 1],
                        )
                        ChannelGet("VIn", v_l2.result, indices=[seg_x])
                        ChannelPut(
                            "KV2L1",
                            v_l2.result,
                            indices=[seg_x, c0_seg, c0_seg],
                            offsets=[0, 0, 0, 0],
                            sizes=[dv_tile // M, lkp // M, M, M],
                            strides=[M, dv_tile * M, dv_tile, 1],
                        )
                        yield_([])

                # ----------------------------------------------------------
                # Herd: [NQ, R] — rows = R q-heads sharing this kv-head
                # ----------------------------------------------------------
                herd_operands = (
                    q_saved_bufs
                    + qpair_bufs
                    + [
                        qk_buf,
                        v_l1,
                        g_l1,
                    ]
                    + gp_l1_pp
                    + [
                        up_l1,
                        sp_l1,
                        seg_x,
                    ]
                )

                @herd(
                    name="herd_0",
                    sizes=[c_nq, c_r],
                    operands=herd_operands,
                    link_with="attn_npu2.o",
                )
                def herd_body(tx, ty, hsx, hsy, *all_args):
                    # Unpack: dk_chunks Q bufs, dk_chunks qpair bufs, then qk, v,
                    # g, n_ob gp buffers (double-buffered output), up, sp, seg_x
                    q_bufs = list(all_args[:dk_chunks])
                    qpair_l = list(all_args[dk_chunks : 2 * dk_chunks])
                    base = 2 * dk_chunks
                    qk = all_args[base]
                    v = all_args[base + 1]
                    g = all_args[base + 2]
                    gp_pp = list(all_args[base + 3 : base + 3 + n_ob])
                    up_buf = all_args[base + 3 + n_ob]
                    sp_buf = all_args[base + 4 + n_ob]
                    h_seg_x = all_args[base + 5 + n_ob]

                    # CAUSAL DMA-TRIANGLE SKIP: loop the q-chunks (lx) IN-CORE,
                    # build-time unrolled. Round lx processes q-seq-tiles
                    # [lx*NQ : lx*NQ+NQ] and consumes only its causal prefix of
                    # cps_lx = (lx+1)*NQ K-blocks (matching the segment's per-lx
                    # stream). q_base = lx*NQ is a build-time constant (no counter).
                    # Online-softmax state (gp/sp/up) re-inits per round (each
                    # q-chunk is an independent attention).
                    for lx in range(num_lq_iters):
                        cps_lx = (lx + 1) * NQ
                        gp = gp_pp[
                            lx % n_ob
                        ]  # double-buffered output (the reference o_ping/o_pong)

                        # === INIT (per round) ===
                        CallOp([], "zero_fill_gp_bf16", [gp])
                        CallOp([], "zero_fill_sp_bf16", [sp_buf])
                        CallOp([], "neg_inf_fill_up_bf16", [up_buf])

                        # === Q PARTITIONED GET (column-block) for round lx ===
                        # tile is in col-block b = tx//2 (q-head), seq-tile
                        # s = ty*2 + tx%2. Gated on tx==2b+lc so only that col reads
                        # the block's row-ty 2-way broadcast (index lc); then
                        # copy_half_tile extracts this col's half into q_bufs.
                        for dk_c in range(dk_chunks):
                            for b in range(R):
                                for lc in range(2):
                                    col_set = IntegerSet.get(
                                        0,
                                        1,
                                        [
                                            AffineSymbolExpr.get(0)
                                            - AffineConstantExpr.get(2 * b + lc)
                                        ],
                                        [True],
                                    )
                                    if_q = affine.AffineIfOp(
                                        col_set, cond_operands=[tx]
                                    )
                                    with InsertionPoint(if_q.then_block):
                                        ChannelGet(
                                            f"Q2L1_{b}",
                                            qpair_l[dk_c],
                                            indices=[ty, ConstantOp(index_type, lc)],
                                        )
                                        CallOp(
                                            [],
                                            "copy_half_tile",
                                            [
                                                qpair_l[dk_c],
                                                q_bufs[dk_c],
                                                ConstantOp(i32, lc),
                                            ],
                                        )
                                        affine.AffineYieldOp([])

                        # === K CHUNK LOOP (static bound cps_lx = causal prefix) ===
                        c_cps_h = ConstantOp(index_type, cps_lx)
                        for chunk_iter in scf_range(0, c_cps_h, 1):
                            # 1. Zero fill G (once per K seq chunk)
                            g1d = CollapseShapeOp(g_l1_1d, g, [[0, 1]])
                            CallOp([], "zero_fill_g_bf16", [g1d])

                            # 2. K get (broadcast to all rows+cols) + matmul.
                            for dk_c in range(dk_chunks):
                                ChannelGet("KV2L1", qk, indices=[h_seg_x, ty, tx])
                                CallOp([], "matmul_a_b_bf16", [q_bufs[dk_c], qk, g1d])

                            # 3. V get into the SAME local buffer (K consumed above).
                            ChannelGet("KV2L1", qk, indices=[h_seg_x, ty, tx])

                            # 4. Causal mask: q_block = lx*NQ + s (s = ty*2 + tx%2),
                            # kv_block = chunk_iter. The per-lx DMA skip is coarse
                            # (whole prefix); the fine diagonal within the prefix is
                            # masked here (the reference masks the diagonal round too).
                            if causal:
                                kv_blk_r = arith.IndexCastOp(i32, chunk_iter).result
                                ty_i32 = arith.IndexCastOp(i32, ty).result
                                tx_i32 = arith.IndexCastOp(i32, tx).result
                                c2_i32 = ConstantOp(i32, 2)
                                s_val = arith.AddIOp(
                                    arith.MulIOp(ty_i32, c2_i32.result).result,
                                    arith.RemUIOp(tx_i32, c2_i32.result).result,
                                )
                                q_block = arith.AddIOp(
                                    ConstantOp(i32, lx * NQ).result, s_val.result
                                )
                                CallOp(
                                    [],
                                    "apply_causal_mask",
                                    [g, q_block.result, kv_blk_r],
                                )

                            # 5. softmax + PV + accumulate (online).
                            s_tmp = AllocOp(up_l1_t, [], [])
                            r_tmp = AllocOp(up_l1_t, [], [])
                            CallOp(
                                [],
                                "fused_softmax",
                                [g1d, up_buf, s_tmp.result, r_tmp.result],
                            )
                            CallOp([], "mul_r_gp", [r_tmp.result, gp])
                            CallOp([], "matmul_g_b_bf16", [g1d, qk, gp])
                            CallOp(
                                [],
                                "accum_sp_r_s",
                                [sp_buf, r_tmp.result, s_tmp.result],
                            )
                            CallOp(
                                [],
                                "vector_copy_32elems",
                                [ConstantOp(i32, 0), s_tmp.result, sp_buf],
                            )
                            DeallocOp(s_tmp)
                            DeallocOp(r_tmp)
                            yield_([])

                        # === OUTPUT for round lx (normalize + split lo/hi puts) ===
                        # NS=1: this tile looped its causal K prefix in-core, so its
                        # (gp, sp) is the final numerator + denominator. Normalize
                        # and emit directly (no cascade merge).
                        CallOp([], "div_gp_sp", [sp_buf, gp])
                        s0o = AffineSymbolExpr.get(0)  # tx
                        s1o = AffineSymbolExpr.get(1)  # ty
                        for b in range(R):
                            lo_set = IntegerSet.get(
                                0,
                                2,
                                [
                                    s0o - AffineConstantExpr.get(2 * b),
                                    AffineConstantExpr.get(2 * b + 1) - s0o,
                                    AffineConstantExpr.get(1) - s1o,
                                ],
                                [False, False, False],
                            )
                            hi_set = IntegerSet.get(
                                0,
                                2,
                                [
                                    s0o - AffineConstantExpr.get(2 * b),
                                    AffineConstantExpr.get(2 * b + 1) - s0o,
                                    s1o - AffineConstantExpr.get(2),
                                ],
                                [False, False, False],
                            )
                            for half_name, half_set in (
                                ("lo", lo_set),
                                ("hi", hi_set),
                            ):
                                if_o = affine.AffineIfOp(
                                    half_set, cond_operands=[tx, ty]
                                )
                                with InsertionPoint(if_o.then_block):
                                    ChannelPut(
                                        f"Gp2L2_{b}_{half_name}",
                                        gp,
                                        indices=[ty, tx],
                                        offsets=[0, 0, 0, 0],
                                        sizes=[tile_size_q // M, M, dv_tile // M, M],
                                        strides=[M * M, M, tile_size_q * M, 1],
                                    )
                                    affine.AffineYieldOp([])

                # Output gather split lo/hi. lo = rows 0,1 (seq-tiles 0..3) into
                # gp_lo (col 2b); hi = rows 2,3 (seq-tiles 4..7) into gp_hi
                # (col 2b+1). Each is a 4-way gather; each half streams out via
                # GpOut (lo then hi) as [lqp/2, dv].
                # Output gather PER ROUND (lx), matching the herd's per-lx Gp2L2
                # puts (FIFO). Each round: lo = rows 0,1 -> gp_lo (col 2b), hi =
                # rows 2,3 -> gp_hi (col 2b+1); each a 4-way gather -> one GpOut.
                # The 4-way gather is a SPATIAL scatter (each get targets a
                # different Gp2L2 bundle index [ty_i, 2b+lc]) -> an scf.forall (its
                # IV is a legal bundle index, unlike scf.for) makes the 2x2 spatial
                # structure explicit. AIR unrolls it to the same 4 BDs. Row (j) and
                # col-local (lc) index the source tile; buf row offset =
                # (2j+lc)*tile_size_q (hi adds +2 to the row index, same offset).
                _gp_off_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(2 * tile_size_q),
                            ),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(tile_size_q),
                            ),
                        )
                    ],
                )
                _gp_tyhi_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(2)
                        )
                    ],
                )

                def _gp_gather(chan, buf, col_map, tyhi):
                    par = scf.ForallOp(
                        lower_bounds=[0, 0], upper_bounds=[2, 2], steps=[1, 1]
                    )
                    with InsertionPoint(par.body):
                        j = par.induction_variables[0]
                        lc = par.induction_variables[1]
                        ty_idx = affine_apply(_gp_tyhi_map, [j]) if tyhi else j
                        ChannelGet(
                            chan,
                            buf,
                            indices=[ty_idx, affine_apply(col_map, [lc])],
                            offsets=[affine_apply(_gp_off_map, [j, lc]), 0],
                            sizes=[tile_size_q, dv_tile],
                            strides=[dv_tile, 1],
                        )
                        scf.InParallelOp()

                for lx in range(num_lq_iters):
                    for b in range(R):
                        # col(lc) = 2b + lc
                        _gp_col_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(2 * b),
                                )
                            ],
                        )
                        gp_lo = gp_lo_bufs[b][lx % n_ob_l2]
                        _gp_gather(f"Gp2L2_{b}_lo", gp_lo.result, _gp_col_map, False)
                        ChannelPut(
                            "GpOut",
                            gp_lo.result,
                            indices=[ConstantOp(index_type, b), c0_seg],
                        )
                        gp_hi = gp_hi_bufs[b][lx % n_ob_l2]
                        _gp_gather(f"Gp2L2_{b}_hi", gp_hi.result, _gp_col_map, True)
                        ChannelPut(
                            "GpOut",
                            gp_hi.result,
                            indices=[
                                ConstantOp(index_type, b),
                                ConstantOp(index_type, 1),
                            ],
                        )

                # Deallocs for segment-level buffers
                for q_buf in q_saved_bufs:
                    DeallocOp(q_buf)
                for qp in qpair_bufs:
                    DeallocOp(qp)
                DeallocOp(qk_buf)
                DeallocOp(v_l1)
                DeallocOp(g_l1)
                for _gpb in gp_l1_pp:
                    DeallocOp(_gpb)
                DeallocOp(up_l1)
                DeallocOp(sp_l1)
                for stage in range(NS):
                    DeallocOp(v_l2_bufs[stage])
                for stage in range(NS):
                    DeallocOp(qk_l2_bufs[stage])
                for q_rbuf in q_relay_l2_bufs:
                    DeallocOp(q_rbuf)
                for _blk in gp_lo_bufs:
                    for g_rbuf in _blk:
                        DeallocOp(g_rbuf)
                for _blk in gp_hi_bufs:
                    for g_rbuf in _blk:
                        DeallocOp(g_rbuf)

            # Output gets PER ROUND (lx): per block (q-head), TWO halves (lo then
            # hi) matching the segment's per-lx GpOut put order. lo = round lx's
            # first lqp/2 seq rows (at row lx*lqp), hi = second half.
            emb_dim_out = num_heads * dv
            half_rows_l = lqp // 2
            # RECTANGULAR in lx (row offset lx*lqp, constant size + channel index),
            # so the round axis is one scf.for that wrap-and-stride folds to a
            # single strided BD per (block, half) endpoint -- collapsing the
            # per-round GpOut drain unroll (num_lq_iters*R*2 gets -> R*2 folded).
            _c_nlq_out = ConstantOp(index_type, num_lq_iters)
            _row_lo_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(lqp)
                    )
                ],
            )
            _row_hi_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_add(
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(lqp)
                        ),
                        AffineConstantExpr.get(half_rows_l),
                    )
                ],
            )
            for lx_iv in scf_range(0, _c_nlq_out, 1):
                out_launch_row = affine_apply(_row_lo_map, [lx_iv])
                out_row_hi = affine_apply(_row_hi_map, [lx_iv])
                for kv_local in range(num_heads_per_unroll):
                    for row_slot in range(R):
                        # q_head = (ly * nhpu + kv_local) * gqa + row_slot.
                        out_col_off = affine_apply(
                            _linmap(
                                num_heads_per_unroll * gqa_group_size * dv,
                                (kv_local * gqa_group_size + row_slot) * dv,
                            ),
                            [ly],
                        )
                        b_idx = ConstantOp(index_type, row_slot)
                        ChannelGet(
                            "GpOut",
                            gp,
                            indices=[b_idx, ConstantOp(index_type, 0)],
                            offsets=[out_launch_row, out_col_off],
                            sizes=[half_rows_l, dv_tile],
                            strides=[emb_dim_out, 1],
                        )
                        ChannelGet(
                            "GpOut",
                            gp,
                            indices=[b_idx, ConstantOp(index_type, 1)],
                            offsets=[out_row_hi, out_col_off],
                            sizes=[half_rows_l, dv_tile],
                            strides=[emb_dim_out, 1],
                        )
                yield_([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="attn_npu2.py",
        description="Flash attention with memtile-relayed L3-to-L1 Q/K/V — "
        "selective Q capture",
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
        "--num-q-tiles",
        type=int,
        default=4,
        dest="num_q_tiles",
        help="Number of tiles to partition the Q chunk into (default: 4). "
        "Under causal masking, lqp / num_q_tiles must equal lkp.",
    )
    parser.add_argument(
        "--num-heads-per-unroll",
        type=int,
        default=2,
        dest="num_heads_per_unroll",
        help="Heads processed per segment instance (default: 2). "
        "Physical columns = num_heads_per_unroll * num_q_tiles.",
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
        help="Number of KV heads (default: num_heads for MHA, " "< num_heads for GQA)",
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
        "--no-causal-skip",
        action="store_false",
        dest="causal_skip",
        default=True,
        help="Disable the causal block-skip (compute-then-mask every K-block). "
        "The skip is ON by default under --causal and is numerically identical; "
        "it just avoids the wasted matmul on fully-future blocks.",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency + GFLOPs in addition to the correctness check",
    )
    args = parser.parse_args()

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")
    if args.num_q_tiles < 1:
        parser.error("--num-q-tiles must be >= 1")
    if args.num_heads_per_unroll < 1:
        parser.error("--num-heads-per-unroll must be >= 1")

    lk = args.lk
    lkp = args.lkp
    lq = args.lq
    lqp = args.lqp
    dk = args.dk
    dv = args.dv
    num_cascade_stages = args.num_cascade_stages
    num_q_tiles = args.num_q_tiles
    num_heads_per_unroll = args.num_heads_per_unroll
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
        num_heads_per_unroll=num_heads_per_unroll,
        causal_skip=args.causal_skip,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner
    from air.backend.xrt import XRTBackend
    from ml_dtypes import bfloat16

    INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16
    rng = np.random.default_rng(42)

    # SEQ-FIRST inputs: [seq, heads * dim]. Use N(0,1) (matching the GPU SDPA
    # test standard — PyTorch uses randn) so the correctness check sees a
    # realistic signed distribution rather than an all-positive one.
    input_q = rng.standard_normal((lq, num_heads * dk)).astype(INPUT_DATATYPE)
    input_k = rng.standard_normal((lk, num_kv_heads * dk)).astype(INPUT_DATATYPE)
    input_v = rng.standard_normal((lk, num_kv_heads * dv)).astype(INPUT_DATATYPE)

    # CPU reference: extract per-head data for attention computation
    inv_sqrt_dk = 1.0 / sqrt(dk)
    sdpa_output_hf = np.zeros((num_heads, lq, dv), dtype=OUTPUT_DATATYPE)
    for h in range(num_heads):
        kv_h = h // gqa_group_size
        Qf = input_q[:, h * dk : (h + 1) * dk].astype(np.float32)
        Kf = input_k[:, kv_h * dk : (kv_h + 1) * dk].astype(np.float32)
        Vf = input_v[:, kv_h * dv : (kv_h + 1) * dv].astype(np.float32)
        scores = Qf @ Kf.T * inv_sqrt_dk
        if causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        P = np.exp(scores - mx)
        P = P / np.sum(P, axis=-1, keepdims=True)
        sdpa_output_hf[h] = (P @ Vf).astype(OUTPUT_DATATYPE)

    # Expected output in seq-first: [lq, num_heads * dv]
    sdpa_output_transposed = (
        sdpa_output_hf.transpose(1, 0, 2).reshape(lq, num_heads * dv).copy()
    )

    # Seq-first: output is 2D [lq, num_heads*dv], so tiling is [1, 1]
    tiling = [1, 1]
    # FLOPs for attention: Q@K^T scales with dk, P@V scales with dv (each is
    # 2*num_heads*lq*lk*<dim>), so total = 2*num_heads*lq*lk*(dk+dv). Causal
    # masking roughly halves the effective work.
    perf_flops = 2.0 * num_heads * lq * lk * (dk + dv)
    if causal:
        perf_flops *= 0.5
    backend_opts = dict(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=tiling,
        output_format=args.output_format,
        instance_name="attention_bf16",
        target_device="npu2",
        report_precision=True,
        n_perf_iters=args.perf_iters,
        perf_flops=(perf_flops if args.perf_iters > 0 else None),
    )

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(**backend_opts)
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_q, input_k, input_v],
                expected_outputs=[sdpa_output_transposed],
                rtol=1.6e-2,
                atol=1e-1,
            )
        )
    elif args.compile_mode == "compile-only":
        # report_precision / n_perf_iters / perf_flops are XRTRunner-only args;
        # strip them for the bare XRTBackend used in compile-only mode.
        runner_only = {"report_precision", "n_perf_iters", "perf_flops"}
        backend = XRTBackend(
            **{k: v for k, v in backend_opts.items() if k not in runner_only}
        )
        module_function = backend.compile(mlir_module)
        print("Compilation complete.")
