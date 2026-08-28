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
    q_tiles_per_core=1,
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
        q_tiles_per_core: q-seq-tiles each core handles per round (default: 1).
            2 halves the round count for a given lq, which is what keeps the
            shim's active-BD count in budget at long sequences.
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
    # head_dim is NOT a tiling axis. Every buffer carries the whole d and the
    # mmul walks it as k-blocks of 8 inside the kernel -- the reference does the
    # same (its per-core q is memref<32x128>, K/V memref<16x128>, d never split).
    # Tiling d instead put dk_chunks puts on one channel per loop iteration and
    # dk_chunks fills on one staging buffer, which defeated BD folding and
    # inflated the outbound lock counts. Capacity comes from a smaller lkp
    # (= tile_size_q), not from splitting d.
    dk_tile = dk
    dk_chunks = 1
    dv_tile = dv
    dv_chunks = 1
    # dv_chunks > 1 (head_dim 128 at lkp 64) keeps ONE score matrix and splits
    # only the PV accumulation, so QK^T / mask / softmax run once per K block --
    # unlike attn_npu2.py, which makes dv_chunks a launch axis and recomputes
    # them per chunk. The cost is dv_chunks live output accumulators in L1.
    #
    # The chunks share ONE [tile_size_q, dv] slab: the accumulator's mmul layout
    # is column-block-major ([dv/M][tile_size_q][M]), so chunk c is exactly the
    # flat range [c*tile_size_q*dv_tile, ...) and the kernel reaches it with a
    # pointer offset. That keeps the tile's output ONE full-dv transfer instead
    # of dv_chunks of them -- the gather is 4 BDs per memtile either way, and 8
    # would exceed the 24 BD IDs a memtile DMA channel can allocate.
    _tsq = lqp // num_q_tiles
    # FULL-d DMA. When head_dim > lkp, chunking dk/dv at the CHANNEL level puts
    # dk_chunks/dv_chunks distinct puts (from distinct memrefs) on one channel
    # inside a single scf.for iteration, which air-opt-memtile-dma-bds cannot
    # fold -- it falls back to unrolling the whole causal-prefix loop into the
    # memtile BD chain (measured: 175 BDs vs the 48 cap; the dh=64 path folds the
    # same 4 rounds into 4). The reference never chunks in the DMA either: its
    # K/V memtile buffer is the full memref<256x128>. So send the whole d in one
    # transfer and do the chunking on pointers in the core.
    full_d_dma = dk_chunks > 1 or dv_chunks > 1
    if full_d_dma:
        assert dk == dv, (
            f"full-d DMA shares one L1 staging buffer for K and V; needs dk == dv "
            f"(got dk={dk}, dv={dv})"
        )
    # The 2-way Q pair broadcast is MANDATORY: replacing it with 2*NR per-tile
    # unicast puts deadlocks the herd on hardware. Verified by a single-variable
    # A/B at dh=64 / L=512 / GQA 1:3 (broadcast PASS, unicast ERT timeout), so
    # it is a property of the unicast channel-bundle scatter, not of head_dim.
    q_pair_bcast = True
    # The staging and the accumulators are never live at the same time
    # (copy_half_tile drains the staging during the Q phase, the accumulators
    # are zero-filled after), so one slab CAN serve as both -- but only while
    # the staging is filled ONCE per round. air-to-aie derives the outbound BD's
    # lock counts from the buffer's fill count, so a slab filled dk_chunks times
    # (one Q broadcast per chunk) but drained once makes the output MM2S acquire
    # dk_chunks tokens while the core releases 1 -> the drain never fires.
    # Confirmed in the lowered IR: at dk_chunks=2 the output BD is
    # AcquireGreaterEqual(lock, 2) against a core Release(lock, 1).
    alias_qpair_gp = dv_chunks > 1 and dk_chunks == 1 and dv == 2 * dk_tile
    # Q staging <-> K/V staging alias. The staging is dead before the first K
    # block arrives, so the two never overlap, and BOTH are inbound-only: no
    # outbound BD sits on the shared buffer, so the fill-count/lock hazard above
    # cannot apply. This is what frees the 16 KB that lets head_dim 128 keep the
    # validated lkp 64 tiling (Q 16K + shared 16K + accumulators 16K + G 8K).
    alias_qpair_kv = full_d_dma and 2 * _tsq * dk_tile == lkp * dk

    # Output accumulator double-buffering (the reference's o_ping/o_pong) keeps
    # round lx's Gp2L2 drain from racing round lx+1's zero_fill. A full-dv slab
    # is single-buffered (there is no L1 room for a second one).
    # The q-round axis is unrolled in-core so cps_lx is a build-time constant.
    # That makes .text scale with the round count against a 16 KB AIE2P program
    # memory: measured 11760 B at 4 rounds, 19328 B at 8. Past 4 rounds fold the
    # axis into an scf.for with an affine causal bound instead -- the reference
    # goes further and keeps its whole core loop L-independent (RTP-driven), so
    # folding is the conservative version of the same idea. Folding costs the
    # accumulator ping-pong (a dynamic slot cannot index a buffer list).
    # The unrolled core emits (rounds x q_tiles_per_core) attention bodies, so
    # that product -- not the round count alone -- is what has to stay small
    # enough for AIE2P program memory. Folding also forces n_ob to 1, which is
    # what keeps q_tiles_per_core accumulators inside L1.
    fold_core_rounds = (lq // lqp) * q_tiles_per_core > 4
    n_ob = (
        1
        if fold_core_rounds or q_tiles_per_core > 1
        else (2 if (lq // lqp) > 1 and dv_chunks == 1 else 1)
    )
    l1_bytes = (
        q_tiles_per_core * _tsq * dk * 2  # Q, one tile set per q-seq-tile
        + (0 if alias_qpair_kv else 2 * _tsq * dk_tile * 2)  # qpair staging
        + (
            0 if alias_qpair_gp else n_ob * q_tiles_per_core * _tsq * dv * 2
        )  # accumulator slabs
        + lkp * dk * 2  # K / V staging (whole d)
        + q_tiles_per_core * _tsq * lkp * 2  # G scores, one per q-seq-tile
        + 4 * _tsq * 2  # up / sp / s_tmp / r_tmp
    )
    assert l1_bytes <= 64 * 1024, (
        f"per-core L1 working set {l1_bytes} B exceeds the 64 KB AIE2P data "
        f"memory (dk_chunks={dk_chunks}, dv_chunks={dv_chunks}, "
        f"tile_size_q={_tsq}, n_ob={n_ob}). At head_dim > lkp the Q staging "
        f"cannot double as the accumulator slab unless dk_chunks == 1, so run "
        f"a smaller lkp (tile_size_q must stay == lkp): head_dim 128 fits at "
        f"lkp 32 / lqp 256 in ~31 KB."
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

    # PHASE 1 TEMPORAL: the array is NB column-BLOCKS (2 physical columns each)
    # x NR physical rows. A block is one q-head of the GQA group; its 2*NR tiles
    # cover 2*NR distinct q-seq-tiles (s = ty*2 + tx%2). K/V are broadcast to the
    # whole array (one kv-head is live per dispatch), Q is partitioned per block.
    # num_heads_per_unroll counts KV-HEADS per dispatch (= segment instances) and
    # the launch head axis iterates kv-head groups.
    #
    # NB (blocks) and NR (rows) are INDEPENDENT: NB follows the GQA group size,
    # NR is the physical row count. They coincide at gqa_group_size == 4 (the
    # Llama-3.2-1B case this was first built for); tying them together would make
    # the q-tiles-per-round (2*NR) shrink with the GQA ratio and stop lqp from
    # dividing the sequence length (gqa 1:3 -> lqp 384 { 2048).
    NB = gqa_group_size
    assert NB * 2 <= 8, (
        f"physical columns = 2 * gqa_group_size ({NB}) must be <= 8; "
        f"got num_heads={num_heads}, num_kv_heads={num_kv_heads}"
    )
    # q_tiles_per_core > 1 gives each core several q-seq-tiles per round, which
    # divides the round count for a given lq. Rounds are what scale the shim's
    # simultaneously-active BD count (the runtime sequence is flat, so a rolled
    # round loop is unrolled into npu insts): at 8 rounds a shim tile carrying 3
    # round-scaled channels needs 24 active BDs against a cap of 16. Two q-tiles
    # per core halves the rounds and takes every channel's peak with it. The
    # tiles a core owns are s, s + 2*NR, ... so tile-set j is exactly output half
    # j -- the lo/hi split becomes the j axis instead of a row split.
    assert num_q_tiles % (2 * q_tiles_per_core) == 0, (
        f"num_q_tiles ({num_q_tiles}) must be divisible by 2 * "
        f"q_tiles_per_core ({q_tiles_per_core})"
    )
    NR = num_q_tiles // (2 * q_tiles_per_core)
    assert NR == 4, (
        f"herd rows = num_q_tiles / (2*q_tiles_per_core) = {NR}, must be 4; "
        f"got num_q_tiles={num_q_tiles}, q_tiles_per_core={q_tiles_per_core}"
    )
    assert q_tiles_per_core in (1, 2), (
        f"q_tiles_per_core must be 1 or 2 (got {q_tiles_per_core}); the output "
        f"gather maps tile-set j onto output half j."
    )
    NC = 2 * NB  # physical columns
    assert num_kv_heads % num_heads_per_unroll == 0, (
        f"num_kv_heads ({num_kv_heads}) must be divisible by "
        f"num_heads_per_unroll ({num_heads_per_unroll})"
    )
    assert num_heads_per_unroll * NC <= 8, (
        f"physical columns = num_heads_per_unroll ({num_heads_per_unroll}) * "
        f"2 * gqa_group_size ({NC}) must be <= 8"
    )
    num_head_groups = num_kv_heads // num_heads_per_unroll

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    M = 8  # mmul_m = mmul_k = mmul_n

    # Derived parameters
    num_lq_iters = lq // lqp
    # Issue each round's output drain inside the K/V loop instead of in a
    # separate pass over the rounds: the shim frees an MM2S BD at its matching
    # wait, so draining afterwards keeps every round's K/V BD active until the
    # launch-iteration boundary. Round-major issue frees round r before r+1 is
    # configured, which is what the reference does (seq_gen.hpp rotates BD ids
    # on round parity and waits only on the output S2MM). Only needed once the
    # round count passes 4, where 8+8+1 would exceed the shim's 16 active BDs.
    _merge_out_into_kv = num_lq_iters > 4
    # The causal triangle gives every round its own DIFFERENT-sized K/V shim
    # transfer, so the round count IS the number of simultaneously in-flight
    # tasks on that MM2S channel. Measured on NPU2: up to 6 rounds run, 7 and
    # 8 deadlock. Past the limit, stream the FULL prefix every round instead --
    # the per-round BDs become identical, AIR folds them into ONE repeated
    # task, and the in-core causal mask (q_block = lx*NQ + s) still discards
    # the extra K blocks, so the result is unchanged. The price is the causal
    # DMA/compute saving (2x the K/V work at large round counts), which is
    # what buys the design the ability to run at all past 6 rounds.
    _MAX_ROUNDS_IN_FLIGHT = 6
    _uniform_cps = num_lq_iters > _MAX_ROUNDS_IN_FLIGHT
    _cps_blocks = lambda lx: (num_lq_iters if _uniform_cps else lx + 1) * NQ
    tile_size_q = lqp // num_q_tiles
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // num_cascade_stages
    lk_per_stage = lkp * chunks_per_stage

    # NQ = q-seq-tiles per round (= 2*NR); distinct from NC, the column count.
    NQ = num_q_tiles
    NS = num_cascade_stages

    # Memory spaces
    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # L1 MemRefTypes (Q and K use dk_tile, not full dk)
    _dk_dma = dk if full_d_dma else dk_tile
    _dv_dma = dv if full_d_dma else dv_tile
    q_l1_t = MemRefType.get([tile_size_q, _dk_dma], bf16, memory_space=l1_space)
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
    k_l1_t = MemRefType.get([lkp, _dk_dma], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv_tile], bf16, memory_space=l1_space)
    g_l1_2d = MemRefType.get([tile_size_q, lkp], bf16, memory_space=l1_space)
    g_l1_1d = MemRefType.get([tile_size_q * lkp], bf16, memory_space=l1_space)
    gp_l1_t = MemRefType.get([tile_size_q, dv_tile], bf16, memory_space=l1_space)
    # One slab holding all dv chunks back to back (== gp_l1_t at dv_chunks == 1).
    gp_slab_l1_t = MemRefType.get([tile_size_q, dv], bf16, memory_space=l1_space)
    up_l1_t = MemRefType.get([tile_size_q, 1], bf16, memory_space=l1_space)

    # L2 MemRefTypes (QK relay uses dk_tile)
    qk_l2_t = MemRefType.get([lkp, _dk_dma], bf16, memory_space=l2_space)
    # Q relay buffer holds a whole row's NQ q-tiles ([lqp, dk_tile]); one QIn get
    # fills it, then NQ disjoint-offset Q2L1 puts scatter per-tile (partition).
    q_relay_l2_t = MemRefType.get([lqp, _dk_dma], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, _dv_dma], bf16, memory_space=l2_space)
    gp_l2_t = MemRefType.get([lqp, dv_tile], bf16, memory_space=l2_space)
    # Half-height output buffer: the reference splits a q-head's output gather across TWO
    # memtiles (first/second half of the sequence), 4 tiles each, to keep each
    # memtile's gather light enough to route alongside the K/V broadcast transit.
    # Full dv wide: the dv_chunks per-tile outputs land in disjoint column slices
    # of the same half-buffer, so one GpOut put still drains a contiguous
    # [lqp/2, dv] block straight into the seq-first L3 output.
    # Output slices. At one q-tile per core these are the two row halves. At
    # two, each half splits again by q-tile set, giving 4 quarter-gathers of
    # 2*NR/2 = 4 sources instead of 2 half-gathers of 8 -- 8 BDs on one memtile
    # channel is what tips the K/V column over its 24 BD-ID budget. Slice
    # (name, j, tyhi) covers output rows [i*rows, (i+1)*rows).
    # Past 4 rounds the K/V column's shim tile is full on K and V alone
    # (one task each per round, 8+8 against its 16 active BDs), so it can carry
    # no output gather at all. Six 4-tile gathers cannot avoid it -- a memtile
    # has six S2MM ports, so two of them (8 flows) never fit on one column.
    # Splitting each block's halves by row instead gives twelve 2-tile gathers,
    # which do fit the five remaining columns; see _out_col for the assignment.
    _row_split_out = q_tiles_per_core == 1 and (lq // lqp) > 4
    # _out_col's table below is a hand-checked column budget for NB == 3 (GQA
    # 3:1, e.g. Llama-3.2-3B) and NB == 4 (GQA 4:1, e.g. Llama-3.2-1B), where
    # the herd spans all 8 columns and sixteen gathers share seven non-K/V
    # columns. Smaller NB leaves the herd narrower than the table assumes, so
    # fail here with the reason rather than letting _out_col raise KeyError
    # halfway through building the module.
    if _row_split_out and NB not in (3, 4):
        raise NotImplementedError(
            f"the row-split output placement past 4 rounds is only mapped for "
            f"the GQA ratio NB = num_heads / num_kv_heads to be 3 or 4; "
            f"got NB={NB} from "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}. Either keep "
            f"lq <= 4 * lqp (= {4 * lqp}) so the four-tile gathers are used, "
            f"or extend _out_col with a verified column budget for this NB."
        )
    if _row_split_out:
        # (name, j, first tile-row, tile-row count) -- one row of the 2xNR block
        # each, ordered so slice i still drains L3 rows [i*lqp/_n_out, ...).
        _out_slices = [
            ("lo", 0, 0, 1),
            ("lo1", 0, 1, 1),
            ("hi", 0, 2, 1),
            ("hi1", 0, 3, 1),
        ]
    else:
        _out_slices = [
            (nm if j == 0 else f"{nm}{j}", j, 2 if hi else 0, 2)
            for j in range(q_tiles_per_core)
            for nm, hi in (("lo", False), ("hi", True))
        ]
    _n_out = len(_out_slices)
    gp_half_l2_t = MemRefType.get([lqp // _n_out, dv], bf16, memory_space=l2_space)

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
    external_func(
        "copy_half_tile",
        [k_l1_t if alias_qpair_kv else qpair_l1_t, q_l1_t, i32],
        link_with="attn_npu2.o",
    )
    external_func("div_gp_sp", [up_l1_t, gp_l1_t], link_with="attn_npu2.o")
    if full_d_dma:
        external_func(
            "matmul_a_b_bf16_chunk",
            [q_l1_t, k_l1_t, g_l1_1d, i32],
            link_with="attn_npu2.o",
        )
        # The Q pair arrives one dk chunk per broadcast; land each in its own
        # chunk slice of the full-d Q buffer.
        external_func(
            "copy_half_tile_at",
            [k_l1_t if alias_qpair_kv else qpair_l1_t, q_l1_t, i32, i32],
            link_with="attn_npu2.o",
        )
    _gp_t = qpair_l1_t if alias_qpair_gp else gp_slab_l1_t
    if dv_chunks > 1:
        # Chunk-indexed variants addressing one dv slice of the accumulator slab
        # (AIE bare-ptr ABI drops subview offsets; pass the index instead).
        external_func("zero_fill_gp_bf16_at", [_gp_t, i32], link_with="attn_npu2.o")
        external_func("mul_r_gp_at", [up_l1_t, _gp_t, i32], link_with="attn_npu2.o")
        external_func(
            "matmul_g_b_bf16_chunk",
            [g_l1_1d, k_l1_t, _gp_t, i32],
            link_with="attn_npu2.o",
        )
        external_func("div_gp_sp_at", [up_l1_t, _gp_t, i32], link_with="attn_npu2.o")
        # Whole-slab forms: one call instead of dv_chunks call sites,
        # which is what keeps the unrolled core inside program memory.
        external_func("zero_fill_gp_bf16_all", [_gp_t], link_with="attn_npu2.o")
        external_func("div_gp_sp_all", [up_l1_t, _gp_t], link_with="attn_npu2.o")
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
    # size=[NR, NC] indexed raw [ty, tx] (mirrors the original per-stage QK2L1_s);
    # only block b's tiles (2b<=tx<=2b+1) consume it (affine-gated in the herd).
    # reference-faithful fanout: per block, per row, ONE 2-way broadcast flow to the
    # row's 2-col pair (bcast_shape [NR,2]) — matches the reference's mem_tile DMA:i -> 2
    # cols. 4 flows/block (one per row) instead of 8 single-tile flows.
    for b in range(NB):
        if q_pair_bcast:
            Channel(f"Q2L1_{b}", size=[NR, 1], broadcast_shape=[NR, 2])
        else:
            # No pair staging buffer at dv_chunks > 1: address each column
            # directly so the tile receives exactly its own q-seq-tile.
            Channel(f"Q2L1_{b}", size=[NR, 2])
    # QIn is PER-BLOCK (one endpoint per col-block) so each block's Q gets its own
    # shim->memtile flow to its own Q memtile (cols 0,2,4,6). A single-endpoint QIn
    # only creates ONE shim->memtile flow, so only one block's Q memtile is fed and
    # the others deadlock (their tiles never receive Q). One endpoint per block.
    Channel("QIn", size=[NB])

    # K+V share ONE consolidated broadcast channel (KV2L1) to ALL NR*NC tiles
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
        broadcast_shape=[num_heads_per_unroll, NR, NC],
    )
    Channel("KIn", size=[num_heads_per_unroll])
    Channel("VIn", size=[num_heads_per_unroll])

    # Output split per col-block into lo (rows 0,1 = seq-tiles 0..3, col 2b) and
    # hi (rows 2,3 = seq-tiles 4..7, col 2b+1) — the reference's 4-way split gather. Each
    # gathers 4 tiles into its own column-pinned memtile, then one GpOut half.
    for b in range(NB):
        for _nm, _j, _rlo, _nrows in _out_slices:
            Channel(f"Gp2L2_{b}_{_nm}", size=[NR, NC])
    # GpOut per (block, half) endpoints so the 8 output streams SPREAD across
    # shim tiles (like the reference's per-memtile DMA:5 output) instead of funneling all 8
    # concurrent memtile sources into ONE shim channel (an 8-to-1 circuit merge
    # that deadlocks).
    Channel("GpOut", size=[NB, _n_out])

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
        # This caps the design at 4 rounds, i.e. lq <= 4*lqp. K, V and the output
        # gather each issue one shim task per round, so at 8 rounds their shared
        # shim tile wants 24 of its 16 simultaneously-active BDs. There is no
        # placement escape: at 8 rounds a shim tile affords only two
        # round-scaled channels, so each column can host at most one gather
        # buffer, the K/V column can host none (K+V alone is 16), and block 1 --
        # whose tiles sit in columns {2,3} -- is then left with a single usable
        # memtile for the 8 flows it must drain. Bounding the in-flight BDs
        # instead (air.preserve_shim_dma_order, or the same awaits with K/V left
        # unpaced) compiles but deadlocks: one K BD
        # carries up to 64 tiles into a 1-tile memtile ring, and the pacing's
        # await-on-drain cannot make progress on a BD that exceeds the ring
        # depth. Lifting this needs the reference's structure -- a runtime-fired
        # ping-pong BD pair driven by RTP loop bounds, whose task count does not
        # scale with the sequence length -- not another placement tweak.
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
                    (
                        AffineConstantExpr.get(num_lq_iters * NQ)
                        if _uniform_cps
                        else AffineExpr.get_mul(
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0), AffineConstantExpr.get(1)
                            ),
                            AffineConstantExpr.get(NQ),
                        )
                    )
                ],
            )

            # Output gets PER ROUND (lx): per block (q-head), TWO halves (lo then
            # hi) matching the segment's per-lx GpOut put order. lo = round lx's
            # first lqp/2 seq rows (at row lx*lqp), hi = second half.
            emb_dim_out = num_heads * dv
            half_rows_l = lqp // _n_out
            # RECTANGULAR in lx (row offset lx*lqp, constant size + channel index),
            # so the round axis is one scf.for that wrap-and-stride folds to a
            # single strided BD per (block, half) endpoint -- collapsing the
            # per-round GpOut drain unroll (num_lq_iters*NB*2 gets -> NB*2 folded).
            _row_lo_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(lqp)
                    )
                ],
            )

            def _row_slice_map(_i):
                return AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(lqp),
                            ),
                            AffineConstantExpr.get(_i * half_rows_l),
                        )
                    ],
                )

            def _emit_out_gets(lx_iv):
                # Slice i is the contiguous row band [i*half_rows_l, ...).
                _out_rows = [
                    (
                        affine_apply(_row_lo_map, [lx_iv])
                        if _i == 0
                        else affine_apply(_row_slice_map(_i), [lx_iv])
                    )
                    for _i in range(_n_out)
                ]
                for kv_local in range(num_heads_per_unroll):
                    for row_slot in range(NB):
                        # q_head = (ly * nhpu + kv_local) * gqa + row_slot.
                        out_col_off = affine_apply(
                            _linmap(
                                num_heads_per_unroll * gqa_group_size * dv,
                                (kv_local * gqa_group_size + row_slot) * dv,
                            ),
                            [ly],
                        )
                        b_idx = ConstantOp(index_type, row_slot)
                        for _i in range(_n_out):
                            ChannelGet(
                                "GpOut",
                                gp,
                                indices=[b_idx, ConstantOp(index_type, _i)],
                                offsets=[_out_rows[_i], out_col_off],
                                sizes=[half_rows_l, dv],
                                strides=[emb_dim_out, 1],
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
                    if full_d_dma:
                        ChannelPut(
                            "KIn",
                            k,
                            indices=[kv_offset_idx],
                            offsets=[0, head_k_off],
                            sizes=[cps_lx, lkp, dk],
                            strides=[lkp * emb_dim_k, emb_dim_k, 1],
                        )
                    else:
                        ChannelPut(
                            "KIn",
                            k,
                            indices=[kv_offset_idx],
                            offsets=[0, head_k_off],
                            sizes=[cps_lx, dk_chunks, lkp, dk_tile],
                            strides=[lkp * emb_dim_k, dk_tile, emb_dim_k, 1],
                        )
                    # V put: causal prefix (cps_lx blocks); bcast to rows. The
                    # dv_chunks axis sits INSIDE the block axis so the stream is
                    # block0/chunk0, block0/chunk1, block1/chunk0, ... matching
                    # the core's per-block "one score matrix, dv_chunks PV" order
                    # (same shape as the K put's dk_chunks axis).
                    ChannelPut(
                        "VIn",
                        v,
                        indices=[kv_offset_idx],
                        offsets=[0, head_v_off],
                        sizes=[cps_lx, lkp, _dv_dma],
                        strides=[lkp * emb_dim_v, emb_dim_v, 1],
                    )
                if _merge_out_into_kv:
                    _emit_out_gets(lx_iv)
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
                    for row_slot in range(NB):
                        # One QIn put PER dk chunk. At full d each put carries
                        # the WHOLE head (the relay is dk wide) and the chunking
                        # happens in the memtile->L1 broadcast, but the relay is
                        # still refilled once per chunk: one fill per drain is
                        # what the memtile's lock accounting expects, and a
                        # single fill feeding dk_chunks broadcasts deadlocks.
                        for _fill in range(dk_chunks * q_tiles_per_core):
                            dk_c = _fill % dk_chunks
                            q_col_off = affine_apply(
                                _linmap(
                                    num_heads_per_unroll * gqa_group_size * dk,
                                    (kv_local * gqa_group_size + row_slot) * dk
                                    + (0 if full_d_dma else dk_c * dk_tile),
                                ),
                                [ly],
                            )
                            ChannelPut(
                                "QIn",
                                q,
                                indices=[ConstantOp(index_type, row_slot)],
                                offsets=[row0, q_col_off],
                                sizes=[lqp, _dk_dma],
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
                # ONE relay buffer per dk / dv chunk. Reusing a single buffer for
                # all chunks makes the per-chunk get/put sequence alias the same
                # memref, which defeats the cyclic-BD folding in air-to-aie and
                # unrolls the K/V loop into the memtile BD chain (175 BDs vs the
                # 48 cap at dk_chunks=dv_chunks=2). The reference alternates
                # distinct buffers (in_0/in_1) for exactly this reason.
                # K/V broadcast memtile. It feeds every compute tile, so it must
                # stay INSIDE the column span -- parking it past NC makes the
                # multicast cross the whole switchbox fabric and the router gives
                # up ("Unable to find a legal routing"). Keep col 3 and move the
                # OUTPUT gathers off it instead (see _out_col below).
                _kv_col = 3
                qk_l2_bufs = []
                for _ in range(NS if full_d_dma else max(NS, dk_chunks)):
                    _kb = AllocOp(qk_l2_t, [], [])
                    _kb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        i32, _kv_col
                    )
                    qk_l2_bufs.append(_kb)
                v_l2_bufs = []
                for _ in range(NS if full_d_dma else max(NS, dv_chunks)):
                    _vb = AllocOp(v_l2_t, [], [])
                    _vb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        i32, _kv_col
                    )
                    v_l2_bufs.append(_vb)
                # One Q relay buffer PER col-block (q-head), each PINNED to its
                # block's even memtile column (0,2,4,6 — the reference's Q memtile columns)
                # via air.memtile_col so place-tiles distributes them instead of
                # clustering all 4 on one central memtile (which piled 32 Q BDs on
                # one MM2S channel). Each memtile then holds ~8 Q BDs (< 24).
                q_relay_l2_bufs = []
                for b in range(NB):
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
                n_ob_l2 = (
                    1
                    if fold_core_rounds
                    else (2 if (num_lq_iters > 1 and dv_chunks == 1) else 1)
                )

                # Output-gather memtiles sit next to their source tiles: lo on
                # col 2b (co-located with that block's Q relay) and hi on 2b+1.
                # Scattering them further to relieve BD pressure is unnecessary
                # once the DMA sends the whole d (full_d_dma) -- and it makes the
                # gather cross columns, which the router rejects.
                # An output gather that lands on the K/V memtile column also
                # shares its SHIM tile, and K/V are by far the heaviest shim
                # channels (one task per kv-head-group per round each): measured
                # 32 + 32 + 8 = 72 tasks on shim (2,0) against 16 on every other
                # shim tile. That is affordable at 4 rounds (both head_dim 64 and
                # 128 pass) but exhausts the tile's 16 active BDs once the rounds
                # double, so at high round counts move just the colliding half to
                # its block's even column -- still inside the block's own column
                # pair {2b, 2b+1}, so the gather routes unchanged.
                # Do NOT consolidate both halves onto one column to keep the K/V
                # column private: three relays on one memtile saturates its
                # switchbox ('aie.connect ... targets same dst'). The shim
                # pressure that sharing causes is relieved by raising
                # num_heads_per_unroll instead, which cuts the kv-head-group
                # launches that scale KIn/VIn.
                # Slice i of block b sits on the block's own column pair. The
                # extra slices q_tiles_per_core adds are steered to whichever of
                # the pair is NOT the K/V column, since that memtile's S2MM
                # channel is the one with no BD-ID headroom.
                def _out_col(b, i):
                    if _row_split_out:
                        if NB == 4:
                            # Sixteen 2-flow gathers, and the herd spans all 8
                            # columns, so every block keeps its own pair and
                            # splits its four slices 2/2. Two per column stays
                            # inside both budgets: four of the six S2MM ports,
                            # and one MM2S each against the two the Q broadcast
                            # leaves free on cols 0/2/4/6.
                            #
                            # Sixteen gathers need eight memtiles, so col 3
                            # carries two even though it is the K/V column --
                            # which is what makes this FIT rather than what
                            # breaks it. A shim logical tile is grouped by
                            # source memtile and holds two S2MM, so col 3's
                            # gathers land on the SAME shim logical tile that
                            # already carries KIn/VIn on its two MM2S, for
                            # eight shim logical tiles total. Spilling off col 3
                            # instead leaves some column with three gathers,
                            # whose odd one needs a second shim tile: ten
                            # logical tiles against the eight physical ones
                            # merge-logical-tiles=false allows, which fails
                            # placement outright.
                            return 2 * b + (i // 2)
                        # NB == 3: twelve 2-flow gathers over the FIVE non-K/V
                        # columns (the herd is only six wide, so there is no
                        # eighth memtile to absorb col 3's share and it stays
                        # private). Every column stays inside both its six S2MM
                        # ports and its six MM2S channels (cols 0/2/4 spend four
                        # of those on the Q broadcast), and each block's spill
                        # lands on a column adjacent to its own pair so the
                        # gather still routes: block 1 reaches col 1 and col 4,
                        # both one hop from its tiles in cols 2 and 3.
                        return {
                            0: [0, 0, 1, 1],
                            1: [2, 2, 1, 4],
                            2: [4, 5, 5, 5],
                        }[
                            b
                        ][i]
                    return 2 * b + (i % 2)

                gp_slice_bufs = []  # [b][slice][n_ob_l2]
                for b in range(NB):
                    # Buffer-major, slice-minor: the double-buffered slots of a
                    # block interleave exactly as they did before the slice axis
                    # existed.
                    _per_slice = [[] for _ in range(_n_out)]
                    for _ in range(n_ob_l2):
                        for i in range(_n_out):
                            _g = AllocOp(gp_half_l2_t, [], [])
                            _g.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(i32, _out_col(b, i))
                            )
                            _g.operation.attributes["air.no_split"] = UnitAttr.get()
                            _per_slice[i].append(_g)
                    gp_slice_bufs.append(_per_slice)

                # L1 allocations passed to herd
                _nqj = 1 if full_d_dma else dk_chunks
                q_saved_bufs = [
                    AllocOp(q_l1_t, [], []) for _ in range(_nqj * q_tiles_per_core)
                ]
                # ONE staging slab reused across dk chunks (copy_half_tile
                # consumes it immediately), doubling as the dv accumulator slab
                # once the Q phase is done (alias_qpair_gp).
                qpair_bufs = [] if alias_qpair_kv else [AllocOp(qpair_l1_t, [], [])]
                qk_buf = AllocOp(k_l1_t, [], [])
                # One score matrix per q-tile the core owns: at q_tiles_per_core
                # > 1 both are live between the shared K get and the shared V get.
                g_l1s = [AllocOp(g_l1_2d, [], []) for _ in range(q_tiles_per_core)]
                # Output accumulator is DOUBLE-BUFFERED across rounds (the reference's
                # o_ping/o_pong): round lx uses gp_l1_pp[lx%n_ob] so round lx+1's
                # zero_fill can't race round lx's Gp2L2 drain (single-buffer +
                # omit_pingpong gave a non-deterministic race that zeroed output).
                # n_ob is forced to 1 at dv_chunks > 1 (no L1 room, see l1_bytes);
                # the round-boundary WAR is serialized in-core instead. When the
                # accumulators alias the Q staging slab there is no buffer here.
                gp_l1_pp = (
                    []
                    if alias_qpair_gp
                    else [
                        AllocOp(gp_slab_l1_t, [], [])
                        for _ in range(n_ob * q_tiles_per_core)
                    ]
                )
                up_l1s = [AllocOp(up_l1_t, [], []) for _ in range(q_tiles_per_core)]
                sp_l1s = [AllocOp(up_l1_t, [], []) for _ in range(q_tiles_per_core)]

                c_nq = ConstantOp(index_type, NC)
                c_r = ConstantOp(index_type, NR)
                c0_seg = ConstantOp(index_type, 0)

                # Per-round (lx) relay, Python-unrolled. Each round's K/V loop
                # uses its OWN qk_l2/v_l2 buffer (qk_l2_bufs[lx]) so the CONSTANT-
                # bound scf.for folds to ONE cyclic ping-pong BD per round on the
                # memtile — the causal triangle is the per-round STATIC trip count.
                for lx in range(num_lq_iters):
                    c_cps_lx = ConstantOp(index_type, _cps_blocks(lx))

                    # Q relay for round lx: one QIn get of the q-head's whole
                    # lqp-row block into q_relay_l2, then NR 2-way broadcasts (per
                    # row) to the col-pair; each col extracts its half in-kernel
                    # (copy_half_tile), mirroring the reference's q+col*lq*dh.
                    # offset[1](ty_i) = (2*ty_i)*(lkp//M)
                    def _q2l1_off_map_for(_j):
                        # row = (_j*2*NR + 2*ty) * tile_size_q, in units of M rows
                        return AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(2 * (lkp // M)),
                                    ),
                                    AffineConstantExpr.get(_j * 2 * NR * (lkp // M)),
                                )
                            ],
                        )

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
                    for b in range(NB):
                        q_relay_l2 = q_relay_l2_bufs[b]
                        # One relay fill per (dk chunk, q-tile): a single fill
                        # feeding several memtile broadcasts is the same
                        # fill-count/lock hazard as a shared staging buffer.
                        for _fill in range(dk_chunks * q_tiles_per_core):
                            dk_c = _fill % dk_chunks
                            _j = _fill // dk_chunks
                            ChannelGet(
                                "QIn",
                                q_relay_l2.result,
                                indices=[ConstantOp(index_type, b)],
                            )
                            for _ in range(1):
                                # NR per-row 2-way broadcasts as a spatial
                                # scf.forall: the Q2L1_b bundle ROW index is the
                                # forall IV (a legal bundle index, unlike scf.for),
                                # col-pair via broadcast_shape. Makes the NR-way row
                                # scatter explicit; AIR unrolls it to the same BDs
                                # (consistent with the Gp2L2 gather).
                                par_q = scf.ForallOp(
                                    lower_bounds=[0], upper_bounds=[NR], steps=[1]
                                )
                                with InsertionPoint(par_q.body):
                                    ty_i = par_q.induction_variables[0]
                                    ChannelPut(
                                        f"Q2L1_{b}",
                                        q_relay_l2.result,
                                        indices=[ty_i, c0_seg],
                                        offsets=[
                                            dk_c * (dk_tile // M) if full_d_dma else 0,
                                            affine_apply(_q2l1_off_map_for(_j), [ty_i]),
                                            0,
                                            0,
                                        ],
                                        sizes=[dk_tile // M, 2 * (lkp // M), M, M],
                                        strides=[M, _dk_dma * M, _dk_dma, 1],
                                    )
                                    scf.InParallelOp()
                    # K+V relay for round lx: cps_lx blocks on ONE broadcast
                    # channel, K then V interleaved (herd's K-then-V gets stay in
                    # FIFO order). Packs both onto the tiles' one shared S2MM.
                    # Per block: dk_chunks K slices then dv_chunks V slices, the
                    # same order the launch-side 4D descriptors emit them in.
                    for chunk_iter in scf_range(0, c_cps_lx, 1):
                        for _dk_c in range(1 if full_d_dma else dk_chunks):
                            qk_l2 = qk_l2_bufs[0 if full_d_dma else _dk_c]
                            ChannelGet("KIn", qk_l2.result, indices=[seg_x])
                            ChannelPut(
                                "KV2L1",
                                qk_l2.result,
                                indices=[seg_x, c0_seg, c0_seg],
                                offsets=[0, 0, 0, 0],
                                sizes=[_dk_dma // M, lkp // M, M, M],
                                strides=[M, _dk_dma * M, _dk_dma, 1],
                            )
                        for _dv_c in range(1 if full_d_dma else dv_chunks):
                            v_l2 = v_l2_bufs[0 if full_d_dma else _dv_c]
                            ChannelGet("VIn", v_l2.result, indices=[seg_x])
                            ChannelPut(
                                "KV2L1",
                                v_l2.result,
                                indices=[seg_x, c0_seg, c0_seg],
                                offsets=[0, 0, 0, 0],
                                sizes=[_dv_dma // M, lkp // M, M, M],
                                strides=[M, _dv_dma * M, _dv_dma, 1],
                            )
                        yield_([])

                # ----------------------------------------------------------
                # Herd: [NC, NR] — NB col-blocks (2 cols each) = the NB q-heads
                # sharing this kv-head; NR rows x 2 local cols = 2*NR q-seq-tiles.
                # ----------------------------------------------------------
                herd_operands = (
                    q_saved_bufs
                    + qpair_bufs
                    + [qk_buf]
                    + g_l1s
                    + gp_l1_pp
                    + up_l1s
                    + sp_l1s
                    + [seg_x]
                )

                @herd(
                    name="herd_0",
                    sizes=[c_nq, c_r],
                    operands=herd_operands,
                    link_with="attn_npu2.o",
                )
                def herd_body(tx, ty, hsx, hsy, *all_args):
                    # Unpack: Q bufs, the qpair staging slab, then qk, g, the gp
                    # accumulator slabs (none when they alias the staging), up,
                    # sp, seg_x.
                    _nqj = 1 if full_d_dma else dk_chunks
                    _n_q = _nqj * q_tiles_per_core
                    # q_bufs_j[j] is the Q tile set for the core's j-th q-seq-tile.
                    q_bufs_j = [
                        list(all_args[j * _nqj : (j + 1) * _nqj])
                        for j in range(q_tiles_per_core)
                    ]
                    if alias_qpair_kv:
                        base = _n_q
                    else:
                        qpair_slab = all_args[_n_q]
                        base = _n_q + 1
                    qk = all_args[base]
                    if alias_qpair_kv:
                        qpair_slab = qk
                    _p = base + 1
                    g_bufs = list(all_args[_p : _p + q_tiles_per_core])
                    _p += q_tiles_per_core
                    n_gp = 0 if alias_qpair_gp else n_ob * q_tiles_per_core
                    gp_pp = list(all_args[_p : _p + n_gp])
                    _p += n_gp
                    up_bufs = list(all_args[_p : _p + q_tiles_per_core])
                    _p += q_tiles_per_core
                    sp_bufs = list(all_args[_p : _p + q_tiles_per_core])
                    _p += q_tiles_per_core
                    h_seg_x = all_args[_p]

                    # CAUSAL DMA-TRIANGLE SKIP: loop the q-chunks (lx) IN-CORE,
                    # build-time unrolled. Round lx processes q-seq-tiles
                    # [lx*NQ : lx*NQ+NQ] and consumes only its causal prefix of
                    # cps_lx = (lx+1)*NQ K-blocks (matching the segment's per-lx
                    # stream). q_base = lx*NQ is a build-time constant (no counter).
                    # Online-softmax state (gp/sp/up) re-inits per round (each
                    # q-chunk is an independent attention).
                    _cps_h_map = AffineMap.get(
                        0,
                        1,
                        [
                            (
                                AffineConstantExpr.get(num_lq_iters * NQ)
                                if _uniform_cps
                                else AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0) + AffineConstantExpr.get(1),
                                    AffineConstantExpr.get(NQ),
                                )
                            )
                        ],
                    )
                    _qbase_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0), AffineConstantExpr.get(NQ)
                            )
                        ],
                    )
                    _core_rounds = (
                        scf_range(0, ConstantOp(index_type, num_lq_iters), 1)
                        if fold_core_rounds
                        else range(num_lq_iters)
                    )
                    for lx in _core_rounds:
                        # cps and the q-tile base are build-time constants when
                        # the axis is unrolled and come from the loop IV when it is
                        # folded. Both are materialised at their point of use so the
                        # unrolled path emits exactly the ops it always did.
                        if fold_core_rounds:
                            _cps_bound = lambda: affine_apply(_cps_h_map, [lx])
                            _q_base = lambda: arith.IndexCastOp(
                                i32, affine_apply(_qbase_map, [lx])
                            ).result
                            slot = 0
                        else:
                            _cps_bound = lambda: ConstantOp(index_type, _cps_blocks(lx))
                            _q_base = lambda: ConstantOp(i32, lx * NQ).result
                            slot = lx % n_ob
                        # One accumulator slab holding all dv chunks; the ping-pong
                        # slot rotates per round (the reference's o_ping/o_pong)
                        # when n_ob > 1.
                        # One accumulator per q-tile the core owns.
                        gps_j = [
                            (
                                qpair_slab
                                if alias_qpair_gp
                                else gp_pp[slot * q_tiles_per_core + j]
                            )
                            for j in range(q_tiles_per_core)
                        ]
                        gp = gps_j[0]

                        def _gp_call(name, pre_args, c, _gp=None):
                            """Call a gp kernel on dv chunk c of an accumulator."""
                            _gp = gps_j[0] if _gp is None else _gp
                            if dv_chunks == 1:
                                CallOp([], name, pre_args + [_gp])
                            else:
                                CallOp(
                                    [],
                                    name + "_at",
                                    pre_args + [_gp, ConstantOp(i32, c)],
                                )

                        # === INIT (per round) ===
                        # NOTE: zero-filling the accumulators must come AFTER the Q
                        # phase when they alias the staging slab -- see below.

                        # === Q PARTITIONED GET (column-block) for round lx ===
                        # tile is in col-block b = tx//2 (q-head), seq-tile
                        # s = ty*2 + tx%2. Gated on tx==2b+lc so only that col reads
                        # the block's row-ty 2-way broadcast (index lc); then
                        # copy_half_tile extracts this col's half into q_bufs.
                        # The dk-chunk loop sits INSIDE the affine.if so a tile
                        # emits ONE guarded region per (b, lc) instead of one per
                        # (b, lc, chunk). Same per-bundle FIFO order (the segment
                        # also streams a block's chunks back to back), a quarter
                        # of the guard scaffolding at dk_chunks=4 -- which is what
                        # keeps the unrolled core inside AIE2P program memory.
                        for b in range(NB):
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
                                if_q = affine.AffineIfOp(col_set, cond_operands=[tx])
                                with InsertionPoint(if_q.then_block):
                                    for _j in range(q_tiles_per_core):
                                        q_bufs = q_bufs_j[_j]
                                        for dk_c in range(dk_chunks):
                                            ChannelGet(
                                                f"Q2L1_{b}",
                                                qpair_slab,
                                                indices=[
                                                    ty,
                                                    ConstantOp(index_type, lc),
                                                ],
                                            )
                                            if full_d_dma:
                                                CallOp(
                                                    [],
                                                    "copy_half_tile_at",
                                                    [
                                                        qpair_slab,
                                                        q_bufs[0],
                                                        ConstantOp(i32, lc),
                                                        ConstantOp(i32, dk_c),
                                                    ],
                                                )
                                            else:
                                                CallOp(
                                                    [],
                                                    "copy_half_tile",
                                                    [
                                                        qpair_slab,
                                                        q_bufs[dk_c],
                                                        ConstantOp(i32, lc),
                                                    ],
                                                )
                                    affine.AffineYieldOp([])

                        # === ACCUMULATOR INIT (per q-tile) ===
                        for _j in range(q_tiles_per_core):
                            if dv_chunks == 1:
                                _gp_call("zero_fill_gp_bf16", [], 0, gps_j[_j])
                            else:
                                CallOp([], "zero_fill_gp_bf16_all", [gps_j[_j]])
                            CallOp([], "zero_fill_sp_bf16", [sp_bufs[_j]])
                            CallOp([], "neg_inf_fill_up_bf16", [up_bufs[_j]])

                        # === K CHUNK LOOP (bound = the round's causal prefix) ===
                        for chunk_iter in scf_range(0, _cps_bound(), 1):
                            # 1. Zero fill each q-tile's G. Done BEFORE the K get
                            # so the single-q-tile case emits its ops in the order
                            # it always did.
                            g1ds = [
                                CollapseShapeOp(g_l1_1d, g_bufs[_j], [[0, 1]])
                                for _j in range(q_tiles_per_core)
                            ]
                            for _j in range(q_tiles_per_core):
                                CallOp([], "zero_fill_g_bf16", [g1ds[_j]])

                            # 2. ONE K get shared by every q-tile this core owns
                            # (a get per q-tile would re-send K and re-inflate the
                            # shim traffic this split exists to reduce), then a
                            # QK^T per q-tile.
                            if full_d_dma:
                                ChannelGet("KV2L1", qk, indices=[h_seg_x, ty, tx])
                                for _j in range(q_tiles_per_core):
                                    for dk_c in range(dk_chunks):
                                        CallOp(
                                            [],
                                            "matmul_a_b_bf16_chunk",
                                            [
                                                q_bufs_j[_j][0],
                                                qk,
                                                g1ds[_j],
                                                ConstantOp(i32, dk_c),
                                            ],
                                        )
                            else:
                                for dk_c in range(dk_chunks):
                                    ChannelGet("KV2L1", qk, indices=[h_seg_x, ty, tx])
                                    for _j in range(q_tiles_per_core):
                                        CallOp(
                                            [],
                                            "matmul_a_b_bf16",
                                            [q_bufs_j[_j][dk_c], qk, g1ds[_j]],
                                        )

                            # 3. V is fetched AFTER the mask/softmax below (it lands
                            # in the same local buffer the K blocks just vacated,
                            # one get per dv chunk).

                            # 4. Causal mask: q_block = lx*NQ + s (s = ty*2 + tx%2),
                            # kv_block = chunk_iter. The per-lx DMA skip is coarse
                            # (whole prefix); the fine diagonal within the prefix is
                            # masked here (the reference masks the diagonal round too).
                            # 4/5. Mask + softmax + accumulator rescale, per
                            # q-tile. The core's j-th tile is seq-tile
                            # s + j*(2*NR), so only q_block changes with j.
                            _stmps, _rtmps = [], []
                            for _j in range(q_tiles_per_core):
                                if causal:
                                    kv_blk_r = arith.IndexCastOp(i32, chunk_iter).result
                                    ty_i32 = arith.IndexCastOp(i32, ty).result
                                    tx_i32 = arith.IndexCastOp(i32, tx).result
                                    c2_i32 = ConstantOp(i32, 2)
                                    s_val = arith.AddIOp(
                                        arith.MulIOp(ty_i32, c2_i32.result).result,
                                        arith.RemUIOp(tx_i32, c2_i32.result).result,
                                    )
                                    if _j:
                                        s_val = arith.AddIOp(
                                            s_val.result,
                                            ConstantOp(i32, _j * 2 * NR).result,
                                        )
                                    q_block = arith.AddIOp(_q_base(), s_val.result)
                                    CallOp(
                                        [],
                                        "apply_causal_mask",
                                        [g_bufs[_j], q_block.result, kv_blk_r],
                                    )
                                s_tmp = AllocOp(up_l1_t, [], [])
                                r_tmp = AllocOp(up_l1_t, [], [])
                                _stmps.append(s_tmp)
                                _rtmps.append(r_tmp)
                                CallOp(
                                    [],
                                    "fused_softmax",
                                    [
                                        g1ds[_j],
                                        up_bufs[_j],
                                        s_tmp.result,
                                        r_tmp.result,
                                    ],
                                )
                                for c in range(dv_chunks):
                                    _gp_call("mul_r_gp", [r_tmp.result], c, gps_j[_j])

                            # 6. ONE V get shared by every q-tile, then a PV per
                            # q-tile. r/s stay live across the get, hence the lists.
                            if full_d_dma:
                                ChannelGet("KV2L1", qk, indices=[h_seg_x, ty, tx])
                                for _j in range(q_tiles_per_core):
                                    for c in range(dv_chunks):
                                        if dv_chunks == 1:
                                            CallOp(
                                                [],
                                                "matmul_g_b_bf16",
                                                [g1ds[_j], qk, gps_j[_j]],
                                            )
                                        else:
                                            CallOp(
                                                [],
                                                "matmul_g_b_bf16_chunk",
                                                [
                                                    g1ds[_j],
                                                    qk,
                                                    gps_j[_j],
                                                    ConstantOp(i32, c),
                                                ],
                                            )
                            else:
                                for c in range(dv_chunks):
                                    ChannelGet("KV2L1", qk, indices=[h_seg_x, ty, tx])
                                    for _j in range(q_tiles_per_core):
                                        _gp_call(
                                            "matmul_g_b_bf16",
                                            [g1ds[_j], qk],
                                            c,
                                            gps_j[_j],
                                        )
                            for _j in range(q_tiles_per_core):
                                CallOp(
                                    [],
                                    "accum_sp_r_s",
                                    [
                                        sp_bufs[_j],
                                        _rtmps[_j].result,
                                        _stmps[_j].result,
                                    ],
                                )
                                CallOp(
                                    [],
                                    "vector_copy_32elems",
                                    [
                                        ConstantOp(i32, 0),
                                        _stmps[_j].result,
                                        sp_bufs[_j],
                                    ],
                                )
                            for _j in range(q_tiles_per_core):
                                DeallocOp(_stmps[_j])
                                DeallocOp(_rtmps[_j])
                            yield_([])

                        # === OUTPUT for round lx (normalize + split lo/hi puts) ===
                        # NS=1: this tile looped its causal K prefix in-core, so its
                        # (gp, sp) is the final numerator + denominator. Normalize
                        # and emit directly (no cascade merge).
                        for _j in range(q_tiles_per_core):
                            if dv_chunks == 1:
                                _gp_call("div_gp_sp", [sp_bufs[_j]], 0, gps_j[_j])
                            else:
                                CallOp([], "div_gp_sp_all", [sp_bufs[_j], gps_j[_j]])
                        s0o = AffineSymbolExpr.get(0)  # tx
                        s1o = AffineSymbolExpr.get(1)  # ty
                        for b in range(NB):
                            # At one q-tile per core the halves are a ROW split
                            # (rows 0,1 -> lo, rows 2,3 -> hi). At two, a core's
                            # tile-set j IS half j (j=0 covers seq-tiles 0..2*NR-1,
                            # j=1 the rest), so every row emits both halves and the
                            # guard is just "is tx in block b".
                            _blk = [
                                s0o - AffineConstantExpr.get(2 * b),
                                AffineConstantExpr.get(2 * b + 1) - s0o,
                            ]

                            def _row_constraints(_rlo, _nrows):
                                # A full half keeps its single-sided guard so the
                                # two-slice IR is unchanged; a single row needs
                                # both sides.
                                if _nrows == 2:
                                    return [
                                        (
                                            (s1o - AffineConstantExpr.get(2))
                                            if _rlo == 2
                                            else (AffineConstantExpr.get(1) - s1o)
                                        )
                                    ]
                                # A single row is an EQUALITY, not a pair of
                                # inequalities: air reads this set to work out
                                # which herd tiles actually reach the conditional
                                # put, and only the equality form pins ty to one
                                # row. With two inequalities it keeps flows from
                                # tiles that never put, and the gather then waits
                                # on data nobody sends.
                                return [s1o - AffineConstantExpr.get(_rlo)]

                            _halves = [
                                (
                                    _nm,
                                    IntegerSet.get(
                                        0,
                                        2,
                                        _blk + _row_constraints(_rlo, _nrows),
                                        [False, False] + [_nrows == 1],
                                    ),
                                    _j,
                                )
                                for _nm, _j, _rlo, _nrows in _out_slices
                            ]
                            for half_name, half_set, _j in _halves:
                                if_o = affine.AffineIfOp(
                                    half_set, cond_operands=[tx, ty]
                                )
                                with InsertionPoint(if_o.then_block):
                                    # ONE put for the whole d. The slab's mmul
                                    # layout is column-block-major over all dv
                                    # chunks ([dv/M][tile_size_q][M]), so the same
                                    # 4D de-tiling descriptor that emits row-major
                                    # [tile_size_q, dv_tile] for one chunk emits
                                    # row-major [tile_size_q, dv] for the slab.
                                    ChannelPut(
                                        f"Gp2L2_{b}_{half_name}",
                                        gps_j[_j],
                                        indices=[ty, tx],
                                        offsets=[0, 0, 0, 0],
                                        sizes=[
                                            tile_size_q // M,
                                            M,
                                            dv // M,
                                            M,
                                        ],
                                        strides=[M * M, M, tile_size_q * M, 1],
                                    )
                                    affine.AffineYieldOp([])
                        if fold_core_rounds:
                            yield_([])

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

                _gp_lc_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_size_q),
                        )
                    ],
                )

                def _gp_row_map(_rlo):
                    return AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(_rlo),
                            )
                        ],
                    )

                # The gather target is [lqp/2, dv] wide and each source tile now
                # sends its whole d in one transfer, so a tile is one full-width
                # row block -- 4 BDs per memtile, the same as at head_dim 64.
                def _gp_gather(chan, buf, col_map, rlo, nrows, b_blk=0):
                    # At one q-tile per core a half comes from 2 of the 4 rows
                    # (tyhi shifts to rows 2,3). At two, half j comes from tile-set
                    # j on EVERY row, so the forall spans all NR rows and the row
                    # shift disappears.
                    if nrows == 1:
                        # One tile row: just two gets with constant bundle
                        # indices. A forall here buys nothing (the row index is
                        # not an IV) and a degenerate [1, 2] one leaves that
                        # index an affine_apply on an always-zero IV, which does
                        # not resolve to a constant channel-bundle index.
                        for _lc in range(2):
                            ChannelGet(
                                chan,
                                buf,
                                indices=[
                                    ConstantOp(index_type, rlo),
                                    ConstantOp(index_type, 2 * b_blk + _lc),
                                ],
                                offsets=[_lc * tile_size_q, 0],
                                sizes=[tile_size_q, dv],
                                strides=[dv, 1],
                            )
                        return
                    par = scf.ForallOp(
                        lower_bounds=[0, 0], upper_bounds=[nrows, 2], steps=[1, 1]
                    )
                    with InsertionPoint(par.body):
                        j = par.induction_variables[0]
                        lc = par.induction_variables[1]
                        ty_idx = j if rlo == 0 else affine_apply(_gp_row_map(rlo), [j])
                        ChannelGet(
                            chan,
                            buf,
                            indices=[ty_idx, affine_apply(col_map, [lc])],
                            offsets=[affine_apply(_gp_off_map, [j, lc]), 0],
                            sizes=[tile_size_q, dv],
                            strides=[dv, 1],
                        )
                        scf.InParallelOp()

                # At dv_chunks == 1 the rounds alternate L2 buffers, so they stay
                # Python-unrolled. At dv_chunks > 1 there is a single buffer and
                # every round's descriptors are identical, so one scf.for folds
                # the round axis into a cyclic BD chain instead of num_lq_iters
                # copies -- the 48-block memtile_dma cap cannot hold the unrolled
                # form once the dv chunks double the gather.
                # Fold the gather's round axis whenever the L2 relay is single
                # buffered (n_ob_l2 == 1), i.e. at dv_chunks > 1 or once the core
                # rounds are folded. Unrolled, a gather costs 4 BDs per round, so
                # two gathers on one memtile blow the 48-block cap at 8 rounds.
                _fold_rounds = dv_chunks > 1 or fold_core_rounds
                _round_iter = (
                    scf_range(0, ConstantOp(index_type, num_lq_iters), 1)
                    if _fold_rounds
                    else range(num_lq_iters)
                )
                for lx in _round_iter:
                    for b in range(NB):
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
                        for _i, (_nm, _j, _rlo, _nrows) in enumerate(_out_slices):
                            _gb = gp_slice_bufs[b][_i][
                                0 if _fold_rounds else lx % n_ob_l2
                            ]
                            _gp_gather(
                                f"Gp2L2_{b}_{_nm}",
                                _gb.result,
                                _gp_col_map,
                                _rlo,
                                _nrows,
                                b,
                            )
                            ChannelPut(
                                "GpOut",
                                _gb.result,
                                indices=[
                                    ConstantOp(index_type, b),
                                    c0_seg if _i == 0 else ConstantOp(index_type, _i),
                                ],
                            )
                    if _fold_rounds:
                        yield_([])

                # Deallocs for segment-level buffers
                for q_buf in q_saved_bufs:
                    DeallocOp(q_buf)
                for qp in qpair_bufs:
                    DeallocOp(qp)
                DeallocOp(qk_buf)
                for _gb in g_l1s:
                    DeallocOp(_gb)
                for _gpb in gp_l1_pp:
                    DeallocOp(_gpb)
                for _ub in up_l1s:
                    DeallocOp(_ub)
                for _sb in sp_l1s:
                    DeallocOp(_sb)
                for _b in v_l2_bufs:
                    DeallocOp(_b)
                for _b in qk_l2_bufs:
                    DeallocOp(_b)
                for q_rbuf in q_relay_l2_bufs:
                    DeallocOp(q_rbuf)
                for _i in range(_n_out):
                    for _blk in gp_slice_bufs:
                        for _gb in _blk[_i]:
                            DeallocOp(_gb)

            if not _merge_out_into_kv:
                _c_nlq_out = ConstantOp(index_type, num_lq_iters)
                for lx_iv in scf_range(0, _c_nlq_out, 1):
                    _emit_out_gets(lx_iv)
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
        "--q-tiles-per-core",
        type=int,
        default=1,
        help="q-seq-tiles each core handles per round (1 or 2). 2 halves "
        "the round count, keeping the shim active-BD count in budget at "
        "long sequences.",
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
        q_tiles_per_core=args.q_tiles_per_core,
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
