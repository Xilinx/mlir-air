# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention on FastFlowLM's CU topology -- NPU2 (AIE2P).

Sibling of ``attn_npu2.py``. Same kernels, same online softmax; what differs is
the spatial mapping, which is transcribed from FLM's Gemma4 prefill attention
(``FLM_Xclbin/Gemma4/attention_DH_256_prefill``) rather than derived here.

FLM's dispatch loop (``gemma4e_npu_sequence.cpp:1132``) is the specification::

    int num_cu = 4;                    // "each CU has 8 CTs and works on 1 head"
    for (head = 0; head < Heads/num_cu; head++)
      for (round = 0; round < down_rounds; round++)   // 128 q rows per round
        for (cu = 0; cu < num_cu; cu++)               // 2 MM2S on IT[cu*2]
          ...
        ... ONE k and ONE v send from IT[3], shared by all four CUs ...

So a **compute unit is a head** and the CU's eight cores split the query rows,
while K and V cross L3 once per ``num_cu`` heads. ``attn_npu2.py`` instead puts
the q-tile on the herd's x axis with a 4-stage cascade over K on its y axis and
iterates heads in the launch, so every head re-reads all of K and V::

    K bytes = (lq/lqp) * num_heads * (dk/dv_tile) * lk * dk * 2

That ``num_heads`` factor is the whole reason this file exists. Under MQA every
head wants the SAME K and V, so the read can be shared -- but only if the heads
are co-resident, which means spatial.

Here each head is its own herd of ``[cu_cols, num_q_tiles/cu_cols]`` pinned at
``at=(cu_cols*head, 2)``, and K/V is declared ``size=[1,1,1]`` with
``broadcast_shape=[num_heads, cu_cols, rows]`` -- ONE put per chunk, reaching
every core of every herd.

The cascade is gone with it. There is no free axis for it once heads are
spatial, and it is not needed: each core walks ALL of K itself and finishes its
own online softmax, exactly as FLM's kernel does. That also takes the merge
(maximum/exp/mul/add over three cascade channels) off the critical path, at the
cost of every core doing the full K walk -- which is free, because this kernel
is bandwidth-bound, not FLOP-bound.

THE MAPPING IS NOT FREE-CHOICE. Three rules, each found by a probe in
``~/air_tools/gemma4_campaign`` that hung until it was obeyed:

* **One Q relay per memtile column.** Two relay buffers pinned to the same
  column deadlock. FLM puts both of a CU's q streams into MT[j*2]; AIR cannot,
  so each core column's relay takes its own memtile column.
* **A broadcast channel's puts must all leave from ONE memtile.** K staged on
  one column and V on another places and then hangs.
* **K and V therefore share ONE shim link**, as a single strided send over an
  interleaved L3 record, which is also what makes the shim budget close: 8 Q
  relays + 1 K/V = 9 MM2S over 8 columns, so the K/V column can still carry a Q
  relay. Two K/V links would need a column with no Q relay on it, leaving seven
  columns for eight relays, and the allocator says so ("no ShimNOCTile has
  sufficient DMA capacity"). FLM already keeps K and V in one BO
  (``v_offset = k_offset + KV_CACHE_SIZE/2``).

Hence ``KV`` is one tensor of ``num_chunks`` records, each holding a K tile
followed by the ``dv_chunks`` V tiles of that chunk. The records are of unequal
width (K is ``lkp*dk``, V is ``lkp*dv_tile``) and that is fine -- one put, four
gets of three shapes.

The d-value chunking is INSIDE the core, not a launch axis: the scores ``g`` are
computed once per K chunk and reused for every ``dv_tile`` slice of the output.
A launch axis would recompute QK^T and re-read all of K once per chunk.

UNDER A SLIDING WINDOW THE K/V DMA IS TRUNCATED, as FLM's is: a round streams
the window plus its own q tiles, not the sequence. Because the window is
BOUNDED the per-round extent is constant and only the start moves, so the round
axis stays traced -- where ``attn_npu2_temporal_causal``'s prefix GROWS and has
to be unrolled in Python to keep every transfer size static. See
``kv_win_table``. Plain ``causal`` (no window) still streams all of K and masks
it, as the sibling does.

L1 budget per core, which is what bounds the tiling::

    q_saved  tile_size_q * dk               * 2
    qk       lkp         * dk               * 2
    v_l1     lkp         * dv_tile          * 2
    gp       tile_size_q * dv_tile * dv_chunks * 2
    g        tile_size_q * lkp              * 2
    up, sp   tile_size_q * 2                * 2
"""

import argparse
from math import sqrt

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt_runner import XRTRunner

KERNEL = "attn_npu2.o"
M = 8  # mmul<8,8,8>


def build_launch(
    lk=2048,
    lkp=32,
    lq=2048,
    lqp=256,
    dk=256,
    dv=256,
    num_q_tiles=8,
    num_heads=4,
    cu_cols=2,
    num_kv_heads=1,
    causal=True,
    window=None,
    dv_tile=128,
    causal_skip=False,
    rounds=None,
    q_round_base=0,
    kv_blocks=None,
):
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    assert (
        num_q_tiles % cu_cols == 0
    ), f"num_q_tiles ({num_q_tiles}) must be divisible by cu_cols ({cu_cols})"
    assert (
        num_heads * cu_cols <= 8
    ), f"a herd per head, {cu_cols} columns wide, needs {num_heads * cu_cols} <= 8"

    # d is NEVER split, as in attn_npu2_temporal_causal: the microkernel is
    # compiled with dk_tile = head_dim, so a builder that chunks d disagrees
    # with the kernel it calls. Capacity at head_dim > lkp comes from a smaller
    # lkp (tile_size_q must stay == lkp), not from d. Verified the hard way:
    # dk_chunks=16 gave a degenerate softmax (uniform output, corr 0.006) while
    # the identical dataflow at dk_chunks=1 passes at 0.9977.
    if dv_tile is None:
        dv_tile = lkp
    assert dv % dv_tile == 0, f"dv ({dv}) must be divisible by dv_tile ({dv_tile})"
    assert dv_tile % M == 0, f"dv_tile ({dv_tile}) must be a multiple of the mmul {M}"
    dv_chunks = dv // dv_tile

    NH = num_heads
    HX = cu_cols
    NQT = num_q_tiles
    HY = NQT // HX

    tile_size_q = lqp // NQT
    if causal:
        assert lq == lk, f"Causal masking requires lq == lk, got lq={lq}, lk={lk}"
        assert tile_size_q == lkp, (
            f"Causal masking requires tile_size_q == lkp, got "
            f"tile_size_q={tile_size_q}, lkp={lkp}"
        )

    # causal_skip is accepted and IGNORED, as in attn_npu2_temporal_causal
    # ("this design has no cascade, and the causal skip is in the DMA rather
    # than a choice"). Guarding the per-block arithmetic on (q_block, kv_block)
    # hangs at this shape -- correctness is unaffected, since apply_causal_mask
    # / apply_window_mask still zero the dead blocks; only the wasted
    # arithmetic stays, and the real cure is truncating the K/V DMA itself.
    del causal_skip
    window_blocks = None
    if window is not None:
        assert causal, "window requires causal=True"
        assert (
            window % lkp == 0
        ), f"window ({window}) must be a multiple of the block size lkp ({lkp})"
        window_blocks = window // lkp

    # The whole point: one K/V stream for every head. GQA would need one stream
    # per kv head, which is a different (and still useful) shape -- refuse it
    # here rather than silently computing the wrong thing.
    assert num_kv_heads == 1, (
        f"attn_npu2_headspatial shares ONE K/V broadcast across all heads, which "
        f"is only correct under MQA; got num_kv_heads={num_kv_heads}. Use "
        f"attn_npu2.py for GQA/MHA."
    )

    num_lq_iters = lq // lqp
    num_chunks = lk // lkp

    # One GROUP of the causal staircase. Plain causal streams every K/V block
    # on every round, but round lx only needs the first NQT*(lx+1) of them, and
    # a growing extent cannot be a single launch (air.api needs the size
    # static). So the round axis is cut into groups, each a launch of its own
    # with a constant, larger-than-it-needs extent, and the groups are stitched
    # into one ELF -- one dispatch, so the per-dispatch cost is paid once.
    n_rounds = num_lq_iters if rounds is None else rounds
    assert q_round_base + n_rounds <= num_lq_iters, (q_round_base, n_rounds)
    if q_round_base or rounds is not None or kv_blocks is not None:
        assert causal and window is None, "grouping is for the plain-causal path"

    # Window truncation of the K/V DMA, which is FLM's gen_swa_engine_seq: per
    # round it computes kv_begin = max(0, Lq_current - window) and issues that
    # round's K/V BDs over kv_length rows only, not the whole sequence
    # (gemma4e_npu_sequence.cpp:1137-1250).
    #
    # A BOUNDED window is why this needs no Python unroll of the round axis.
    # temporal_causal's causal prefix GROWS, so its per-round transfer size
    # changes and the round axis has to be unrolled to keep every size static;
    # here the extent is constant -- the window plus this round's own q tiles --
    # and only the start moves. A static size with a moving start is exactly
    # what air.api's subscript rule allows, so the round axis stays traced and
    # the core program is untouched.
    #
    # A round covers q blocks [NQT*lx, NQT*lx + NQT), and q block qb reads k
    # blocks [qb - window_blocks, qb], so the union is NQT + window_blocks
    # blocks starting at NQT*lx - window_blocks. Rounds before the window fills
    # clamp to 0 and over-read FORWARD, which the causal half of the mask
    # already discards; exact per-round extents would save another 12% of the
    # K/V traffic and cost the unroll, which is not a trade worth making.
    kv_win_table = None
    if window_blocks is not None and NQT + window_blocks < num_chunks:
        kv_win_blocks = NQT + window_blocks
        kv_win_table = [max(0, NQT * lx - window_blocks) for lx in range(num_lq_iters)]
        assert all(t + kv_win_blocks <= num_chunks for t in kv_win_table), (
            f"window stream [{kv_win_blocks} blocks] runs past the sequence; "
            f"table={kv_win_table}, num_chunks={num_chunks}"
        )
    kv_stream_blocks = kv_win_blocks if kv_win_table is not None else num_chunks
    if kv_blocks is not None:
        assert NQT * (q_round_base + n_rounds) <= kv_blocks <= num_chunks, kv_blocks
        kv_stream_blocks = kv_blocks

    g_flat = tile_size_q * lkp
    # One interleaved L3 record per K chunk: the K tile, then that chunk's V
    # tiles. One shim link carries all of it.
    kv_rec = lkp * dk + dv_chunks * lkp * dv_tile

    # ---------------------------------------------------------------- kernels
    zero_fill_g = air.extern("zero_fill_g_bf16", link_with=KERNEL)
    zero_fill_gp = air.extern("zero_fill_gp_bf16", link_with=KERNEL)
    zero_fill_sp = air.extern("zero_fill_sp_bf16", link_with=KERNEL)
    neg_inf_fill_up = air.extern("neg_inf_fill_up_bf16", link_with=KERNEL)
    matmul_a_b = air.extern("matmul_a_b_bf16", link_with=KERNEL)
    matmul_g_b = air.extern("matmul_g_b_bf16", link_with=KERNEL)
    fused_softmax = air.extern("fused_softmax", link_with=KERNEL)
    mul_r_gp = air.extern("mul_r_gp", link_with=KERNEL)
    accum_sp_r_s = air.extern("accum_sp_r_s", link_with=KERNEL)
    vector_copy = air.extern("vector_copy_32elems", link_with=KERNEL, scalars=[i32])
    div_gp_sp = air.extern("div_gp_sp", link_with=KERNEL)
    if not causal:
        apply_mask = None
    elif window_blocks is not None:
        apply_mask = air.extern(
            "apply_window_mask", link_with=KERNEL, scalars=[i32, i32, i32]
        )
    else:
        apply_mask = air.extern(
            "apply_causal_mask", link_with=KERNEL, scalars=[i32, i32]
        )

    # --------------------------------------------------------------- channels
    # One endpoint per (head, core column) -- FLM's two shim MM2S per CU, split
    # onto two memtile columns because AIR will not let two relays share one.
    qin = air.channel("QIn", size=[NH, HX])
    # ONE CHANNEL PER (head, column), selected in the core by a guard on tx
    # rather than by a coordinate-valued bundle index. Two reasons, both from
    # attn_npu2_temporal_causal (q2l1[b] / gp2l2[(b,nm)]):
    #   - air.parallel's docstring: "The trips share one set of buffer
    #     descriptors. Writing the same transfers out as a Python for unrolls
    #     them into that many independent DMAs."
    #   - it keeps the head and column COMPILE-TIME constants, so no derived
    #     index (tx//2) is ever needed -- which is what makes the CU mapping
    #     expressible at all.
    q2l1 = [
        [air.channel(f"Q2L1_{h_i}_{c}", size=[HY]) for c in range(HX)]
        for h_i in range(NH)
    ]
    # ONE shim link for K and V both (see the module docstring), and ONE
    # broadcast out of the staging memtile to every core of every herd. K and V
    # share the core's second S2MM: a compute tile has exactly two, Q takes one,
    # so they must share the other or routing fails with "connecting (East: N)
    # to (DMA: 0) ... targets same dst". FLM does the same -- attn.mlir's
    # ^bb3/^bb5 is a 2-BD ring on S2MM 1 carrying both, alternated in-kernel by
    # `is_in_ping`. Gets on one channel into different buffers lower to that.
    kvin = air.channel("KVIn", size=[1])
    kv2l1 = air.channel("KV2L1", size=[1, 1, 1], broadcast_shape=[NH, HX, HY])
    # ONE gather channel per (head, column), carrying every dv chunk. A channel
    # per chunk would double the flows into the memtile, and 4 cores x 2 chunks
    # plus the Q relay's own inbound is 9 against the 6 S2MM channels a memtile
    # has -- "Unable to find a legal routing". Two puts from one core on one
    # channel share a flow.
    gp2l2 = [
        [air.channel(f"Gp2L2_{h_i}_{c}", size=[HY]) for c in range(HX)]
        for h_i in range(NH)
    ]
    gpout = air.channel("GpOut", size=[NH, HX])

    # ---------------------------------------------------------------- tensors
    # SEQ-FIRST at L3, the layout the model already carries: [seq, heads*dh].
    # Q used to be [heads, seq, dk] and the output [heads*dv_chunks, seq,
    # dv_tile], so the host had to transpose into and out of them on every
    # dispatch -- 97 ms per prefill, measured. As BD strides both are free.
    Q = air.tensor([lq, num_heads * dk], bf16)
    KV = air.tensor([num_chunks * kv_rec], bf16)
    GP = air.tensor([lq, num_heads * dv], bf16)

    rows_per_relay = HY * tile_size_q

    with air.launch([range(n_rounds)], name="attention_bf16") as launch:

        @launch.body
        def _(lx):
            # Q: one send per (head, core column). The head is a COLUMN range of
            # the seq-first tensor, so the relay's rows come out strided -- one
            # 2-D BD, same bytes, no host transpose.
            for h_i in range(NH):
                for c in range(HX):
                    q_row0 = (q_round_base + lx) * lqp + c * rows_per_relay
                    qin.put(
                        Q[
                            q_row0 : q_row0 + rows_per_relay,
                            h_i * dk : (h_i + 1) * dk,
                        ],
                        indices=[h_i, c],
                    )

            # K and V: ONE send for all heads (MQA), one record per K chunk --
            # and, under a sliding window, only this round's records. The case
            # table is FLM's kv_begin, evaluated at build time; scf.index_switch
            # picks the round's entry and it lands as the put's slice start.
            if kv_win_table is not None:
                kv_start = ops.switch(lx, [t * kv_rec for t in kv_win_table]).as_index()
            else:
                kv_start = 0
            kvin.put(
                KV[kv_start : kv_start + kv_stream_blocks * kv_rec].reshape(
                    kv_stream_blocks, kv_rec
                )
            )

            with air.segment([range(1), range(1)], name="attn_seg") as seg:

                @seg.body
                def _(seg_x, seg_y):
                    # FLM's memtile map (attn_aie.py:88-117): a Q relay per CU
                    # column, K/V staged on a central column so its switchbox
                    # carries nothing else, gathers on the core's own column.
                    # split=False on the relays because air-split-l2-memref
                    # would otherwise partition each into per-tile slices and
                    # overflow the shim; the gathers ARE split, since
                    # air.parallel fills them slice-wise.
                    def qcol(h_i, c):
                        return HX * h_i + c

                    # FLM stages K/V on MT[3]. It shares that column with a Q
                    # relay here, which is what one shim link for K and V buys.
                    kv_col = min(3, NH * HX - 1)

                    q_l2 = [
                        [
                            air.alloc(
                                [rows_per_relay, dk],
                                bf16,
                                scope=seg.private(),
                                column=qcol(h_i, c),
                                split=False,
                            )
                            for c in range(HX)
                        ]
                        for h_i in range(NH)
                    ]
                    k_l2 = air.alloc(
                        [lkp, dk], bf16, scope=seg.private(), column=kv_col
                    )
                    # One staging buffer per V chunk: reusing a single buffer
                    # would serialise the broadcast against its own refill.
                    v_l2 = [
                        air.alloc(
                            [lkp, dv_tile], bf16, scope=seg.private(), column=kv_col
                        )
                        for _ in range(dv_chunks)
                    ]
                    gp_l2 = [
                        [
                            air.alloc(
                                [rows_per_relay, dv],
                                bf16,
                                scope=seg.private(),
                                column=qcol(h_i, c),
                            )
                            for c in range(HX)
                        ]
                        for h_i in range(NH)
                    ]

                    # Q relay: ONE get of the relay's whole block, then an
                    # air.parallel that SLICES it per core. Re-getting the relay
                    # once per core instead unrolls into that many independent
                    # DMAs against a single L3 put, and hangs. This is
                    # temporal_causal's shape (q_relay[b] sliced by ty_i).
                    for h_i in range(NH):
                        for c in range(HX):
                            qin.get(q_l2[h_i][c], indices=[h_i, c])
                            for r in air.parallel(0, HY):
                                row0 = r * tile_size_q
                                q2l1[h_i][c].put(
                                    q_l2[h_i][c][row0 : row0 + tile_size_q, :]
                                    .reshape(tile_size_q // M, M, dk // M, M)
                                    .transpose(2, 0, 1, 3),
                                    indices=[r],
                                )

                    # K/V relay: one record in, one broadcast put per tile. One
                    # put per loop iteration is what lets air-to-aie fold this
                    # into a cyclic BD chain -- FLM's ^bb3/^bb5 ping-pong ring.
                    for _ in air.sequential(0, kv_stream_blocks):
                        kvin.get(k_l2)
                        kv2l1.put(
                            k_l2.reshape(lkp // M, M, dk // M, M).transpose(2, 0, 1, 3),
                            indices=[0, 0, 0],
                        )
                        for z in range(dv_chunks):
                            kvin.get(v_l2[z])
                            kv2l1.put(
                                v_l2[z]
                                .reshape(lkp // M, M, dv_tile // M, M)
                                .transpose(2, 0, 1, 3),
                                indices=[0, 0, 0],
                            )

                    def one_head(h_i):
                        """One CU: a herd of HX x HY cores, all on one head."""
                        # The q-block counter must survive between launch
                        # iterations, so it stays a segment-scope per-core
                        # buffer (8 bytes, no budget impact) while the tiles
                        # above are herd-local.
                        # Slot 2 is the round's kv_begin, in blocks, when the
                        # K/V DMA is windowed: the same max(0, q_base - window)
                        # the launch's case table holds, recomputed where the
                        # mask needs it.
                        ctr = (
                            air.alloc([3], i32, scope=seg.per_core())
                            if causal
                            else None
                        )

                        with air.herd(
                            [range(HX), range(HY)],
                            name=f"herd_{h_i}",
                            shape=(HX, HY),
                            at=(HX * h_i, 2),
                            link_with=KERNEL,
                        ) as h:

                            @h.body
                            def _(tx, ty):
                                # Per HERD, not segment scope: a seg.per_core()
                                # pool is shared by every herd in the segment, so
                                # four heads' worth of buffers would land on each
                                # core and blow the 64 KB budget.
                                q_saved = air.alloc(
                                    [tile_size_q, dk], bf16, scope=h.private()
                                )
                                qk = air.alloc([lkp, dk], bf16, scope=h.private())
                                v_l1 = air.alloc(
                                    [lkp, dv_tile], bf16, scope=h.private()
                                )
                                g = air.alloc(
                                    [tile_size_q, lkp], bf16, scope=h.private()
                                )
                                gp = [
                                    air.alloc(
                                        [tile_size_q, dv_tile], bf16, scope=h.private()
                                    )
                                    for _ in range(dv_chunks)
                                ]
                                up = air.alloc(
                                    [tile_size_q, 1], bf16, scope=h.private()
                                )
                                sp = air.alloc(
                                    [tile_size_q, 1], bf16, scope=h.private()
                                )
                                for z in range(dv_chunks):
                                    zero_fill_gp(gp[z])
                                zero_fill_sp(sp)
                                neg_inf_fill_up(up)

                                if causal:
                                    first = ops.equal(ctr[1:2], 0)
                                    ctr[0:1] = ops.select(first, 0, ctr[0:1])
                                    ctr[1:2] = ops.select(first, 1, ctr[1:2])
                                if kv_win_table is not None:
                                    ctr[2:3] = ops.maximum(ctr[0:1] - window_blocks, 0)

                                # Guarded per column: tx == c is an EQUALITY, not
                                # a pair of inequalities. AIR reads the condition
                                # to decide which tiles actually reach the put,
                                # and only the pinned form keeps it from creating
                                # flows for tiles that never send -- after which
                                # the gather waits on data nobody sends.
                                for c in range(HX):
                                    with ops.branch(tx == c):
                                        q2l1[h_i][c].get(q_saved, indices=[ty])

                                for chunk in air.sequential(0, kv_stream_blocks):
                                    # The core's q block within the sequence:
                                    # the relay column contributes HY blocks.
                                    q_block = (
                                        q_round_base * NQT + ctr[0] + tx * HY + ty
                                        if causal
                                        else None
                                    )
                                    # The mask indexes the SEQUENCE, but a
                                    # truncated stream delivers the window's
                                    # blocks starting at zero, so shift by the
                                    # same kv_begin the DMA used. ctr[2] carries
                                    # it rather than the launch index, for the
                                    # reason ctr[0] exists at all: the herd body
                                    # is outlined and counts its own rounds.
                                    kv_block = (
                                        ctr[2] + chunk
                                        if kv_win_table is not None
                                        else chunk
                                    )

                                    zero_fill_g(g.reshape(g_flat))
                                    kv2l1.get(qk, indices=[h_i, tx, ty])
                                    matmul_a_b(q_saved, qk, g.reshape(g_flat))

                                    s_tmp = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    r_tmp = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )

                                    if causal:
                                        if window_blocks is not None:
                                            apply_mask(
                                                g, q_block, kv_block, window_blocks
                                            )
                                        else:
                                            apply_mask(g, q_block, kv_block)
                                    fused_softmax(g.reshape(g_flat), up, s_tmp, r_tmp)

                                    # The scores are computed ONCE and reused for
                                    # every dv slice of the output; one staging
                                    # buffer, used before the next get refills it.
                                    for z in range(dv_chunks):
                                        kv2l1.get(v_l1, indices=[h_i, tx, ty])
                                        mul_r_gp(r_tmp, gp[z])
                                        matmul_g_b(g.reshape(g_flat), v_l1, gp[z])

                                    accum_sp_r_s(sp, r_tmp, s_tmp)
                                    vector_copy(0, s_tmp, sp)

                                # No cascade: this core owns the whole reduction
                                # for its (head, q-tile), so it normalises and
                                # drains.
                                for z in range(dv_chunks):
                                    div_gp_sp(sp, gp[z])
                                for c in range(HX):
                                    with ops.branch(tx == c):
                                        for z in range(dv_chunks):
                                            gp2l2[h_i][c].put(
                                                gp[z]
                                                .reshape(
                                                    dv_tile // M,
                                                    tile_size_q // M,
                                                    M,
                                                    M,
                                                )
                                                .transpose(1, 2, 0, 3),
                                                indices=[ty],
                                            )

                                if causal:
                                    # WRAP, do not just advance. L1 survives
                                    # between dispatches, so the boot flag alone
                                    # leaves the second dispatch of the same ELF
                                    # starting at q block num_lq_iters*NQT --
                                    # every block then reads as future, the mask
                                    # kills the whole row and the softmax divides
                                    # 0/0. An 8-head model runs this ELF twice.
                                    # Same idiom as the sibling's head counter
                                    # ("head_next >= num_head_groups").
                                    adv = ctr[0:1] + NQT
                                    ctr[0:1] = ops.select(adv >= n_rounds * NQT, 0, adv)

                    for h_i in range(NH):
                        one_head(h_i)

                    for h_i in range(NH):
                        for c in range(HX):
                            # Python range, not air.parallel: this unrolls to
                            # CONSTANT bundle indices, which is the spelling
                            # temporal_causal uses for its small grids.
                            # air.parallel here leaves the index an affine.apply
                            # that need not resolve to a constant, and the
                            # gather then waits on data nobody sends.
                            for r in range(HY):
                                row0 = r * tile_size_q
                                for z in range(dv_chunks):
                                    gp2l2[h_i][c].get(
                                        gp_l2[h_i][c][
                                            row0 : row0 + tile_size_q,
                                            z * dv_tile : (z + 1) * dv_tile,
                                        ],
                                        indices=[r],
                                    )
                            # gp_l2 is already [rows, head_dim] in natural order;
                            # the seq-first L3 tensor wants exactly that, so the
                            # dv-chunk-major shuffle is gone from both sides.
                            gpout.put(gp_l2[h_i][c], indices=[h_i, c])

            for h_i in range(NH):
                for c in range(HX):
                    row0 = (q_round_base + lx) * lqp + c * rows_per_relay
                    gpout.get(
                        GP[row0 : row0 + rows_per_relay, h_i * dv : (h_i + 1) * dv],
                        indices=[h_i, c],
                    )

    return launch


def build_module(**kwargs):
    """The MLIR module -- the llms/ builders' contract, as in the sibling."""
    return build_launch(**kwargs).build(target="npu2")


def interleave_kv(k, v, lkp):
    """Lay K and V out as the one stream the single shim link reads.

    ``k`` is [lk, dk] and ``v`` is [dv_chunks, lk, dv_tile]; the result is one
    flat array of ``lk/lkp`` records, each a K tile followed by that chunk's V
    tiles. FLM keeps K and V in a single BO for the same reason.
    """
    lk, dk = k.shape
    dv_chunks, _, dv_tile = v.shape
    num_chunks = lk // lkp
    rec = lkp * dk + dv_chunks * lkp * dv_tile
    out = np.empty(num_chunks * rec, dtype=k.dtype)
    for c in range(num_chunks):
        base = c * rec
        out[base : base + lkp * dk] = k[c * lkp : (c + 1) * lkp, :].reshape(-1)
        base += lkp * dk
        for z in range(dv_chunks):
            n = lkp * dv_tile
            out[base : base + n] = v[z, c * lkp : (c + 1) * lkp, :].reshape(-1)
            base += n
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="attn_npu2_headspatial.py")
    p.add_argument("--lk", type=int, default=2048)
    p.add_argument("--lkp", type=int, default=32)
    p.add_argument("--lq", type=int, default=2048)
    p.add_argument("--lqp", type=int, default=256)
    p.add_argument("--dk", type=int, default=256)
    p.add_argument("--dv", type=int, default=256)
    p.add_argument("--num-q-tiles", type=int, default=8, dest="num_q_tiles")
    p.add_argument("--num-heads", type=int, default=4, dest="num_heads")
    p.add_argument("--cu-cols", type=int, default=2, dest="cu_cols")
    p.add_argument("--dv-tile", type=int, default=128, dest="dv_tile")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--causal-skip", action="store_true", dest="causal_skip")
    p.add_argument("--no-causal", action="store_true", dest="no_causal")
    p.add_argument("-p", "--print-module-only", action="store_true")
    args = p.parse_args()

    mod = build_module(
        lk=args.lk,
        lkp=args.lkp,
        lq=args.lq,
        lqp=args.lqp,
        dk=args.dk,
        dv=args.dv,
        num_q_tiles=args.num_q_tiles,
        num_heads=args.num_heads,
        cu_cols=args.cu_cols,
        num_kv_heads=1,
        causal=not args.no_causal,
        window=args.window,
        dv_tile=args.dv_tile,
        causal_skip=args.causal_skip,
    )
    if args.print_module_only:
        print(mod)
        raise SystemExit(0)

    from ml_dtypes import bfloat16

    lk, lq, dk, dv = args.lk, args.lq, args.dk, args.dv
    nh, dvt = args.num_heads, args.dv_tile
    dv_chunks = dv // dvt

    rng = np.random.default_rng(42)
    input_q_hf = rng.uniform(0, 4.0, (nh, lq, dk)).astype(bfloat16)
    # seq-first at L3: [seq, heads*dk]
    input_q = np.ascontiguousarray(input_q_hf.transpose(1, 0, 2)).reshape(lq, nh * dk)
    input_k = rng.uniform(0, 4.0, (lk, dk)).astype(bfloat16)
    input_v_orig = rng.uniform(0, 4.0, (lk, dv)).astype(bfloat16)
    input_v = input_v_orig.reshape(lk, dv_chunks, dvt).transpose(1, 0, 2).copy()
    input_kv = interleave_kv(input_k, input_v, args.lkp)

    # float64 reference, per the campaign's Phase 1 gate.
    inv_sqrt_dk = 1.0 / sqrt(dk)
    out = np.zeros((nh, lq, dv), dtype=bfloat16)
    kf = input_k.astype(np.float64)
    vf = input_v_orig.astype(np.float64)
    for head in range(nh):
        scores = input_q_hf[head].astype(np.float64) @ kf.T * inv_sqrt_dk
        mask = np.zeros(scores.shape, dtype=bool)
        if not args.no_causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        if args.window is not None:
            rows = np.arange(scores.shape[0])[:, None]
            cols = np.arange(scores.shape[1])[None, :]
            mask |= cols <= rows - args.window
        scores = np.where(mask, -1e9, scores)
        pr = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        pr = pr / np.sum(pr, axis=-1, keepdims=True)
        out[head] = (pr @ vf).astype(bfloat16)

    # seq-first at L3: [seq, heads*dv]
    expected = np.ascontiguousarray(out.transpose(1, 0, 2)).reshape(lq, nh * dv)

    runner = XRTRunner(
        omit_while_true_loop=False,
        omit_pingpong="all",
        runtime_loop_tiling_sizes=[1],
        output_format="elf",
        instance_name="attention_bf16",
    )
    raise SystemExit(
        runner.run_test(
            mod,
            inputs=[input_q, input_kv],
            expected_outputs=[expected],
            atol=0.15,
            rtol=0.04,
            max_mismatch_percentage=0.5,
            min_correlation=0.99,
        )
    )
