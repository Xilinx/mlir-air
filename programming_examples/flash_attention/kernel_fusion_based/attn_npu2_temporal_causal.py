# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention on air.api -- NPU2, temporal causal.

The cascade is gone. Each core loops its own causal prefix of K blocks in
time rather than splitting the K reduction across a column of cores, so a
tile's (numerator, denominator) is already final when the loop ends: it
normalises and emits, with no merge. That is what makes the causal triangle
cheap -- round ``lx`` streams only ``(lx + 1) * NQ`` K blocks instead of all of
them, so the DMA skips the upper triangle rather than masking it.

The array is (rows x columns) = ``NR x 2 * gqa_group_size``. A column block is
one q-head of a GQA group; the two columns within it split the q-sequence
tiles, and a row is a pair of them. K and V broadcast to every tile at once
from one central memtile -- only one kv-head is live per dispatch -- while Q
fans out per column block, so the two never contend for the same S2MM.

Layout is sequence-first, as in ``attn_npu2_seqfirst``: ``[seq, heads * d]``.

Four things about this port are worth knowing.

**Buffer aliasing is two names for one allocation.** The predecessor
deliberately overlaps the Q pair staging with either the accumulator slab or
the K/V staging, because the two are never live at once and the 64 KB does not
hold both. In the DSL that is exactly what it sounds like: one ``air.alloc``,
two Python names. The conditions under which it is safe are unchanged and the
comments explaining them are kept, because they are about lock counts in the
lowered IR rather than about the frontend.

**The output gather is one ``air.parallel`` over a grid.** It is a 4-way
spatial scatter over (row, column-within-block), and both indices name a
channel bundle -- which is why it cannot be ``air.sequential``: ``air-place-herds``
refuses a temporal induction variable as a bundle index. One ``scf.forall``
with two induction variables, not two nested loops.

**The row guards stay equalities.** A single output row is ``ty == r``, not a
pair of inequalities bracketing it. The predecessor's comment is emphatic and
it is right: AIR reads the condition to decide which tiles actually reach the
put, and only the pinned form keeps it from creating flows for tiles that never
send -- after which the gather waits on data nobody sends. ``ops.branch(ty == r)``
emits ``arith.cmpi eq``, which ``SpecializeScfIfPattern`` folds per tile exactly
as the affine path folds an equality set.

**The per-round loop is unrolled in Python, not traced.** ``cps_lx`` -- the
round's causal prefix -- has to be a build-time constant for the DMA skip to
exist at all, so the round axis is a Python ``for`` and the herd body is emitted
once per round. That is what ``fold_core_rounds`` trades away when the round
count grows past what AIE2P program memory holds.
"""

import argparse
from math import sqrt

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

KERNEL = "attn_npu2.o"
M = 8  # mmul_m = mmul_k = mmul_n
# Rounds whose causal prefix can be in flight at once. Past this, stream the
# FULL prefix every round instead: the per-round descriptors become identical,
# AIR folds them into one repeated task, and the in-core mask still discards the
# extra K blocks, so the result is unchanged. The price is the causal DMA saving
# -- 2x the K/V work -- which is what buys the design the ability to run at all.
#
# 6 when this design landed (7 and 8 deadlocked). 8 runs now that the shim-BD
# rebalance is in, and 8 rounds is what a 4096 prefill needs at lqp=512;
# restoring the triangle there is worth 12% of prefill TTFT on llama-3.2-1B.
#
# Raising it further will not help until an R > 8 shape compiles at all. When
# one does, the better move is to QUANTIZE the prefix -- group rounds so each
# streams its group's last prefix, which the mask makes correct for any group
# size and which holds the work at (G+1)/2G of the square rather than at 1.0.
_MAX_ROUNDS_IN_FLIGHT = 8

# K/V shim descriptors the unrolled triangle may hold open at once. A shim tile
# has 16 and the Q relay and output gather need the rest; 8 is what
# llama32_1b_q4nx runs at (4 rounds, one kv-head per unroll) and 16 is what
# llama32_3b failed to build at. Only reachable when the prefix VARIES -- a
# uniform prefix emits one descriptor whatever the round count.
_SHIM_KV_DESCRIPTORS = 8


def build_launch(
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
    # num_cascade_stages and causal_skip are accepted and ignored: this design
    # has no cascade, and the causal skip is in the DMA rather than a choice.
    # Six llms/ call sites still pass them, so they stay in the signature.
    del num_cascade_stages, causal_skip
    assert lq % lqp == 0, f"lq ({lq}) must be divisible by lqp ({lqp})"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp ({lqp}) must be divisible by num_q_tiles ({num_q_tiles})"
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    dk_tile = min(dk, lkp)
    dv_tile = min(dv, lkp)
    dk_chunks = dk // dk_tile
    dv_chunks = dv // dv_tile
    if num_kv_heads is None:
        num_kv_heads = num_heads
    gqa_group_size = num_heads // num_kv_heads

    # Whole-d DMA: chunking K/V in the transfer makes the memtile BD chain
    # unroll the causal-prefix loop (measured 175 BDs against a 48 cap), so send
    # the whole depth in one go and chunk on pointers inside the core instead.
    full_d_dma = dk_chunks > 1 or dv_chunks > 1
    if full_d_dma:
        assert dk == dv, (
            f"full-d DMA shares one L1 staging buffer for K and V; needs dk == dv "
            f"(got dk={dk}, dv={dv})"
        )
    # The 2-way Q pair broadcast is mandatory: 2*NR per-tile unicast puts
    # deadlock the herd on hardware (A/B'd at dh=64 / L=512 / GQA 1:3).
    q_pair_bcast = True

    NB = gqa_group_size
    assert NB * 2 <= 8, f"physical columns = 2 * gqa_group_size ({NB}) must be <= 8"
    assert num_q_tiles % (2 * q_tiles_per_core) == 0
    NR = num_q_tiles // (2 * q_tiles_per_core)
    assert NR == 4, f"herd rows must be 4, got {NR}"
    assert q_tiles_per_core in (1, 2)
    NC = 2 * NB
    assert num_kv_heads % num_heads_per_unroll == 0
    assert num_heads_per_unroll * NC <= 8
    num_head_groups = num_kv_heads // num_heads_per_unroll

    NQ = num_q_tiles
    num_lq_iters = lq // lqp
    tile_size_q = lqp // num_q_tiles
    _merge_out_into_kv = num_lq_iters > 4
    # A varying prefix is a varying transfer size, air.api needs sizes static,
    # so the triangle costs one shim descriptor per round per kv-head rather
    # than one for the whole loop. A shim tile holds 16 simultaneously-active
    # BDs and the Q relay and output gather want their share, so past
    # _SHIM_KV_DESCRIPTORS the K/V descriptors alone would exhaust it:
    #
    #   'aiex.dma_configure_task' op Too many simultaneously active buffer
    #   descriptors on tile (0,0), which supports up to 16.
    #
    # Uniform is the documented way out and is what the round-count cap below
    # already does -- every round streams the full prefix, the in-core mask
    # discards the extra blocks, and the descriptors collapse back to one
    # because they are identical. It costs the causal DMA saving and it builds.
    #
    # `not causal` is the third way in, and it is a correctness one rather than
    # a budget one: the prefix is only sound because the in-core mask discards
    # what lies past the diagonal, and with no mask a round that streams
    # (lx + 1) * NQ blocks simply does not see the rest of K. Non-causal
    # attention has to read all of it.
    _kv_descriptors = num_lq_iters * num_heads_per_unroll * 2  # K and V
    _uniform_cps = (
        not causal
        or num_lq_iters > _MAX_ROUNDS_IN_FLIGHT
        or _kv_descriptors > _SHIM_KV_DESCRIPTORS
    )

    def cps_blocks(lx):
        """The round's causal prefix, in K blocks.

        ``lx`` is a Python int while the round axis is unrolled and a loop
        variable once it is folded; the arithmetic is the same either way, and
        air.sequential takes the result as a run-time bound in the second case.
        """
        return (num_lq_iters if _uniform_cps else lx + 1) * NQ

    _dk_dma = dk if full_d_dma else dk_tile
    _dv_dma = dv if full_d_dma else dv_tile

    # Accumulator / staging aliasing. Both are safe only under the conditions
    # below, which are about lock counts in the lowered IR: air-to-aie derives
    # an outbound BD's lock count from the buffer's fill count, so a slab filled
    # more than once but drained once never drains.
    alias_qpair_gp = dv_chunks > 1 and dk_chunks == 1 and dv == 2 * dk_tile
    alias_qpair_kv = full_d_dma and 2 * tile_size_q * dk_tile == lkp * dk

    fold_core_rounds = num_lq_iters * q_tiles_per_core > 4
    n_ob = (
        1
        if fold_core_rounds or q_tiles_per_core > 1
        else (2 if num_lq_iters > 1 and dv_chunks == 1 else 1)
    )

    l1_bytes = (
        q_tiles_per_core * tile_size_q * dk * 2
        + (0 if alias_qpair_kv else 2 * tile_size_q * dk_tile * 2)
        + (0 if alias_qpair_gp else n_ob * q_tiles_per_core * tile_size_q * dv * 2)
        + lkp * dk * 2
        + q_tiles_per_core * tile_size_q * lkp * 2
        + 4 * tile_size_q * 2
    )
    assert l1_bytes <= 64 * 1024, (
        f"per-core L1 working set {l1_bytes} B exceeds the 64 KB AIE2P data "
        f"memory (dk_chunks={dk_chunks}, dv_chunks={dv_chunks})"
    )

    _row_split_out = q_tiles_per_core == 1 and num_lq_iters > 4
    # out_col's table below is a hand-checked column budget for NB == 3 (GQA
    # 3:1) and NB == 4 (GQA 4:1), where the herd spans all 8 columns and sixteen
    # gathers share the seven non-K/V ones. A smaller NB leaves the herd
    # narrower than the table assumes, so fail here with the reason rather than
    # building a module on an unverified placement.
    if _row_split_out and NB not in (3, 4):
        raise NotImplementedError(
            f"the row-split output placement past 4 rounds is only mapped for "
            f"the GQA ratio NB = num_heads / num_kv_heads to be 3 or 4; got "
            f"NB={NB} from num_heads={num_heads}, num_kv_heads={num_kv_heads}. "
            f"Either keep lq <= 4 * lqp (= {4 * lqp}) so the four-tile gathers "
            f"are used, or extend out_col with a verified column budget for "
            f"this NB."
        )
    if _row_split_out:
        out_slices = [
            ("lo", 0, 0, 1),
            ("lo1", 0, 1, 1),
            ("hi", 0, 2, 1),
            ("hi1", 0, 3, 1),
        ]
    else:
        out_slices = [
            (nm if j == 0 else f"{nm}{j}", j, 2 if hi else 0, 2)
            for j in range(q_tiles_per_core)
            for nm, hi in (("lo", False), ("hi", True))
        ]
    n_out = len(out_slices)
    n_ob_l2 = 1 if (dv_chunks > 1 or fold_core_rounds) else min(2, num_lq_iters)
    g_flat = tile_size_q * lkp

    # ---------------------------------------------------------------- kernels
    zero_fill_g = air.extern("zero_fill_g_bf16", link_with=KERNEL)
    zero_fill_gp = air.extern("zero_fill_gp_bf16", link_with=KERNEL)
    zero_fill_sp = air.extern("zero_fill_sp_bf16", link_with=KERNEL)
    neg_inf_fill_up = air.extern("neg_inf_fill_up_bf16", link_with=KERNEL)
    fused_softmax = air.extern("fused_softmax", link_with=KERNEL)
    accum_sp_r_s = air.extern("accum_sp_r_s", link_with=KERNEL)
    vector_copy = air.extern("vector_copy_32elems", link_with=KERNEL, scalars=[i32])
    apply_causal_mask = (
        air.extern("apply_causal_mask", link_with=KERNEL, scalars=[i32, i32])
        if causal
        else None
    )
    copy_half_tile = (
        air.extern("copy_half_tile_at", link_with=KERNEL, scalars=[i32, i32])
        if full_d_dma
        else air.extern("copy_half_tile", link_with=KERNEL, scalars=[i32])
    )
    matmul_a_b = (
        air.extern("matmul_a_b_bf16_chunk", link_with=KERNEL, scalars=[i32])
        if full_d_dma
        else air.extern("matmul_a_b_bf16", link_with=KERNEL)
    )
    # The dv-chunked accumulator kernels take the chunk index as a scalar: the
    # AIE bare-pointer ABI drops a subview's offset, so the slice is selected
    # inside the kernel rather than by narrowing the memref.
    if dv_chunks == 1:
        mul_r_gp = air.extern("mul_r_gp", link_with=KERNEL)
        div_gp_sp = air.extern("div_gp_sp", link_with=KERNEL)
        matmul_g_b = air.extern("matmul_g_b_bf16", link_with=KERNEL)
        zero_fill_gp_all = None
        div_gp_sp_all = None
    else:
        mul_r_gp = air.extern("mul_r_gp_at", link_with=KERNEL, scalars=[i32])
        div_gp_sp = air.extern("div_gp_sp_at", link_with=KERNEL, scalars=[i32])
        matmul_g_b = air.extern(
            "matmul_g_b_bf16_chunk" if full_d_dma else "matmul_g_b_bf16_at",
            link_with=KERNEL,
            scalars=[i32],
        )
        zero_fill_gp_all = air.extern("zero_fill_gp_bf16_all", link_with=KERNEL)
        div_gp_sp_all = air.extern("div_gp_sp_all", link_with=KERNEL)

    # --------------------------------------------------------------- channels
    # Q fans out per column block so each block gets its own shim -> memtile
    # flow; a single endpoint feeds one block and the rest deadlock waiting.
    q2l1 = [
        (
            air.channel(f"Q2L1_{b}", size=[NR, 1], broadcast_shape=[NR, 2])
            if q_pair_bcast
            else air.channel(f"Q2L1_{b}", size=[NR, 2])
        )
        for b in range(NB)
    ]
    qin = air.channel("QIn", size=[NB])
    # K and V share one consolidated broadcast to every tile: only one kv-head
    # is live per dispatch, so the whole array wants the same K/V, and packing
    # them onto one S2MM leaves the other for Q.
    kv2l1 = air.channel(
        "KV2L1",
        size=[num_heads_per_unroll, 1, 1],
        broadcast_shape=[num_heads_per_unroll, NR, NC],
    )
    kin = air.channel("KIn", size=[num_heads_per_unroll])
    vin = air.channel("VIn", size=[num_heads_per_unroll])
    # One gather channel per (block, output slice), so the output streams spread
    # across shim tiles instead of merging 8-to-1 onto one.
    gp2l2 = {
        (b, nm): air.channel(f"Gp2L2_{b}_{nm}", size=[NR, NC])
        for b in range(NB)
        for nm, _j, _rlo, _nrows in out_slices
    }
    gpout = air.channel("GpOut", size=[NB, n_out])

    # ---------------------------------------------------------------- tensors
    emb_q = num_heads * dk
    emb_k = num_kv_heads * dk
    emb_v = num_kv_heads * dv
    emb_out = num_heads * dv
    Q = air.tensor([lq, emb_q], bf16)
    K = air.tensor([lk, emb_k], bf16)
    V = air.tensor([lk, emb_v], bf16)
    GP = air.tensor([lq, emb_out], bf16)

    half_rows = lqp // n_out

    with air.launch([range(num_head_groups)], name="attention_bf16") as launch:

        @launch.body
        def _(ly):

            def out_gets(lx):
                """Drain one round's output halves back to L3."""
                for kv_local in range(num_heads_per_unroll):
                    for row_slot in range(NB):
                        col = (
                            ly * (num_heads_per_unroll * gqa_group_size * dv)
                            + (kv_local * gqa_group_size + row_slot) * dv
                        )
                        for i in range(n_out):
                            row = lx * lqp + i * half_rows
                            gpout.get(
                                GP[row : row + half_rows, col : col + dv],
                                indices=[row_slot, i],
                            )

            # K and V: one round's causal prefix, cps_blocks(lx) blocks of lkp
            # rows. That count is a transfer SIZE and air.api needs those
            # static, so a varying prefix unrolls the round axis in Python --
            # matching the memtile relay and the core, which consume exactly
            # cps_blocks(lx) per round. Over-sending here does not merely waste
            # bandwidth, it desynchronises the stream.
            kv_rounds = (
                air.sequential(0, num_lq_iters) if _uniform_cps else range(num_lq_iters)
            )
            for lx in kv_rounds:
                for kv_local in range(num_heads_per_unroll):
                    k_col = ly * (num_heads_per_unroll * dk) + kv_local * dk
                    v_col = ly * (num_heads_per_unroll * dv) + kv_local * dv
                    rows = cps_blocks(lx) * lkp
                    if full_d_dma:
                        kin.put(
                            K[0:rows, k_col : k_col + dk].reshape(rows // lkp, lkp, dk),
                            indices=[kv_local],
                        )
                    else:
                        kin.put(
                            K[0:rows, k_col : k_col + dk]
                            .reshape(rows // lkp, lkp, dk_chunks, dk_tile)
                            .transpose(0, 2, 1, 3),
                            indices=[kv_local],
                        )
                    vin.put(
                        V[0:rows, v_col : v_col + _dv_dma].reshape(
                            rows // lkp, lkp, _dv_dma
                        ),
                        indices=[kv_local],
                    )
                if _merge_out_into_kv:
                    out_gets(lx)

            # Q: one send per (block, fill) of a [lqp, _dk_dma] window.
            for lx in air.sequential(0, num_lq_iters):
                row0 = lx * lqp
                for kv_local in range(num_heads_per_unroll):
                    for row_slot in range(NB):
                        for fill in range(dk_chunks * q_tiles_per_core):
                            dk_c = fill % dk_chunks
                            q_col = (
                                ly * (num_heads_per_unroll * gqa_group_size * dk)
                                + (kv_local * gqa_group_size + row_slot) * dk
                                + (0 if full_d_dma else dk_c * dk_tile)
                            )
                            qin.put(
                                Q[row0 : row0 + lqp, q_col : q_col + _dk_dma],
                                indices=[row_slot],
                            )

            with air.segment(
                [range(num_heads_per_unroll), range(1)], name="attn_seg"
            ) as seg:

                @seg.body
                def _(seg_x, seg_y):
                    # Placement. The K/V broadcast sits alone on a central
                    # odd column so its switchbox carries nothing else; the Q
                    # relays pin to the even columns so place-tiles spreads them
                    # rather than piling every Q descriptor on one MM2S; and the
                    # relays are unsplit because air-split-l2-memref would
                    # otherwise partition each into per-tile slices and overflow
                    # the shim. The output columns are a hand-checked budget.
                    kv_col = 3

                    def out_col(b, i):
                        if _row_split_out:
                            if NB == 4:
                                return 2 * b + (i // 2)
                            return {
                                0: [0, 0, 1, 1],
                                1: [2, 2, 1, 4],
                                2: [4, 5, 5, 5],
                            }[
                                b
                            ][i]
                        return 2 * b + (i % 2)

                    qk_l2 = [
                        air.alloc(
                            [lkp, _dk_dma], bf16, scope=seg.private(), column=kv_col
                        )
                        for _ in range(1 if full_d_dma else dk_chunks)
                    ]
                    v_l2 = [
                        air.alloc(
                            [lkp, _dv_dma], bf16, scope=seg.private(), column=kv_col
                        )
                        for _ in range(1 if full_d_dma else dv_chunks)
                    ]
                    q_relay = [
                        air.alloc(
                            [lqp, _dk_dma],
                            bf16,
                            scope=seg.private(),
                            column=2 * b,
                            split=False,
                        )
                        for b in range(NB)
                    ]
                    gp_slice = [
                        [
                            [
                                air.alloc(
                                    [half_rows, dv],
                                    bf16,
                                    scope=seg.private(),
                                    column=out_col(b, i),
                                    split=False,
                                )
                                for _ in range(n_ob_l2)
                            ]
                            for i in range(n_out)
                        ]
                        for b in range(NB)
                    ]

                    # L1, per core, allocated here so it survives the whole
                    # segment and reaches the herd as an operand.
                    nqj = 1 if full_d_dma else dk_chunks
                    q_bufs_j = [
                        [
                            air.alloc(
                                [tile_size_q, _dk_dma], bf16, scope=seg.per_core()
                            )
                            for _ in range(nqj)
                        ]
                        for _ in range(q_tiles_per_core)
                    ]
                    qk = air.alloc([lkp, _dk_dma], bf16, scope=seg.per_core())
                    # Aliases, not copies. The Q pair staging is dead before the
                    # first K block lands and both are inbound-only, so no
                    # outbound BD sits on the shared buffer and the fill-count
                    # lock hazard cannot apply.
                    qpair = (
                        qk
                        if alias_qpair_kv
                        else air.alloc(
                            [2 * tile_size_q, dk_tile], bf16, scope=seg.per_core()
                        )
                    )
                    g_bufs = [
                        air.alloc([tile_size_q, lkp], bf16, scope=seg.per_core())
                        for _ in range(q_tiles_per_core)
                    ]
                    gp_pp = (
                        []
                        if alias_qpair_gp
                        else [
                            air.alloc([tile_size_q, dv], bf16, scope=seg.per_core())
                            for _ in range(n_ob * q_tiles_per_core)
                        ]
                    )
                    up_bufs = [
                        air.alloc([tile_size_q, 1], bf16, scope=seg.per_core())
                        for _ in range(q_tiles_per_core)
                    ]
                    sp_bufs = [
                        air.alloc([tile_size_q, 1], bf16, scope=seg.per_core())
                        for _ in range(q_tiles_per_core)
                    ]

                    # Q relay: L3 -> memtile -> the block's row pairs, packed
                    # into the 8x8 blocks the mmul consumes.
                    for lx in range(num_lq_iters):
                        for b in range(NB):
                            for fill in range(dk_chunks * q_tiles_per_core):
                                dk_c = fill % dk_chunks
                                j = fill // dk_chunks
                                qin.get(q_relay[b], indices=[b])
                                for ty_i in air.parallel(0, NR):
                                    rows = 2 * tile_size_q
                                    row0 = (j * 2 * NR + 2 * ty_i) * tile_size_q
                                    packed = (
                                        q_relay[b][row0 : row0 + rows, :]
                                        .reshape(rows // M, M, _dk_dma // M, M)
                                        .transpose(2, 0, 1, 3)
                                    )
                                    lo = dk_c * (dk_tile // M) if full_d_dma else 0
                                    q2l1[b].put(
                                        packed[lo : lo + dk_tile // M, :, :, :],
                                        indices=[ty_i, 0],
                                    )

                        # K then V, one broadcast each per temporal chunk.
                        for _chunk in air.sequential(0, cps_blocks(lx)):
                            for dk_c in range(1 if full_d_dma else dk_chunks):
                                buf = qk_l2[0 if full_d_dma else dk_c]
                                kin.get(buf, indices=[seg_x])
                                kv2l1.put(
                                    buf.reshape(lkp // M, M, _dk_dma // M, M).transpose(
                                        2, 0, 1, 3
                                    ),
                                    indices=[seg_x, 0, 0],
                                )
                            for dv_c in range(1 if full_d_dma else dv_chunks):
                                buf = v_l2[0 if full_d_dma else dv_c]
                                vin.get(buf, indices=[seg_x])
                                kv2l1.put(
                                    buf.reshape(lkp // M, M, _dv_dma // M, M).transpose(
                                        2, 0, 1, 3
                                    ),
                                    indices=[seg_x, 0, 0],
                                )

                    with air.herd(
                        [range(NC), range(NR)],
                        name="herd_0",
                        shape=(NC, NR),
                        link_with=KERNEL,
                    ) as h:

                        def emit_out(b, nm, j, rlo, nrows, gps_j, tx, ty):
                            """One tile's finished half, into the block's gather.

                            The block guard is the column pair 2b..2b+1;
                            conjunction is nesting in this DSL, so it is two
                            regions rather than one set with two inequalities.

                            The row guard is where the *form* matters. A single
                            row is an equality, not a pair of inequalities
                            bracketing it: AIR reads the condition to decide
                            which tiles actually reach the put, and only the
                            pinned form keeps it from creating flows for tiles
                            that never send -- after which the gather waits on
                            data nobody sends.

                            The slab's mmul layout is column-block-major over
                            every dv chunk, so one de-tiling view emits
                            row-major [tile_size_q, dv] for the whole slab.
                            """
                            packed = (
                                gps_j[j]
                                .reshape(tile_size_q // M, dv // M, M, M)
                                .transpose(1, 2, 0, 3)
                            )

                            def put():
                                gp2l2[(b, nm)].put(packed, indices=[ty, tx])

                            with ops.branch(tx >= 2 * b):
                                with ops.branch(tx <= 2 * b + 1):
                                    if nrows == 1:
                                        with ops.branch(ty == rlo):
                                            put()
                                    elif rlo == 0:
                                        with ops.branch(ty <= 1):
                                            put()
                                    else:
                                        with ops.branch(ty >= rlo):
                                            put()

                        @h.body
                        def _(tx, ty):
                            # The round axis is unrolled in Python while it is
                            # small, so cps_lx is a build-time constant and the
                            # DMA skip exists at all. That makes .text scale
                            # with the round count against 16 KB of AIE2P
                            # program memory, so past the threshold it folds
                            # into one scf.for with a run-time causal bound --
                            # which costs the accumulator ping-pong, since a
                            # dynamic slot cannot index a buffer list.
                            core_rounds = (
                                air.sequential(0, num_lq_iters)
                                if fold_core_rounds
                                else range(num_lq_iters)
                            )
                            for lx in core_rounds:
                                slot = 0 if fold_core_rounds else lx % n_ob
                                gps_j = [
                                    (
                                        qpair
                                        if alias_qpair_gp
                                        else gp_pp[slot * q_tiles_per_core + j]
                                    )
                                    for j in range(q_tiles_per_core)
                                ]

                                def gp_call(kernel, pre, c, gp_buf):
                                    """A gp kernel on dv chunk c, or the whole slab."""
                                    if dv_chunks == 1:
                                        kernel(*pre, gp_buf)
                                    else:
                                        kernel(*pre, gp_buf, c)

                                # Q capture. A tile takes only its own column's
                                # half of the block's row-pair broadcast; the
                                # dk-chunk loop sits inside the guard so a tile
                                # emits one guarded region per (block, column)
                                # rather than one per chunk, which is what keeps
                                # the unrolled core inside program memory.
                                for b in range(NB):
                                    for lc in range(2):
                                        with ops.branch(tx == 2 * b + lc):
                                            for j in range(q_tiles_per_core):
                                                for dk_c in range(dk_chunks):
                                                    q2l1[b].get(qpair, indices=[ty, lc])
                                                    if full_d_dma:
                                                        copy_half_tile(
                                                            qpair,
                                                            q_bufs_j[j][0],
                                                            lc,
                                                            dk_c,
                                                        )
                                                    else:
                                                        copy_half_tile(
                                                            qpair,
                                                            q_bufs_j[j][dk_c],
                                                            lc,
                                                        )

                                # Accumulators are zeroed AFTER the Q phase:
                                # when they alias the staging slab, the copy
                                # above is what drains it.
                                for j in range(q_tiles_per_core):
                                    if dv_chunks == 1:
                                        zero_fill_gp(gps_j[j])
                                    else:
                                        zero_fill_gp_all(gps_j[j])
                                    zero_fill_sp(sp_bufs[j])
                                    neg_inf_fill_up(up_bufs[j])

                                # This round's causal prefix, in time.
                                for chunk in air.sequential(0, cps_blocks(lx)):
                                    for j in range(q_tiles_per_core):
                                        zero_fill_g(g_bufs[j].reshape(g_flat))

                                    # One K get shared by every q-tile the core
                                    # owns; a get per tile would re-send K.
                                    if full_d_dma:
                                        kv2l1.get(qk, indices=[seg_x, ty, tx])
                                        for j in range(q_tiles_per_core):
                                            for dk_c in range(dk_chunks):
                                                matmul_a_b(
                                                    q_bufs_j[j][0],
                                                    qk,
                                                    g_bufs[j].reshape(g_flat),
                                                    dk_c,
                                                )
                                    else:
                                        for dk_c in range(dk_chunks):
                                            kv2l1.get(qk, indices=[seg_x, ty, tx])
                                            for j in range(q_tiles_per_core):
                                                matmul_a_b(
                                                    q_bufs_j[j][dk_c],
                                                    qk,
                                                    g_bufs[j].reshape(g_flat),
                                                )

                                    s_tmps, r_tmps = [], []
                                    for j in range(q_tiles_per_core):
                                        if causal:
                                            # seq-tile s = ty*2 + tx%2, and the
                                            # core's j-th tile is s + j*2*NR.
                                            q_block = (
                                                lx * NQ + ty * 2 + tx % 2 + j * 2 * NR
                                            )
                                            apply_causal_mask(g_bufs[j], q_block, chunk)
                                        s_tmp = air.alloc(
                                            [tile_size_q, 1], bf16, scope=h.private()
                                        )
                                        r_tmp = air.alloc(
                                            [tile_size_q, 1], bf16, scope=h.private()
                                        )
                                        s_tmps.append(s_tmp)
                                        r_tmps.append(r_tmp)
                                        fused_softmax(
                                            g_bufs[j].reshape(g_flat),
                                            up_bufs[j],
                                            s_tmp,
                                            r_tmp,
                                        )
                                        for c in range(dv_chunks):
                                            gp_call(mul_r_gp, [r_tmp], c, gps_j[j])

                                    # One V get shared likewise; r/s stay live
                                    # across it, hence the lists.
                                    if full_d_dma:
                                        kv2l1.get(qk, indices=[seg_x, ty, tx])
                                        for j in range(q_tiles_per_core):
                                            for c in range(dv_chunks):
                                                if dv_chunks == 1:
                                                    matmul_g_b(
                                                        g_bufs[j].reshape(g_flat),
                                                        qk,
                                                        gps_j[j],
                                                    )
                                                else:
                                                    matmul_g_b(
                                                        g_bufs[j].reshape(g_flat),
                                                        qk,
                                                        gps_j[j],
                                                        c,
                                                    )
                                    else:
                                        for c in range(dv_chunks):
                                            kv2l1.get(qk, indices=[seg_x, ty, tx])
                                            for j in range(q_tiles_per_core):
                                                gp_call(
                                                    matmul_g_b,
                                                    [g_bufs[j].reshape(g_flat), qk],
                                                    c,
                                                    gps_j[j],
                                                )
                                    for j in range(q_tiles_per_core):
                                        accum_sp_r_s(sp_bufs[j], r_tmps[j], s_tmps[j])
                                        vector_copy(0, s_tmps[j], sp_bufs[j])

                                # NS = 1: this tile looped its whole causal
                                # prefix, so (gp, sp) is already final. Normalise
                                # and emit -- there is no cascade to merge.
                                for j in range(q_tiles_per_core):
                                    if dv_chunks == 1:
                                        gp_call(div_gp_sp, [sp_bufs[j]], 0, gps_j[j])
                                    else:
                                        div_gp_sp_all(sp_bufs[j], gps_j[j])

                                for b in range(NB):
                                    for nm, j, rlo, nrows in out_slices:
                                        emit_out(b, nm, j, rlo, nrows, gps_j, tx, ty)

                    # Output gather, per round. Each half is a 4-way spatial
                    # scatter over (row, column-within-block) -- one air.parallel
                    # grid, because both indices name a channel bundle and
                    # air-place-herds refuses a temporal induction variable
                    # there.
                    fold_rounds = dv_chunks > 1 or fold_core_rounds
                    rounds = (
                        air.sequential(0, num_lq_iters)
                        if fold_rounds
                        else range(num_lq_iters)
                    )
                    for lx in rounds:
                        for b in range(NB):
                            for i, (nm, j, rlo, nrows) in enumerate(out_slices):
                                buf = gp_slice[b][i][0 if fold_rounds else lx % n_ob_l2]
                                if nrows == 1:
                                    # One tile row: two gets at constant bundle
                                    # indices. A degenerate [1, 2] grid would
                                    # leave the row index an affine.apply on an
                                    # always-zero variable, which does not
                                    # resolve to a constant bundle index.
                                    for lc in range(2):
                                        gp2l2[(b, nm)].get(
                                            buf[
                                                lc * tile_size_q : lc * tile_size_q
                                                + tile_size_q,
                                                :,
                                            ],
                                            indices=[rlo, 2 * b + lc],
                                        )
                                else:
                                    for r, lc in air.parallel([range(nrows), range(2)]):
                                        row = (r * 2 + lc) * tile_size_q
                                        gp2l2[(b, nm)].get(
                                            buf[row : row + tile_size_q, :],
                                            indices=[r + rlo, 2 * b + lc],
                                        )
                                gpout.put(buf, indices=[b, i])

            if not _merge_out_into_kv:
                for lx in air.sequential(0, num_lq_iters):
                    out_gets(lx)

    return launch


def build_module(**kwargs):
    """The MLIR module. Return type is the llms/ builders' contract.

    programming_examples/llms/llama32_1b/llama32_1b_prefill.py and
    programming_examples/llms/shared/infra/fa_temporal.py import this name and
    hand the result straight to KernelCache.compile_and_cache, which stringifies
    it into air.mlir -- so it must be a module, not the LaunchContext that
    build_launch returns.
    """
    return build_launch(**kwargs).build(target="npu2")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="attn_npu2_temporal_causal.py",
        description="Flash attention, NPU2, temporal causal -- each core loops "
        "its own causal prefix of K blocks instead of splitting the reduction "
        "across a cascade",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--lk", type=int, default=512)
    parser.add_argument("--lkp", type=int, default=64)
    parser.add_argument("--lq", type=int, default=512)
    parser.add_argument("--lqp", type=int, default=256)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--num-q-tiles", type=int, default=4)
    parser.add_argument("--num-cascade-stages", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--num-heads-per-unroll", type=int, default=2)
    parser.add_argument("--q-tiles-per-core", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--perf-iters", type=int, default=0)
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"], default="elf"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-only"],
        default="compile-and-run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lk, lkp, lq, lqp = args.lk, args.lkp, args.lq, args.lqp
    dk, dv = args.dk, args.dv
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads
    gqa_group_size = num_heads // num_kv_heads
    causal = args.causal

    launch = build_launch(
        lk=lk,
        lkp=lkp,
        lq=lq,
        lqp=lqp,
        dk=dk,
        dv=dv,
        num_q_tiles=args.num_q_tiles,
        num_cascade_stages=args.num_cascade_stages,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        num_heads_per_unroll=args.num_heads_per_unroll,
        q_tiles_per_core=args.q_tiles_per_core,
    )
    mlir_module = launch.build(target="npu2")
    if args.print_module_only:
        print(mlir_module)
        return 0

    from ml_dtypes import bfloat16

    rng = np.random.default_rng(42)
    # Sequence-first inputs. N(0, 1) rather than a uniform positive range, so
    # the check sees the signed distribution the GPU SDPA tests use.
    input_q = rng.standard_normal((lq, num_heads * dk)).astype(bfloat16)
    input_k = rng.standard_normal((lk, num_kv_heads * dk)).astype(bfloat16)
    input_v = rng.standard_normal((lk, num_kv_heads * dv)).astype(bfloat16)

    inv_sqrt_dk = 1.0 / sqrt(dk)
    per_head = np.zeros((num_heads, lq, dv), dtype=bfloat16)
    for head in range(num_heads):
        kv_h = head // gqa_group_size
        qf = input_q[:, head * dk : (head + 1) * dk].astype(np.float32)
        kf = input_k[:, kv_h * dk : (kv_h + 1) * dk].astype(np.float32)
        vf = input_v[:, kv_h * dv : (kv_h + 1) * dv].astype(np.float32)
        scores = qf @ kf.T * inv_sqrt_dk
        if causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        p = np.exp(scores - mx)
        p = p / np.sum(p, axis=-1, keepdims=True)
        per_head[head] = (p @ vf).astype(bfloat16)
    expected = per_head.transpose(1, 0, 2).reshape(lq, num_heads * dv).copy()

    # Q@K^T scales with dk and P@V with dv, so 2*heads*lq*lk*(dk+dv); causal
    # masking roughly halves the effective work.
    perf_flops = 2.0 * num_heads * lq * lk * (dk + dv)
    if causal:
        perf_flops *= 0.5

    opts = dict(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=[1, 1],
        output_format=args.output_format,
        instance_name="attention_bf16",
        target_device=launch.target,
    )
    if args.compile_mode == "compile-only":
        backend = XRTBackend(**opts)
        backend.compile(mlir_module)
        print("Compilation complete.")
        return 0

    runner = XRTRunner(
        report_precision=True,
        n_perf_iters=args.perf_iters,
        perf_flops=(perf_flops if args.perf_iters > 0 else None),
        **opts,
    )
    return runner.run_test(
        mlir_module,
        inputs=[input_q, input_k, input_v],
        expected_outputs=[expected],
        rtol=1.6e-2,
        atol=1e-1,
    )


if __name__ == "__main__":
    exit(main())
