# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention on air.api -- NPU2, sequence-first input layout.

Q, K and V all reach L1 through the memtile: per-stage ``QKIn``/``QK2L1`` and
``VIn``/``V2L1`` channels relay L3 -> L2 -> L1. Q tiles are captured
selectively -- every tile receives all NQ Q sends and copies only the one whose
index matches its own ``tx`` -- and the per-stage partial results merge along a
cascade, last stage to first.

Multi-head runs through a segment unroll: the segment carries its own
iteration space of ``num_heads_per_unroll`` (1 on NPU1, whose 4x4 array does one
head at a time), the QK/V channels carry a leading head axis, and the cascade
channels stay 2-D because they are private to one segment instance. MHA, GQA
and causal masking are all supported.

Default design parameters:
  lk=512, lkp=64, lq=512, lqp=256, dk=64, dv=64
  num_q_tiles=4, num_cascade_stages=4, num_heads=1

Q, K, V and the output are laid out sequence-first -- ``[seq, heads * d]``,
with the heads interleaved along the feature axis -- rather than head-first
``[heads, seq, d]``. That is the layout the surrounding pipeline already has, so
consuming it directly removes a host-side transpose per tensor per layer.

Only the L3 access patterns change. Head ``h``'s slice is a *column* range
rather than a leading index, so the send is a two-dimensional slice of the
tensor re-described with the same reshape-and-transpose the other variants use:
the row range splits into (tile, row) and the column range into (chunk,
element), and the chunk axis lifts over the row axis. The row stride is the
whole embedding width, which is what makes it a strided view rather than a flat
one.

Differences from the NPU1 variant, beyond mmul<8,8,8> and attn_npu2.o:

* ``num_heads_per_unroll`` defaults to 2 and is a parameter. It multiplies the
  physical columns -- ``num_heads_per_unroll * num_q_tiles``, which must be at
  most 8 -- so the 8-column part runs two heads at once.
* Under ``causal``, fully-future K-blocks always skip their matmul, softmax and
  PV rather than computing them and masking to -inf. It is not a flag here, as
  it is in ``attn_npu2``: numerically it is identical -- a fully masked block
  contributes exp(-inf) = 0 -- and it saves the wasted block-matmul over the
  causal upper triangle, which grows with sequence length. The channel gets
  stay unconditional so the channels stay balanced; a skipped block leaves the
  stage's neutral local, so the cascade merge is an identity.

That skip is the one place this needs a branch on a value rather than a
predicated write, because what it skips is a ``func.call`` and ``ops.select``
evaluates both arms. The q-block index is rank 0, so the comparison opens a
region.

DMA channel budget per compute tile is 2 S2MM + 2 MM2S:
  S2MM 0: QK (Q selective capture, then K chunks)
  S2MM 1: V, per stage via the memtile
  MM2S 0: cascade or output
  MM2S 1: cascade

Three things about this port are worth knowing.

**Every hand-built AffineMap is gone.** The predecessor spelled each offset as
an ``AffineMap.get`` plus an ``affine_apply``; here they are Python arithmetic
on the launch and tile coordinates, and the DSL emits the same
``affine.apply``. The access patterns went the same way: what was an offsets /
sizes / strides triple written out by hand is a ``reshape`` and a ``transpose``
of the tensor, which is a view and moves nothing. The 4-D Q send, for instance,
is the sequence axis split into (tile, row) and the depth axis into (chunk,
element), with the chunk axis lifted over the row axis.

**The causal counter is predicated writes, not branches.** Each core keeps a
small i32 tile in L1 holding a boot flag, a q-block index and a head index, and
carries it across launch iterations. The predecessor updates it inside
``scf.if``; every arm of those branches only ever stores to that tile, so they
are ``ops.select`` here -- which emits the same load, compare, add, select and
store. The q-block index reaches ``apply_causal_mask`` as ``ctr[0] + tx``: rank
zero, so it is passed as a value rather than as a memref.

**``ops.branch`` replaces ``affine.if``.** The stage and cascade dispatch was an
``IntegerSet`` per case; the conditions are plain comparisons on ``ty``, so they
are written as such. The middle-stage arm nests inside the first-stage
``otherwise()`` rather than testing a two-sided range, which is the shape
``cascade_reduction`` and ``matvec_cascade`` already use.
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
M = 8  # mmul_m = mmul_k = mmul_n, the AIE2P mmul<8,8,8>
K_MMUL = 8


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
):
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
    if causal:
        assert lq == lk, f"Causal masking requires lq == lk, got lq={lq}, lk={lk}"
        assert lqp // num_q_tiles == lkp, (
            f"Causal masking requires tile_size_q == lkp, got "
            f"tile_size_q={lqp // num_q_tiles}, lkp={lkp}"
        )

    # Skipping fully-future blocks is unconditional under causal here.
    causal_skip = causal
    window_blocks = None

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_kv_heads > 0, f"num_kv_heads must be positive, got {num_kv_heads}"
    assert (
        num_heads % num_kv_heads == 0
    ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    gqa_group_size = num_heads // num_kv_heads

    assert num_heads % num_heads_per_unroll == 0, (
        f"num_heads ({num_heads}) must be divisible by num_heads_per_unroll "
        f"({num_heads_per_unroll})"
    )
    num_head_groups = num_heads // num_heads_per_unroll

    num_lq_iters = lq // lqp
    tile_size_q = lqp // num_q_tiles
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // num_cascade_stages
    lk_per_stage = lkp * chunks_per_stage

    NQ = num_q_tiles
    NS = num_cascade_stages
    H = num_heads_per_unroll

    # L1 tile shapes. G is allocated 2-D for the matmul and handed to the
    # kernels flat, which is the one place a collapse is needed.
    g_flat = tile_size_q * lkp

    # ---------------------------------------------------------------- kernels
    zero_fill_g = air.extern("zero_fill_g_bf16", link_with=KERNEL)
    zero_fill_gp = air.extern("zero_fill_gp_bf16", link_with=KERNEL)
    zero_fill_sp = air.extern("zero_fill_sp_bf16", link_with=KERNEL)
    neg_inf_fill_up = air.extern("neg_inf_fill_up_bf16", link_with=KERNEL)
    matmul_a_b = air.extern("matmul_a_b_bf16", link_with=KERNEL)
    matmul_g_b = air.extern("matmul_g_b_bf16", link_with=KERNEL)
    fused_softmax = air.extern("fused_softmax", link_with=KERNEL)
    maximum_up_u = air.extern("maximum_up_u_bf16", link_with=KERNEL)
    exp_up_minus_u = air.extern("exp_up_minus_u", link_with=KERNEL)
    mul_r_gp = air.extern("mul_r_gp", link_with=KERNEL)
    accum_sp_r_s = air.extern("accum_sp_r_s", link_with=KERNEL)
    vector_copy = air.extern("vector_copy_32elems", link_with=KERNEL, scalars=[i32])
    copy_tile = air.extern("copy_tile", link_with=KERNEL)
    div_gp_sp = air.extern("div_gp_sp", link_with=KERNEL)
    add_gp_g = air.extern("add_gp_g", link_with=KERNEL)
    apply_mask = (
        air.extern("apply_causal_mask", link_with=KERNEL, scalars=[i32, i32])
        if causal
        else None
    )

    # --------------------------------------------------------------- channels
    # Declared in the order the predecessor declares them, which is the order
    # they are printed in.
    qk2l1, qkin, v2l1, vin = [], [], [], []
    for s in range(NS):
        qk2l1.append(
            air.channel(f"QK2L1_{s}", size=[H, 1, 1], broadcast_shape=[H, 1, NQ])
        )
        qkin.append(air.channel(f"QKIn_{s}", size=[H]))
    for s in range(NS):
        v2l1.append(
            air.channel(f"V2L1_{s}", size=[H, 1, 1], broadcast_shape=[H, 1, NQ])
        )
        vin.append(air.channel(f"VIn_{s}", size=[H]))
    cascade_gp = air.channel(
        "cascade_gp", size=[NQ, NS - 1], channel_type="npu_cascade"
    )
    cascade_up = air.channel(
        "cascade_up", size=[NQ, NS - 1], channel_type="npu_cascade"
    )
    cascade_sp = air.channel(
        "cascade_sp", size=[NQ, NS - 1], channel_type="npu_cascade"
    )
    gp2l2 = air.channel("Gp2L2", size=[NQ, 1])
    gpout = air.channel("GpOut", size=[H])

    # ---------------------------------------------------------------- tensors
    # Sequence-first: heads are interleaved along the feature axis, so a head is
    # a column range rather than a leading index.
    emb_q = num_heads * dk
    emb_k = num_kv_heads * dk
    emb_v = num_kv_heads * dv
    emb_out = num_heads * dv
    Q = air.tensor([lq, emb_q], bf16)
    K = air.tensor([lk, emb_k], bf16)
    V = air.tensor([lk, emb_v], bf16)
    GP = air.tensor([lq, emb_out], bf16)

    grid = [range(num_lq_iters), range(num_head_groups)]
    if dv_chunks > 1:
        grid.append(range(dv_chunks))

    with air.launch(grid, name="attention_bf16") as launch:
        # The body is registered at the grid's own arity -- the DSL checks that
        # they agree -- so the third coordinate is bound to 0 when the value
        # dimension fits in one tile and there is no third axis.
        def run(lx, ly, lz):

            head_base = ly * H

            for head_local in range(H):
                head_idx = head_base + head_local
                kv_head_idx = (
                    head_idx if gqa_group_size == 1 else head_idx // gqa_group_size
                )

                q_row = lx * lqp
                q_col = head_idx * dk
                k_col = kv_head_idx * dk
                v_col = kv_head_idx * dv + lz * dv_tile

                # Q: a [lqp, dk] window of the embedding, re-described as NQ
                # tiles each split into dk_chunks depth slices, with the chunk
                # axis lifted over the row axis so one send is a
                # [tile_size_q, dk_tile] block. The row stride is the whole
                # embedding width, which is what makes this a strided view
                # rather than a flat one.
                for s in range(NS):
                    qkin[s].put(
                        Q[q_row : q_row + lqp, q_col : q_col + dk]
                        .reshape(NQ, tile_size_q, dk_chunks, dk_tile)
                        .transpose(0, 2, 1, 3),
                        indices=[head_local],
                    )

                # K: the same split over this stage's chunks.
                for s in range(NS):
                    row = s * lk_per_stage
                    qkin[s].put(
                        K[row : row + chunks_per_stage * lkp, k_col : k_col + dk]
                        .reshape(chunks_per_stage, lkp, dk_chunks, dk_tile)
                        .transpose(0, 2, 1, 3),
                        indices=[head_local],
                    )

                # V needs no permutation, only the chunk split.
                for s in range(NS):
                    row = s * lk_per_stage
                    vin[s].put(
                        V[
                            row : row + chunks_per_stage * lkp,
                            v_col : v_col + dv_tile,
                        ].reshape(chunks_per_stage, lkp, dv_tile),
                        indices=[head_local],
                    )

            with air.segment([range(H), range(1)], name="attn_seg") as seg:

                @seg.body
                def _(seg_x, seg_y):
                    qk_l2 = [
                        air.alloc([lkp, dk_tile], bf16, scope=seg.private())
                        for _ in range(NS)
                    ]
                    v_l2 = [
                        air.alloc([lkp, dv_tile], bf16, scope=seg.private())
                        for _ in range(NS)
                    ]
                    gp_l2 = air.alloc([lqp, dv_tile], bf16, scope=seg.private())

                    # L1, allocated here so it survives the whole segment and
                    # reaches the herd as an operand.
                    q_saved = [
                        air.alloc([tile_size_q, dk_tile], bf16, scope=seg.per_core())
                        for _ in range(dk_chunks)
                    ]
                    qk = air.alloc([lkp, dk_tile], bf16, scope=seg.per_core())
                    v_l1 = air.alloc([lkp, dv_tile], bf16, scope=seg.per_core())
                    g = air.alloc([tile_size_q, lkp], bf16, scope=seg.per_core())
                    gp = air.alloc([tile_size_q, dv_tile], bf16, scope=seg.per_core())
                    up = air.alloc([tile_size_q, 1], bf16, scope=seg.per_core())
                    sp = air.alloc([tile_size_q, 1], bf16, scope=seg.per_core())
                    ctr = (
                        air.alloc(
                            [4 if dv_chunks > 1 else 3], i32, scope=seg.per_core()
                        )
                        if causal
                        else None
                    )

                    # The memtile relay. One get per L3 send, one put per L1
                    # receive, and the put re-describes the [lkp, dk_tile] tile
                    # in the 4x8 blocks the mmul instruction consumes.
                    for s in range(NS):
                        for _ in air.sequential(0, NQ * dk_chunks):
                            qkin[s].get(qk_l2[s], indices=[seg_x])
                            qk2l1[s].put(
                                qk_l2[s]
                                .reshape(lkp // M, M, dk_tile // M, M)
                                .transpose(2, 0, 1, 3),
                                indices=[seg_x, 0, 0],
                            )
                        for _ in air.sequential(0, chunks_per_stage * dk_chunks):
                            qkin[s].get(qk_l2[s], indices=[seg_x])
                            qk2l1[s].put(
                                qk_l2[s]
                                .reshape(lkp // M, M, dk_tile // M, M)
                                .transpose(2, 0, 1, 3),
                                indices=[seg_x, 0, 0],
                            )

                    for s in range(NS):
                        for _ in air.sequential(0, chunks_per_stage):
                            vin[s].get(v_l2[s], indices=[seg_x])
                            v2l1[s].put(
                                v_l2[s]
                                .reshape(lkp // M, M, dv_tile // M, M)
                                .transpose(2, 0, 1, 3),
                                indices=[seg_x, 0, 0],
                            )

                    with air.herd(
                        [range(NQ), range(NS)],
                        name="herd_0",
                        shape=(NQ, NS),
                        link_with=KERNEL,
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            zero_fill_gp(gp)
                            zero_fill_sp(sp)
                            neg_inf_fill_up(up)

                            if causal:
                                # Boot: set the counters the first time this
                                # core runs. Predicated writes rather than a
                                # branch -- every arm only stores. ctr[1] is
                                # written last because the predicate reads it.
                                first = ops.equal(ctr[1:2], 0)
                                ctr[0:1] = ops.select(first, 0, ctr[0:1])
                                ctr[2:3] = ops.select(first, 0, ctr[2:3])
                                if dv_chunks > 1:
                                    ctr[3:4] = ops.select(first, 0, ctr[3:4])
                                ctr[1:2] = ops.select(first, 1, ctr[1:2])

                            # Q selective capture: receive all NQ * dk_chunks
                            # sends, keep the one this column owns.
                            for qt in range(NQ):
                                for dk_c in range(dk_chunks):
                                    for s in range(NS):
                                        with ops.branch(ty == s):
                                            qk2l1[s].get(qk, indices=[seg_x, ty, tx])
                                    with ops.branch(tx == qt):
                                        copy_tile(qk, q_saved[dk_c])

                            for chunk in air.sequential(0, chunks_per_stage):
                                # This block's place in the mask. q_block is a
                                # value the core carries in L1; kv_block is a
                                # coordinate.
                                q_block = ctr[0] + tx if causal else None
                                kv_block = ty * chunks_per_stage + chunk

                                def live():
                                    """The region for a block the mask keeps.

                                    kv_block <= q_block, and inside the window
                                    when there is one. The lower bound is
                                    inclusive: that block is the ragged edge
                                    apply_window_mask half-keeps, not a dead
                                    one. Conjunction is nesting in this DSL, so
                                    the window bound is a second region inside
                                    the first rather than an arith.andi.
                                    """
                                    outer = ops.branch(q_block >= kv_block)
                                    outer.__enter__()
                                    if window_blocks is None:
                                        return [outer]
                                    inner = ops.branch(
                                        q_block - window_blocks <= kv_block
                                    )
                                    inner.__enter__()
                                    return [inner, outer]

                                def close(regions):
                                    for r in reversed(regions):
                                        r.__exit__(None, None, None)

                                zero_fill_g(g.reshape(g_flat))

                                for dk_c in range(dk_chunks):
                                    # The gets stay unconditional even for a
                                    # skipped block: the channels have to stay
                                    # balanced or the herd deadlocks. Only the
                                    # arithmetic is elided.
                                    for s in range(NS):
                                        with ops.branch(ty == s):
                                            qk2l1[s].get(qk, indices=[seg_x, ty, tx])
                                    if causal_skip:
                                        regions = live()
                                        matmul_a_b(q_saved[dk_c], qk, g.reshape(g_flat))
                                        close(regions)
                                    else:
                                        matmul_a_b(q_saved[dk_c], qk, g.reshape(g_flat))

                                for s in range(NS):
                                    with ops.branch(ty == s):
                                        v2l1[s].get(v_l1, indices=[seg_x, ty, tx])

                                def softmax_accumulate():
                                    if causal:
                                        if window_blocks is not None:
                                            apply_mask(
                                                g, q_block, kv_block, window_blocks
                                            )
                                        else:
                                            apply_mask(g, q_block, kv_block)
                                    s_tmp = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    r_tmp = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    fused_softmax(g.reshape(g_flat), up, s_tmp, r_tmp)
                                    mul_r_gp(r_tmp, gp)
                                    matmul_g_b(g.reshape(g_flat), v_l1, gp)
                                    accum_sp_r_s(sp, r_tmp, s_tmp)
                                    vector_copy(0, s_tmp, sp)

                                if causal_skip:
                                    regions = live()
                                    softmax_accumulate()
                                    close(regions)
                                else:
                                    softmax_accumulate()

                            # Cascade merge, north to south.
                            with ops.branch(ty == NS - 1) as north:
                                cascade_gp.put(gp, indices=[tx, ty - 1])
                                cascade_up.put(up, indices=[tx, ty - 1])
                                cascade_sp.put(sp, indices=[tx, ty - 1])

                            with north.otherwise():

                                def merge():
                                    """Fold the neighbour's partials into ours.

                                    Returns the buffers holding the merged
                                    result, which the caller either forwards or
                                    normalises and drains.
                                    """
                                    gp_c = air.alloc(
                                        [tile_size_q, dv_tile], bf16, scope=h.private()
                                    )
                                    up_c = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    sp_c = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    cascade_gp.get(gp_c, indices=[tx, ty])
                                    cascade_up.get(up_c, indices=[tx, ty])
                                    cascade_sp.get(sp_c, indices=[tx, ty])
                                    up_s = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    vector_copy(0, up, up_s)
                                    maximum_up_u(up_c, up)
                                    rc = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    exp_up_minus_u(up_c, up, rc)
                                    rl = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    exp_up_minus_u(up_s, up, rl)
                                    mul_r_gp(rc, gp_c)
                                    mul_r_gp(rl, gp)
                                    add_gp_g(gp, gp_c)
                                    st = air.alloc(
                                        [tile_size_q, 1], bf16, scope=h.private()
                                    )
                                    zero_fill_sp(st)
                                    accum_sp_r_s(sp_c, rc, st)
                                    accum_sp_r_s(sp, rl, st)
                                    vector_copy(0, st, sp_c)
                                    return gp_c, sp_c

                                with ops.branch(ty == 0) as south:
                                    # Southernmost: normalise and drain.
                                    gp_c, sp_c = merge()
                                    div_gp_sp(sp_c, gp_c)
                                    gp2l2.put(
                                        gp_c.reshape(
                                            tile_size_q // M, dv_tile // M, M, M
                                        ).transpose(1, 2, 0, 3),
                                        indices=[tx, 0],
                                    )

                                with south.otherwise():
                                    gp_c, _sp_c = merge()
                                    cascade_gp.put(gp_c, indices=[tx, ty - 1])
                                    cascade_up.put(up, indices=[tx, ty - 1])
                                    cascade_sp.put(_sp_c, indices=[tx, ty - 1])

                            if causal:
                                # Advance the q block once per head-group cycle,
                                # and only on the last dv chunk when there is
                                # more than one. Nested selects rather than
                                # nested ifs, for the reason above.
                                head_next = ctr[2:3] + 1
                                wrapped = head_next >= num_head_groups
                                q_adv = ops.select(wrapped, ctr[0:1] + NQ, ctr[0:1])
                                head_adv = ops.select(wrapped, 0, head_next)
                                if dv_chunks > 1:
                                    last_dv = ctr[3:4] >= dv_chunks - 1
                                    ctr[0:1] = ops.select(last_dv, q_adv, ctr[0:1])
                                    ctr[2:3] = ops.select(last_dv, head_adv, ctr[2:3])
                                    ctr[3:4] = ops.select(last_dv, 0, ctr[3:4] + 1)
                                else:
                                    ctr[0:1] = q_adv
                                    ctr[2:3] = head_adv

                    # Gather the NQ column results into the L2 output tile.
                    for col in air.parallel(0, NQ):
                        gp2l2.get(
                            gp_l2[
                                col * tile_size_q : col * tile_size_q + tile_size_q, :
                            ],
                            indices=[col, 0],
                        )

                    gpout.put(gp_l2, indices=[seg_x])

            for head_local in range(H):
                head_idx = head_base + head_local
                out_row = lx * lqp
                out_col = head_idx * dv + lz * dv_tile
                gpout.get(
                    GP[out_row : out_row + lqp, out_col : out_col + dv_tile],
                    indices=[head_local],
                )

        if dv_chunks > 1:

            @launch.body
            def _(lx, ly, lz):
                run(lx, ly, lz)

        else:

            @launch.body
            def _(lx, ly):
                run(lx, ly, 0)

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="attn_npu2_seqfirst.py",
        description="Flash attention with memtile-relayed L3-to-L1 Q/K/V -- "
        "selective Q capture (NPU2/AIE2P, sequence-first layout)",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--lk", type=int, default=512)
    parser.add_argument("--lkp", type=int, default=64)
    parser.add_argument("--lq", type=int, default=512)
    parser.add_argument("--lqp", type=int, default=256)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--num-cascade-stages", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-heads-per-unroll", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    # Both carried from the predecessor's CLI. --num-q-tiles was hardcoded to 4
    # by the conversion and --perf-iters dropped entirely; neither is exercised
    # by a lit, which is exactly why losing them was invisible.
    parser.add_argument(
        "--num-q-tiles",
        type=int,
        default=4,
        dest="num_q_tiles",
        help="Number of tiles to partition the Q chunk into (default: 4). "
        "Under causal masking, lqp / num_q_tiles must equal lkp.",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters and report latency "
        "and achieved TFLOP/s alongside the correctness check",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        help="Output format (default: elf, as on NPU2)",
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
    num_q_tiles = args.num_q_tiles
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads
    causal = args.causal
    gqa_group_size = num_heads // num_kv_heads

    launch = build_module(
        lk=lk,
        lkp=lkp,
        lq=lq,
        lqp=lqp,
        dk=dk,
        dv=dv,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=args.num_cascade_stages,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        num_heads_per_unroll=args.num_heads_per_unroll,
    )
    mlir_module = launch.build(target="npu2")
    if args.print_module_only:
        print(mlir_module)
        return 0

    from ml_dtypes import bfloat16

    rng = np.random.default_rng(42)
    # SEQ-FIRST inputs: [seq, heads * dim]. This is the whole point of the
    # variant -- the builder declares air.tensor([lq, num_heads * dk]) and the
    # heads are a slice of the trailing axis, not a leading axis. Feeding the
    # head-first arrays attn_npu2.py uses does NOT fail: the buffers are the
    # same size, so XRT copies them happily and the kernel reads transposed
    # data. That is what it did, for a correlation of 0.497.
    #
    # N(0,1) rather than uniform(0, 4), matching the GPU SDPA test standard
    # (PyTorch uses randn), so the check sees a signed distribution.
    input_q = rng.standard_normal((lq, num_heads * dk)).astype(bfloat16)
    input_k = rng.standard_normal((lk, num_kv_heads * dk)).astype(bfloat16)
    input_v = rng.standard_normal((lk, num_kv_heads * dv)).astype(bfloat16)

    inv_sqrt_dk = 1.0 / sqrt(dk)
    sdpa_output_hf = np.zeros((num_heads, lq, dv), dtype=bfloat16)
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
        sdpa_output_hf[head] = (p @ vf).astype(bfloat16)

    # And seq-first on the way out too: [lq, num_heads * dv].
    expected = sdpa_output_hf.transpose(1, 0, 2).reshape(lq, num_heads * dv).copy()

    # The output is 2-D, so the runtime loop tiling is 2-D.
    tiling = [1, 1]
    # Q@K^T scales with dk and P@V with dv, each 2*num_heads*lq*lk*<dim>;
    # causal masking roughly halves the effective work.
    perf_flops = 2.0 * num_heads * lq * lk * (dk + dv)
    if causal:
        perf_flops *= 0.5
    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            verbose=args.verbose,
            runtime_loop_tiling_sizes=tiling,
            output_format=args.output_format,
            instance_name="attention_bf16",
            target_device=launch.target,
        )
        backend.compile(mlir_module)
        print("Compilation complete.")
        return 0

    runner = XRTRunner(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=tiling,
        output_format=args.output_format,
        instance_name="attention_bf16",
        target_device=launch.target,
        report_precision=True,
        n_perf_iters=args.perf_iters,
        perf_flops=(perf_flops if args.perf_iters > 0 else None),
    )
    # The predecessor's tolerances, unchanged. The conversion had loosened them
    # to atol=0.15 / rtol=0.04 plus a 0.5% mismatch allowance -- a weaker gate
    # than the thing being replaced, which is not a trade a conversion gets to
    # make on its own.
    return runner.run_test(
        mlir_module,
        inputs=[input_q, input_k, input_v],
        expected_outputs=[expected],
        rtol=1.6e-2,
        atol=1e-1,
    )


if __name__ == "__main__":
    exit(main())
