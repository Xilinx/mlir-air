# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention with memtile-relayed dataflow on air.api -- NPU1 (AIE2).

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

Key differences from the NPU2 variant:
  - M=4 (mmul<4,8,4>) instead of M=8 (mmul<8,8,8>)
  - num_heads_per_unroll=1
  - LUT-based exp (no aie::exp2 on AIE2)
  - 1/sqrt(dk) scaling inside fused_softmax
  - links attn_npu1.o

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

KERNEL = "attn_npu1.o"
M = 4  # mmul_m = mmul_n = 4 for the AIE2 mmul<4,8,4>
K_MMUL = 8  # mmul_k = 8


def build_launch(
    lk=512,
    lkp=64,
    lq=512,
    lqp=256,
    dk=64,
    dv=64,
    num_q_tiles=4,
    num_cascade_stages=4,
    num_heads=1,
    num_kv_heads=None,
    causal=False,
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

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_kv_heads > 0, f"num_kv_heads must be positive, got {num_kv_heads}"
    assert (
        num_heads % num_kv_heads == 0
    ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    gqa_group_size = num_heads // num_kv_heads

    num_heads_per_unroll = 1
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
    apply_causal_mask = (
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
    Q = air.tensor([num_heads, lq, dk], bf16)
    K = air.tensor([num_kv_heads, lk, dk], bf16)
    # V and the output use a transposed L3 layout so that one dv_tile is
    # contiguous: [heads * dv_chunks, seq, dv_tile] rather than [heads, seq, dv].
    V = air.tensor([num_kv_heads * dv_chunks, lk, dv_tile], bf16)
    GP = air.tensor([num_heads * dv_chunks, lq, dv_tile], bf16)

    # Flat views, because every L3 access pattern here is a run of elements at a
    # computed offset rather than a subscript of the declared rank.
    q_flat = Q.reshape(num_heads * lq * dk)
    k_flat = K.reshape(num_kv_heads * lk * dk)
    v_flat = V.reshape(num_kv_heads * dv_chunks * lk * dv_tile)
    gp_flat = GP.reshape(num_heads * dv_chunks * lq * dv_tile)

    grid = [range(num_lq_iters), range(num_head_groups)]
    if dv_chunks > 1:
        grid.append(range(dv_chunks))

    with air.launch(grid, name="attention_bf16") as launch:
        # The body is registered at the grid's own arity -- the DSL checks that
        # they agree -- so the third coordinate is bound to 0 when the value
        # dimension fits in one tile and there is no third axis.
        def run(lx, ly, lz):

            q_launch_off = lx * (lqp * dk)
            out_launch_off = lx * (lqp * dv_tile)
            head_base = ly * H

            for head_local in range(H):
                head_idx = head_base + head_local
                kv_head_idx = (
                    head_idx if gqa_group_size == 1 else head_idx // gqa_group_size
                )

                q_off = head_idx * (lq * dk) + q_launch_off
                k_off = kv_head_idx * (lk * dk)
                v_off = (kv_head_idx * dv_chunks + lz) * (lk * dv_tile)
                out_off = (head_idx * dv_chunks + lz) * (lq * dv_tile) + out_launch_off

                # Q: NQ tiles, each split into dk_chunks depth slices, with the
                # chunk axis lifted over the row axis so one send is a
                # [tile_size_q, dk_tile] block.
                for s in range(NS):
                    qkin[s].put(
                        q_flat[q_off : q_off + lqp * dk]
                        .reshape(NQ, tile_size_q, dk_chunks, dk_tile)
                        .transpose(0, 2, 1, 3),
                        indices=[head_local],
                    )

                # K: the same split over this stage's chunks.
                for s in range(NS):
                    off = k_off + s * lk_per_stage * dk
                    qkin[s].put(
                        k_flat[off : off + chunks_per_stage * lkp * dk]
                        .reshape(chunks_per_stage, lkp, dk_chunks, dk_tile)
                        .transpose(0, 2, 1, 3),
                        indices=[head_local],
                    )

                # V needs no permutation: its L3 layout is already dv_tile-major.
                for s in range(NS):
                    off = v_off + s * lk_per_stage * dv_tile
                    vin[s].put(
                        v_flat[off : off + chunks_per_stage * lkp * dv_tile].reshape(
                            chunks_per_stage, lkp, dv_tile
                        ),
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
                                .reshape(lkp // M, M, dk_tile // K_MMUL, K_MMUL)
                                .transpose(2, 0, 1, 3),
                                indices=[seg_x, 0, 0],
                            )
                        for _ in air.sequential(0, chunks_per_stage * dk_chunks):
                            qkin[s].get(qk_l2[s], indices=[seg_x])
                            qk2l1[s].put(
                                qk_l2[s]
                                .reshape(lkp // M, M, dk_tile // K_MMUL, K_MMUL)
                                .transpose(2, 0, 1, 3),
                                indices=[seg_x, 0, 0],
                            )

                    for s in range(NS):
                        for _ in air.sequential(0, chunks_per_stage):
                            vin[s].get(v_l2[s], indices=[seg_x])
                            v2l1[s].put(
                                v_l2[s]
                                .reshape(lkp // K_MMUL, K_MMUL, dv_tile // M, M)
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
                                zero_fill_g(g.reshape(g_flat))

                                for dk_c in range(dk_chunks):
                                    for s in range(NS):
                                        with ops.branch(ty == s):
                                            qk2l1[s].get(qk, indices=[seg_x, ty, tx])
                                    matmul_a_b(q_saved[dk_c], qk, g.reshape(g_flat))

                                for s in range(NS):
                                    with ops.branch(ty == s):
                                        v2l1[s].get(v_l1, indices=[seg_x, ty, tx])

                                if causal:
                                    apply_causal_mask(
                                        g,
                                        ctr[0] + tx,
                                        ty * chunks_per_stage + chunk,
                                    )

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
                                            dv_tile // M, tile_size_q // M, M, M
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
                out_off = (head_idx * dv_chunks + lz) * (lq * dv_tile) + out_launch_off
                gpout.get(
                    gp_flat[out_off : out_off + lqp * dv_tile],
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


def build_module(**kwargs):
    """The MLIR module. Return type is the llms/ builders' contract.

    The llms/ prefill builders import this name and hand the result straight to
    KernelCache.compile_and_cache, which stringifies it into air.mlir -- so it
    must be a module, not the LaunchContext that build_launch returns.
    """
    return build_launch(**kwargs).build(target="npu1")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="attn_npu1.py",
        description="Flash attention with memtile-relayed L3-to-L1 Q/K/V -- "
        "selective Q capture (NPU1/AIE2 variant)",
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
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"], default="xclbin"
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
    num_q_tiles = 4
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else num_heads
    causal = args.causal
    gqa_group_size = num_heads // num_kv_heads

    launch = build_launch(
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
    )
    mlir_module = launch.build(target="npu1")
    if args.print_module_only:
        print(mlir_module)
        return 0

    from ml_dtypes import bfloat16

    rng = np.random.default_rng(42)
    val_range = 4.0
    input_q = rng.uniform(0, val_range, (num_heads, lq, dk)).astype(bfloat16)
    input_k = rng.uniform(0, val_range, (num_kv_heads, lk, dk)).astype(bfloat16)
    input_v_orig = rng.uniform(0, val_range, (num_kv_heads, lk, dv)).astype(bfloat16)
    dv_chunks_host = dv // lkp
    input_v = (
        input_v_orig.reshape(num_kv_heads, lk, dv_chunks_host, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_kv_heads * dv_chunks_host, lk, lkp)
        .copy()
    )

    inv_sqrt_dk = 1.0 / sqrt(dk)
    sdpa_output = np.zeros((num_heads, lq, dv), dtype=bfloat16)
    for head in range(num_heads):
        kv_h = head // gqa_group_size
        qf = input_q[head].astype(np.float32)
        kf = input_k[kv_h].astype(np.float32)
        vf = input_v_orig[kv_h].astype(np.float32)
        scores = qf @ kf.T * inv_sqrt_dk
        if causal:
            mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        p = np.exp(scores - mx)
        p = p / np.sum(p, axis=-1, keepdims=True)
        sdpa_output[head] = (p @ vf).astype(bfloat16)

    expected = (
        sdpa_output.reshape(num_heads, lq, dv_chunks_host, lkp)
        .transpose(0, 2, 1, 3)
        .reshape(num_heads * dv_chunks_host, lq, lkp)
        .copy()
    )

    tiling = [1, 1, 1] if dv_chunks_host > 1 else [1, 1]
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
    )
    return runner.run_test(
        mlir_module,
        inputs=[input_q, input_k, input_v],
        expected_outputs=[expected],
        atol=0.15,
        rtol=0.04,
        max_mismatch_percentage=0.5,
        min_correlation=0.99,
    )


if __name__ == "__main__":
    exit(main())
