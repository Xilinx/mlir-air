# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention as a pipeline of single-purpose cores on NPU2.

Each column takes one 64-row query block through a chain of herds, one stage
per core::

    --stages 3:  Q K^T -> softmax -> P V, rescaled as it accumulates
    --stages 4:  Q K^T -> softmax -> P V -> rescale

Query blocks are spread over up to eight columns, and K and V are broadcast to
every column, so there is no split of the key sequence and no cascade. The
stages hand 64x64 tiles to one another over tile DMA (``--handoff l1``) or
through a MemTile buffer per column (``--handoff l2``). With ``--causal-skip``
the 3-stage pipeline skips the arithmetic for blocks above the diagonal; their
tiles still stream.

The math is the ``kernel_fusion_based/attn_npu2.cc`` microkernels, wrapped by
``attn_stages.cc`` for the tiles the stages pass between them.

Two things here are worth reading as DSL rather than as attention:

* **A hand-off tile carries its row vectors.** The softmax core sends the
  rescale factor and the running row sum after the 64x64 tile in the same
  buffer, so every core keeps to two inbound DMA streams.
* **The causal q-block index is per-core L1 state.** The herds do not see the
  launch coordinates, so each core counts the query blocks it has processed in
  a small ``i32`` tile, written with ``ops.select`` and wrapped after the last
  block so that a rerun on the same hardware context starts again at block 0.
"""

import argparse
from math import sqrt

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

KERNEL = "attn_stages.o"
M = 8
B = 64  # query block == key block == head dim
MAX_COLS = 8


def build_launch(
    lk=512,
    lq=512,
    dk=64,
    dv=64,
    num_heads=2,
    causal=False,
    stages=3,
    handoff="l1",
    causal_skip=False,
):
    assert lq == lk, "one query block per key block"
    assert dk == B and dv == B, f"head dims must be {B}"
    assert stages in (3, 4)
    assert handoff in ("l1", "l2")
    assert not causal_skip or (
        causal and stages == 3
    ), "--causal-skip needs --causal and --stages 3"
    L, H = lq, num_heads
    NC = min(MAX_COLS, L // B)
    assert L % (B * NC) == 0, f"lq must be a multiple of {B * NC}"
    nchunks = L // B
    assert nchunks % 2 == 0, "key blocks are double-buffered"
    T = B * B
    TP = T + 2 * B  # tile + rescale factor + row sum
    q_iters = L // (B * NC)

    def q_counter(h, tx):
        """This core's query block and a closure that advances it, once per
        launch iteration. Heads cycle fastest in the launch grid."""
        c = air.alloc([3], i32, scope=h.private())
        first = ops.equal(c[1:2], 0)
        c[0:1] = ops.select(first, 0, c[0:1])
        c[2:3] = ops.select(first, 0, c[2:3])
        c[1:2] = ops.select(first, 1, c[1:2])

        def advance():
            head_next = c[2:3] + 1
            wrapped = head_next >= H
            q_next = c[0:1] + NC
            q_next = ops.select(q_next >= q_iters * NC, 0, q_next)
            c[0:1] = ops.select(wrapped, q_next, c[0:1])
            c[2:3] = ops.select(wrapped, 0, head_next)

        return c[0] + tx, advance

    zero_fill_g = air.extern("zero_fill_g_bf16", link_with=KERNEL)
    zero_fill_gp = air.extern("zero_fill_gp_bf16", link_with=KERNEL)
    zero_fill_sp = air.extern("zero_fill_sp_bf16", link_with=KERNEL)
    neg_inf_fill_up = air.extern("neg_inf_fill_up_bf16", link_with=KERNEL)
    matmul_a_b = air.extern("matmul_a_b_bf16", link_with=KERNEL)
    fused_softmax = air.extern("fused_softmax", link_with=KERNEL)
    accum_sp_r_s = air.extern("accum_sp_r_s", link_with=KERNEL)
    vector_copy = air.extern("vector_copy_32elems", link_with=KERNEL, scalars=[i32])
    apply_mask = air.extern("apply_causal_mask", link_with=KERNEL, scalars=[i32, i32])
    sm_out = air.extern("sm_out", link_with=KERNEL)
    pv_step = air.extern("pv_step", link_with=KERNEL)
    pv_final = air.extern("pv_final", link_with=KERNEL)
    pv_fresh = air.extern("pv_fresh", link_with=KERNEL)
    rs_step = air.extern("rs_step", link_with=KERNEL)
    rs_final = air.extern("rs_final", link_with=KERNEL)

    qin = air.channel("QIn", size=[NC])
    kin = air.channel("KIn")
    vin = air.channel("VIn")
    q2l1 = air.channel("Q2L1", size=[NC, 1])
    k2l1 = air.channel("K2L1", size=[1, 1], broadcast_shape=[NC, 1])
    v2l1 = air.channel("V2L1", size=[1, 1], broadcast_shape=[NC, 1])
    o2l2 = air.channel("O2L2", size=[NC, 1])
    oout = air.channel("OOut", size=[NC])

    # Stage-to-stage hand-offs. Through L2, each is a producer->L2 channel and
    # an L2->consumer channel; core to core, the two are the same channel.
    hand_names = ["G", "P"] + (["GV"] if stages == 4 else [])
    up_ch, dn_ch = {}, {}
    for n in hand_names:
        if handoff == "l1":
            up_ch[n] = dn_ch[n] = air.channel(f"{n}_L1L1", size=[NC, 1])
        else:
            up_ch[n] = air.channel(f"{n}_L1L2", size=[NC, 1])
            dn_ch[n] = air.channel(f"{n}_L2L1", size=[NC, 1])

    Q = air.tensor([H, L, B], bf16)
    K = air.tensor([H, L, B], bf16)
    V = air.tensor([H, L, B], bf16)
    O = air.tensor([H, L, B], bf16)
    q_flat = Q.reshape(H * L * B)
    k_flat = K.reshape(H * L * B)
    v_flat = V.reshape(H * L * B)
    o_flat = O.reshape(H * L * B)

    def blocked(buf):
        return buf.reshape(B // M, M, B // M, M).transpose(2, 0, 1, 3)

    def unblocked(buf):
        return buf.reshape(B // M, B // M, M, M).transpose(1, 2, 0, 3)

    with air.launch([range(q_iters), range(H)], name="attention_bf16") as launch:

        @launch.body
        def _(lx, ly):
            q_off = ly * (L * B) + lx * (NC * T)
            kv_off = ly * (L * B)
            for c in range(NC):
                qin.put(q_flat[q_off + c * T : q_off + (c + 1) * T], indices=[c])
            kin.put(k_flat[kv_off : kv_off + L * B].reshape(nchunks, B, B))
            vin.put(v_flat[kv_off : kv_off + L * B].reshape(nchunks, B, B))

            with air.segment(name="attn_seg") as seg:

                @seg.body
                def _():
                    # One L2 buffer per column, so each column's Q and O move
                    # on their own.
                    q_l2 = [
                        air.alloc([B, B], bf16, scope=seg.private(), column=c)
                        for c in range(NC)
                    ]
                    k_l2 = air.alloc([B, B], bf16, scope=seg.private())
                    v_l2 = air.alloc([B, B], bf16, scope=seg.private())
                    o_l2 = [
                        air.alloc([B, B], bf16, scope=seg.private(), column=c)
                        for c in range(NC)
                    ]

                    for c in range(NC):
                        qin.get(q_l2[c], indices=[c])
                        q2l1.put(blocked(q_l2[c]), indices=[c, 0])
                    for _ in air.sequential(0, nchunks):
                        kin.get(k_l2)
                        k2l1.put(blocked(k_l2), indices=[0, 0])
                    for _ in air.sequential(0, nchunks):
                        vin.get(v_l2)
                        v2l1.put(blocked(v_l2), indices=[0, 0])

                    if handoff == "l2":
                        for n in hand_names:
                            for c in range(NC):
                                bufs = [
                                    air.alloc([TP], bf16, scope=seg.private(), column=c)
                                    for _ in range(2)
                                ]
                                for _ in air.sequential(0, nchunks // 2):
                                    for b in range(2):
                                        up_ch[n].get(bufs[b], indices=[c, 0])
                                        dn_ch[n].put(bufs[b], indices=[c, 0])

                    # Stage 1: S = Q K^T.
                    with air.herd(
                        [range(NC), range(1)],
                        name="h_qk",
                        shape=(NC, 1),
                        at=(0, 2),
                        link_with=KERNEL,
                    ) as hqk:

                        @hqk.body
                        def _(tx, ty):
                            q_l1 = air.alloc([B, B], bf16, scope=hqk.private())
                            q2l1.get(q_l1, indices=[tx, ty])
                            kb = [
                                air.alloc([B, B], bf16, scope=hqk.private())
                                for _ in range(2)
                            ]
                            gb = [
                                air.alloc([TP], bf16, scope=hqk.private())
                                for _ in range(2)
                            ]
                            if causal_skip:
                                q_blk, q_adv = q_counter(hqk, tx)
                            for j in air.sequential(0, nchunks // 2):
                                for b in range(2):
                                    k2l1.get(kb[b], indices=[tx, ty])
                                    if causal_skip:
                                        with ops.branch(q_blk >= j * 2 + b):
                                            zero_fill_g(gb[b])
                                            matmul_a_b(q_l1, kb[b], gb[b])
                                    else:
                                        zero_fill_g(gb[b])
                                        matmul_a_b(q_l1, kb[b], gb[b])
                                    up_ch["G"].put(gb[b], indices=[tx, ty])
                            if causal_skip:
                                q_adv()

                    # Stage 2: online softmax over the key blocks.
                    with air.herd(
                        [range(NC), range(1)],
                        name="h_sm",
                        shape=(NC, 1),
                        at=(0, 3),
                        link_with=KERNEL,
                    ) as hsm:

                        @hsm.body
                        def _(tx, ty):
                            up = air.alloc([B, 1], bf16, scope=hsm.private())
                            sp = air.alloc([B, 1], bf16, scope=hsm.private())
                            s_t = air.alloc([B, 1], bf16, scope=hsm.private())
                            r_t = air.alloc([B, 1], bf16, scope=hsm.private())
                            gb = [
                                air.alloc([TP], bf16, scope=hsm.private())
                                for _ in range(2)
                            ]
                            # Send from separate buffers, not the ones
                            # received into.
                            pb = [
                                air.alloc([TP], bf16, scope=hsm.private())
                                for _ in range(2)
                            ]
                            neg_inf_fill_up(up)
                            zero_fill_sp(sp)
                            if causal:
                                q_blk, q_adv = q_counter(hsm, tx)

                            def softmax(g, kv_blk):
                                if causal:
                                    apply_mask(g, q_blk, kv_blk)
                                fused_softmax(g, up, s_t, r_t)
                                accum_sp_r_s(sp, r_t, s_t)
                                vector_copy(0, s_t, sp)

                            for j in air.sequential(0, nchunks // 2):
                                for b in range(2):
                                    dn_ch["G"].get(gb[b], indices=[tx, ty])
                                    if causal_skip:
                                        with ops.branch(q_blk >= j * 2 + b):
                                            softmax(gb[b], j * 2 + b)
                                            sm_out(gb[b], r_t, sp, pb[b])
                                    else:
                                        softmax(gb[b], j * 2 + b)
                                        sm_out(gb[b], r_t, sp, pb[b])
                                    up_ch["P"].put(pb[b], indices=[tx, ty])
                            if causal:
                                q_adv()

                    # Stage 3: O = P V, rescaled per block (3 stages) or
                    # passed on fresh (4 stages).
                    with air.herd(
                        [range(NC), range(1)],
                        name="h_pv",
                        shape=(NC, 1),
                        at=(0, 4),
                        link_with=KERNEL,
                    ) as hpv:

                        @hpv.body
                        def _(tx, ty):
                            pb = [
                                air.alloc([TP], bf16, scope=hpv.private())
                                for _ in range(2)
                            ]
                            vb = [
                                air.alloc([B, B], bf16, scope=hpv.private())
                                for _ in range(2)
                            ]
                            if stages == 3:
                                gp = air.alloc([B, B], bf16, scope=hpv.private())
                                s_l = air.alloc([B, 1], bf16, scope=hpv.private())
                                zero_fill_gp(gp)
                            else:
                                ob = [
                                    air.alloc([TP], bf16, scope=hpv.private())
                                    for _ in range(2)
                                ]
                            if causal_skip:
                                q_blk, q_adv = q_counter(hpv, tx)
                            for j in air.sequential(0, nchunks // 2):
                                for b in range(2):
                                    dn_ch["P"].get(pb[b], indices=[tx, ty])
                                    v2l1.get(vb[b], indices=[tx, ty])
                                    if causal_skip:
                                        with ops.branch(q_blk >= j * 2 + b):
                                            pv_step(pb[b], vb[b], gp, s_l)
                                    elif stages == 3:
                                        pv_step(pb[b], vb[b], gp, s_l)
                                    else:
                                        pv_fresh(pb[b], vb[b], ob[b])
                                        up_ch["GV"].put(ob[b], indices=[tx, ty])
                            if causal_skip:
                                q_adv()
                            if stages == 3:
                                pv_final(s_l, gp)
                                o2l2.put(unblocked(gp), indices=[tx, ty])

                    # Stage 4: O = O * r + P V per block, then normalise.
                    if stages == 4:
                        with air.herd(
                            [range(NC), range(1)],
                            name="h_rs",
                            shape=(NC, 1),
                            at=(0, 5),
                            link_with=KERNEL,
                        ) as hrs:

                            @hrs.body
                            def _(tx, ty):
                                ib = [
                                    air.alloc([TP], bf16, scope=hrs.private())
                                    for _ in range(2)
                                ]
                                acc = air.alloc([B, B], bf16, scope=hrs.private())
                                s_l = air.alloc([B, 1], bf16, scope=hrs.private())
                                zero_fill_gp(acc)
                                for _ in air.sequential(0, nchunks // 2):
                                    for b in range(2):
                                        dn_ch["GV"].get(ib[b], indices=[tx, ty])
                                        rs_step(ib[b], acc, s_l)
                                rs_final(s_l, acc)
                                o2l2.put(unblocked(acc), indices=[tx, ty])

                    for c in range(NC):
                        o2l2.get(o_l2[c], indices=[c, 0])
                        oout.put(o_l2[c], indices=[c])

            o_off = ly * (L * B) + lx * (NC * T)
            for c in range(NC):
                oout.get(o_flat[o_off + c * T : o_off + (c + 1) * T], indices=[c])

    return launch


def build_module(**kwargs):
    launch = build_launch(**kwargs)
    return launch.build(target="npu2"), launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="attn_npu2.py",
        description="Flash attention as a pipeline of single-purpose cores",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--lk", type=int, default=512)
    parser.add_argument("--lq", type=int, default=512)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        choices=[3, 4],
        help="3: QK, softmax, PV with the rescale folded into PV. "
        "4: QK, softmax, PV, rescale.",
    )
    parser.add_argument(
        "--handoff",
        default="l1",
        choices=["l1", "l2"],
        help="Pass tiles between stages core to core (l1) or through a "
        "MemTile buffer (l2)",
    )
    parser.add_argument(
        "--causal-skip",
        action="store_true",
        help="Skip the arithmetic for blocks above the diagonal (3 stages)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Run the kernel this many times on one hardware context and check "
        "the last output, so state the cores carry across runs is covered.",
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


def reference(q, k, v, causal):
    num_heads, lq, d = q.shape
    out = np.zeros((num_heads, lq, v.shape[2]), dtype=np.float32)
    mask = np.triu(np.ones((lq, k.shape[1]), dtype=bool), k=1)
    for h in range(num_heads):
        s = q[h].astype(np.float32) @ k[h].astype(np.float32).T / sqrt(d)
        if causal:
            s = np.where(mask, -1e9, s)
        p = np.exp(s - s.max(-1, keepdims=True))
        out[h] = (p / p.sum(-1, keepdims=True)) @ v[h].astype(np.float32)
    return out


def main():
    args = parse_args()
    mlir_module, launch = build_module(
        lk=args.lk,
        lq=args.lq,
        dk=args.dk,
        dv=args.dv,
        num_heads=args.num_heads,
        causal=args.causal,
        stages=args.stages,
        handoff=args.handoff,
        causal_skip=args.causal_skip,
    )
    if args.print_module_only:
        print(mlir_module)
        return 0

    backend_kwargs = dict(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        runtime_loop_tiling_sizes=[1, 1],
        output_format=args.output_format,
        instance_name="attention_bf16",
        target_device=launch.target,
    )
    if args.compile_mode == "compile-only":
        XRTBackend(**backend_kwargs).compile(mlir_module)
        print("Compilation complete.")
        return 0

    from ml_dtypes import bfloat16

    rng = np.random.default_rng(42)
    shape_q = (args.num_heads, args.lq, args.dk)
    shape_kv = (args.num_heads, args.lk, args.dk)
    input_q = rng.uniform(0, 4, shape_q).astype(bfloat16)
    input_k = rng.uniform(0, 4, shape_kv).astype(bfloat16)
    input_v = rng.uniform(0, 4, shape_kv).astype(bfloat16)
    expected = reference(input_q, input_k, input_v, args.causal).astype(bfloat16)

    runner = XRTRunner(
        n_warmup_iters=args.num_runs - 1,
        n_perf_iters=1 if args.num_runs > 1 else 0,
        **backend_kwargs,
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
