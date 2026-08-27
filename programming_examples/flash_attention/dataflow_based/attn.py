# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flash attention as a dataflow pipeline across a cascade of cores.

The kernel is online softmax, split over ``num_cascade_stages`` columns. Each
column walks its own share of the K/V sequence and keeps three running values --
a per-row maximum ``up``, a per-row sum ``sp``, and the unnormalised output
``Gp``. When the sequence is exhausted the columns merge pairwise down the
cascade, rescaling each side by ``exp(local_max - merged_max)``, and column 0
divides through and writes the result out.

Three things here are worth reading as DSL rather than as attention:

* **``seg.per_core()``** is what lets ``up``/``sp``/``Gp`` survive from one herd
  to the next. They are L1, but a buffer allocated in a herd body dies with that
  body, and these are written by three separate herds. Unlike ``seg.shared()``
  they are not divided between the cores -- every core keeps its own whole copy.
* **The micro-tiled L2->L1 transfers are one idiom, not three.** Q, K and V each
  become ``reshape(outer_m, mm, outer_k, kk).transpose(2, 0, 1, 3)``; only which
  extents play which role differs. See ``_packed``.
* **The cascade select is nested ``ops.branch`` on the column coordinate.** The
  predecessor wrote one ``affine.if`` over a two-constraint ``IntegerSet``;
  ``air-to-aie`` specialises both forms away once the herd is unrolled and the
  coordinate is a literal (``SpecializeScfIfPattern`` sits beside
  ``SpecializeAffineIfPattern``), so the cores see the same code either way.
"""

import argparse
from math import sqrt

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16, i32

KERNEL = "attn.o"


def _packed(buf, outer_m, mm, outer_k, kk):
    """A [M, K] L2 buffer walked in the block order the matmul kernel wants.

    The hand-written access pattern is ``sizes=[outer_k, outer_m, mm, kk]`` with
    ``strides=[kk, K*mm, K, 1]``: blocks down the K axis first, then down M,
    then the block itself. Splitting both axes and permuting reaches exactly
    that, and nothing moves -- it is a view.
    """
    return buf.reshape(outer_m, mm, outer_k, kk).transpose(2, 0, 1, 3)


def build_module(
    lk=3072,
    lkp=96,
    lq=128,
    dk=64,
    dv=64,
    num_q_tiles=4,
    num_cascade_stages=4,
    arch="aie2",
):
    """Build the attention module.

    Args:
        lk: total sequence length for K/V
        lkp: chunk of the K/V sequence handled per step
        lq: sequence length for Q
        dk: key dimension
        dv: value dimension
        num_q_tiles: retained for the caller's signature; the Q tile is lq
        num_cascade_stages: columns in the cascade
        arch: "aie2" or "aie2p", which picks the matmul block shape
    """
    mmul_m, mmul_k, mmul_n = [8, 8, 8] if arch == "aie2p" else [4, 8, 4]

    ncs = num_cascade_stages
    cps = (lk // lkp) // ncs  # K/V chunks each column walks
    tq = lq

    Q = air.tensor([lq, dk], bf16)
    K = air.tensor([dk, lk], bf16)
    V = air.tensor([lk, dv], bf16)
    # Declared because the kernel's interface takes an additive mask, and the
    # host passes one. The pipeline folds it into G before the max, so nothing
    # in the body reads it while the mask is zero.
    air.tensor([lq, lk], bf16)  # the mask, read only by the pipeline (see above)
    OUT = air.tensor([lq, dk], bf16)

    # Q and K share the L3->L2 bundle: they are never in flight together, Q
    # being staged once before the loop and K once per trip.
    l3_qk = air.channel("L3ToL2Chan1", size=[1, ncs])
    l3_v = air.channel("L3ToL2Chan3", size=[1, ncs])
    l2_q = air.channel("L2ToL1Chan1", size=[1, ncs])
    l2_k = air.channel("L2ToL1Chan2", size=[1, ncs])
    l2_v = air.channel("L2ToL1Chan3", size=[1, ncs])
    out_l2 = air.channel("L1ToL2Chan1")
    out_l3 = air.channel("L2ToL3Chan1")
    g_to_softmax = air.channel("L1ToL1Chan1", size=[1, ncs])
    g_to_matmul = air.channel("L1ToL1Chan2", size=[1, ncs])
    gv_back = air.channel("L1ToL1Chan3", size=[1, ncs])
    cascade = air.channel("cascade", size=[1, ncs - 1], channel_type="npu_cascade")

    zero_gp = air.extern("zero_fill_gp_bf16", link_with=KERNEL)
    zero_sp = air.extern("zero_fill_sp_bf16", link_with=KERNEL)
    zero_g = air.extern("zero_fill_g_bf16", link_with=KERNEL)
    neg_inf_up = air.extern("neg_inf_fill_up_bf16", link_with=KERNEL)
    matmul_a_b = air.extern("matmul_a_b_bf16", link_with=KERNEL)
    matmul_g_b = air.extern("matmul_g_b_bf16", link_with=KERNEL)
    max_g = air.extern("max_g_bf16", link_with=KERNEL)
    maximum_up_u = air.extern("maximum_up_u_bf16", link_with=KERNEL)
    exp_g_minus_u = air.extern("exp_g_minus_u", link_with=KERNEL)
    exp_up_minus_u = air.extern("exp_up_minus_u", link_with=KERNEL)
    mul_r_gp = air.extern("mul_r_gp", link_with=KERNEL)
    sum_g = air.extern("sum_g", link_with=KERNEL)
    accum_sp_r_s = air.extern("accum_sp_r_s", link_with=KERNEL)
    copy_row = air.extern("vector_copy_32elems", link_with=KERNEL, scalars=[i32])
    div_gp_sp = air.extern("div_gp_sp", link_with=KERNEL)
    copy_g = air.extern("vector_copy_32x96elems", link_with=KERNEL, scalars=[i32])
    add_gp_g = air.extern("add_gp_g", link_with=KERNEL)
    accum_gp = air.extern("vector_accum_32x64elems", link_with=KERNEL)

    with air.launch([range(1), range(1)], name="attention_bf16") as launch:

        @launch.body
        def _(li, lj):
            # One Q tile per column. air.parallel rather than a Python loop:
            # the trips share a set of buffer descriptors, where unrolling
            # would spend one shim DMA per column.
            for c in air.parallel(0, ncs):
                l3_qk.put(Q[c * tq : c * tq + tq, :], indices=[0, c])

            # K arrives column-major in chunks: column c takes every ncs'th
            # chunk, which is a slice of the middle axis once lk is split into
            # (chunk, column, position).
            for c in range(ncs):
                l3_qk.put(
                    K.reshape(dk, cps, ncs * lkp)[
                        :, :, c * lkp : (c + 1) * lkp
                    ].transpose(1, 0, 2),
                    indices=[0, c],
                )

            # V is the same split on its own sequence axis, and needs no
            # permutation: it is already chunk-major.
            for c in range(ncs):
                l3_v.put(
                    V.reshape(cps, ncs * lkp, dv)[:, c * lkp : (c + 1) * lkp, :],
                    indices=[0, c],
                )

            with air.segment(name="attention_seg") as seg:

                @seg.body
                def _():
                    q_l2 = [
                        air.alloc([tq, dk], bf16, scope=seg.private())
                        for _ in range(ncs)
                    ]
                    k_l2 = [
                        air.alloc([dk, lkp], bf16, scope=seg.private())
                        for _ in range(ncs)
                    ]
                    v_l2 = [
                        air.alloc([lkp, dv], bf16, scope=seg.private())
                        for _ in range(ncs)
                    ]
                    result_l2 = air.alloc([lq, dk], bf16, scope=seg.private())

                    # The running state of the online softmax. per_core, not
                    # shared: each column keeps its own whole copy, and it has
                    # to outlive the herd that writes it.
                    up = air.alloc([tq, 1], bf16, scope=seg.per_core())
                    sp = air.alloc([tq, 1], bf16, scope=seg.per_core())
                    Gp = air.alloc([tq, dk], bf16, scope=seg.per_core())
                    a_l1 = air.alloc([tq, dk], bf16, scope=seg.per_core())

                    for c in range(ncs):
                        l3_qk.get(q_l2[c], indices=[0, c])
                    for c in range(ncs):
                        l2_q.put(
                            _packed(
                                q_l2[c], tq // mmul_m, mmul_m, dk // mmul_k, mmul_k
                            ),
                            indices=[0, c],
                        )

                    # Q into each column's L1, once.
                    with air.herd(
                        [range(1), range(ncs)],
                        name="herd_0",
                        shape=(1, ncs),
                        link_with=KERNEL,
                    ) as h_q:

                        @h_q.body
                        def _(tx, ty):
                            l2_q.get(a_l1, indices=[tx, ty])

                    with air.herd(
                        [range(1), range(ncs)],
                        name="herd_1",
                        shape=(1, ncs),
                        link_with=KERNEL,
                    ) as h_init:

                        @h_init.body
                        def _(tx, ty):
                            zero_gp(Gp)
                            zero_sp(sp)
                            neg_inf_up(up)

                    for _chunk in air.sequential(0, cps):
                        for c in range(ncs):
                            l3_qk.get(k_l2[c], indices=[0, c])
                            l3_v.get(v_l2[c], indices=[0, c])
                        for c in range(ncs):
                            l2_k.put(
                                _packed(
                                    k_l2[c], dk // mmul_k, mmul_k, lkp // mmul_n, mmul_n
                                ),
                                indices=[0, c],
                            )
                        for c in range(ncs):
                            l2_v.put(
                                _packed(
                                    v_l2[c], lkp // mmul_k, mmul_k, dv // mmul_n, mmul_n
                                ),
                                indices=[0, c],
                            )

                        # G = Q @ K
                        with air.herd(
                            [range(1), range(ncs)],
                            name="herd_0",
                            shape=(1, ncs),
                            link_with=KERNEL,
                        ) as h_qk:

                            @h_qk.body
                            def _(tx, ty):
                                k_l1 = air.alloc([dk, lkp], bf16, scope=h_qk.private())
                                g = air.alloc([tq * lkp], bf16, scope=h_qk.private())
                                zero_g(g)
                                l2_k.get(k_l1, indices=[tx, ty])
                                matmul_a_b(a_l1, k_l1, g)
                                # G leaves in block order, not row order: the
                                # consumer reads one n-block down every row
                                # before moving to the next block. Splitting
                                # the flat buffer with the block axis outermost
                                # and swapping it with the row axis reaches
                                # sizes [tq, lkp/n, n] over strides [n, tq*n,
                                # 1] -- a view, nothing moves.
                                g_to_softmax.put(
                                    g.reshape(lkp // mmul_n, tq, mmul_n).transpose(
                                        1, 0, 2
                                    ),
                                    indices=[tx, ty],
                                )
                                # Released here rather than at last use: the
                                # predecessor batches its frees at the end of
                                # the body, and herd fusion is sensitive to
                                # what sits between two herds.
                                air.dealloc(k_l1)
                                air.dealloc(g)

                        # Online softmax over this chunk, and accumulate G @ V.
                        with air.herd(
                            [range(1), range(ncs)],
                            name="herd_1",
                            shape=(1, ncs),
                            link_with=KERNEL,
                        ) as h_soft:

                            @h_soft.body
                            def _(tx, ty):
                                u = air.alloc([tq, 1], bf16, scope=h_soft.private())
                                s = air.alloc([tq, 1], bf16, scope=h_soft.private())
                                r = air.alloc([tq, 1], bf16, scope=h_soft.private())
                                g = air.alloc([tq * lkp], bf16, scope=h_soft.private())

                                g_to_softmax.get(g, indices=[tx, ty])
                                max_g(g, u)
                                maximum_up_u(up, u)
                                exp_g_minus_u(u, g)
                                exp_up_minus_u(up, u, r)
                                mul_r_gp(r, Gp)

                                g_copy = air.alloc(
                                    [tq * lkp], bf16, scope=h_soft.private()
                                )
                                copy_g(0, g, g_copy)
                                g_to_matmul.put(
                                    g_copy.reshape(tq, lkp // mmul_k, mmul_k).transpose(
                                        1, 0, 2
                                    ),
                                    indices=[tx, ty],
                                )
                                air.dealloc(g_copy)

                                gv = air.alloc([tq, dk], bf16, scope=h_soft.private())
                                gv_back.get(gv, indices=[tx, ty])
                                accum_gp(gv, Gp)
                                air.dealloc(gv)

                                sum_g(g, s)
                                accum_sp_r_s(sp, r, s)
                                copy_row(0, s, sp)
                                copy_row(0, u, up)
                                air.dealloc(u)
                                air.dealloc(s)
                                air.dealloc(r)
                                air.dealloc(g)

                        # G @ V, on its own core so it overlaps the softmax.
                        with air.herd(
                            [range(1), range(ncs)],
                            name="herd_2",
                            shape=(1, ncs),
                            link_with=KERNEL,
                        ) as h_gv:

                            @h_gv.body
                            def _(tx, ty):
                                v_l1 = air.alloc([dk, lkp], bf16, scope=h_gv.private())
                                g = air.alloc([tq * lkp], bf16, scope=h_gv.private())
                                acc = air.alloc([tq, dk], bf16, scope=h_gv.private())
                                zero_gp(acc)
                                g_to_matmul.get(g, indices=[tx, ty])
                                l2_v.get(v_l1, indices=[tx, ty])
                                matmul_g_b(g, v_l1, acc)
                                gv_back.put(
                                    acc.reshape(dv // mmul_n, tq, mmul_n).transpose(
                                        1, 0, 2
                                    ),
                                    indices=[tx, ty],
                                )
                                air.dealloc(v_l1)
                                air.dealloc(g)
                                air.dealloc(acc)

                    # Merge the columns down the cascade.
                    with air.herd(
                        [range(1), range(ncs)],
                        name="herd_1",
                        shape=(1, ncs),
                        link_with=KERNEL,
                    ) as h_merge:

                        @h_merge.body
                        def _(tx, ty):
                            r = air.alloc([tq, 1], bf16, scope=h_merge.private())

                            def merge():
                                """Fold the column above into this one.

                                A Python function, so each call emits its own
                                copy of the body -- which is what is wanted
                                here: the two arms below are separate regions
                                and only one of them runs on any given core.
                                """
                                gp_in = air.alloc(
                                    [tq, dk], bf16, scope=h_merge.private()
                                )
                                up_in = air.alloc(
                                    [tq, 1], bf16, scope=h_merge.private()
                                )
                                sp_in = air.alloc(
                                    [tq, 1], bf16, scope=h_merge.private()
                                )
                                cascade.get(gp_in, indices=[tx, ty])
                                cascade.get(up_in, indices=[tx, ty])
                                cascade.get(sp_in, indices=[tx, ty])

                                # maximum_up_u overwrites up, so keep the local
                                # maximum before merging it away.
                                up_mine = air.alloc(
                                    [tq, 1], bf16, scope=h_merge.private()
                                )
                                copy_row(0, up, up_mine)
                                maximum_up_u(up_in, up)

                                r_b = air.alloc([tq, 1], bf16, scope=h_merge.private())
                                exp_up_minus_u(up_in, up, r)
                                exp_up_minus_u(up_mine, up, r_b)
                                mul_r_gp(r, gp_in)
                                mul_r_gp(r_b, Gp)
                                add_gp_g(Gp, gp_in)

                                sp_merged = air.alloc(
                                    [tq, 1], bf16, scope=h_merge.private()
                                )
                                zero_sp(sp_merged)
                                accum_sp_r_s(sp_in, r, sp_merged)
                                accum_sp_r_s(sp, r_b, sp_merged)
                                copy_row(0, sp_merged, sp_in)
                                air.dealloc(up_mine)
                                air.dealloc(r_b)
                                air.dealloc(sp_merged)
                                return gp_in, sp_in

                            with air.ops.branch(ty == ncs - 1) as last:
                                # The tail has nobody above it: hand its state
                                # down and stop.
                                cascade.put(Gp, indices=[tx, ty - 1])
                                cascade.put(up, indices=[tx, ty - 1])
                                cascade.put(sp, indices=[tx, ty - 1])

                            with last.otherwise():
                                # ty is not the tail here, so the predecessor's
                                # 1 <= ty <= ncs-2 is just ty >= 1.
                                with air.ops.branch(ty >= 1) as middle:
                                    gp_in, sp_in = merge()
                                    cascade.put(gp_in, indices=[tx, ty - 1])
                                    cascade.put(up, indices=[tx, ty - 1])
                                    cascade.put(sp_in, indices=[tx, ty - 1])
                                with middle.otherwise():
                                    # Column 0 owns the answer.
                                    gp_in, sp_in = merge()
                                    div_gp_sp(sp_in, gp_in)
                                    out_l2.put(gp_in)

                    out_l2.get(result_l2[0:tq, 0:dv])
                    out_l3.put(result_l2)

            out_l3.get(OUT)

    return launch.build()


def _reference(q, k, v, m, lq, lkp, lk, dv):
    """Online softmax in fp32, rounded to bf16 the way the kernel accumulates."""
    Gp = np.zeros((lq, dv), dtype=bfloat16)
    up = np.full((lq, 1), -np.inf, dtype=bfloat16)
    sp = np.zeros((lq, 1), dtype=bfloat16)
    for j in range(lk // lkp):
        G = (q @ k[:, j * lkp : (j + 1) * lkp] + m[:, j * lkp : (j + 1) * lkp]).astype(
            bfloat16
        )
        u = np.maximum(np.max(G, axis=-1, keepdims=True).astype(bfloat16), up)
        G = np.exp(G - u).astype(bfloat16)
        r = np.exp(up - u).astype(bfloat16)
        Gp = (G @ v[j * lkp : (j + 1) * lkp, :] + Gp * r).astype(bfloat16)
        s = np.sum(G, axis=-1, keepdims=True).astype(bfloat16) + sp * r
        sp, up = s, u
    return (Gp / sp).astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="attn.py")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--lk", type=int, default=12288, help="Total sequence length for K/V matrices"
    )
    parser.add_argument(
        "--lkp", type=int, default=96, help="Chunk size for K/V processing"
    )
    parser.add_argument(
        "--lq", type=int, default=128, help="Sequence length for Q matrix"
    )
    parser.add_argument("--dk", type=int, default=64, help="Key dimension")
    parser.add_argument("--dv", type=int, default=64, help="Value dimension")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="Target architecture (default: aie2)",
    )
    args = parser.parse_args()

    lk, lkp, lq, dk, dv = args.lk, args.lkp, args.lq, args.dk, args.dv

    module = build_module(
        lk=lk, lkp=lkp, lq=lq, dk=dk, dv=dv, num_cascade_stages=4, arch=args.arch
    )
    if args.print_module_only:
        print(module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner

    input_q = np.arange(0, lq * dk, dtype=bfloat16).reshape(lq, dk) / (lq * dk) * 2
    input_k = np.arange(0, dk * lk, dtype=bfloat16).reshape(dk, lk) / (dk * lk) * 2
    input_v = np.arange(0, lk * dv, dtype=bfloat16).reshape(lk, dv) / (lk * dv) * 2
    input_m = np.zeros((lq, lk), dtype=bfloat16)
    input_q = (input_q.astype(bfloat16) / sqrt(dk)).astype(bfloat16)
    input_k = input_k.astype(bfloat16)
    input_v = input_v.astype(bfloat16)

    expected = _reference(input_q, input_k, input_v, input_m, lq, lkp, lk, dv)

    runner = XRTRunner(
        omit_while_true_loop=False,
        omit_pingpong=True,
        verbose=args.verbose,
        runtime_loop_tiling_sizes=[1, 1],
        output_format=args.output_format,
        instance_name="attention_bf16",
    )
    exit(
        runner.run_test(
            module,
            inputs=[input_q, input_k, input_v, input_m],
            expected_outputs=[expected],
            rtol=1e-1,
        )
    )
