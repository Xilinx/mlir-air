# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Attention as three separate operators on NPU2, intermediates in DDR.

::

    qk:  S = Q K^T, masked with --causal   (Q, K) -> S
    sm:  P = softmax(S), row by row        (S)    -> P
    pv:  O = P V                           (P, V) -> O

Each operator is its own binary and runs on the whole array; every core owns
one 64-row query block per launch iteration. S and P go to DDR as the
microkernels' native 64x64 tiles, laid out ``[head, key block, query block,
64 * 64]`` so that one column's tiles for one key block are contiguous. The
softmax operator reads S three times, for the row maxima, the row sums and P,
so that the sums are taken over the same exponentials P is built from. The
math is the ``kernel_fusion_based/attn_npu2.cc`` microkernels.

Running the script builds the three operators into ``qk/``, ``sm/`` and
``pv/`` under the working directory, then runs them one after another and
checks O against NumPy.
"""

import argparse
import os
import shutil
import tempfile
from math import sqrt

import filelock
import numpy as np

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend

KERNEL = "attn_npu2.o"
OPS = ("qk", "sm", "pv")
M = 8
B = 64  # query block == key block == head dim
T = B * B


def geometry(lq):
    """(query blocks, columns, rows): up to an 8x4 array of cores."""
    nqb = lq // B
    ncols = min(8, nqb)
    nrows = min(4, nqb // ncols)
    assert nqb % (ncols * nrows) == 0, f"lq must be a multiple of {B * ncols}"
    return nqb, ncols, nrows


def blocked(buf):
    return buf.reshape(B // M, M, B // M, M).transpose(2, 0, 1, 3)


def unblocked(buf):
    return buf.reshape(B // M, B // M, M, M).transpose(1, 2, 0, 3)


def build_launch(op, lk=512, lq=512, dk=64, dv=64, num_heads=2, causal=False):
    assert op in OPS
    assert lq == lk, "one query block per key block"
    assert dk == B and dv == B, f"head dims must be {B}"
    nqb, NC, NR = geometry(lq)
    nkb = nqb
    assert nkb % 2 == 0, "key blocks are double-buffered"
    per_iter = NC * NR
    q_iters = nqb // per_iter
    L, H = lq, num_heads

    zero_fill_g = air.extern("zero_fill_g_bf16", link_with=KERNEL)
    zero_fill_gp = air.extern("zero_fill_gp_bf16", link_with=KERNEL)
    zero_fill_sp = air.extern("zero_fill_sp_bf16", link_with=KERNEL)
    neg_inf_fill_up = air.extern("neg_inf_fill_up_bf16", link_with=KERNEL)
    matmul_a_b = air.extern("matmul_a_b_bf16", link_with=KERNEL)
    matmul_g_b = air.extern("matmul_g_b_bf16", link_with=KERNEL)
    fused_softmax = air.extern("fused_softmax", link_with=KERNEL)
    max_g = air.extern("max_g_bf16", link_with=KERNEL)
    maximum_up_u = air.extern("maximum_up_u_bf16", link_with=KERNEL)
    accum_sp_r_s = air.extern("accum_sp_r_s", link_with=KERNEL)
    vector_copy = air.extern("vector_copy_32elems", link_with=KERNEL, scalars=[i32])
    exp_g_minus_u = air.extern("exp_g_minus_u", link_with=KERNEL)
    div_gp_sp = air.extern("div_gp_sp", link_with=KERNEL)
    copy_tile = air.extern("copy_tile", link_with=KERNEL)
    apply_mask = air.extern("apply_causal_mask", link_with=KERNEL, scalars=[i32, i32])

    # Tile streams between DDR and one column's cores, through that column's
    # MemTile: tin = DDR->L2, t2l1 = L2->core, t2l2 = core->L2, tout = L2->DDR.
    tin = air.channel("TIn", size=[NC])
    t2l1 = air.channel("T2L1", size=[NC, NR])
    t2l2 = air.channel("T2L2", size=[NC, NR])
    tout = air.channel("TOut", size=[NC])
    # The operand every core needs (K for qk, V for pv), broadcast.
    bin_ = air.channel("BIn")
    b2l1 = air.channel("B2L1", size=[1, 1], broadcast_shape=[NC, NR])

    s_shape = [H, nkb, nqb, T]
    if op == "qk":
        A = air.tensor([H, L, B], bf16)  # Q
        Bt = air.tensor([H, L, B], bf16)  # K
        R = air.tensor(s_shape, bf16)  # S
    elif op == "sm":
        A = air.tensor(s_shape, bf16)  # S
        R = air.tensor(s_shape, bf16)  # P
    else:
        A = air.tensor(s_shape, bf16)  # P
        Bt = air.tensor([H, L, B], bf16)  # V
        R = air.tensor([H, L, B], bf16)  # O

    def col_tiles(t, h, q0):
        """[nkb, NR, T] view of one column's tiles of an S-shaped tensor."""
        return t[h, :, q0 : q0 + NR, :]

    with air.launch([range(q_iters), range(H)], name="attention_bf16") as launch:

        @launch.body
        def _(lx, ly):
            for c in range(NC):
                q0 = lx * per_iter + c * NR
                if op == "qk":
                    tin.put(A[ly, q0 * B : (q0 + NR) * B, :], indices=[c])
                elif op == "sm":
                    tin.put(col_tiles(A, ly, q0), indices=[c])  # maxima
                    tin.put(col_tiles(A, ly, q0), indices=[c])  # sums
                    tin.put(col_tiles(A, ly, q0), indices=[c])  # P
                else:
                    tin.put(col_tiles(A, ly, q0), indices=[c])
            if op != "sm":
                bin_.put(Bt.reshape(H * nkb, B, B)[ly * nkb : (ly + 1) * nkb, :, :])

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    b_l2 = air.alloc([B, B], bf16, scope=seg.private(), column=0)
                    in_l2 = [
                        [
                            air.alloc([B, B], bf16, scope=seg.private(), column=c)
                            for _ in range(NR)
                        ]
                        for c in range(NC)
                    ]
                    out_l2 = [
                        [
                            air.alloc([B, B], bf16, scope=seg.private(), column=c)
                            for _ in range(NR)
                        ]
                        for c in range(NC)
                    ]

                    n_in = {"qk": 1, "sm": 3 * nkb, "pv": nkb}[op]
                    n_out = {"qk": nkb, "sm": nkb, "pv": 1}[op]
                    xf = blocked if op == "qk" else (lambda b: b)
                    for c in range(NC):
                        for _ in air.sequential(0, n_in):
                            for r in range(NR):
                                tin.get(in_l2[c][r], indices=[c])
                                t2l1.put(xf(in_l2[c][r]), indices=[c, r])
                        for _ in air.sequential(0, n_out):
                            for r in range(NR):
                                t2l2.get(out_l2[c][r], indices=[c, r])
                                tout.put(out_l2[c][r], indices=[c])
                    if op != "sm":
                        for _ in air.sequential(0, nkb):
                            bin_.get(b_l2)
                            b2l1.put(blocked(b_l2), indices=[0, 0])

                    with air.herd(
                        [range(NC), range(NR)],
                        name="h",
                        shape=(NC, NR),
                        at=(0, 2),
                        link_with=KERNEL,
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            ib = [
                                air.alloc([T], bf16, scope=h.private())
                                for _ in range(2)
                            ]
                            if op == "qk":
                                q_l1 = air.alloc([B, B], bf16, scope=h.private())
                                t2l1.get(q_l1, indices=[tx, ty])
                                kb = [
                                    air.alloc([B, B], bf16, scope=h.private())
                                    for _ in range(2)
                                ]
                                if causal:
                                    # This core's query block, counted in L1
                                    # and wrapped so a rerun starts at block 0.
                                    # Heads cycle fastest in the launch grid.
                                    ctr = air.alloc([3], i32, scope=h.private())
                                    first = ops.equal(ctr[1:2], 0)
                                    ctr[0:1] = ops.select(first, 0, ctr[0:1])
                                    ctr[2:3] = ops.select(first, 0, ctr[2:3])
                                    ctr[1:2] = ops.select(first, 1, ctr[1:2])
                                    q_block = ctr[0] * per_iter + tx * NR + ty
                                for j in air.sequential(0, nkb // 2):
                                    for b in range(2):
                                        b2l1.get(kb[b], indices=[tx, ty])
                                        zero_fill_g(ib[b])
                                        matmul_a_b(q_l1, kb[b], ib[b])
                                        if causal:
                                            apply_mask(ib[b], q_block, j * 2 + b)
                                        t2l2.put(ib[b], indices=[tx, ty])
                                if causal:
                                    q_next = ctr[0:1] + 1
                                    q_next = ops.select(q_next >= q_iters, 0, q_next)
                                    head_next = ctr[2:3] + 1
                                    wrapped = head_next >= H
                                    ctr[0:1] = ops.select(wrapped, q_next, ctr[0:1])
                                    ctr[2:3] = ops.select(wrapped, 0, head_next)
                            elif op == "sm":
                                up = air.alloc([B, 1], bf16, scope=h.private())
                                sp = air.alloc([B, 1], bf16, scope=h.private())
                                s_t = air.alloc([B, 1], bf16, scope=h.private())
                                r_t = air.alloc([B, 1], bf16, scope=h.private())
                                ob = [
                                    air.alloc([T], bf16, scope=h.private())
                                    for _ in range(2)
                                ]
                                neg_inf_fill_up(up)
                                zero_fill_sp(sp)
                                for _ in air.sequential(0, nkb // 2):
                                    for b in range(2):
                                        t2l1.get(ib[b], indices=[tx, ty])
                                        max_g(ib[b], r_t)
                                        maximum_up_u(up, r_t)
                                        vector_copy(0, r_t, up)
                                # up already holds the final maxima, so the
                                # rescale is 1 and this pass only sums.
                                for _ in air.sequential(0, nkb // 2):
                                    for b in range(2):
                                        t2l1.get(ib[b], indices=[tx, ty])
                                        fused_softmax(ib[b], up, s_t, r_t)
                                        accum_sp_r_s(sp, r_t, s_t)
                                        vector_copy(0, s_t, sp)
                                for _ in air.sequential(0, nkb // 2):
                                    for b in range(2):
                                        t2l1.get(ib[b], indices=[tx, ty])
                                        exp_g_minus_u(up, ib[b])
                                        div_gp_sp(sp, ib[b])
                                        copy_tile(ib[b], ob[b])
                                        t2l2.put(ob[b], indices=[tx, ty])
                            else:
                                gp = air.alloc([B, B], bf16, scope=h.private())
                                vb = [
                                    air.alloc([B, B], bf16, scope=h.private())
                                    for _ in range(2)
                                ]
                                zero_fill_gp(gp)
                                for _ in air.sequential(0, nkb // 2):
                                    for b in range(2):
                                        t2l1.get(ib[b], indices=[tx, ty])
                                        b2l1.get(vb[b], indices=[tx, ty])
                                        matmul_g_b(ib[b], vb[b], gp)
                                t2l2.put(unblocked(gp), indices=[tx, ty])

            for c in range(NC):
                q0 = lx * per_iter + c * NR
                if op == "pv":
                    tout.get(R[ly, q0 * B : (q0 + NR) * B, :], indices=[c])
                else:
                    tout.get(col_tiles(R, ly, q0), indices=[c])

    return launch


def build_module(op, **kwargs):
    launch = build_launch(op, **kwargs)
    return launch.build(target="npu2"), launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="attn_npu2.py",
        description="Attention as three operators with intermediates in DDR",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--op",
        choices=OPS,
        default="qk",
        help="Operator to print with -p",
    )
    parser.add_argument("--lk", type=int, default=512)
    parser.add_argument("--lq", type=int, default=512)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Run the three operators this many times and check the last "
        "output, so state the cores carry across runs is covered.",
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
    shape_kwargs = dict(
        lk=args.lk,
        lq=args.lq,
        dk=args.dk,
        dv=args.dv,
        num_heads=args.num_heads,
        causal=args.causal,
    )
    if args.print_module_only:
        print(build_module(args.op, **shape_kwargs)[0])
        return 0

    root = os.getcwd()
    backends, artifacts = {}, {}
    for op in OPS:
        d = os.path.join(root, op)
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(root, KERNEL), d)
        os.chdir(d)
        mlir_module, launch = build_module(op, **shape_kwargs)
        backends[op] = XRTBackend(
            omit_while_true_loop=False,
            omit_pingpong="all",
            verbose=args.verbose,
            runtime_loop_tiling_sizes=[1, 1],
            output_format=args.output_format,
            instance_name="attention_bf16",
            target_device=launch.target,
        )
        artifacts[op] = backends[op].compile(mlir_module)
        os.chdir(root)
    if args.compile_mode == "compile-only":
        print("Compilation complete.")
        return 0

    from ml_dtypes import bfloat16

    rng = np.random.default_rng(42)
    H, L = args.num_heads, args.lq
    nqb = L // B
    input_q = rng.uniform(0, 4, (H, L, args.dk)).astype(bfloat16)
    input_k = rng.uniform(0, 4, (H, L, args.dk)).astype(bfloat16)
    input_v = rng.uniform(0, 4, (H, L, args.dv)).astype(bfloat16)
    s_shape = (H, nqb, nqb, T)

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fns = {}
        for op in OPS:
            # The artifact paths are relative to the directory it was built in.
            os.chdir(os.path.join(root, op))
            fns[op] = backends[op].load(artifacts[op])
        os.chdir(root)
        for _ in range(args.num_runs):
            S = fns["qk"](input_q, input_k, np.zeros(s_shape, bfloat16))[-1]
            P = fns["sm"](S.reshape(s_shape), np.zeros(s_shape, bfloat16))[-1]
            O = fns["pv"](P.reshape(s_shape), input_v, np.zeros((H, L, B), bfloat16))
        for op in OPS:
            backends[op].unload()

    actual = np.asarray(O[-1]).reshape(H, L, args.dv).astype(np.float32)
    expected = reference(input_q, input_k, input_v, args.causal)
    corr = np.corrcoef(expected.ravel(), actual.ravel())[0, 1]
    mismatch = 100 * np.mean(~np.isclose(actual, expected, atol=0.15, rtol=0.04))
    print(f"Output correlation: {corr:.6f} (threshold: 0.99)")
    print(f"Mismatched elements: {mismatch:.3f}% (threshold: 0.5%)")
    if corr >= 0.99 and mismatch <= 0.5:
        print("PASS!")
        return 0
    print("FAIL!")
    return 1


if __name__ == "__main__":
    exit(main())
