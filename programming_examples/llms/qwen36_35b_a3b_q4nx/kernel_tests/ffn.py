# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The residual stream and the MoE combine, the projections given, in two
# single-core runs (one core's DMA holds 16 buffer descriptors):
#   experts  each routed expert's GLU, then the score-weighted sum of their
#            down projections
#   stream   input norm, residual add of the mixer output, post norm, the
#            gated shared expert plus the routed sum, final residual add
import os
import tempfile

import filelock
import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16

import common
import qwen36_35b_a3b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt import XRTBackend
from reference import rmsnorm

D, I, K = W.D, W.I_EXP, W.TOP_K


def ext(name, scalars):
    return air.extern(name, link_with="ffn.o", scalars=scalars)


def build_experts():
    glu, sacc, zero = (
        ext("glu_silu", [i32]),
        ext("score_acc", [i32, i32]),
        ext("acc_zero", [i32]),
    )
    GU = air.tensor([K * 2 * I], bf16)
    DN = air.tensor([K * D], bf16)
    SC = air.tensor([16], f32)
    H = air.tensor([K * I], bf16)
    ACC = air.tensor([D], f32)
    b_in, s_in, h_out, a_out = (air.channel(n) for n in ("BIn", "SIn", "HOut", "AOut"))

    with air.launch(name="ffn_experts") as launch:

        @launch.body
        def _():
            b_in.put(GU)
            b_in.put(DN)
            s_in.put(SC)
            h_out.get(H)
            a_out.get(ACC)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="ffn", shape=(1,)) as he:

                        @he.body
                        def _(tx):
                            p = he.private()
                            g, u, h = (air.alloc([I], bf16, scope=p) for _ in range(3))
                            dn = air.alloc([D], bf16, scope=p)
                            acc = air.alloc([D], f32, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            s_in.get(sc)
                            for e in air.sequential(0, K):
                                b_in.get(g)
                                b_in.get(u)
                                glu(h, g, u, I)
                                h_out.put(h)
                            zero(acc, D)
                            for e in air.sequential(0, K):
                                b_in.get(dn)
                                sacc(acc, dn, sc, e, D)
                            a_out.put(acc)

    return launch.build(target="npu2")


def build_stream():
    norm, res = ext("rms_norm", [i32]), ext("residual_add", [i32])
    sg, out = ext("shared_gate", [i32]), ext("moe_out", [i32])
    IN = air.tensor([6 * D], bf16)  # x, m, n1, n2, w_sg, s
    ACC = air.tensor([D], f32)
    OUT = air.tensor([3 * D], bf16)
    b_in, a_in, y_out = (air.channel(n) for n in ("BIn", "AIn", "YOut"))

    with air.launch(name="ffn_stream") as launch:

        @launch.body
        def _():
            b_in.put(IN)
            a_in.put(ACC)
            y_out.get(OUT)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="ffn", shape=(1,)) as he:

                        @he.body
                        def _(tx):
                            p = he.private()
                            xb, m, n1, n2, w, s, y = (
                                air.alloc([D], bf16, scope=p) for _ in range(7)
                            )
                            acc = air.alloc([D], f32, scope=p)
                            for buf in (xb, m, n1, n2, w, s):
                                b_in.get(buf)
                            a_in.get(acc)
                            norm(y, xb, n1, D)
                            y_out.put(y)
                            res(m, xb, m, D)
                            norm(y, xb, n2, D)
                            y_out.put(y)
                            sg(y, w, s, D)
                            out(m, s, acc, D)
                            res(y, xb, m, D)
                            y_out.put(y)

    return launch.build(target="npu2")


def check(name, got, want, thr=0.9999):
    g_, w_ = np.asarray(got, np.float64).ravel(), np.asarray(want, np.float64).ravel()
    c = g_ @ w_ / np.linalg.norm(g_) / np.linalg.norm(w_)
    e = np.abs(g_ - w_).max() / np.abs(w_).max()
    print(f"{name:10s} corr {c:.7f} max rel err {e:.2e} {'ok' if c > thr else 'BAD'}")
    return c > thr


def run(module, name, *args):
    be = XRTBackend(instance_name=name)
    art = be.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        out = be.load(art)(*args)
        be.unload()
    return out


if __name__ == "__main__":
    common.compile_kernel("ffn")
    w = W.Qwen36Q4nx()
    rng = np.random.default_rng(0)
    bf = lambda a: np.asarray(a).astype(bfloat16)  # noqa: E731
    t = lambda a: torch.from_numpy(bf(a).astype(np.float64))  # noqa: E731
    r = lambda v: v.to(torch.bfloat16).double()  # noqa: E731

    gu = 2 * rng.standard_normal((K, 2, I))
    dn = 0.05 * rng.standard_normal((K, D))
    sc = np.zeros(16, np.float32)
    sc[:K] = np.sort(rng.dirichlet(np.ones(K)))[::-1]
    out = run(
        build_experts(),
        "ffn_experts",
        bf(gu).ravel(),
        bf(dn).ravel(),
        sc,
        np.zeros(K * I, bfloat16),
        np.zeros(D, np.float32),
    )
    routed = (torch.from_numpy(sc[:K].astype(np.float64))[:, None] * t(dn)).sum(0)
    ok = check("glu", out[3], (F.silu(t(gu)[:, 0]) * t(gu)[:, 1]).numpy())
    ok &= check("routed", out[4], routed.numpy(), 0.99999)

    n1, n2 = w.norms(0)
    sgate = w.router(0)[1]
    x = w.embed([7])[0]
    m = 0.5 * rng.standard_normal(D)
    s = 0.05 * rng.standard_normal(D)
    inp = np.concatenate([bf(v).ravel() for v in (x, m, n1, n2, sgate, s)])
    out = run(build_stream(), "ffn_stream", inp, out[4], np.zeros(3 * D, bfloat16))
    got = out[2].reshape(3, D)
    # FLM's roundings: the residual stream, the gate's dot product, the gated
    # shared output and the routed sum are bf16
    x2 = r(t(x) + t(m))
    h2 = rmsnorm(x2, t(n2))
    shared = r(torch.sigmoid(r(r(h2) @ t(sgate))) * t(s))
    final = r(x2 + r(shared + r(routed)))
    ok &= check("input norm", got[0], rmsnorm(t(x), t(n1)).numpy())
    ok &= check("post norm", got[1], h2.numpy())
    ok &= check("final", got[2], final.numpy())
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
