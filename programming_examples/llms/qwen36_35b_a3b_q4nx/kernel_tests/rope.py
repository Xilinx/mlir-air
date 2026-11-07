# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# q/k RMSNorm and partial RoPE (the leading 64 of each 256-wide head) on one
# core, through fused_decode_gdn's rope kernel, on a layer's real projections.
import argparse
import os
import tempfile

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

import common
import qwen36_35b_a3b_q4nx_weights as W
from air import api as air
from air.api import bf16, i32
from air.backend.xrt import XRTBackend
from reference import rmsnorm, rope

DH, NQ, NKV = W.DH, W.NQ, W.NKV
DQ, DK = NQ * DH, NKV * DH
QKV = DQ + 2 * DK


def build_module():
    rp = air.extern("rope_compute", link_with="rope.o", scalars=[i32])
    X = air.tensor([QKV], bf16)
    RW = air.tensor([3 * DH], bf16)
    Q = air.tensor([DQ], bf16)
    K = air.tensor([DK], bf16)
    V = air.tensor([DK], bf16)
    # the qkv slab, then the rope weights, on the core's one inbound flow
    # k then v share an outbound flow: a core has two
    x_in, q_out, kv_out = (air.channel(n) for n in ("XIn", "QO", "KVO"))
    k_out = v_out = kv_out

    with air.launch(name="rope") as launch:

        @launch.body
        def _():
            x_in.put(X)
            x_in.put(RW)
            q_out.get(Q)
            k_out.get(K)
            v_out.get(V)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="rope", shape=(1,)) as he:

                        @he.body
                        def _(tx):
                            p = he.private()
                            qkv = air.alloc([QKV], bf16, scope=p)
                            rw = air.alloc([3 * DH], bf16, scope=p)
                            q = air.alloc([DQ], bf16, scope=p)
                            k = air.alloc([DK], bf16, scope=p)
                            v = air.alloc([DK], bf16, scope=p)
                            x_in.get(qkv)
                            x_in.get(rw)
                            rp(q, k, v, qkv, rw, 0)
                            q_out.put(q)
                            k_out.put(k)
                            v_out.put(v)

    return launch.build(target="npu2")


def rope_weights(pos, qn, kn):
    """[cos 32 | sin 32 | 0 .. DH | q_norm DH | k_norm DH] at position pos."""
    inv = 1.0 / (W.THETA ** (np.arange(0, W.ROPE_DIM, 2) / W.ROPE_DIM))
    f = pos * inv
    rw = np.zeros(3 * DH, np.float32)
    rw[: W.ROPE_DIM // 2] = np.cos(f)
    rw[W.ROPE_DIM // 2 : W.ROPE_DIM] = np.sin(f)
    rw[DH : 2 * DH], rw[2 * DH :] = qn, kn
    return rw.astype(bfloat16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--pos", type=int, default=1234)
    a = ap.parse_args()
    common.compile_kernel("rope", engine=True)
    w = W.Qwen36Q4nx()
    q_w, _, k_w, v_w, _, qn, kn = w.attn(a.layer)
    x = w.embed([42])[0]
    x = x / np.sqrt((x * x).mean() + W.EPS)
    qkv = np.concatenate([q_w @ x, k_w @ x, v_w @ x]).astype(bfloat16)
    be = XRTBackend(instance_name="rope")
    art = be.compile(build_module())
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = run(
            qkv,
            rope_weights(a.pos, qn, kn),
            *(np.zeros(n, bfloat16) for n in (DQ, DK, DK)),
        )
        be.unload()
    t = lambda v: torch.from_numpy(v.astype(np.float32))  # noqa: E731
    pos = torch.tensor([a.pos])
    q = rope(rmsnorm(t(qkv[:DQ]).view(1, NQ, DH), t(qn)), pos)[0].numpy()
    k = rope(rmsnorm(t(qkv[DQ : DQ + DK]).view(1, NKV, DH), t(kn)), pos)[0].numpy()
    ok = True
    for n, g, wnt in (
        ("q", out[2], q),
        ("k", out[3], k),
        ("v", out[4], qkv[DQ + DK :]),
    ):
        g_, w_ = g.astype(np.float64).ravel(), wnt.astype(np.float64).ravel()
        c = g_ @ w_ / np.linalg.norm(g_) / np.linalg.norm(w_)
        e = np.abs(g_ - w_).max() / np.abs(w_).max()
        good = c > 0.9999
        ok &= good
        print(f"{n}: corr {c:.7f} max rel err {e:.2e} {'ok' if good else 'BAD'}")
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
