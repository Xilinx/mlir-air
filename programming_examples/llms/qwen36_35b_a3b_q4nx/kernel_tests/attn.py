# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Decode attention for one KV head and its 8 query heads (one CU, 1x8x1, head
# dim 256), on fused_decode_gdn's FLM-derived kernels: the qk core scores each
# 16-key block into L1 it shares with the kv core above it, which accumulates
# the values and normalizes at the end.
import argparse
import os
import tempfile

import filelock
import numpy as np
from ml_dtypes import bfloat16

import common
import qwen36_35b_a3b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt import XRTBackend

DH, QH = W.DH, W.NQ // W.NKV
S_BLK = ((QH * 16 + 16 + 63) // 64) * 64
BLK = 16 * DH


def build_module(nb, L):
    qk = air.extern("attn_qk_blk", link_with="attn_qk.o", scalars=[i32, i32])
    kv = air.extern("attn_kv_blk", link_with="attn_kv.o", scalars=[i32, i32])
    fin = air.extern("attn_kv_fin", link_with="attn_kv.o")
    Q = air.tensor([QH * DH], bf16)
    K = air.tensor([nb, BLK], bf16)
    V = air.tensor([nb, BLK], bf16)
    O = air.tensor([QH * DH], bf16)
    # q, then the k blocks, on the qk core's one inbound flow
    qk_in, v_in, o_out = (air.channel(n) for n in ("QKIn", "VIn", "OOut"))
    q_in = k_in = qk_in

    with air.launch(name="attn") as launch:

        @launch.body
        def _():
            q_in.put(Q)
            k_in.put(K)
            v_in.put(V)
            o_out.get(O)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    s = air.alloc([S_BLK], bf16, scope=seg.shared())
                    with air.herd([range(1)], name="qk", shape=(1,), at=(3, 2)) as hq:

                        @hq.body
                        def _(tx):
                            p = hq.private()
                            q = air.alloc([QH * DH], bf16, scope=p)
                            kb = air.alloc([BLK], bf16, scope=p)
                            m = air.alloc([16], bf16, scope=p)
                            cc = air.alloc([8], f32, scope=p)
                            q_in.get(q)
                            for b in air.sequential(0, nb):
                                k_in.get(kb)
                                qk(q, kb, m, cc, s, b, L)

                    with air.herd([range(1)], name="kv", shape=(1,), at=(3, 3)) as hk:

                        @hk.body
                        def _(tx):
                            p = hk.private()
                            vb = air.alloc([BLK], bf16, scope=p)
                            y = air.alloc([QH * DH], f32, scope=p)
                            l = air.alloc([16], f32, scope=p)
                            o = air.alloc([QH * DH], bf16, scope=p)
                            for b in air.sequential(0, nb):
                                v_in.get(vb)
                                kv(s, vb, y, l, b, L)
                            fin(y, l, o)
                            o_out.put(o)

    return launch.build(target="npu2")


def pack(q, k, v):
    """q [QH, DH], k / v [nb * 16, DH] -> the kernels' layouts: q and o
    [DH/8][QH][8], a k block [DH/8][16][8], a v block [2][DH/8][8][8]."""
    nb = k.shape[0] // 16
    qp = q.reshape(QH, DH // 8, 8).transpose(1, 0, 2)
    kp = k.reshape(nb, 16, DH // 8, 8).transpose(0, 2, 1, 3)
    vp = v.reshape(nb, 2, 8, DH // 8, 8).transpose(0, 1, 3, 2, 4)
    c = lambda a: np.ascontiguousarray(a.astype(bfloat16))  # noqa: E731
    return c(qp).reshape(-1), c(kp).reshape(nb, BLK), c(vp).reshape(nb, BLK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--L", type=str, default="1,16,37,100")
    a = ap.parse_args()
    lengths = [int(x) for x in a.L.split(",")]
    if a.print_module_only:
        print(build_module(-(-lengths[0] // 16), lengths[0]))
        raise SystemExit(0)
    common.compile_kernel("attn_qk", engine=True)
    common.compile_kernel("attn_kv", engine=True)
    rng = np.random.default_rng(0)
    ok = True
    for L in lengths:
        nb = -(-L // 16)
        q = rng.standard_normal((QH, DH)).astype(bfloat16)
        k = np.zeros((nb * 16, DH), bfloat16)
        v = np.zeros((nb * 16, DH), bfloat16)
        k[:L] = rng.standard_normal((L, DH))
        v[:L] = rng.standard_normal((L, DH))
        be = XRTBackend(instance_name="attn")
        art = be.compile(build_module(nb, L))
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            run = be.load(art)
            qp, kp, vp = pack(q, k, v)
            out = run(qp, kp, vp, np.zeros(QH * DH, bfloat16))[3]
            be.unload()
        o = out.reshape(DH // 8, QH, 8).transpose(1, 0, 2).reshape(QH, DH)
        s = q.astype(np.float64) @ k[:L].astype(np.float64).T * W.DH**-0.5
        p = np.exp(s - s.max(1, keepdims=True))
        want = (p / p.sum(1, keepdims=True)) @ v[:L].astype(np.float64)
        g = o.astype(np.float64)
        c = float(g.ravel() @ want.ravel() / np.linalg.norm(g) / np.linalg.norm(want))
        e = np.abs(g - want).max() / np.abs(want).max()
        good = c > 0.999
        ok &= good
        print(f"L={L}: corr {c:.6f} max rel err {e:.2e} {'ok' if good else 'BAD'}")
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
