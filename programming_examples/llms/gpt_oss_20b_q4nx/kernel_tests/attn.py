# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One attention unit (16 query heads, 2 KV heads, sinks) over L cached keys:
# per 16-key block the qk step then the kv step, then the final scale, all on
# one core. Checked against plain attention with sinks.
import argparse
import math

import numpy as np
import torch
from ml_dtypes import bfloat16

import common  # noqa: F401  (sets sys.path)
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt_runner import XRTRunner

DH = 64
QN = 16 * DH + 16  # 16 heads, then their sinks
BLK = 2 * 16 * DH  # 2 KV heads x 16 keys


def build_module(L):
    nb = (L + 15) // 16
    qk = air.extern("gpt_attn_qk_blk", link_with="attn.o", scalars=[i32, i32, i32])
    kv = air.extern("gpt_attn_kv_blk", link_with="attn.o", scalars=[i32])
    fin = air.extern("gpt_attn_kv_fin", link_with="attn.o")
    Q = air.tensor([QN], bf16)
    K = air.tensor([nb, BLK], bf16)
    V = air.tensor([nb, BLK], bf16)
    O = air.tensor([16 * DH], bf16)
    # FLM's two cores per unit: qk scores a block, kv folds it into y
    qk_in = air.channel("QKIn")  # q, then the k blocks
    kv_in = air.channel("KVIn")  # sinks (the tail of q), then the v blocks
    s_ch = air.channel("S")

    with air.launch(name="attn") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    qk_in.put(Q)
                    kv_in.put(Q[16 * DH : QN])
                    for b in air.sequential(0, nb):
                        qk_in.put(K[b, :])
                        kv_in.put(V[b, :])
                    with air.herd([range(1)], name="qk", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            q = air.alloc([QN], bf16, scope=h.private())
                            k = air.alloc([BLK], bf16, scope=h.private())
                            s = air.alloc([16 * 16 + 64], bf16, scope=h.private())
                            m = air.alloc([16], f32, scope=h.private())
                            c = air.alloc([16], f32, scope=h.private())
                            qk_in.get(q)
                            for b in air.sequential(0, nb):
                                qk_in.get(k)
                                qk(q, k, m, c, s, b, L, 0)
                                s_ch.put(s)

                    with air.herd([range(1)], name="kv", shape=(1,)) as h2:

                        @h2.body
                        def _(tx):
                            sink = air.alloc([16], bf16, scope=h2.private())
                            s = air.alloc([16 * 16 + 64], bf16, scope=h2.private())
                            v = air.alloc([BLK], bf16, scope=h2.private())
                            y = air.alloc([16 * DH], f32, scope=h2.private())
                            l = air.alloc([16], f32, scope=h2.private())
                            ml = air.alloc([16], f32, scope=h2.private())
                            o = air.alloc([16 * DH], bf16, scope=h2.private())
                            kv_in.get(sink)
                            for b in air.sequential(0, nb):
                                s_ch.get(s)
                                kv_in.get(v)
                                kv(s, v, y, l, ml, b)
                            fin(y, l, sink, ml, o)
                            air.ops.store(o, O[:])

    return launch.build(target="npu2")


def pack(q, sinks, k, v):
    """Logical q [16, 64], sinks [16], k/v [2, L, 64] -> the kernel's layouts."""
    L = k.shape[1]
    nb = (L + 15) // 16
    kp = np.zeros((2, nb * 16, DH), np.float32)
    vp = np.zeros_like(kp)
    kp[:, :L], vp[:, :L] = k, v
    qq = q.reshape(2, 8, 8, 8).transpose(0, 2, 1, 3)  # hhalf, dchunk, head, d
    qbuf = np.concatenate([qq.ravel(), sinks])
    kb = kp.reshape(2, nb, 2, 8, 8, 8).transpose(
        1, 0, 4, 2, 3, 5
    )  # blk, kvh, dchunk, khalf, key, d
    vb = vp.reshape(2, nb, 2, 8, 8, 8).transpose(
        1, 2, 0, 4, 3, 5
    )  # blk, khalf, kvh, dchunk, key, d
    return (
        qbuf.astype(bfloat16),
        kb.reshape(nb, BLK).astype(bfloat16),
        vb.reshape(nb, BLK).astype(bfloat16),
    )


def reference(q, sinks, k, v):
    """Attention of 16 heads (8 per KV head) with sinks, fp32."""
    q, k, v = (torch.from_numpy(a) for a in (q, k, v))
    kr = k.repeat_interleave(8, 0)  # [16, L, 64]
    vr = v.repeat_interleave(8, 0)
    s = (q[:, None, :] @ kr.transpose(1, 2))[:, 0] / math.sqrt(DH)  # [16, L]
    sink = torch.from_numpy(sinks)[:, None]
    p = torch.softmax(torch.cat((s, sink), -1), -1)[:, :-1]
    o = (p[:, None, :] @ vr)[:, 0]  # [16, 64]
    return o.numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    module = build_module(a.L)
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("attn")
    rng = np.random.default_rng(a.seed)

    def bfr(*shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(bfloat16).astype(np.float32)

    q, k, v = bfr(16, DH), bfr(2, a.L, DH), bfr(2, a.L, DH)
    sinks = bfr(16, scale=2.0)
    qb, kb, vb = pack(q, sinks, k, v)
    o = reference(q, sinks, k, v)
    o_kernel_layout = o.reshape(2, 8, 8, 8).transpose(0, 2, 1, 3).ravel()
    # the qk and kv blocks each take a 2368-byte frame
    runner = XRTRunner(instance_name="attn", stack_size=4096)
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[qb, kb, vb],
            expected_outputs=[o_kernel_layout.astype(bfloat16)],
            # FLM rounds scores to bf16 and uses LUT exp / approximate 1/x
            rtol=5e-2,
            atol=5e-2,
            min_correlation=0.999,
        )
    )
