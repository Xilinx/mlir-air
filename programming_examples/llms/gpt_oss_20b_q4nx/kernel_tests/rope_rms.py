# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The dense kernels the other LLMs use, built for gpt-oss-20b: rope_compute
# (q/k/v bias, then half-split RoPE with a YaRN table whose attention scale is
# folded into cos/sin) and rms_norm_aie / residual_add_aie (eps 1e-5).
import argparse

import torch
from ml_dtypes import bfloat16

import common  # noqa: F401  (sets sys.path)
import gpt_oss_20b_q4nx_weights as g
from air import api as air
from air.api import bf16, i32
from air.backend.xrt_runner import XRTRunner

DQ, DK = g.N_Q_HEADS * g.DH, g.N_KV_HEADS * g.DH
QKV = DQ + 2 * DK
ROPE_W = g.DH + QKV  # cos[32] sin[32], then the q|k|v bias


def build_rope():
    rope = air.extern("rope_compute", link_with="rope.o", scalars=[i32])
    QKVT = air.tensor([QKV], bf16)
    RW = air.tensor([ROPE_W], bf16)
    OUT = air.tensor([QKV], bf16)  # q | k | v
    out = air.channel("Out")
    with air.launch(name="rope") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    out.get(OUT[0:DQ])
                    out.get(OUT[DQ : DQ + DK])
                    out.get(OUT[DQ + DK : QKV])
                    with air.herd([range(1)], name="rope", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            qkv = air.alloc([QKV], bf16, scope=h.private())
                            rw = air.alloc([ROPE_W], bf16, scope=h.private())
                            q = air.alloc([DQ], bf16, scope=h.private())
                            k = air.alloc([DK], bf16, scope=h.private())
                            v = air.alloc([DK], bf16, scope=h.private())
                            air.ops.load(qkv, QKVT[:])
                            air.ops.load(rw, RW[:])
                            rope(q, k, v, qkv, rw, 0)
                            out.put(q)
                            out.put(k)
                            out.put(v)

    return launch.build(target="npu2")


def build_rms():
    rms = air.extern("rms_norm_aie", link_with="rms_residual.o", scalars=[i32])
    res = air.extern("residual_add_aie", link_with="rms_residual.o")
    IN = air.tensor([3, g.D], bf16)  # x, branch, norm weight
    OUT = air.tensor([2, g.D], bf16)  # x + branch, rmsnorm(x + branch) * w
    a_in = air.channel("AIn")
    out = air.channel("Out")
    with air.launch(name="rms") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    for i in range(3):
                        a_in.put(IN[i, :])
                    out.get(OUT[0, :])
                    out.get(OUT[1, :])
                    with air.herd([range(1)], name="rms", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            x = air.alloc([g.D], bf16, scope=h.private())
                            br = air.alloc([g.D], bf16, scope=h.private())
                            nw = air.alloc([g.D], bf16, scope=h.private())
                            s = air.alloc([g.D], bf16, scope=h.private())
                            y = air.alloc([g.D], bf16, scope=h.private())
                            a_in.get(x)
                            a_in.get(br)
                            a_in.get(nw)
                            res(s, x, br)
                            rms(y, s, nw, 0)
                            out.put(s)
                            out.put(y)

    return launch.build(target="npu2")


def bf(t):
    return t.to(torch.bfloat16).float()


def run_rope(layer, pos, seed):
    w = g.GptOssQ4nx()
    a = w.attn(layer)
    qkv = bf(3 * torch.randn(QKV, generator=torch.Generator().manual_seed(seed)))
    bias = bf(torch.cat((a["bq"], a["bk"], a["bv"])))
    cos, sin = g.rope_cos_sin([pos])
    rw = torch.cat((bf(cos[0]), bf(sin[0]), bias))
    x = bf(qkv + bias)
    q = g.apply_rope(x[:DQ].reshape(-1, g.DH), bf(cos), bf(sin)).ravel()
    k = g.apply_rope(x[DQ : DQ + DK].reshape(-1, g.DH), bf(cos), bf(sin)).ravel()
    want = torch.cat((q, k, x[DQ + DK :]))
    common.compile_kernel("rope", shared=True)
    return XRTRunner(instance_name="rope").run_test(
        build_rope(),
        inputs=[qkv.numpy().astype(bfloat16), rw.numpy().astype(bfloat16)],
        expected_outputs=[want.numpy().astype(bfloat16)],
        rtol=2e-2,
        atol=5e-2,
        min_correlation=0.9999,
    )


def run_rms(layer, seed):
    w = g.GptOssQ4nx()
    gen = torch.Generator().manual_seed(seed)
    x, br = bf(torch.randn(g.D, generator=gen)), bf(
        0.3 * torch.randn(g.D, generator=gen)
    )
    nw = bf(w.norms(layer)[1])
    s = bf(x + br)
    want = torch.stack((s, g.rmsnorm(s, nw)))
    common.compile_kernel("rms_residual", shared=True)
    return XRTRunner(instance_name="rms").run_test(
        build_rms(),
        inputs=[torch.stack((x, br, nw)).numpy().astype(bfloat16)],
        expected_outputs=[want.numpy().astype(bfloat16)],
        rtol=2e-2,
        atol=2e-2,
        min_correlation=0.9999,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rope", "rms"])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--pos", type=int, default=37)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rc = (
        run_rope(a.layer, a.pos, a.seed)
        if a.what == "rope"
        else run_rms(a.layer, a.seed)
    )
    raise SystemExit(rc)
