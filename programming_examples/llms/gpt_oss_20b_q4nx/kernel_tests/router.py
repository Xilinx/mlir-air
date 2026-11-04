# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Router GEMV (bundle's decode-layout weights), top-4 selection + softmax, and
# the weighted combine of four experts' down projections, chained on one core.
import argparse
import os
import tempfile

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

import common  # noqa: F401  (sets sys.path)
import gpt_oss_20b_q4nx_weights as g
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt import XRTBackend

NB = g.D // 64  # 45 router blocks of 32x64
NCH = g.D_PAD // 128  # 23 combine chunks


def build_module():
    k = {
        n: air.extern(n, link_with="router.o", scalars=s)
        for n, s in (
            ("router_bias_init", []),
            ("router_mvm", [i32]),
            ("router_flush", []),
            ("router_sort", [i32]),
            ("router_softmax", []),
            ("moe_combine", []),
        )
    }
    # FLM feeds the router bias as the first block of its weight stream
    # FLM feeds the router bias as the first block of its weight stream
    skip = os.environ.get("ROUTER_SKIP", "").split(",")
    for n in skip:
        if n in k:
            k[n] = lambda *a: None
    WR = air.tensor([NB + 1, 32 * 64], bf16)
    X = air.tensor([g.D], bf16)
    DOWN = air.tensor([NCH, g.TOP_K * 128], bf16)
    OUTB = air.tensor([32 + 32 + 16 + NCH * 128], bf16)  # logits, sorted, topw, out
    IDX = air.tensor([32], i32)
    w_in = air.channel("WIn")
    a_in = air.channel("AIn")  # x, then the down chunks
    b_out = air.channel("BOut")
    i_out = air.channel("IOut")

    with air.launch(name="router") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    w_in.put(WR)
                    a_in.put(X)
                    a_in.put(DOWN)
                    # one get per core put: each put ends its packet
                    b_out.get(OUTB[0:32])
                    for c in air.sequential(0, NCH):
                        b_out.get(OUTB[80 + c * 128 : 208 + c * 128])
                    b_out.get(OUTB[32:64])
                    b_out.get(OUTB[64:80])
                    i_out.get(IDX)

                    with air.herd([range(1)], name="router", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            x = air.alloc([g.D], bf16, scope=h.private())
                            w = air.alloc([32 * 64], bf16, scope=h.private())
                            acc = air.alloc([32], f32, scope=h.private())
                            logits = air.alloc([32], bf16, scope=h.private())
                            sorted_ = air.alloc([32], bf16, scope=h.private())
                            topw = air.alloc([16], bf16, scope=h.private())
                            idx = air.alloc([32], i32, scope=h.private())
                            dn = air.alloc([g.TOP_K * 128], bf16, scope=h.private())
                            o = air.alloc([128], bf16, scope=h.private())
                            w_in.get(w)
                            k["router_bias_init"](w, acc)
                            a_in.get(x)
                            for j in air.sequential(0, NB):
                                w_in.get(w)
                                k["router_mvm"](w, x, acc, j)
                            k["router_flush"](acc, logits)
                            b_out.put(logits)
                            k["router_flush"](acc, sorted_)
                            k["router_sort"](sorted_, idx, 0)
                            k["router_softmax"](sorted_, topw)
                            for c in air.sequential(0, NCH):
                                a_in.get(dn)
                                k["moe_combine"](dn, topw, o)
                                b_out.put(o)
                            # after the loop: AIR holds a put buffer's lock until
                            # its last read, so putting topw earlier deadlocks
                            # behind the o chunks on the same channel
                            b_out.put(sorted_)
                            b_out.put(topw)
                            i_out.put(idx)

    return launch.build(target="npu2")


def bf16_floor(t):
    b = t.contiguous().view(torch.int32)
    up = ((b & 0xFFFF) != 0) & (b < 0)
    return ((b & ~0xFFFF) + up.to(torch.int32) * 0x10000).view(torch.float32)


def bf(a):
    return np.asarray(a, np.float32).astype(bfloat16)


def inputs(layer, token, seed):
    w = g.GptOssQ4nx()
    x = w.embed(torch.tensor([token]))[0]
    x = g.rmsnorm(x, w.norms(layer)[1]).to(torch.bfloat16).float()
    wr = w.raw(f"model.layers.{layer}.mlp.router.weight").reshape(NB, 32 * 64)
    wr_logical, br = w.router(layer)
    down = torch.randn(g.TOP_K, g.D_PAD, generator=torch.Generator().manual_seed(seed))
    down = down.reshape(g.TOP_K, NCH, 128).permute(1, 0, 2).reshape(NCH, -1)
    wr = torch.cat((torch.zeros(1, 32 * 64, dtype=torch.bfloat16), wr))
    wr[0, :32] = br.to(torch.bfloat16)
    return dict(
        wr=wr.view(torch.int16).numpy().view(bfloat16),
        x=bf(x),
        down=bf(down),
        ref_logits=(wr_logical @ x + br),
    )


def run(module, d):
    backend = XRTBackend()
    art = backend.compile(module)
    outs = [np.zeros(32 + 32 + 16 + NCH * 128, bfloat16), np.zeros(32, np.int32)]
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        f = backend.load(art)
        res = f(d["wr"], d["x"], d["down"], *outs)
    ob, idx = res[3].view(bfloat16), res[4].view(np.int32)
    return ob[:32], ob[32:64], idx, ob[64:80], ob[80:].reshape(NCH, 128)


def check(d, logits, sorted_, idx, topw, out):
    ok = True
    lg = logits.astype(np.float32)
    ref = d["ref_logits"].numpy()
    err = np.abs(lg - ref)
    ulp = np.abs(ref) * 2.0**-7
    print(f"logits: max |err| {err.max():.4f}, worst in ulps {(err / ulp).max():.2f}")
    ok &= bool((err <= 2 * ulp + 1e-3).all())
    # bubble sort is stable descending on the device's own logits
    want = np.argsort(-lg, kind="stable")
    ok &= np.array_equal(idx, want) and np.array_equal(
        sorted_.astype(np.float32), lg[want]
    )
    print(
        f"top4 ids {idx[:4].tolist()} (reference {np.argsort(-ref, kind='stable')[:4].tolist()})"
    )
    sm = torch.softmax(torch.from_numpy(lg[want[:4]]), 0).numpy()
    tw = topw[:4].astype(np.float32)
    print(f"top4 weights {tw.tolist()} vs softmax {sm.round(4).tolist()}")
    ok &= bool(np.abs(tw - sm).max() < 2e-2)
    dn = torch.from_numpy(d["down"].astype(np.float32)).reshape(NCH, g.TOP_K, 128)
    want_out = bf16_floor((dn * torch.from_numpy(tw)[None, :, None]).sum(1)).numpy()
    print(f"combine exact: {np.array_equal(out.astype(np.float32), want_out)}")
    ok &= np.array_equal(out.astype(np.float32), want_out)
    print("PASS!" if ok else "failed.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=12650)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("router", moe=True)
    d = inputs(a.layer, a.token, a.seed)
    raise SystemExit(0 if check(d, *run(module, d)) else 1)
