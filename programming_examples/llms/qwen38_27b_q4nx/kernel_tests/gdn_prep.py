# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Gated DeltaNet layer's step inputs on one core: the conv4 + SiLU of the new
# q/k/v projection against three earlier ones, q/k l2-normalized, alpha / beta
# from their projection, packed per value head as the step cores take them.
import argparse
import os
import tempfile

import filelock
import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16

import common
import qwen38_27b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt import XRTBackend

# the element-wise atol, over max |want|: twice the worst measured over seeds
# 0-2 (3.2e-2, alpha)
ATOL = 6e-2

HD, NV, NK = W.DK, W.NVH, W.NK
NCH = W.QKV // HD  # conv chunks of 128 channels
SLOT = HD + 32
RB, CB = 32, 256
# alpha / beta rows [alpha NV | beta NV] in 32-row blocks
NRB_AB = 2 * NV // RB


def build_module():
    conv = air.extern("gdn_conv", link_with="gdn_prep.o", scalars=[i32])
    norm = air.extern("gdn_l2norm", link_with="gdn_prep.o", scalars=[i32, i32])
    zero = air.extern("ab_zero", link_with="gdn_prep.o")
    mvm = air.extern("ab_mvm", link_with="gdn_prep.o", scalars=[i32, i32])
    act = air.extern("ab_activate", link_with="gdn_prep.o")
    pack = air.extern("gdn_pack", link_with="gdn_prep.o", scalars=[i32])
    WX = air.tensor([NCH, 8 * HD], bf16)
    AB = air.tensor([NRB_AB, W.D // CB, CB * RB], bf16)
    X = air.tensor([W.D], bf16)
    ADT = air.tensor([2 * NV], f32)
    KVQ = air.tensor([NV, 3 * SLOT], bf16)
    # the bf16 inputs in the order the core takes them, on one flow
    b_in, adt_in, kvq_out = (air.channel(n) for n in ("BIn", "ADTIn", "KVQOut"))
    wx_in = x_in = ab_in = b_in

    with air.launch(name="gdn_prep") as launch:

        @launch.body
        def _():
            b_in.put(WX)
            b_in.put(X)
            b_in.put(AB)
            adt_in.put(ADT)
            kvq_out.get(KVQ)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="prep", shape=(1,)) as he:

                        @he.body
                        def _(tx):
                            p = he.private()
                            wx = air.alloc([8 * HD], bf16, scope=p)
                            cv = air.alloc([W.QKV], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            x = air.alloc([W.D], bf16, scope=p)
                            ab = air.alloc([2 * NV], f32, scope=p)
                            adt = air.alloc([2 * NV], f32, scope=p)
                            kvq = air.alloc([3 * SLOT], bf16, scope=p)
                            # q chunks (normed, scaled), k (normed), v
                            for lo, hi, scale in ((0, NK, 1), (NK, 2 * NK, 0)):
                                for c in air.sequential(lo, hi):
                                    wx_in.get(wx)
                                    conv(wx, cv, c)
                                    norm(cv, c, scale)
                            for c in air.sequential(2 * NK, NCH):
                                wx_in.get(wx)
                                conv(wx, cv, c)
                            x_in.get(x)
                            adt_in.get(adt)
                            zero(ab)
                            for rb in air.sequential(0, NRB_AB):
                                for cb in air.sequential(0, W.D // CB):
                                    ab_in.get(blk)
                                    mvm(blk, x, ab, rb, cb)
                            act(ab, adt)
                            for h in air.sequential(0, NV):
                                pack(cv, ab, kvq, h)
                                kvq_out.put(kvq)

    return launch.build(target="npu2")


def reference(w, layer, xs):
    """xs [4, D]: three earlier tokens' normed inputs, then this one's. ->
    (qkv, q, k, v, alpha, beta), exact math; q and k per key head, v, alpha
    and beta per value head."""
    qkv_w, _, _, al, be, conv, _, a, dt = (torch.from_numpy(t) for t in w.gdn(layer))
    x = torch.from_numpy(xs.astype(np.float32))
    qkv = (x @ qkv_w.T).to(torch.bfloat16).float()  # the projection's output
    y = F.silu(sum(conv[:, j] * qkv[j] for j in range(4)))
    q, k, v = y.split([NK * HD, NK * HD, NV * HD])
    q = F.normalize(q.view(NK, HD), dim=-1, eps=0) * HD**-0.5
    k = F.normalize(k.view(NK, HD), dim=-1, eps=0)
    alpha = torch.exp(a * F.softplus(x[3] @ al.T + dt))
    beta = torch.sigmoid(x[3] @ be.T)
    return qkv, q, k, v.view(NV, HD), alpha, beta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("gdn_prep")
    w = W.Qwen38Q4nx()
    rng = np.random.default_rng(a.seed)
    xs = w.embed(rng.integers(0, W.VOCAB, 4))
    xs = (xs / np.sqrt((xs * xs).mean(-1, keepdims=True) + W.EPS)).astype(bfloat16)
    qkv, q, k, v, alpha, beta = reference(w, a.layer, xs)
    _, _, _, al, be, conv, _, A, dt = w.gdn(a.layer)
    # each conv chunk: [x0, w0, ..., x3, w3], x3 this token's projection
    qb = qkv.numpy().astype(bfloat16).reshape(4, NCH, HD)
    wb = conv.astype(bfloat16).T.reshape(4, NCH, HD)
    wx = np.stack([qb, wb], 1).transpose(2, 0, 1, 3).reshape(NCH, 8 * HD)
    abw = np.concatenate([al, be]).reshape(NRB_AB, RB, W.D // CB, CB)
    abw = abw.transpose(0, 2, 3, 1)
    abw = np.ascontiguousarray(abw.astype(bfloat16)).reshape(NRB_AB, W.D // CB, CB * RB)
    adt = np.concatenate([A, dt]).astype(np.float32)
    be_ = XRTBackend(instance_name="gdn_prep")
    art = be_.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be_.load(art)
        out = run(wx, abw, xs[3], adt, np.zeros((NV, 3 * SLOT), bfloat16))
        be_.unload()
    kvq = out[4].reshape(NV, 3, SLOT)
    f32s = kvq.view(np.uint16)[:, :2, HD : HD + 2].copy().view(np.float32)[..., 0]
    got = dict(
        k=kvq[:, 0, :HD].astype(np.float32),
        v=kvq[:, 1, :HD].astype(np.float32),
        q=kvq[:, 2, :HD].astype(np.float32),
        alpha=f32s[:, 0],
        beta=f32s[:, 1],
    )
    rep = lambda t: t.repeat_interleave(NV // NK, 0).numpy()  # noqa: E731
    want = dict(k=rep(k), v=v.numpy(), q=rep(q), alpha=alpha.numpy(), beta=beta.numpy())
    ok = True
    for n in ("q", "k", "v", "alpha", "beta"):
        ok &= common.check(n, got[n], want[n], ATOL)
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
