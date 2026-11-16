# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The verify pass's Gated DeltaNet step inputs for two tokens on one core, as
# the conv core takes them (qwen38_27b_q4nx_verify.py): the conv of each
# 128-channel chunk from the cache [x0, x1, x2] and the two tokens' rows of a
# two-token round, q/k l2-normalized, alpha / beta of both tokens, packed per
# value head and token. Each token against the float64 reference. alpha /
# beta come from the host here (the verify pass computes them on another
# core, with ab_mvm2 / ab_activate2: ab_mvm / ab_activate on each token).
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

# the element-wise atol, over max |want|, as gdn_prep.py's
ATOL = 6e-2

HD, NV, NK = W.DK, W.NVH, W.NK
NCH = W.QKV // HD
SLOT = HD + 32
RB, CB = 32, 256
NRB_AB = 2 * NV // RB
CONV_Q, CONV_K, NRD = NK * HD // 512, 2 * NK * HD // 512, W.QKV // 512
RND = 1024  # a two-token round


def build_module():
    k_ = lambda n, s=(): air.extern(n, link_with="gdn_prep2.o", scalars=list(s))  # noqa
    conv2, l2n_t = k_("gdn_conv_cache2", [i32]), k_("gdn_l2norm_t", [i32] * 3)
    pack_t = k_("gdn_pack_t", [i32, i32])
    HX = air.tensor([NCH, 7 * HD], bf16)
    U = air.tensor([NRD, RND], bf16)
    ABT = air.tensor([4 * NV], f32)  # [token][alpha NV | beta NV], activated
    KVQ = air.tensor([NV, 2, 3 * SLOT], bf16)
    HO = air.tensor([NCH, 3 * HD], bf16)
    b_in, u_in, ab_in = (air.channel(n) for n in ("BIn", "UIn", "ABIn"))
    kvq_out, ho_out = air.channel("KVQOut"), air.channel("HOOut")

    with air.launch(name="gdn_prep2") as launch:

        @launch.body
        def _():
            b_in.put(HX)
            u_in.put(U)
            ab_in.put(ABT)
            kvq_out.get(KVQ)
            ho_out.get(HO)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="prep", shape=(1,)) as he:

                        @he.body
                        def _(tx):
                            p = he.private()
                            u = air.alloc([RND], bf16, scope=p)
                            hx = air.alloc([7 * HD], bf16, scope=p)
                            ho = air.alloc([3 * HD], bf16, scope=p)
                            cv = air.alloc([2 * W.QKV], bf16, scope=p)
                            ab = air.alloc([4 * NV], f32, scope=p)
                            kq = [
                                air.alloc([3 * SLOT], bf16, scope=p) for _ in range(2)
                            ]
                            for lo, hi, norm in (
                                (0, CONV_Q, 1),
                                (CONV_Q, CONV_K, 0),
                                (CONV_K, NRD, None),
                            ):
                                for rd in air.sequential(lo, hi):
                                    u_in.get(u)
                                    for k in range(4):
                                        c = rd * 4 + k
                                        b_in.get(hx)
                                        conv2(hx, u, cv, ho, c)
                                        ho_out.put(ho)
                                        if norm is not None:
                                            for t in range(2):
                                                l2n_t(cv, c, norm, t)
                            ab_in.get(ab)
                            for h in air.sequential(0, NV):
                                for t in range(2):
                                    pack_t(cv, ab, kq[t], h, t)
                                    kvq_out.put(kq[t])

    return launch.build(target="npu2")


def rounds(a, b):
    """Two tokens' rows [QKV] -> the two-token rounds [NRD][col][core][tok][32]."""
    ab = np.stack([a, b])  # [tok, QKV]
    ab = ab.reshape(2, NRD, 4, 4, 32)  # [tok][round][col][core][32]
    return np.ascontiguousarray(ab.transpose(1, 2, 3, 0, 4)).reshape(NRD, RND)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    common.compile_kernel("gdn_prep2")
    w = W.Qwen38Q4nx()
    rng = np.random.default_rng(a.seed)
    # five tokens' normed inputs: three earlier (the cache), then the two
    xs = w.embed(rng.integers(0, W.VOCAB, 5))
    xs = (xs / np.sqrt((xs * xs).mean(-1, keepdims=True) + W.EPS)).astype(bfloat16)
    qkv_w, _, _, al, be, conv, _, A, dt = (
        torch.from_numpy(np.asarray(t, np.float32)) for t in w.gdn(a.layer)
    )
    x = torch.from_numpy(xs.astype(np.float32))
    qkv = (x @ qkv_w.T).to(torch.bfloat16).float()  # [5, QKV]
    cw = conv.T.contiguous() if conv.shape[0] != 4 else conv  # [4, QKV]
    hx = np.concatenate([qkv[:3].numpy(), cw.numpy()]).astype(bfloat16)  # [7, QKV]
    hx = np.ascontiguousarray(hx.reshape(7, NCH, HD).transpose(1, 0, 2)).reshape(
        NCH, 7 * HD
    )
    abt = []
    for t in range(2):
        abt += [
            torch.exp(A.ravel() * F.softplus(x[3 + t] @ al.T + dt.ravel())),
            torch.sigmoid(x[3 + t] @ be.T),
        ]
    abt = torch.cat(abt).numpy().astype(np.float32)
    u = rounds(qkv[3].numpy().astype(bfloat16), qkv[4].numpy().astype(bfloat16))
    be_ = XRTBackend(instance_name="gdn_prep2")
    art = be_.compile(build_module())
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be_.load(art)
        out = run(
            hx,
            u,
            abt,
            np.zeros((NV, 2, 3 * SLOT), bfloat16),
            np.zeros((NCH, 3 * HD), bfloat16),
        )
        be_.unload()
    kvq = out[3].reshape(NV, 2, 3, SLOT)
    # the new conv cache: per chunk [x2, a, b], copies of bf16 rows, exact
    ho_want = np.concatenate(
        [
            hx.reshape(NCH, 7, HD)[:, 2],
            qkv[3].numpy().astype(bfloat16).reshape(NCH, HD),
            qkv[4].numpy().astype(bfloat16).reshape(NCH, HD),
        ],
        axis=1,
    )
    ho_got = np.asarray(out[4]).reshape(NCH, 3 * HD)
    ok = bool(np.array_equal(ho_got.view(np.uint16), ho_want.view(np.uint16)))
    print(f"conv cache [x2, a, b]: {'exact' if ok else 'BAD'}")
    for t in range(2):
        hist = qkv[t : t + 4]  # the conv's four inputs for token t
        y = F.silu(sum(cw[j] * hist[j] for j in range(4)))
        q, k, v = y.split([NK * HD, NK * HD, NV * HD])
        q = F.normalize(q.view(NK, HD), dim=-1, eps=0) * HD**-0.5
        k = F.normalize(k.view(NK, HD), dim=-1, eps=0)
        alpha = torch.exp(A.ravel() * F.softplus(x[3 + t] @ al.T + dt.ravel()))
        beta = torch.sigmoid(x[3 + t] @ be.T)
        rep = lambda z: z.repeat_interleave(NV // NK, 0).numpy()  # noqa: E731
        f32s = (
            kvq[:, t]
            .view(np.uint16)[:, :2, HD : HD + 2]
            .copy()
            .view(np.float32)[..., 0]
        )
        got = dict(
            k=kvq[:, t, 0, :HD],
            v=kvq[:, t, 1, :HD],
            q=kvq[:, t, 2, :HD],
            alpha=f32s[:, 0],
            beta=f32s[:, 1],
        )
        want = dict(
            k=rep(k),
            v=v.view(NV, HD).numpy(),
            q=rep(q),
            alpha=alpha.numpy(),
            beta=beta.numpy(),
        )
        for n in got:
            ok &= common.check(
                f"token {t} {n}", np.asarray(got[n], np.float64), want[n], ATOL
            )
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
