# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.6-35B-A3B text model on the CPU, float32, on the bundle's
dequantized weights: the reference the NPU decode is checked against. Experts are dequantized as the router picks them.

Reference(w).forward(ids) runs the new tokens `ids` after those already seen
(the KV cache, conv state and recurrent state carry over) and returns the
logits of the last one.
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import qwen36_35b_a3b_q4nx_weights as W


def rmsnorm(x, w, eps=W.EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rope(x, pos):
    """Rotate-half RoPE on the first ROPE_DIM dims of x [L, heads, DH]."""
    half = W.ROPE_DIM // 2
    inv = 1.0 / (W.THETA ** (torch.arange(0, W.ROPE_DIM, 2).float() / W.ROPE_DIM))
    f = pos.float()[:, None] * inv[None]  # [L, half]
    cos, sin = f.cos()[:, None], f.sin()[:, None]
    x1, x2 = x[..., :half], x[..., half : W.ROPE_DIM]
    return torch.cat(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., W.ROPE_DIM :]], -1
    )


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


class Reference:
    def __init__(self, w, layers=None):
        self.w = w
        self.layers = list(range(W.N_LAYERS)) if layers is None else layers
        self.p = {}  # per-layer dense weights, as torch
        self.reset()

    def reset(self):
        self.pos = 0
        self.kv = {}  # attention layer -> (K [L, NKV, DH], V)
        self.conv = {}  # GDN layer -> last CONV - 1 inputs [CONV - 1, QKV]
        self.state = {}  # GDN layer -> S [NVH, DK, DV]

    def weights(self, l):
        if l not in self.p:
            w = self.w
            mix = w.attn(l) if W.is_attention(l) else w.gdn(l)
            self.p[l] = dict(
                mix=[t(a) for a in mix],
                norms=[t(a) for a in w.norms(l)],
                router=[t(a) for a in w.router(l)],
                shared=[t(a) for a in w.shared(l)],
            )
        return self.p[l]

    def gdn(self, l, x):
        qkv_w, z_w, out_w, al_w, be_w, conv_w, norm_w, a, dt = self.weights(l)["mix"]
        L = x.shape[0]
        qkv = x @ qkv_w.T  # [L, QKV]
        prev = self.conv.get(l, torch.zeros(W.CONV - 1, W.QKV))
        hist = torch.cat([prev, qkv])  # [CONV - 1 + L, QKV]
        self.conv[l] = hist[-(W.CONV - 1) :].clone()
        # causal depthwise conv: out[t] = sum_j conv_w[:, j] * in[t - 3 + j]
        conv = sum(conv_w[:, j] * hist[j : j + L] for j in range(W.CONV))
        conv = F.silu(conv)
        q, k, v = conv.split([W.NK * W.DK, W.NK * W.DK, W.NVH * W.DV], -1)
        q = l2norm(q.view(L, W.NK, W.DK)).repeat_interleave(W.NVH // W.NK, 1)
        k = l2norm(k.view(L, W.NK, W.DK)).repeat_interleave(W.NVH // W.NK, 1)
        v = v.view(L, W.NVH, W.DV)
        q = q * W.DK**-0.5
        beta = torch.sigmoid(x @ be_w.T)  # [L, NVH]
        g = a * F.softplus(x @ al_w.T + dt)  # a = -exp(A_log)
        S = self.state.get(l, torch.zeros(W.NVH, W.DK, W.DV))
        out = torch.empty(L, W.NVH, W.DV)
        for i in range(L):
            S = S * g[i].exp()[:, None, None]
            mem = (S * k[i][:, :, None]).sum(1)  # [NVH, DV]
            delta = (v[i] - mem) * beta[i][:, None]
            S = S + k[i][:, :, None] * delta[:, None, :]
            out[i] = (S * q[i][:, :, None]).sum(1)
        self.state[l] = S
        z = (x @ z_w.T).view(L, W.NVH, W.DV)
        o = rmsnorm(out, norm_w) * F.silu(z)
        return o.reshape(L, -1) @ out_w.T

    def attention(self, l, x):
        q_w, g_w, k_w, v_w, o_w, qn, kn = self.weights(l)["mix"]
        L = x.shape[0]
        pos = torch.arange(self.pos, self.pos + L)
        q = rope(rmsnorm((x @ q_w.T).view(L, W.NQ, W.DH), qn), pos)
        k = rope(rmsnorm((x @ k_w.T).view(L, W.NKV, W.DH), kn), pos)
        v = (x @ v_w.T).view(L, W.NKV, W.DH)
        if l in self.kv:
            k = torch.cat([self.kv[l][0], k])
            v = torch.cat([self.kv[l][1], v])
        self.kv[l] = (k, v)
        rep = W.NQ // W.NKV
        kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = torch.einsum("qhd,khd->hqk", q, kk) * W.DH**-0.5
        n = k.shape[0]
        mask = torch.arange(n)[None] > (pos[:, None])  # key after query
        s = s.masked_fill(mask[None], float("-inf"))
        o = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv).reshape(L, -1)
        o = o * torch.sigmoid(x @ g_w.T)
        return o @ o_w.T

    def moe(self, l, x):
        """The MoE block; also returns each row's (experts, weights)."""
        p = self.weights(l)
        router, sgate = p["router"]
        probs = (x @ router.T).softmax(-1)
        top, idx = probs.topk(W.TOP_K, -1)
        top = top / top.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for r in range(x.shape[0]):
            for e, s in zip(idx[r].tolist(), top[r]):
                eg, eu, ed = (t(a) for a in self.w.expert(l, e))
                out[r] += s * ((F.silu(x[r] @ eg.T) * (x[r] @ eu.T)) @ ed.T)
        sg, su, sd = p["shared"]
        shared = (F.silu(x @ sg.T) * (x @ su.T)) @ sd.T
        out = out + torch.sigmoid(x @ sgate)[:, None] * shared
        return out, (idx, top)

    def layer(self, l, x, collect=None):
        n1, n2 = self.weights(l)["norms"]
        h = rmsnorm(x, n1)
        mix = self.attention(l, h) if W.is_attention(l) else self.gdn(l, h)
        x = x + mix
        m, route = self.moe(l, rmsnorm(x, n2))
        if collect is not None:
            collect[l] = dict(mix=mix, route=route, out=x + m)
        return x + m

    @torch.no_grad()
    def forward(self, ids, collect=None, lm=True):
        x = t(self.w.embed(ids))
        for l in self.layers:
            x = self.layer(l, x, collect)
        self.pos += len(ids)
        if not lm:
            return x
        h = rmsnorm(x[-1:], t(self.w.final_norm()))
        return (h @ t(self.w.lm_head).T)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    w = W.Qwen36Q4nx()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    ref = Reference(w)
    t0 = time.time()
    logits = ref.forward(ids)
    out = []
    for _ in range(a.steps):
        nxt = int(logits.argmax())
        out.append(nxt)
        logits = ref.forward([nxt])
    print(f"{a.prompt!r} -> {tok.decode(out)!r}  ids {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
