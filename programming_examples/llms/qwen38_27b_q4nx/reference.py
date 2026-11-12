# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.8-27B text model on the CPU, float32, on the bundle's dequantized
weights: the reference the NPU decode is checked against.

Reference(w).forward(ids) runs the new tokens `ids` after those already seen
(the KV cache, conv state and recurrent state carry over) and returns the
logits of the last one.

A layer dequantized to float32 is about 1.5 GB, all 64 about 98 GB, so the
reference keeps only the `cache_layers` most recently used layers (default
$QWEN38_REF_CACHE_LAYERS, else 8) and dequantizes the others as it reaches
them; the LM head is applied in row slices.
"""

import argparse
import collections
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import qwen38_27b_q4nx_weights as W

LM_SLICE = 8192  # lm_head rows dequantized at a time


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
    def __init__(self, w, layers=None, cache_layers=None):
        self.w = w
        self.layers = list(range(W.N_LAYERS)) if layers is None else layers
        if cache_layers is None:
            cache_layers = int(os.environ.get("QWEN38_REF_CACHE_LAYERS", "8"))
        self.cache_layers = cache_layers
        self.p = collections.OrderedDict()  # layer -> weights as torch, LRU
        self.reset()

    def reset(self):
        self.pos = 0
        self.kv = {}  # attention layer -> (K [L, NKV, DH], V)
        self.conv = {}  # GDN layer -> last CONV - 1 inputs [CONV - 1, QKV]
        self.state = {}  # GDN layer -> S [NVH, DK, DV]

    def weights(self, l):
        if l in self.p:
            self.p.move_to_end(l)
            return self.p[l]
        w = self.w
        mix = w.attn(l) if W.is_attention(l) else w.gdn(l)
        p = dict(
            mix=[t(a) for a in mix],
            norms=[t(a) for a in w.norms(l)],
            mlp=[t(a) for a in w.mlp(l)],
        )
        if self.cache_layers > 0:
            self.p[l] = p
            while len(self.p) > self.cache_layers:
                self.p.popitem(last=False)
        return p

    def gdn(self, l, x, p):
        qkv_w, z_w, out_w, al_w, be_w, conv_w, norm_w, a, dt = p["mix"]
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

    def attention(self, l, x, p):
        q_w, g_w, k_w, v_w, o_w, qn, kn = p["mix"]
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

    @staticmethod
    def mlp(x, p):
        gate, up, down = p["mlp"]
        return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T

    def layer(self, l, x, collect=None):
        p = self.weights(l)
        n1, n2 = p["norms"]
        h = rmsnorm(x, n1)
        mix = self.attention(l, h, p) if W.is_attention(l) else self.gdn(l, h, p)
        x = x + mix
        out = x + self.mlp(rmsnorm(x, n2), p)
        if collect is not None:
            collect[l] = dict(mix=mix, out=out)
        return out

    def lm_head(self, h):
        """Logits [VOCAB] of the final-normed row h [D]."""
        return torch.cat(
            [
                t(self.w.lm_head_rows(r, min(r + LM_SLICE, W.VOCAB))) @ h
                for r in range(0, W.VOCAB, LM_SLICE)
            ]
        )

    @torch.no_grad()
    def forward(self, ids, collect=None, lm=True):
        x = t(self.w.embed(ids))
        for l in self.layers:
            x = self.layer(l, x, collect)
        self.pos += len(ids)
        if not lm:
            return x
        return self.lm_head(rmsnorm(x[-1], t(self.w.final_norm())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    w = W.Qwen38Q4nx()
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
