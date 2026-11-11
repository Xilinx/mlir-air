# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-Coder-30B-A3B text model on the CPU, float32, on the bundle's
dequantized weights: the reference the NPU is checked against. Experts are
dequantized as the router picks them.

Reference(w).forward(ids) runs the new tokens `ids` after those already seen
(the KV cache carries over) and returns the logits of the last one.
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import qwen3_coder_30b_a3b_q4k_weights as W


def rmsnorm(x, w, eps=W.EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):
    """Rotate-half RoPE over the whole head of x [L, heads, DH]."""
    half = W.DH // 2
    inv = 1.0 / (W.THETA ** (torch.arange(0, W.DH, 2).float() / W.DH))
    f = pos.float()[:, None] * inv[None]  # [L, half]
    cos, sin = f.cos()[:, None], f.sin()[:, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


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
        self.kv = {}  # layer -> (K [L, NKV, DH] after norm and RoPE, V)

    def weights(self, l):
        if l not in self.p:
            w = self.w
            self.p[l] = dict(
                attn=[t(a) for a in w.attn(l)],
                norms=[t(a) for a in w.norms(l)],
                router=t(w.router(l)),
            )
        return self.p[l]

    def attention(self, l, x):
        q_w, k_w, v_w, o_w, qn, kn = self.weights(l)["attn"]
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
        mask = torch.arange(k.shape[0])[None] > pos[:, None]  # key after query
        s = s.masked_fill(mask[None], float("-inf"))
        o = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv).reshape(L, -1)
        return o @ o_w.T

    def moe(self, l, x):
        """The MoE block; also returns each row's (experts, weights)."""
        probs = (x @ self.weights(l)["router"].T).softmax(-1)
        top, idx = probs.topk(W.TOP_K, -1)
        top = top / top.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for r in range(x.shape[0]):
            for e, s in zip(idx[r].tolist(), top[r]):
                eg, eu, ed = (t(a) for a in self.w.expert(l, e))
                out[r] += s * ((F.silu(x[r] @ eg.T) * (x[r] @ eu.T)) @ ed.T)
        return out, (idx, top)

    def layer(self, l, x, collect=None):
        n1, n2 = self.weights(l)["norms"]
        a = self.attention(l, rmsnorm(x, n1))
        x = x + a
        m, route = self.moe(l, rmsnorm(x, n2))
        if collect is not None:
            collect[l] = dict(mix=a, route=route, out=x + m)
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
    w = W.Qwen3CoderQ4k()
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
