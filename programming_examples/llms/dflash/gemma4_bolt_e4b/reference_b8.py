# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU reference for the eight-token decoder: gemma4_e2b_q4nx_weights's
forward over the first `layers` layers, keeping every layer boundary and
every token's logits (no softcap)."""

import numpy as np

from weights_b8 import gw


def forward(qm, ids, layers, logits=True):
    """xs [layers + 1, T, D] (xs[0] the embeddings) and logits [T, VOCAB]."""
    ids = [int(t) for t in ids]
    T = len(ids)
    x = qm.embed_rows("model.embed_tokens.weight", ids) * gw.EMBED_SCALE
    pli_all = gw.per_layer_inputs(qm, ids, x)
    freqs = qm.rope_freqs()
    xs, kv = [x], {}
    for L in range(layers):
        w, nm, pw = qm.layer_weights(L), qm.layer_norms(L), qm.layer_ple(L)
        dh = gw.head_dim(L)
        r = x
        x1 = gw._rmsnorm(x, nm["input"])
        q = gw._rmsnorm((x1 @ w["q"].T).reshape(T, gw.N_Q_HEADS, dh), nm["q_norm"])
        rope = [gw.rope_lut(t, L, rope_freqs=freqs) for t in range(T)]
        q = np.stack([gw.apply_rope(q[t], *rope[t]) for t in range(T)])
        if gw.owns_kv(L):
            k = (x1 @ w["k"].T).reshape(T, gw.N_KV_HEADS, dh)
            v = gw._rmsnorm((x1 @ w["v"].T).reshape(T, gw.N_KV_HEADS, dh), None)
            k = gw._rmsnorm(k, nm["k_norm"])
            k = np.stack([gw.apply_rope(k[t], *rope[t]) for t in range(T)])
            kv[L] = (k, v)
        k, v = kv[gw.kv_source_layer(L)]
        grp = gw.N_Q_HEADS // gw.N_KV_HEADS
        win = gw.SLIDING_WINDOW if gw.is_sliding(L) else 0
        mask = gw._sliding_mask(T, T, win)
        o = np.empty((T, gw.N_Q_HEADS, dh), np.float32)
        for h in range(gw.N_Q_HEADS):
            s = q[:, h] @ k[:, h // grp].T * gw.ATTN_SCALE + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o[:, h] = s @ v[:, h // grp]
        o1 = r + gw._rmsnorm(o.reshape(T, -1) @ w["o"].T, nm["post_attn"])
        h = gw._rmsnorm(o1, nm["pre_ffn"])
        h = gw._gelu_tanh(h @ w["gate"].T) * (h @ w["up"].T)
        o2 = o1 + gw._rmsnorm(h @ w["down"].T, nm["post_ffn"])
        g = gw._gelu_tanh(o2 @ pw["inp_gate"].T) * pli_all[:, L, :]
        x = (o2 + gw._rmsnorm(g @ pw["per_layer_projection"].T, nm["post_ple"])) * nm[
            "out_scale"
        ]
        xs.append(x)
    if not logits:
        return np.stack(xs), None
    xf = gw._rmsnorm(x, qm.globals()["final_norm"])
    out = np.empty((T, gw.VOCAB), np.float32)
    for r0 in range(0, gw.VOCAB, 16384):
        r1 = min(r0 + 16384, gw.VOCAB)
        out[:, r0:r1] = xf @ qm.lm_head_rows(r0, r1).T
    return np.stack(xs), out
