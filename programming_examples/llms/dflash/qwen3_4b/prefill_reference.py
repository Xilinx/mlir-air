# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Prompt reference extracted from pr1942-port 67831a05 dflash_phase2_device.py.
# Uses the current-main Qwen weight reader and pure NumPy arithmetic; no NPU imports.
import numpy as np


def forward_prompt_with_taps(model, prompt_ids, tap_slots):
    """qwen3_8b_q4nx_weights.forward_prompt, extended to also return the
    tapped hidden states (output-after-k-layers, k in tap_slots) at every
    prompt position. Duplicated (not monkeypatched) so the shared reference
    function stays untouched for every other example that relies on it."""
    from ml_dtypes import bfloat16
    from qwen3_8b_q4nx_weights import (
        _bf,
        _rmsnorm,
        _apply_rope_halfsplit,
        _silu,
        NUM_LAYERS,
        DK,
        N_Q_HEADS,
        N_KV_HEADS,
        Q_PER_KV,
        DH,
        DQ,
        ROPE_THETA,
        ATTN_SCALE,
        _DH_2,
    )

    embed, final_norm, lm_head = model.embed_norm_lmhead()
    ids = np.asarray(prompt_ids, dtype=np.int64)
    P = ids.shape[0]
    x = _bf(embed[ids].astype(np.float32))

    Kc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    Vc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    pos = np.arange(P)
    taps = {s: None for s in tap_slots}
    if 0 in taps:
        taps[0] = x.copy()

    for L in range(NUM_LAYERS):
        w = model.layer_weights(L)
        n_in, n_pa = model.layer_rms(L)
        qn, kn = model.layer_qk_norm(L)

        residual = x
        h = _rmsnorm(x, n_in)
        q = _bf(h @ w["q"]).reshape(P, N_Q_HEADS, DH).astype(np.float32)
        k = _bf(h @ w["k"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)
        v = _bf(h @ w["v"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)

        inv = 1.0 / (ROPE_THETA ** (np.arange(_DH_2) / _DH_2))
        ang = pos[:, None] * inv[None, :]
        cos = np.cos(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        q = _apply_rope_halfsplit(_rmsnorm(q, qn), cos, sin)
        k = _apply_rope_halfsplit(_rmsnorm(k, kn), cos, sin)
        Kc[L] = _bf(k).reshape(P, DK)
        Vc[L] = _bf(v).reshape(P, DK)

        o = np.zeros((P, N_Q_HEADS, DH), np.float32)
        for hq in range(N_Q_HEADS):
            hk = hq // Q_PER_KV
            scores = (q[:, hq, :] @ k[:, hk, :].T) * ATTN_SCALE
            mask = pos[None, :] > pos[:, None]
            scores = np.where(mask, -1e30, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            e = np.exp(scores)
            attn = e / e.sum(axis=-1, keepdims=True)
            o[:, hq, :] = attn @ v[:, hk, :]
        attn_out = _bf(o.reshape(P, DQ) @ w["o"])
        x = _bf(residual + attn_out)

        residual = x
        h2 = _rmsnorm(x, n_pa)
        act = _bf(_silu(_bf(h2 @ w["gate"])) * _bf(h2 @ w["up"]))
        x = _bf(residual + _bf(act @ w["down"]))

        if (L + 1) in taps:
            taps[L + 1] = x.copy()

    xf = _rmsnorm(x, final_norm)
    logits = (xf.astype(np.float32) @ lm_head.astype(np.float32).T).astype(np.float32)
    return Kc, Vc, logits, taps
