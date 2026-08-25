# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU attention fallback, for bringing the prefill up without the NPU FA ELF.

Not on the production path: `run_attn_block(..., cpu_attn=True)` uses it to
separate "the projection front end is wrong" from "FlashAttention is wrong",
which is otherwise one undifferentiated cosine.
"""

import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """Causal grouped-query attention in f32.

    LFM2's attention is plain causal GQA -- no sliding window, no logit
    softcapping, no attention sinks. QK-norm is NOT applied here: it belongs to
    the projection stage and must already be baked into `q` and `k`, along with
    RoPE.

    Args:
        q: (seq_len, n_heads * head_dim)    -- projected, QK-normed, RoPE'd.
        k: (seq_len, n_kv_heads * head_dim) -- projected, QK-normed, RoPE'd.
        v: (seq_len, n_kv_heads * head_dim) -- projected.
        n_heads: query heads (32 for LFM2-1.2B).
        n_kv_heads: key/value heads (8 for LFM2-1.2B).

    Returns:
        (seq_len, n_heads * head_dim) attention output, f32.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads

    # (seq, heads, head_dim) -> (heads, seq, head_dim)
    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)

    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        scores = q[h] @ k[kv_idx].T * scale + causal_mask
        out_heads[h] = _softmax(scores, axis=-1) @ v[kv_idx]

    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
