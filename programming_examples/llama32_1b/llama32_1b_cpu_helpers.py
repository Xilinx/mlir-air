# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small NumPy CPU helpers shared by production prefill/decode + verify.

This file used to be a full F32 CPU forward-pass implementation of the model
(plus a standalone `--verify` CLI that compared the F32 forward against HF
transformers F32). With the verify subsystem rewritten to compare directly
against HF transformers in bf16 (see verify/), that whole F32 reference
chain became redundant. What is kept here is the small set of NumPy helpers
that production still imports:

  - rms_norm           : LM-head GEMV final-norm (inference.py prefill end,
                         and every decode step).
  - attention_reference: prefill cpu_attn=True fallback (full GQA attention
                         in F32 on host; used when the NPU FlashAttention
                         kernel is unavailable for the configured head_dim).
  - softmax            : kept because attention_reference uses it; not
                         imported anywhere else.
"""

import numpy as np


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight.

    Args:
        x: (M, N) input array in F32.
        weight: (N,) learned scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        (M, N) normalized and scaled array in F32.
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def softmax(x, axis=-1):
    """Numerically stable softmax (used by attention_reference)."""
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """Multi-head attention with Grouped Query Attention (GQA), causal mask.

    Args:
        q: (seq_len, n_heads * head_dim) -- already projected and RoPE'd.
        k: (seq_len, n_kv_heads * head_dim) -- already projected and RoPE'd.
        v: (seq_len, n_kv_heads * head_dim) -- already projected.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (for GQA).

    Returns:
        (seq_len, n_heads * head_dim) attention output (F32).
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads

    # Reshape to per-head views: (seq, n_*_heads, head_dim) -> (n_*_heads, seq, head_dim)
    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)

    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        scores = q[h] @ k[kv_idx].T * scale
        scores = scores + causal_mask
        probs = softmax(scores, axis=-1)
        out_heads[h] = probs @ v[kv_idx]

    # (n_heads, seq, head_dim) -> (seq, n_heads * head_dim)
    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
