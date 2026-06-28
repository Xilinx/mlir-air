# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small NumPy CPU helpers shared by production prefill/decode + verify.

Mirrors llama32_1b_cpu_helpers.py. Kept helpers are the ones production
prefill/decode import at runtime:

  - rms_norm           : LM-head GEMV final-norm + (Qwen3) QK-norm building block.
  - qk_norm_per_head   : Qwen3 per-head RMSNorm over head_dim, applied to Q and K
                         after projection and before RoPE.
  - attention_reference: prefill cpu_attn=True fallback (full GQA attention in F32).
  - softmax            : used by attention_reference.
"""

import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight.

    Qwen3 uses eps=1e-6 (rms_norm_eps in config.json).
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def qk_norm_per_head(x, weight, n_heads, head_dim, eps=1e-6):
    """Qwen3 per-head RMSNorm.

    Args:
        x: (seq, n_heads*head_dim) projected Q or K (pre-RoPE).
        weight: (head_dim,) q_norm or k_norm weight.
    Returns:
        (seq, n_heads*head_dim) normed, same layout.
    """
    x = np.asarray(x, dtype=np.float32)
    seq = x.shape[0]
    xh = x.reshape(seq, n_heads, head_dim)
    rms = np.sqrt(np.mean(xh * xh, axis=-1, keepdims=True) + eps)
    xh = (xh / rms) * np.asarray(weight, dtype=np.float32)
    return xh.reshape(seq, n_heads * head_dim)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """Multi-head GQA attention with causal mask (F32). q/k already RoPE'd."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads

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

    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
