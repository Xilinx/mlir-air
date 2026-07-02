# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small NumPy CPU helpers for Qwen2.5-1.5B production prefill/decode + verify.

Mirrors llama32_1b_cpu_helpers.py. Qwen2.5 has QKV bias (no QK-norm).
Kept helpers: rms_norm, attention_reference, softmax. The QKV bias is fused into
the NPU attention-input ELFs (rms_qkv_bias_rope; see prefill/decode) and applied
on-device, so no dedicated host-side bias helper is needed here.
"""

import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """RMS norm; Qwen2.5 uses eps=1e-6 (rms_norm_eps)."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """GQA attention with causal mask (F32). q/k already RoPE'd (and bias-added)."""
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
        scores = q[h] @ k[kv_idx].T * scale + causal_mask
        probs = softmax(scores, axis=-1)
        out_heads[h] = probs @ v[kv_idx]
    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
