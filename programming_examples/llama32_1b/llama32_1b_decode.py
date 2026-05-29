# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache.
Runs prefill first to populate KV cache, then decodes token-by-token.

Usage:
    cd build_peano
    python3 ../llama32_1b_decode.py --compile-only
    python3 ../llama32_1b_decode.py --run-only --n-tokens 10 --profile
    python3 ../llama32_1b_decode.py --run-only --n-tokens 1 --verify
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama32_1b_weights import LlamaConfig
from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import (
    RGR_BACKEND,
    OGF_BACKEND,
    LM_GEMV_BACKEND,
)

# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------


def compile_decode_kernels(cache, config):
    """Compile the 3 merged decode kernels."""
    from kernel_builder.external_kernels import compile_all_external_kernels

    compile_all_external_kernels(head_dim=config.head_dim)

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling decode kernels (2-call merged pipeline)...")
    print(f"{'='*60}\n")

    # 1. rms_gemv_rope: RMSNorm + QKV GEMV + RoPE Q+K (6 launches, 13 args)
    from multi_launch_builder.rms_gemv_rope_multi import (
        build_rms_gemv_rope_module,
    )

    cache.compile_and_cache(
        "rms_gemv_rope",
        build_rms_gemv_rope_module(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim),
        {"verbose": cache.verbose, **RGR_BACKEND},
    )

    # 2. o_gemv_ffn: O GEMV + Add + RMSNorm + Gate/Up GEMV + SiLU*mul
    #                + Down GEMV + Add (8 launches, 15 args)
    from multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module

    cache.compile_and_cache(
        "o_gemv_ffn",
        build_o_gemv_ffn_module(emb_dim, hidden_dim),
        {"verbose": cache.verbose, **OGF_BACKEND},
    )

    # 3. LM Head GEMV multi-launch: 8-partition GEMV in one ELF
    from multi_launch_builder.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )

    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(emb_dim),
        {"verbose": cache.verbose, **LM_GEMV_BACKEND},
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(
    q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim
):
    """Single-query attention with KV cache.

    Args:
        q: (emb_dim,) — query vector for current token
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys [0:current_pos+1]
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values [0:current_pos+1]
        current_pos: current token position (0-indexed)
        n_heads: number of Q heads (32)
        n_kv_heads: number of KV heads (8)
        head_dim: head dimension (64)

    Returns:
        attn_out: (emb_dim,) — attention output
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)  # (n_kv, seq, hd)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale  # (seq,)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]  # (hd,)

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
):
    """Run one transformer block for a single decode token.

    Args:
        x_bf16: (emb_dim,) input token embedding
        layer_weights: LayerWeights for this layer
        cache: KernelCache
        config: LlamaConfig
        k_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's K cache
        v_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's V cache
        current_pos: current token position
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT

    Returns:
        output: (emb_dim,) — block output.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = None  # Set by caller via layer_weights._layer_idx
    if hasattr(layer_weights, "_layer_idx"):
        layer_idx = layer_weights._layer_idx

    def _run(name, backend, *inputs, static_indices=None, **kwargs):
        # Per-layer BO key: same XRT context, separate BOs for weight isolation
        bk = (
            f"{name}_L{layer_idx}" if static_indices and layer_idx is not None else None
        )
        return cache.load_and_run(
            name,
            backend,
            *inputs,
            bo_key=bk,
            static_input_indices=static_indices,
            **kwargs,
        )

    # --- Call 1: rms_gemv_rope (6 launches, 13 args) ---
    # RMSNorm + Q/K/V GEMV + RoPE Q + RoPE K
    x_in = x_bf16.flatten().astype(bfloat16)
    w_norm = layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16)
    normed_buf = np.zeros(emb_dim, dtype=bfloat16)
    wq = layer_weights._wq_t
    q_buf = np.zeros(emb_dim, dtype=bfloat16)
    wk = layer_weights._wk_t
    k_buf = np.zeros(kv_dim, dtype=bfloat16)
    wv = layer_weights._wv_t
    v_buf = np.zeros(kv_dim, dtype=bfloat16)

    # RoPE LUT for current position
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, 64)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)
    q_roped_buf = np.zeros(emb_dim, dtype=bfloat16)
    k_roped_buf = np.zeros(kv_dim, dtype=bfloat16)

    results = _run(
        "rms_gemv_rope",
        RGR_BACKEND,
        x_in,  # arg0
        w_norm,  # arg1
        normed_buf,  # arg2 (intermediate)
        wq,  # arg3 (static)
        q_buf,  # arg4 (intermediate)
        wk,  # arg5 (static)
        k_buf,  # arg6 (intermediate)
        wv,  # arg7 (static)
        v_buf,  # arg8 (intermediate/output)
        lut_q,  # arg9
        lut_k,  # arg10
        q_roped_buf,  # arg11 (intermediate/output)
        k_roped_buf,  # arg12 (intermediate/output)
        output_indices=[8, 11, 12],
        static_indices={3, 5, 7},
        intermediate_indices={2, 4, 6, 8, 11, 12},
    )
    v = results[8].astype(bfloat16)
    q_roped = results[11].reshape(n_heads, head_dim).astype(bfloat16)
    k_roped = results[12].reshape(n_kv_heads, head_dim).astype(bfloat16)

    # Update KV cache
    k_cache_layer[:, current_pos, :] = k_roped
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU Attention ---
    # Single-query attention against the growing K/V cache. CPU-side because
    # at head_dim=64 the NPU FA kernel's per-call overhead dominates the
    # single-query workload.
    with cache.profiler.time_cpu("decode_attention_cpu"):
        attn_out = decode_attention_cpu(
            q_roped.flatten(),
            k_cache_layer,
            v_cache_layer,
            current_pos,
            n_heads,
            n_kv_heads,
            head_dim,
        )

    # --- Call 2: o_gemv_ffn (8 launches, 15 args) ---
    # O GEMV + Add + RMSNorm + Gate/Up GEMV + SiLU*mul + Down GEMV + Add
    wo = layer_weights._wo_t
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = x_bf16.flatten().astype(bfloat16)
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_norm2 = layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16)
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_gate = layer_weights._wgate_t
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_up = layer_weights._wup_t
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_down = layer_weights._wdown_t
    down_buf = np.zeros(emb_dim, dtype=bfloat16)
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    results = _run(
        "o_gemv_ffn",
        OGF_BACKEND,
        wo,  # arg0 (static)
        attn_out,  # arg1
        proj_buf,  # arg2 (intermediate)
        x_residual,  # arg3
        res1_buf,  # arg4 (intermediate)
        w_norm2,  # arg5
        normed2_buf,  # arg6 (intermediate)
        w_gate,  # arg7 (static)
        gate_buf,  # arg8 (intermediate)
        w_up,  # arg9 (static)
        up_buf,  # arg10 (intermediate)
        swiglu_buf,  # arg11 (intermediate)
        w_down,  # arg12 (static)
        down_buf,  # arg13 (intermediate)
        output_buf,  # arg14 (intermediate/output)
        output_indices=[14],
        static_indices={0, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
    )
    return results[14].astype(bfloat16)
