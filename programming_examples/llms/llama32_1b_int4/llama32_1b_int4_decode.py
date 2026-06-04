# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B int4-AWQ Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache, weights in int4-AWQ.

Mirror of `../llama32_1b/llama32_1b_decode.py` (bf16 decode) — always int4.
Compiles the two merged decode ELFs (`rms_qkv_int4_rope`, `o_gemv_ffn_int4`)
plus the shared `lm_head_gemv` ELF, then runs per-token via the CPU
attention helper.

Weights are read from per-layer packed-BO attrs (`_wq_packed`/...) set up
by `llama32_1b_int4_weights.load_weights_awq`.

Usage (standalone decode-only smoke):
    cd build_peano
    python3 ../llama32_1b_int4_decode.py --compile-only
For full e2e (prefill + decode + chat), use `llama32_1b_int4_inference.py`.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROG_EXAMPLES = os.path.dirname(_THIS_DIR)
_LLAMA_BF16 = os.path.join(_PROG_EXAMPLES, "llama32_1b")
# Insert in reverse priority order so the LAST insert wins sys.path[0].
# `_THIS_DIR` MUST end up first so `multi_launch_builder` resolves to the
# int4 dir's package (the bf16 sibling has a same-named one). Don't use
# the "already present" skip — Python auto-inserts the script's dir, and
# skipping would leave _THIS_DIR below _LLAMA_BF16.
for _p in (_PROG_EXAMPLES, _LLAMA_BF16, _THIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from llama32_1b_weights import LlamaConfig  # noqa: E402
from llama_kernel_builder.cache import KernelCache  # noqa: E402
from llama_kernel_builder.backend_presets import (  # noqa: E402
    RGR_INT4_BACKEND,
    OGF_INT4_BACKEND,
    LM_GEMV_BACKEND,
)

# Cache of dead-ABI placeholders passed to o_gemv_ffn_int4. Reallocating
# the 32 MB hidden×emb buffer per call costs ~15 ms/token.
_DEAD_PLACEHOLDERS = {}


def _dead_buf(shape, dtype=bfloat16):
    key = (shape if isinstance(shape, tuple) else (shape,), dtype)
    buf = _DEAD_PLACEHOLDERS.get(key)
    if buf is None:
        buf = np.zeros(shape, dtype=dtype)
        _DEAD_PLACEHOLDERS[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------


def compile_decode_kernels(cache, config):
    """Compile the 3 int4 decode kernels (rms_qkv_int4_rope, o_gemv_ffn_int4,
    lm_head_gemv)."""
    from llama_kernel_builder.external_kernels import compile_all_external_kernels

    compile_all_external_kernels(head_dim=config.head_dim, quant="awq")

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling int4 decode kernels...")
    print(f"{'='*60}\n")

    from multi_launch_builder.rms_qkv_int4_rope_multi import (
        build_rms_qkv_int4_rope_module,
    )

    cache.compile_and_cache(
        "rms_qkv_int4_rope",
        build_rms_qkv_int4_rope_module(
            emb_dim=emb_dim,
            kv_dim=kv_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        ),
        {"verbose": cache.verbose, **RGR_INT4_BACKEND},
    )

    from multi_launch_builder.o_gemv_ffn_int4_multi import (
        build_o_gemv_ffn_int4_module,
    )

    cache.compile_and_cache(
        "o_gemv_ffn_int4",
        build_o_gemv_ffn_int4_module(emb_dim=emb_dim, hidden_dim=hidden_dim),
        {"verbose": cache.verbose, **OGF_INT4_BACKEND},
    )

    # lm_head_gemv is the same module as bf16 — load it directly from the
    # bf16 sibling by file path. Can't use `from multi_launch_builder.X` here
    # because Python has already pinned `multi_launch_builder` to the int4
    # dir's package (we imported the int4 stitchers above), and lm_head_gemv
    # only exists in the bf16 dir's namesake. (LM head stays bf16 — AMD's
    # AWQ checkpoint keeps it un-quantized, and we tie to embed_table.)
    import importlib.util

    _lm_head_path = os.path.join(
        _LLAMA_BF16, "multi_launch_builder", "lm_head_gemv_multi.py"
    )
    _spec = importlib.util.spec_from_file_location(
        "_bf16_lm_head_gemv_multi", _lm_head_path
    )
    _lm_head_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_lm_head_mod)
    build_lm_head_gemv_module = _lm_head_mod.build_lm_head_gemv_module

    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(emb_dim),
        {"verbose": cache.verbose, **LM_GEMV_BACKEND},
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache) — identical to bf16 sibling.
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
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block (int4)
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
    """Run one int4 transformer block for a single decode token.

    Reads packed weights from `layer_weights._wq_packed` / `_wk_packed` /
    `_wv_packed` / `_wo_packed` / `_wgateup_packed` / `_wdown_packed` plus
    the shared `_packed_rms_buf` (preloaded by the inference driver).

    Returns:
        output: (emb_dim,) — block output (bf16).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)

    def _run(name, backend, *inputs, static_indices=None, **kwargs):
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

    # --- Call 1: rms_qkv_int4_rope (6 launches, 13 args) ---
    # Same ABI as bf16's rms_gemv_rope; slots 3/5/7 are packed-uint8 BOs.
    x_in = x_bf16.flatten().astype(bfloat16)
    w_norm = layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16)
    normed_buf = np.zeros(emb_dim, dtype=bfloat16)
    wq = layer_weights._wq_packed
    q_buf = np.zeros(emb_dim, dtype=bfloat16)
    wk = layer_weights._wk_packed
    k_buf = np.zeros(kv_dim, dtype=bfloat16)
    wv = layer_weights._wv_packed
    v_buf = np.zeros(kv_dim, dtype=bfloat16)

    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, head_dim)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)
    q_roped_buf = np.zeros(emb_dim, dtype=bfloat16)
    k_roped_buf = np.zeros(kv_dim, dtype=bfloat16)

    results = _run(
        "rms_qkv_int4_rope",
        RGR_INT4_BACKEND,
        x_in,  # arg0
        w_norm,  # arg1
        normed_buf,  # arg2 (intermediate)
        wq,  # arg3 (static, packed-i8)
        q_buf,  # arg4 (intermediate)
        wk,  # arg5 (static, packed-i8)
        k_buf,  # arg6 (intermediate)
        wv,  # arg7 (static, packed-i8)
        v_buf,  # arg8 (intermediate/output)
        lut_q,  # arg9
        lut_k,  # arg10
        q_roped_buf,  # arg11 (intermediate/output)
        k_roped_buf,  # arg12 (intermediate/output)
        output_indices=[8, 11, 12],
        # arg1 (w_norm) is per-layer constant and pre-written by
        # _preload_decode_weights — include it in the static set so the
        # decode loop skips the per-token re-sync. arg6 (`packed_rms_buf`)
        # in the bf16 sibling is similar.
        static_indices={1, 3, 5, 7},
        intermediate_indices={2, 4, 6, 8, 11, 12},
    )
    v = results[8].astype(bfloat16)
    q_roped = results[11].reshape(n_heads, head_dim).astype(bfloat16)
    k_roped = results[12].reshape(n_kv_heads, head_dim).astype(bfloat16)

    k_cache_layer[:, current_pos, :] = k_roped
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

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

    # --- Call 2: o_gemv_ffn_int4 (3 stages, 15-arg ABI) ---
    # Same arg slots as bf16; slots 0/7/12 hold packed-uint8 BOs.
    wo = layer_weights._wo_packed
    x_residual = x_bf16.flatten().astype(bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_down = layer_weights._wdown_packed
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    arg6 = layer_weights._packed_rms_buf  # [2, emb_dim] bf16
    arg7 = layer_weights._wgateup_packed
    z_emb = _dead_buf(emb_dim)
    z_hidden = _dead_buf(hidden_dim)
    z_hidden_emb = _dead_buf((hidden_dim, emb_dim))

    results = _run(
        "o_gemv_ffn_int4",
        OGF_INT4_BACKEND,
        wo,  # arg0  wo               (static, packed-i8)
        attn_out,  # arg1  attn_out         (input)
        z_emb,  # arg2  (dead)
        x_residual,  # arg3  x_residual       (input)
        z_emb,  # arg4  (dead)
        z_emb,  # arg5  (dead)
        arg6,  # arg6  packed RMS input (static)
        arg7,  # arg7  w_gateup         (static, packed-i8)
        z_hidden,  # arg8  (dead)
        z_hidden_emb,  # arg9  (dead)
        z_hidden,  # arg10 (dead)
        swiglu_buf,  # arg11 swiglu           (intermediate)
        w_down,  # arg12 wdown            (static, packed-i8)
        z_emb,  # arg13 (dead)
        output_buf,  # arg14 output           (output)
        output_indices=[14],
        static_indices={0, 6, 7, 12},
        intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
    )
    return results[14].astype(bfloat16)


# ---------------------------------------------------------------------------
# Standalone smoke entry: --compile-only
# ---------------------------------------------------------------------------


def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    config = LlamaConfig()
    cache = KernelCache(cache_dir=args.cache_dir, verbose=args.verbose)
    compile_decode_kernels(cache, config)


if __name__ == "__main__":
    _main()
