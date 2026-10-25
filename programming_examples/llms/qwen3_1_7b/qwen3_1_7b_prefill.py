# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-1.7B Prefill on MLIR-AIR (NPU2) — single-block (Phase 2) path.

Qwen3 diverges from LLAMA-3.2 in two ways that break the llama
`build_rms_gemms_rope_module` fusion:

  1. QK-norm: a per-head RMSNorm over head_dim is applied to Q and K AFTER
     the projection GEMM and BEFORE RoPE. RoPE linearity does NOT let us
     commute the (nonlinear) QK-norm past RoPE, so the llama RMSNorm+QKV+RoPE
     fused ELF (RoPE immediately after the GEMM) is wrong. We instead build a
     Qwen-specific 8-launch ELF that does RMSNorm + Q/K/V GEMM + per-head
     QK-norm(Q,K) + RoPE(Q,K) all on the NPU (rms_qkv_qknorm_rope).

  2. GQA dims (Qwen3-1.7B): emb_dim=2048, q_dim=n_heads*head_dim=16*128=2048,
     kv_dim=n_kv_heads*head_dim=8*128=1024, hidden=6144.
        q_proj : 2048 -> 2048   (16 heads x 128)
        k/v    : 2048 -> 1024   (8 heads x 128)
        o_proj : 2048 -> 2048   (SQUARE — q_dim == emb_dim)
     Because q_dim == emb_dim the O GEMM is square (2048x2048x2048), so the
     O-proj + FFN stage is the shared `build_o_ffn_module`. Each GEMM's
     method and tiles come from the kernel registry per shape, and they need
     not agree: at 512 and 1024 the O GEMM is drain while Gate/Up/Down are
     fused-cast.

Attention runs NPU FlashAttention by default (cpu_attn=False builds the
flash_attn ELF); CPU attention is an optional fallback (cpu_attn=True).
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Add programming_examples/ and llms/ to path for shared.* + registry imports.

from qwen3_1_7b_weights import LlamaConfig
from qwen3_1_7b_cpu_helpers import attention_reference

# programming_examples/ is published as the air_examples package rather than put
# on sys.path: every directory under it would otherwise become a top-level
# module name and shadow any installed package that shares it. Registered
# before the first air_examples import below, which needs it.
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]

from air_examples.llms.shared.infra.cache import KernelCache, Profiler

import types

# ---------------------------------------------------------------------------
# Builder 1 (FUSED): RMSNorm + Q/K/V GEMM + per-head QK-norm(Q,K) + RoPE(Q,K).
#   8-launch ELF that does the entire attention-input stage on the NPU. See
#   shared/builders/rms_qkv_qknorm_rope_multi.py.
# ---------------------------------------------------------------------------


def build_rms_qkv_qknorm_rope_module(seq_len, config):
    from air_examples.llms.shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_module as _build,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    return _build(
        seq_len,
        emb_dim,
        q_dim,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        qknorm_eps=1e-6,
    )


# ---------------------------------------------------------------------------
# Builder 2: O proj + Residual + FFN, the shared builder. It takes the O GEMM
#   as square (emb_dim x emb_dim), which holds here since q_dim == emb_dim.
# ---------------------------------------------------------------------------


def _o_ffn_scratch(seq_len, emb_dim, hidden_dim):
    """The f32 C-scratch the o_ffn ELF declares after its 15 base args: one
    (rows, cols) shape per fused-cast GEMM, in builder order (O, Gate, Up,
    Down), and their arg indices. Mirrors build_o_ffn_module's own
    alloc_gemm_scratch call, so the host passes exactly the args the ELF has."""
    from air_examples.llms.shared.builders.o_ffn_multi import o_ffn_gemm_specs

    o_spec, g_spec, d_spec = o_ffn_gemm_specs(seq_len, emb_dim, hidden_dim)
    shapes, idx = [], set()
    for spec, cols in (
        (o_spec, emb_dim),
        (g_spec, hidden_dim),
        (g_spec, hidden_dim),
        (d_spec, emb_dim),
    ):
        if spec["needs_f32_scratch"]:
            idx.add(15 + len(shapes))
            shapes.append((seq_len, cols))
    return shapes, idx


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _rms_qkv_qknorm_rope_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _o_ffn_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# Scratch-arg indices for the rms_qkv_qknorm_rope ELF (registry-driven;
# GQA -> 1 fused-cast scratch on Q). Set by compile_all_kernels so the block
# runner + preload know the fused ELF's scratch-arg layout.
_FUSED_SCRATCH_FOR = None


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=False):
    global _FUSED_SCRATCH_FOR
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(
        f"\n{'='*60}\nCompiling Qwen3 prefill kernels (seq_len={seq_len})...\n{'='*60}\n"
    )

    from air_examples.llms.shared.infra.external_kernels import compile_rope
    from air_examples.llms.shared.builders.gemm_builder import compile_gemm_objects
    from air_examples.llms.shared.builders.rms_qkv_qknorm_rope_multi import (
        rms_qkv_qknorm_rope_gemm_specs,
    )
    from air_examples.llms.shared.builders.o_ffn_multi import (
        build_o_ffn_module,
        o_ffn_gemm_specs,
    )

    assert q_dim == emb_dim, (
        f"the shared o_ffn builder takes a square O GEMM; q_dim={q_dim}, "
        f"emb_dim={emb_dim}"
    )

    # One mm.o per GEMM spec for co-linking; rope.o (head_dim=128) for the rope ELFs.
    compile_gemm_objects(
        rms_qkv_qknorm_rope_gemm_specs(seq_len, emb_dim, q_dim, kv_dim)
        + o_ffn_gemm_specs(seq_len, emb_dim, hidden_dim)
    )
    compile_rope()

    print("\n--- rms_qkv_qknorm_rope (FUSED: RMSNorm+QKV+QK-norm+RoPE, 8 launches) ---")
    fused_mod, fused_scratch = build_rms_qkv_qknorm_rope_module(seq_len, config)
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope", fused_mod, _rms_qkv_qknorm_rope_backend(verbose)
    )

    print("\n--- o_ffn (O proj + Residual + FFN) ---")
    cache.compile_and_cache(
        "o_ffn",
        build_o_ffn_module(seq_len, emb_dim, hidden_dim),
        _o_ffn_backend(verbose),
    )

    # Flash Attention (head-first, head_dim=128). Skip if using CPU fallback.
    if not cpu_attn:
        print("\n--- flash_attn (head-first FA, head_dim=128) ---")
        from air_examples.llms.shared.infra.fa_headfirst import compile_headfirst_fa

        compile_headfirst_fa(cache, seq_len, n_heads, n_kv_heads, head_dim, verbose)
    else:
        print("\n--- Skipping flash_attn (CPU attention fallback) ---")

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")


# ---------------------------------------------------------------------------
# Prefill weight pre-load (BO reuse — opt-buffer-object-reuse B1)
# ---------------------------------------------------------------------------


def _fused_qknorm_rope_call(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """Issue one rms_qkv_qknorm_rope ELF call (fused prefill attention-input).

    Used by BOTH preload_prefill_weights (warmup, x_in zeroed) and the block
    runner (x_in = real hidden). Returns the cache.load_and_run result tuple
    (output_indices=[8, 15, 16] -> v, q_roped, k_roped). The single owner of
    the fused arg layout + index sets so the warmup BO set lines up exactly.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0 x_in (dynamic)
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),  # 1 norm_w (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2 normed (inter)
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),  # 3 wq (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 4 q (inter)
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 5 wk (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 6 k (inter)
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 7 wv (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 8 v (inter/out)
        np.asarray(lw.q_norm, dtype=bfloat16).reshape(head_dim),  # 9 q_norm (static)
        np.asarray(lw.k_norm, dtype=bfloat16).reshape(head_dim),  # 10 k_norm (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 11 q_n (inter)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 12 k_n (inter)
        lut_q,  # 13 lut_q (static)
        lut_k,  # 14 lut_k (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 15 q_roped (inter/out)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 16 k_roped (inter/out)
    ]
    inter = {2, 4, 6, 8, 11, 12, 15, 16}
    nxt = 17
    for sc, cols in zip(_FUSED_SCRATCH_FOR or [], (q_dim, kv_dim, kv_dim)):
        if sc is not None:
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return cache.load_and_run(
        "rms_qkv_qknorm_rope",
        _rms_qkv_qknorm_rope_backend(verbose),
        *args,
        output_indices=[8, 15, 16],
        static_input_indices={1, 3, 5, 7, 9, 10, 13, 14},
        intermediate_indices=inter,
        bo_key=f"rms_qkv_qknorm_rope_L{layer_idx}",
        shared_nonstatic=True,
    )


def _o_ffn_call(
    cache, lw, config, seq_len, layer_idx, attn_out, x_resid, verbose=False
):
    """Issue one o_ffn ELF call (O proj + Residual + FFN).

    Used by both preload_prefill_weights (warmup, activations zeroed) and the
    block runner, so the warmup BO set lines up with the real call. Returns the
    cache.load_and_run result tuple; arg 14 is the block output.
    """
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim

    def act(cols):
        return np.zeros((seq_len, cols), dtype=bfloat16)

    args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0
        np.asarray(lw.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),  # 1 wo
        act(emb_dim),  # 2 proj
        np.asarray(x_resid, dtype=bfloat16).reshape(seq_len, emb_dim),  # 3
        act(emb_dim),  # 4 res1
        np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),  # 5 ffn_norm
        act(emb_dim),  # 6 normed2
        np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 7
        act(hidden_dim),  # 8 gate
        np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 9
        act(hidden_dim),  # 10 up
        act(hidden_dim),  # 11 swiglu
        np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),  # 12
        act(emb_dim),  # 13 down
        np.zeros(seq_len * emb_dim, dtype=bfloat16),  # 14 output
    ]
    shapes, scratch = _o_ffn_scratch(seq_len, emb_dim, hidden_dim)
    args += [np.zeros(shape, dtype=np.float32) for shape in shapes]
    return cache.load_and_run(
        "o_ffn",
        _o_ffn_backend(verbose),
        *args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14} | scratch,
        bo_key=f"o_ffn_L{layer_idx}",
        shared_nonstatic=True,
    )


def preload_prefill_weights(weights, config, cache, seq_len, rope_lut_bf16):
    """Pre-load all prefill block weights into per-layer BOs once.

    Mirrors llama's preload_prefill_weights / IRON's prepare_runtime: a warmup
    XRT call per layer per ELF allocates the bo_key-keyed BO set and performs
    the host->device write of the *static* weight args. During the real prefill
    pass, ``static_input_indices`` then skips those weight writes (they are
    unchanged), so only the small dynamic activation inputs are re-written.

    Without this, the first (and only) prefill pass writes every weight inside
    the timed region, o_ffn's weights included.

    The warmup call layout MUST match ``run_transformer_block_qwen3`` exactly
    (same arg count, same static_input_indices / intermediate_indices, same
    bo_key) or the reused BO set would not line up at inference time.
    """
    if hasattr(weights, "_prefill_weights_preloaded"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads

    print("Pre-loading prefill block weights (per-layer BOs)...")
    profiler_enabled = cache.profiler.enabled
    cache.profiler.enabled = False

    # RoPE LUTs (seq-first, repeated per head) — same for all layers.
    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # One fused ELF: RMSNorm + Q/K/V GEMM + per-head QK-norm + RoPE.
        _fused_qknorm_rope_call(
            cache,
            lw,
            config,
            seq_len,
            lut_q,
            lut_k,
            layer_idx,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
        )

        # o_ffn: allocate + write wo/ffn_norm/w_gate/w_up/w_down ({1,5,7,9,12}).
        zeros = np.zeros((seq_len, emb_dim), dtype=bfloat16)
        _o_ffn_call(cache, lw, config, seq_len, layer_idx, zeros, zeros)

    cache.profiler.enabled = profiler_enabled
    weights._prefill_weights_preloaded = True
    print(f"  Pre-loaded {config.n_layers} prefill layers.")


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------


def run_transformer_block_qwen3(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=True,
    verbose=False,
):
    """Run one Qwen3 transformer block on NPU (kernels pre-compiled in cache)."""
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    inter = {}

    # RoPE LUTs (seq-first, repeated per head).
    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # ---- Stages A-C: one ELF = RMSNorm + Q/K/V GEMM + QK-norm(Q,K) + RoPE(Q,K).
    res = _fused_qknorm_rope_call(
        cache,
        layer_weights,
        config,
        seq_len,
        lut_q,
        lut_k,
        layer_idx,
        x_bf16,
        verbose=verbose,
    )
    v = res[8].reshape(seq_len, kv_dim)
    q_roped = res[15].reshape(seq_len, q_dim)
    k_roped = res[16].reshape(seq_len, kv_dim)
    inter["v"] = v
    inter["q_roped"] = q_roped
    inter["k_roped"] = k_roped

    # ---- Stage D: GQA attention ----
    if cpu_attn:
        # HOST cpu fallback (FP32 causal GQA reference).
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)
    else:
        # NPU head-first FlashAttention (head_dim=128). q_roped/k_roped are
        # post-QK-norm post-RoPE seq-first; v is the raw projection seq-first.
        from air_examples.llms.shared.infra.fa_headfirst import npu_fa_headfirst

        attn_out = npu_fa_headfirst(
            cache,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(k_roped),
            np.ascontiguousarray(v),
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            verbose=verbose,
        )
    inter["attn_out"] = attn_out

    # ---- Stage E: O proj + Residual + FFN ----
    results = _o_ffn_call(
        cache, layer_weights, config, seq_len, layer_idx, attn_out, x_bf16, verbose
    )
    output_bf16 = results[14].reshape(seq_len, emb_dim)
    inter["ffn_out"] = output_bf16
    return output_bf16, inter
