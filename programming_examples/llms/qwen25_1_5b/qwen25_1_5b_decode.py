# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-1.5B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache. Mirrors the Phase-2
prefill (qwen25_1_5b_prefill.py) deltas, applied to the M=1 GEMV path:

  1. QKV BIAS fused on-device (Qwen2 family, attention_bias=True; NOT QK-norm —
     that is Qwen3). A single fused rms_qkv_bias_rope_gemv ELF does
     RMSNorm + Q/K/V GEMV + per-channel bias-add(Q,K,V) + RoPE(Q,K); the bias
     (bq/bk/bv) is a kernel INPUT applied inside the NPU ELF, not on the host.
     HF Qwen2 order: proj -> +bias -> RoPE(Q,K) -> attention; V is bias-added
     and used directly.

  2. Dims: emb=1536, q_dim=1536 (12*128), kv_dim=256 (2*128),
     hidden=8960, head_dim=128. o_proj is SQUARE (q_dim==emb_dim==1536).
     The fused llama/qwen3 `o_gemv_ffn` ELF stitches O-GEMV + SwiGLU +
     Down-GEMV via the cascade `matvec_2tile_add` / `matvec_swiglu_rms`
     builders, whose K-chunk requires K % 512 == 0 — which qwen2.5's K=8960
     (Down) does NOT satisfy. Rather than re-derive cascade
     tile params for non-512 K, decode uses STANDALONE per-projection GEMV ELFs
     (the plain `matvec` builder, 3-arg A@B) for O / Gate / Up / Down, exactly
     the (kernel, shape, tile) configs Phase 1 verified PASS, and does the
     residual add / FFN RMSNorm / SwiGLU on the host (single-token, cheap,
     exact). This trades a few extra XRT dispatches for guaranteed correctness;
     Phase 5 can fuse later.

  3. LM-head vocab = 151936. Per-partition GEMV broadcasts the K=emb input
     vector with a push_queue repeat_count ~= n_part/32 - 1, capped at [0:255]
     → n_part <= 8192. Use 19 partitions × 8192 (19*8192 = 155648 >= 151936;
     last partition zero-padded, logits truncated to vocab on host). Same as
     Qwen3-0.6B (shared vocab 151936).

Decode attention is CPU (decode_attention_cpu); single-token attention is
trivial on host (head_dim=128 FA risk is irrelevant at M=1).

Phase-1-verified decode GEMV tile configs (see docs phase1_kernels.md):
  Q/O   1536×1536 : tile_m=8  m_input=8  herd_m=8
  K/V   256×1536  : tile_m=8  m_input=8  herd_m=8
  Gate/Up 8960×1536 : tile_m=8 m_input=8 herd_m=8
  Down  1536×8960 : tile_m=2  m_input=2  herd_m=8   (K=8960 → L2 limit)
"""

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from qwen25_1_5b_weights import LlamaConfig
from qwen25_1_5b_cpu_helpers import rms_norm


# LM-head decode partitioning. vocab=151936. Per-partition GEMV broadcasts the
# K=emb_dim input vector with a push_queue repeat_count ~= n_part/32 - 1, capped
# at [0:255] → n_part <= 8192. 19 * 8192 = 155648 >= 151936; the final partition
# carries zero-padded rows (logits truncated to vocab on host).
_LM_N_PARTITIONS = 19
_LM_N_PART = 8192  # % 64 == 0; n_part/32 - 1 = 255 (at the repeat-count limit)

# Phase-1-verified decode GEMV tile configs (tile_m, m_input, herd_m).
_GEMV_QO = (8, 8, 8)        # 1536×1536 (Q proj, O proj)
_GEMV_KV = (8, 8, 8)        # 256×1536 (K proj, V proj)
_GEMV_GATEUP = (8, 8, 8)    # 8960×1536 (Gate proj, Up proj)
_GEMV_DOWN = (2, 2, 8)      # 1536×8960 (Down proj; K=8960 → L2-bound tile_m=2)


# ---------------------------------------------------------------------------
# Builder 1 (FUSED): RMSNorm + Q/K/V GEMV + bias-add(Q,K,V) + RoPE(Q,K), M=1.
#   One ELF (RMSNorm+QKV+bias+RoPE) for the decode attention-input stage.
# ---------------------------------------------------------------------------


def build_rms_qkv_bias_rope_gemv_module(emb_dim, q_dim, kv_dim, config, eps=1e-6):
    from shared.builders.rms_qkv_bias_rope_multi import (
        build_rms_qkv_bias_rope_gemv_module as _build,
    )

    q_tm, q_mi, q_hm = _GEMV_QO
    return _build(
        emb_dim, q_dim, kv_dim,
        config.n_heads, config.n_kv_heads, config.head_dim,
        tile_m=q_tm, m_input=q_mi, herd_m=q_hm, eps=eps,
    )


# ---------------------------------------------------------------------------
# Builder 3: standalone single-projection GEMV ELF (plain matvec, A@B).
#   Used for O / Gate / Up / Down — the residual / RMSNorm / SwiGLU around
#   them is done on the host (single token, cheap, exact). Avoids the cascade
#   builders' K % 512 == 0 requirement which qwen2.5 dims violate.
# ---------------------------------------------------------------------------


def build_gemv_module(m, k, tile_m, m_input, herd_m=8, name="gemv"):
    """Standalone GEMV ELF: C[m] = A[m,k] @ B[k]. 3-arg func.

    Func args: %arg0 A (m,k)  %arg1 B (k,)  %arg2 C (m,)

    The raw `matvec` builder names its func @matvec_bf16; for ELF output the
    backend's instance_name must match the module's func name (else the loaded
    kernel symbol `main:<instance_name>` is not found). We stitch the single
    GEMV slice through stitch_elf so the public func is renamed to `name`,
    matching the per-projection instance_name in the backend kwargs.
    """
    _mv_dir = os.path.join(
        _PROG_EXAMPLES, "matrix_vector_multiplication", "bf16"
    )
    if _mv_dir not in sys.path:
        sys.path.insert(0, _mv_dir)
    from matvec import build_module as build_gemv
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    gemv_ir = str(build_gemv(m, k, tile_m, m_input, herd_m, bfloat16, bfloat16))
    base_args = [
        FuncArg("%arg0", f"memref<{m}x{k}xbf16>"),
        FuncArg("%arg1", f"memref<{k}xbf16>"),
        FuncArg("%arg2", f"memref<{m}xbf16>"),
    ]
    # GEMV func args: {0: weight (MxK), 1: input (K,), 2: output (M,)}.
    slices = [
        KernelSlice(
            gemv_ir, "g", {0: 0, 1: 1, 2: 2},
            extern_syms={"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"},
        )
    ]
    return stitch_elf(name, base_args, slices)


# ---------------------------------------------------------------------------
# Builder 4: LM-head GEMV (19 partitions, n_part=8192 for vocab 151936).
# ---------------------------------------------------------------------------


def build_lm_head_gemv_qwen_module(emb_dim):
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    return build_lm_head_gemv_module(
        emb_dim=emb_dim,
        n_partitions=_LM_N_PARTITIONS,
        n_part=_LM_N_PART,
        tile_m=8,
        m_input=4,
        herd_m=8,
    )


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _rms_qkv_bias_rope_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_bias_rope_gemv",
    }


def _gemv_backend(verbose=False, name="gemv"):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
    }


def _lm_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
    }


# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------


def compile_decode_kernels(cache, config, verbose=False):
    """Compile the Qwen2.5 decode kernels."""
    from shared.infra.external_kernels import compile_mv, compile_rope

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}\nCompiling Qwen2.5 decode kernels...\n{'='*60}\n")

    # External .o: GEMV (mv.o), RoPE (rope.o).
    compile_mv()
    compile_rope()

    print("\n--- rms_qkv_bias_rope_gemv (FUSED: RMSNorm+QKV+bias+RoPE, M=1) ---")
    cache.compile_and_cache(
        "rms_qkv_bias_rope_gemv",
        build_rms_qkv_bias_rope_gemv_module(emb_dim, q_dim, kv_dim, config, eps=1e-6),
        _rms_qkv_bias_rope_gemv_backend(verbose),
    )

    # Standalone projection GEMVs. O proj is SQUARE (M=emb, K=q_dim=emb).
    print(f"\n--- o_gemv (O proj GEMV, {emb_dim}x{q_dim}) ---")
    o_tm, o_mi, o_hm = _GEMV_QO
    cache.compile_and_cache(
        "o_gemv", build_gemv_module(emb_dim, q_dim, o_tm, o_mi, o_hm, name="o_gemv"),
        _gemv_backend(verbose, "o_gemv"),
    )
    print(f"\n--- gate_gemv (Gate proj GEMV, {hidden_dim}x{emb_dim}) ---")
    g_tm, g_mi, g_hm = _GEMV_GATEUP
    cache.compile_and_cache(
        "gate_gemv", build_gemv_module(hidden_dim, emb_dim, g_tm, g_mi, g_hm, name="gate_gemv"),
        _gemv_backend(verbose, "gate_gemv"),
    )
    print(f"\n--- up_gemv (Up proj GEMV, {hidden_dim}x{emb_dim}) ---")
    cache.compile_and_cache(
        "up_gemv", build_gemv_module(hidden_dim, emb_dim, g_tm, g_mi, g_hm, name="up_gemv"),
        _gemv_backend(verbose, "up_gemv"),
    )
    print(f"\n--- down_gemv (Down proj GEMV, {emb_dim}x{hidden_dim}) ---")
    d_tm, d_mi, d_hm = _GEMV_DOWN
    cache.compile_and_cache(
        "down_gemv", build_gemv_module(emb_dim, hidden_dim, d_tm, d_mi, d_hm, name="down_gemv"),
        _gemv_backend(verbose, "down_gemv"),
    )

    print(f"\n--- lm_head_gemv ({_LM_N_PARTITIONS}-partition, vocab 151936) ---")
    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_qwen_module(emb_dim),
        _lm_gemv_backend(verbose),
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim):
    """Single-query GQA attention with KV cache.

    Args:
        q: (q_dim,) — RoPE'd (and bias-added) query vector for the current token.
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys (after bias+RoPE).
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values (after bias).
        current_pos: current token position (0-indexed).
    Returns:
        attn_out: (q_dim,) bfloat16.
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
# Single decode transformer block
# ---------------------------------------------------------------------------


def _fused_bias_rope_gemv_call(cache, lw, config, lut_q, lut_k, suffix, x_in, verbose=False):
    """Issue one rms_qkv_bias_rope_gemv ELF call (fused decode attention-input).

    Used by BOTH _preload_decode_weights and run_decode_block. Single owner of
    the fused arg layout. output_indices=[14,17,18] -> v_b, q_roped, k_roped.
    RoPE LUTs (15,16) are position-dependent → NOT static.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    args = [
        np.ascontiguousarray(x_in).astype(bfloat16).reshape(emb_dim),       # 0 x_in
        lw.attn_norm.reshape(emb_dim).astype(bfloat16),                     # 1 norm_w (static)
        np.zeros(emb_dim, dtype=bfloat16),                                  # 2 normed (inter)
        lw._wq_t,                                                           # 3 wq (static)
        np.zeros(q_dim, dtype=bfloat16),                                    # 4 q (inter)
        lw._wk_t,                                                           # 5 wk (static)
        np.zeros(kv_dim, dtype=bfloat16),                                   # 6 k (inter)
        lw._wv_t,                                                           # 7 wv (static)
        np.zeros(kv_dim, dtype=bfloat16),                                   # 8 v (inter)
        np.asarray(lw.bq, dtype=bfloat16).reshape(q_dim),                  # 9 bq (static)
        np.asarray(lw.bk, dtype=bfloat16).reshape(kv_dim),                 # 10 bk (static)
        np.asarray(lw.bv, dtype=bfloat16).reshape(kv_dim),                 # 11 bv (static)
        np.zeros(q_dim, dtype=bfloat16),                                    # 12 q_b (inter)
        np.zeros(kv_dim, dtype=bfloat16),                                   # 13 k_b (inter)
        np.zeros(kv_dim, dtype=bfloat16),                                   # 14 v_b (inter/out)
        np.ascontiguousarray(lut_q).astype(bfloat16),                      # 15 lut_q (NON-static)
        np.ascontiguousarray(lut_k).astype(bfloat16),                      # 16 lut_k (NON-static)
        np.zeros(q_dim, dtype=bfloat16),                                    # 17 q_roped (inter/out)
        np.zeros(kv_dim, dtype=bfloat16),                                   # 18 k_roped (inter/out)
    ]
    return cache.load_and_run(
        "rms_qkv_bias_rope_gemv",
        _rms_qkv_bias_rope_gemv_backend(verbose),
        *args,
        output_indices=[14, 17, 18],
        static_input_indices={1, 3, 5, 7, 9, 10, 11},
        intermediate_indices={2, 4, 6, 8, 12, 13, 14, 17, 18},
        bo_key=f"rms_qkv_bias_rope_gemv{suffix}" if suffix else None,
    )


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
    verbose=False,
):
    """Run one Qwen2.5 transformer block for a single decode token.

    Stages: rms_qkv_bias_rope_gemv (NPU, fused RMSNorm+QKV+bias+RoPE)
    -> KV-cache write -> CPU attention -> O GEMV (NPU) -> residual (host)
    -> FFN RMSNorm (host) -> Gate/Up GEMV (NPU) -> SwiGLU (host)
    -> Down GEMV (NPU) -> FFN residual (host).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)
    _suffix = f"_L{layer_idx}" if layer_idx is not None else None
    inter = {}

    x_in = x_bf16.flatten().astype(bfloat16)

    # RoPE LUT for the current single position.
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, head_dim)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)

    # --- Stages A-C (FUSED): one ELF = RMSNorm + Q/K/V GEMV + bias(Q,K,V)
    # + RoPE(Q,K). Output: v (bias-only), q_roped, k_roped. ---
    res = _fused_bias_rope_gemv_call(
        cache, layer_weights, config, lut_q, lut_k, _suffix, x_in, verbose
    )
    v = res[14].astype(bfloat16)
    q_roped = res[17].astype(bfloat16)
    k_roped = res[18].astype(bfloat16)
    inter["v"] = v
    inter["q_roped"] = q_roped
    inter["k_roped"] = k_roped

    # --- Update KV cache (K after bias+RoPE, V after bias) ---
    k_cache_layer[:, current_pos, :] = k_roped.reshape(n_kv_heads, head_dim)
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU attention ---
    with cache.profiler.time_cpu("decode_attention_cpu"):
        attn_out = decode_attention_cpu(
            q_roped, k_cache_layer, v_cache_layer, current_pos,
            n_heads, n_kv_heads, head_dim,
        )
    inter["attn_out"] = attn_out

    # --- Stage E: O proj GEMV + residual + FFN (RMSNorm/SwiGLU on host) ---
    ro = cache.load_and_run(
        "o_gemv",
        _gemv_backend(verbose, "o_gemv"),
        layer_weights._wo_t,                        # arg0 wo (static) (emb, q_dim)
        np.ascontiguousarray(attn_out),             # arg1 attn_out (q_dim,)
        np.zeros(emb_dim, dtype=bfloat16),          # arg2 proj (emb,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"o_gemv{_suffix}" if _suffix else None,
    )
    proj = ro[2].astype(np.float32)

    # residual 1 (host)
    res1 = proj + x_in.astype(np.float32)
    inter["res1"] = res1.astype(bfloat16)

    # FFN RMSNorm (host, eps=1e-6)
    normed2 = rms_norm(res1.reshape(1, emb_dim), layer_weights.ffn_norm, eps=1e-6).reshape(emb_dim).astype(bfloat16)

    rg = cache.load_and_run(
        "gate_gemv",
        _gemv_backend(verbose, "gate_gemv"),
        layer_weights._wgate_t,                     # arg0 w_gate (static) (hidden, emb)
        np.ascontiguousarray(normed2),              # arg1 normed2 (emb,)
        np.zeros(hidden_dim, dtype=bfloat16),       # arg2 gate (hidden,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"gate_gemv{_suffix}" if _suffix else None,
    )
    gate = rg[2].astype(np.float32)
    ru = cache.load_and_run(
        "up_gemv",
        _gemv_backend(verbose, "up_gemv"),
        layer_weights._wup_t,                       # arg0 w_up (static) (hidden, emb)
        np.ascontiguousarray(normed2),              # arg1 normed2 (emb,)
        np.zeros(hidden_dim, dtype=bfloat16),       # arg2 up (hidden,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"up_gemv{_suffix}" if _suffix else None,
    )
    up = ru[2].astype(np.float32)
    # SwiGLU (host)
    swig = ((gate / (1.0 + np.exp(-gate))) * up).astype(bfloat16)
    rd = cache.load_and_run(
        "down_gemv",
        _gemv_backend(verbose, "down_gemv"),
        layer_weights._wdown_t,                     # arg0 w_down (static) (emb, hidden)
        np.ascontiguousarray(swig),                 # arg1 swiglu (hidden,)
        np.zeros(emb_dim, dtype=bfloat16),          # arg2 down (emb,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"down_gemv{_suffix}" if _suffix else None,
    )
    down = rd[2].astype(np.float32)

    out = (down + res1).astype(bfloat16)
    inter["ffn_out"] = out
    return out, inter
