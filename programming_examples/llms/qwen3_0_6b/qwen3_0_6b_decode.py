# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-0.6B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache. Mirrors
llama32_1b_decode.py but applies the same two Qwen3 deltas the Phase-2
prefill handled:

  1. QK-norm: a per-head RMSNorm over head_dim on Q and K AFTER the GEMV
     projection and BEFORE RoPE. RoPE's linearity does NOT let us commute
     the (nonlinear) QK-norm past it, so we CANNOT use the llama
     `rms_gemv_rope` ELF (which fuses RoPE right after the GEMV). We instead
     build a Qwen-specific fused decode ELF that does RMSNorm + Q/K/V GEMV +
     per-head QK-norm + RoPE (M=1) entirely on the NPU
     (rms_qkv_qknorm_rope_gemv).

  2. Decoupled head_dim: n_heads*head_dim = 2048 != hidden_size = 1024.
        q_proj : 1024 -> 2048   (16 heads x 128)
        k/v    : 1024 -> 1024   (8 heads x 128)
        o_proj : 2048 -> 1024   (NOT square)
     The llama `rms_gemv_rope` asserts q_total==emb_dim; the llama
     `o_gemv_ffn` stage-1 O-GEMV is square (emb x emb). We build Qwen
     variants: the Q GEMV is M=q_dim, the O GEMV is M=emb_dim, K=q_dim.

  3. LM-head vocab = 151936 (not 128256). 151936 is not divisible by
     8*64 the llama way; we pad each of the 8 partitions to n_part=19008
     (8*19008 = 152064 >= 151936; 19008 % 64 == 0). The last partition
     carries 128 zero rows (logits truncated to vocab on host).

Decode attention is CPU (decode_attention_cpu), matching llama.
"""

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

from qwen3_0_6b_weights import LlamaConfig
from shared.infra.cache import KernelCache


def build_rms_qkv_qknorm_rope_gemv_module(config):
    """Fused decode ELF: RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE (M=1)."""
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_gemv_module as _build,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    return _build(emb_dim, q_dim, kv_dim, n_heads, n_kv_heads, head_dim, qknorm_eps=1e-6)


def _rms_qkv_qknorm_rope_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope_gemv",
    }


# LM-head decode partitioning. vocab=151936.
# Per-partition GEMV broadcasts the K=emb_dim input vector with a hardware
# push_queue repeat_count ~= n_part/32 - 1, capped at the [0:255] range. So
# n_part must be <= 8192 (8192/32 - 1 = 255). 19 * 8192 = 155648 >= 151936;
# the final partition carries 3712 zero-padded rows (logits truncated to
# vocab on host).
_LM_N_PARTITIONS = 19
_LM_N_PART = 8192  # % 64 == 0; n_part/32 - 1 = 255 (at the repeat-count limit)


# ---------------------------------------------------------------------------
# Builder 1: o_gemv_ffn (decoupled O GEMV) + Residual + RMSNorm + SwiGLU FFN.
#   Copy of shared build_o_gemv_ffn_module but stage 1's O GEMV is
#   M=emb_dim, K=q_dim (attn_out is q_dim wide), wo is (emb_dim, q_dim).
#   Stages 2/3 (RMSNorm+SwiGLU, down GEMV) stay emb/hidden.
# ---------------------------------------------------------------------------


def build_o_gemv_ffn_qwen_module(emb_dim, q_dim, hidden_dim):
    """3-launch decode ELF: O-proj(decoupled) + residual + RMSNorm + SwiGLU.

    15-arg ABI mirrors the shared o_gemv_ffn (dead args kept), with two
    decoupled shapes:
      %arg0  wo        (emb_dim, q_dim)   <- DECOUPLED (was emb x emb)
      %arg1  attn_out  (q_dim,)           <- DECOUPLED (was emb)
      ... rest identical to shared o_gemv_ffn.
    """
    # Import o_gemv_ffn_multi first: its module-level sys.path.insert adds the
    # matvec_2tile_add / matvec_swiglu_rms source dirs to the path.
    from shared.builders.o_gemv_ffn_multi import (
        _STAGE2_TILE_M,
        _STAGE2_M_INPUT,
        _STAGE2_HERD_COLS,
        _STAGE2_N_CASCADE,
        _EXTERNS,
    )
    from matvec_2tile_add import build_module as build_2tile_add
    from matvec_swiglu_rms import build_module as build_swiglu_rms
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    # Stage 1: O GEMV is M=emb_dim (output), K=q_dim (input). DECOUPLED.
    stage1 = build_2tile_add(emb_dim, q_dim, m=8, k=512, n_cores=8)
    # Stage 2: RMSNorm + interleaved gate/up GEMV + SwiGLU. emb/hidden.
    stage2 = build_swiglu_rms(
        2 * hidden_dim,
        emb_dim,
        _STAGE2_TILE_M,
        _STAGE2_M_INPUT,
        _STAGE2_HERD_COLS,
        _STAGE2_N_CASCADE,
        bfloat16,
        bfloat16,
    )
    # Stage 3: down GEMV M=emb_dim, K=hidden_dim.
    stage3 = build_2tile_add(emb_dim, hidden_dim, m=8, k=512, n_cores=8)

    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}x{q_dim}xbf16>"),   # wo (DECOUPLED)
        FuncArg("%arg1", f"memref<{q_dim}xbf16>"),             # attn_out (DECOUPLED)
        FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}xbf16>"),           # x_residual
        FuncArg("%arg4", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<2x{emb_dim}xbf16>"),         # packed RMS input
        FuncArg("%arg7", f"memref<{2 * hidden_dim}x{emb_dim}xbf16>"),  # gate/up
        FuncArg("%arg8", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{hidden_dim}xbf16>"),       # swiglu
        FuncArg("%arg12", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # wdown
        FuncArg("%arg13", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{emb_dim}xbf16>"),          # output
    ]
    prelude = (
        f"    %arg6_row0_strided = memref.subview %arg6[0, 0] [1, {emb_dim}] [1, 1]\n"
        f"        : memref<2x{emb_dim}xbf16> to memref<{emb_dim}xbf16, strided<[1]>>\n"
        f"    %arg6_row0 = memref.cast %arg6_row0_strided\n"
        f"        : memref<{emb_dim}xbf16, strided<[1]>> to memref<{emb_dim}xbf16>"
    )
    slices = [
        KernelSlice(str(stage1), "s1", {0: 0, 1: 1, 2: 3},
                    arg_aliases={3: "%arg6_row0"}, extern_syms=_EXTERNS),
        KernelSlice(str(stage2), "s2", {0: 7, 1: 6, 2: 11}, extern_syms=_EXTERNS),
        KernelSlice(str(stage3), "s3", {0: 12, 1: 11, 3: 14},
                    arg_aliases={2: "%arg6_row0"}, extern_syms=_EXTERNS),
    ]
    module = stitch_elf(
        "o_gemv_ffn",
        base_args,
        slices,
        prelude=prelude,
        allow_unreferenced_args={2, 4, 5, 8, 9, 10, 13},
    )
    print(f"  o_gemv_ffn_qwen module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# ---------------------------------------------------------------------------
# Builder 2: LM-head GEMV (8 partitions, n_part=19008 for vocab 151936).
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


def _o_gemv_ffn_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_gemv_ffn",
        "use_lock_race_condition_fix": False,
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
    """Compile the Qwen3 decode kernels."""
    from shared.infra.external_kernels import (
        compile_mv,
        compile_mv_bf16,
        compile_rope,
        compile_silu_and_mul,
    )

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    q_dim = config.n_heads * config.head_dim

    print(f"\n{'='*60}\nCompiling Qwen3 decode kernels...\n{'='*60}\n")

    # External .o kernels: GEMV (mv.o), 2tile-add/swiglu (mv_bf16.o), RoPE.
    compile_mv()
    compile_mv_bf16()
    compile_rope()
    compile_silu_and_mul()

    print("\n--- rms_qkv_qknorm_rope_gemv (FUSED: RMSNorm+QKV+QK-norm+RoPE, 8 launches) ---")
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope_gemv",
        build_rms_qkv_qknorm_rope_gemv_module(config),
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
    )

    print("\n--- o_gemv_ffn (O GEMV decoupled + Residual + FFN) ---")
    cache.compile_and_cache(
        "o_gemv_ffn",
        build_o_gemv_ffn_qwen_module(emb_dim, q_dim, hidden_dim),
        _o_gemv_ffn_backend(verbose),
    )

    print("\n--- lm_head_gemv (8-partition, vocab 151936) ---")
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
        q: (q_dim,) — RoPE'd query vector for the current token.
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys [0:current_pos+1].
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values.
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
    """Run one Qwen3 transformer block for a single decode token.

    Stages: rms_qkv_qknorm_rope_gemv (NPU: RMSNorm + Q/K/V GEMV + per-head
    QK-norm + RoPE) -> KV-cache write -> CPU attention -> o_gemv_ffn (NPU).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)

    # RoPE LUT for this position (position-dependent — NOT static).
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, head_dim)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)

    # --- One ELF = RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE ---
    res = cache.load_and_run(
        "rms_qkv_qknorm_rope_gemv",
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
        x_bf16.flatten().astype(bfloat16),                 # 0 x_in
        layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16),  # 1 norm_w (static)
        np.zeros(emb_dim, dtype=bfloat16),                 # 2 normed
        layer_weights._wq_t,                               # 3 wq (static)
        np.zeros(q_dim, dtype=bfloat16),                   # 4 q
        layer_weights._wk_t,                               # 5 wk (static)
        np.zeros(kv_dim, dtype=bfloat16),                  # 6 k
        layer_weights._wv_t,                               # 7 wv (static)
        np.zeros(kv_dim, dtype=bfloat16),                  # 8 v
        np.asarray(layer_weights.q_norm, bfloat16).reshape(head_dim),  # 9 q_norm (static)
        np.asarray(layer_weights.k_norm, bfloat16).reshape(head_dim),  # 10 k_norm (static)
        np.zeros(q_dim, dtype=bfloat16),                   # 11 q_n
        np.zeros(kv_dim, dtype=bfloat16),                  # 12 k_n
        lut_q,                                             # 13 lut_q (DYNAMIC — position-dependent)
        lut_k,                                             # 14 lut_k (DYNAMIC)
        np.zeros(q_dim, dtype=bfloat16),                   # 15 q_roped
        np.zeros(kv_dim, dtype=bfloat16),                  # 16 k_roped
        output_indices=[8, 15, 16],
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices={2, 4, 6, 8, 11, 12, 15, 16},
        bo_key=f"rms_qkv_qknorm_rope_gemv_L{layer_idx}" if layer_idx is not None else None,
    )
    v = res[8].astype(bfloat16)
    q_roped = res[15].astype(bfloat16)
    k_roped = res[16].astype(bfloat16)

    # --- Update KV cache (K after qk-norm AND rope; V raw projection) ---
    k_cache_layer[:, current_pos, :] = k_roped.reshape(n_kv_heads, head_dim)
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU attention ---
    with cache.profiler.time_cpu("decode_attention_cpu"):
        attn_out = decode_attention_cpu(
            q_roped, k_cache_layer, v_cache_layer, current_pos,
            n_heads, n_kv_heads, head_dim,
        )

    # --- Stage E: O-proj (decoupled) + Residual + RMSNorm + SwiGLU ---
    return _run_o_gemv_ffn(
        attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose
    )


def _run_o_gemv_ffn(attn_out, x_bf16, layer_weights, config, cache, layer_idx, verbose=False):
    """Decode Stage E: O-proj(decoupled) + Residual + RMSNorm + SwiGLU FFN.

    Shared by the fused and legacy decode paths so the o_gemv_ffn arg layout +
    BO indices have a single owner.
    """
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    z_emb = np.zeros(emb_dim, dtype=bfloat16)
    z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
    z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)
    results = cache.load_and_run(
        "o_gemv_ffn",
        _o_gemv_ffn_backend(verbose),
        layer_weights._wo_t,                      # arg0 wo (static, decoupled)
        attn_out,                                 # arg1 attn_out (q_dim)
        z_emb,                                    # arg2 (dead)
        x_bf16.flatten().astype(bfloat16),        # arg3 x_residual
        z_emb,                                    # arg4 (dead)
        z_emb,                                    # arg5 (dead)
        layer_weights._packed_rms_buf,            # arg6 packed (static)
        layer_weights._wgateup_t,                 # arg7 gate/up (static)
        z_hidden,                                 # arg8 (dead)
        z_hidden_emb,                             # arg9 (dead)
        z_hidden,                                 # arg10 (dead)
        z_hidden,                                 # arg11 swiglu
        layer_weights._wdown_t,                   # arg12 wdown (static)
        z_emb,                                    # arg13 (dead)
        z_emb,                                    # arg14 output
        output_indices=[14],
        static_input_indices={0, 6, 7, 12},
        intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
        bo_key=f"o_gemv_ffn_L{layer_idx}" if layer_idx is not None else None,
    )
    return results[14].astype(bfloat16)
