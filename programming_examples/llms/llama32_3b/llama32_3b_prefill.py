# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-3B Prefill on MLIR-AIR (NPU2).

Pure-Llama deployment: Llama-3.2-3B has the IDENTICAL kernel sequence as
Llama-3.2-1B (RMSNorm -> QKV GEMM -> RoPE -> attention -> O -> add -> RMSNorm
-> Gate/Up -> SwiGLU -> Down -> add; NO QK-norm, NO bias). The only deltas are
dims (emb 3072, n_heads 24, 28 layers, hidden 8192) and head_dim=128 (vs 64).

Because the llama32_1b builders + block runner are fully config-driven (they
take seq_len / emb_dim / n_heads / head_dim / hidden_dim as arguments and
resolve all GEMM tiles from the kernel_registry per shape), we reuse the
``rms_gemms_rope`` and ``o_ffn`` builders + ``preload_prefill_weights``
directly with the 3B config.

ATTENTION FORK (head_dim=128): llama32_1b is head_dim=64 and uses the
SEQ-FIRST FlashAttention kernel (``attn_npu2_seqfirst``), which CANNOT express
the dv_chunks=2 tiling head_dim=128 requires (and its dk_chunks>1 path hangs).
So Llama-3.2-3B (head_dim=128) MUST use the shared HEAD-FIRST FA wrapper
(``shared.infra.fa_headfirst``), exactly like the Qwen head_dim=128 siblings.

We therefore fork ``compile_all_kernels`` and ``run_transformer_block`` here:
the body mirrors llama32_1b's verbatim (same rms_gemms_rope + o_ffn ELFs,
same BO-reuse layout) EXCEPT Step 7 calls ``npu_fa_headfirst`` instead of the
seq-first ``flash_attn`` ELF. ``preload_prefill_weights`` is reused unchanged
(it does not touch attention).

Prefill attention defaults to NPU head-first FlashAttention (cpu_attn=False).
Pass cpu_attn=True to fall back to the FP32 host attention reference.
"""

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Make llms/ and programming_examples/ importable, and the sibling
# llama32_1b/ package (we reuse its config-driven builders verbatim).
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG = _LLMS_DIR.parent
_LLAMA1B = _LLMS_DIR / "llama32_1b"
for _p in (str(_PROG), str(_LLMS_DIR), str(_LLAMA1B), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the bit-for-bit-identical, config-driven prefill weight pre-load and
# the private backend/scratch helpers from llama32_1b. Only the two
# attention-touching entry points (compile_all_kernels, run_transformer_block)
# are forked below to swap seq-first FA -> head-first FA.
from llama32_1b_prefill import (  # noqa: E402,F401
    preload_prefill_weights,
    _o_ffn_run_backend,
    _rms_gemms_rope_run_backend,
    _rms_scratch_specs,
)
from llama32_1b_cpu_helpers import attention_reference  # noqa: E402


def compile_all_kernels(cache, config, seq_len, cpu_attn=False):
    """Pre-compile all unique kernel configs to cache.

    Mirrors llama32_1b.compile_all_kernels but compiles the HEAD-FIRST FA ELF
    (head_dim=128) via the shared wrapper instead of the seq-first kernel.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling unique kernels (Llama-3.2-3B, seq_len={seq_len})...")
    print(f"{'='*60}\n")

    from shared.infra.external_kernels import compile_gemm_mm

    compile_gemm_mm(
        tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )

    # 1. RMSNorm + QKV GEMMs + RoPE Q+K: one ELF (registry-driven per-GEMM method).
    from shared.builders.rms_gemms_rope_multi import build_rms_gemms_rope_module

    cache.compile_and_cache(
        "rms_gemms_rope",
        build_rms_gemms_rope_module(
            seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
        ),
        {"verbose": cache.verbose, **_rms_gemms_rope_run_backend()},
    )

    # 2. O GEMM + Residual Add + FFN (registry-driven fused-cast GEMMs).
    from shared.builders.o_ffn_multi import build_o_ffn_module

    o_ffn_backend = {
        "verbose": cache.verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }
    cache.compile_and_cache(
        "o_ffn", build_o_ffn_module(seq_len, emb_dim, hidden_dim), o_ffn_backend
    )

    # 3. Flash Attention (head-first, head_dim=128). Skip if using CPU fallback.
    if not cpu_attn:
        print("\n--- flash_attn (head-first FA, head_dim=128) ---")
        from shared.infra.fa_headfirst import compile_headfirst_fa

        compile_headfirst_fa(
            cache, seq_len, n_heads, n_kv_heads, head_dim, cache.verbose
        )
    else:
        print("  Skipping flash_attn compilation (using CPU attention fallback)")

    cache._save_manifest()
    print(
        f"\nAll {len(cache.artifacts)} kernels compiled and cached to {cache.cache_dir}/"
    )
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


def run_transformer_block(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=False,
    verbose=False,
):
    """Execute one Llama-3.2-3B transformer block on NPU.

    Identical to llama32_1b.run_transformer_block (same rms_gemms_rope + o_ffn
    ELFs, same BO-reuse arg cache) EXCEPT Step 7 uses the shared HEAD-FIRST FA
    wrapper (head_dim=128) instead of the seq-first ELF.
    """
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    n_total = seq_len * emb_dim
    kv_dim = n_kv_heads * head_dim

    intermediates = {}

    _arg_cache = getattr(run_transformer_block, "_arg_cache", {})
    run_transformer_block._arg_cache = _arg_cache

    # 1-6. RMSNorm + Q/K/V Projection + RoPE Q+K [6-launch multi-launch ELF]
    _rms_key = f"rms_gemms_rope_L{layer_idx}"
    if _rms_key not in _arg_cache:
        _rms_args = [
            None,  # arg0: x_in (dynamic)
            np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),
            np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),
            np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
            np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),
        ]
        _scratch_arrays, _scratch_inter = _rms_scratch_specs(seq_len, emb_dim, kv_dim)
        _rms_args.extend(_scratch_arrays)
        _arg_cache[_rms_key] = (_rms_args, _scratch_inter)
    cached_args, _scratch_inter = _arg_cache[_rms_key]
    cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

    _rms_inter = {2, 4, 6, 8, 11, 12} | _scratch_inter
    results = cache.load_and_run(
        "rms_gemms_rope",
        _rms_gemms_rope_run_backend(),
        *cached_args,
        output_indices=[8, 11, 12],
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices=_rms_inter,
        bo_key=_rms_key,
    )
    v = results[8].reshape(seq_len, kv_dim)
    q_roped = results[11].reshape(seq_len, n_heads * head_dim)
    k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)
    intermediates["v"] = v
    intermediates["k_roped"] = k_roped
    intermediates["q_roped"] = q_roped

    # 7. Attention GQA.
    if cpu_attn:
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
        # post-RoPE seq-first (pure Llama: no QK-norm, no bias); v is the raw
        # V projection seq-first. Real (un-padded) dims.
        from shared.infra.fa_headfirst import npu_fa_headfirst

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
    intermediates["attn_out"] = attn_out

    # 8-15. O GEMM + Residual Add + FFN [8-launch multi-launch ELF]
    _offn_key = f"o_ffn_L{layer_idx}"
    if _offn_key not in _arg_cache:
        offn_args = [
            None,  # arg0: attn_out (dynamic)
            np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            None,  # arg3: x_residual (dynamic)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
                emb_dim, hidden_dim
            ),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(
                hidden_dim, emb_dim
            ),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros(n_total, dtype=bfloat16),
            np.zeros((seq_len, emb_dim), dtype=np.float32),
            np.zeros((seq_len, hidden_dim), dtype=np.float32),
            np.zeros((seq_len, hidden_dim), dtype=np.float32),
            np.zeros((seq_len, emb_dim), dtype=np.float32),
        ]
        _arg_cache[_offn_key] = offn_args
    cached_args = _arg_cache[_offn_key]
    cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
    cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

    _out_idx = 14
    _inter = {2, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18}
    results = cache.load_and_run(
        "o_ffn",
        _o_ffn_run_backend(),
        *cached_args,
        output_indices=[_out_idx],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices=_inter,
        bo_key=_offn_key,
    )
    output_bf16 = results[_out_idx].reshape(seq_len, emb_dim)
    intermediates["ffn_out"] = output_bf16

    return output_bf16, intermediates
