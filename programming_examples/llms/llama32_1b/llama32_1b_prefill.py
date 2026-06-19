# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Prefill on MLIR-AIR (NPU2)

Orchestrates sequential NPU kernel invocations for a LLAMA-3.2-1B prefill
(seq_len=2048). Supports kernel caching (compile once, run many) and profiling.

Architecture:
  1. Compile phase: Build MLIR for each unique kernel config, compile via
     aircc, save binaries to prefill_kernel_cache/ directory.
  2. Run phase: Construct XRTCompileArtifact from cached binary paths,
     load via XRTBackend.load(), execute on NPU.

There are only 8 unique kernel configs across 16 layers (176 invocations):
  - 1 RMSNorm config
  - 1 GEMM config (O projection)
  - 1 Attention GEMMs multi-launch (Q+K+V in one ELF, 3 air.launch ops)
  - 1 FFN multi-launch (Gate+Up+SiLU*mul+Down in one ELF, 4 air.launch ops)
  - 2 RoPE configs (Q heads, K heads)
  - 1 Flash Attention config
  - 1 Eltwise Add config

Usage:
  python3 llama32_1b_prefill.py --model ... --seq-len 2048 --n-layers 16
    --compile-only    # Just compile kernels to cache
    --run-only        # Run using cached kernels (skip compilation)
    --profile         # Enable timing instrumentation
"""

import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Add parent directory to path for kernel imports
_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
# Also add llms/ for sibling LLM packages (llama_kernel_builder).
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from llama32_1b_weights import LlamaConfig, load_weights, generate_rope_lut
from llama32_1b_cpu_helpers import attention_reference
from llama_kernel_builder.cache import KernelCache, Profiler
from llama_kernel_builder.backend_presets import (
    SIMPLE_BACKEND,
    RMS_GEMMS_ROPE_BACKEND,
    O_FFN_BACKEND,
)

# ---------------------------------------------------------------------------
# Kernel compilation definitions
# ---------------------------------------------------------------------------

# Prefill GEMM is always the external high-precision path, registry-driven: every
# GEMM's method (fused-cast vs drain) + tile sizes come from the kernel_registry JSON
# per shape (gemm_registry_config). o_ffn = 4 fused-cast GEMMs (mm_m64.o); rms =
# Q fused-cast (mm_m64.o) + K/V drain (mm_m32.o), mixed in one ELF. All GPU-standard
# 9.3e-3 precision. The external GEMM herds need BD-ID-recycling tiling [2,2].


def _o_ffn_run_backend():
    from llama_kernel_builder.backend_presets import O_FFN_BACKEND as _base

    return {**_base, "runtime_loop_tiling_sizes": [2, 2]}


def _rms_gemms_rope_run_backend():
    from llama_kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND as _base

    return {**_base, "runtime_loop_tiling_sizes": [2, 2]}


def _rms_scratch_specs(seq_len, emb_dim, kv_dim):
    """Registry-driven f32 C-scratch args for the Q/K/V GEMMs of rms_gemms_rope,
    in builder order (Q, K, V). Returns (list_of_scratch_arrays, set_of_indices).

    One scratch array per GEMM whose registry method is fused-cast; indices start
    at 13. This MUST mirror build_rms_gemms_rope_module's own per-shape
    gemm_registry_config lookup (same shapes, same Q->K->V order) so the args we
    pass match the scratch args the compiled ELF declares. For GQA
    (kv_dim<emb_dim) that resolves to Q only (arg13); for MHA (kv_dim==emb_dim)
    to Q,K,V (arg13,14,15). Hardcoding "Q only" was the GQA assumption that
    produced zero K/V at kv_dim==emb_dim.
    """
    from llama32_1b.gemm_builder import gemm_registry_config

    q_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    arrays, inter = [], set()
    nxt = 13
    for spec, cols in ((q_spec, emb_dim), (k_spec, kv_dim), (v_spec, kv_dim)):
        if spec["needs_f32_scratch"]:
            arrays.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return arrays, inter


# Each kernel config is defined as a dict with:
#   build_fn: callable that returns an MLIR module
#   backend_kwargs: dict for XRTBackend constructor
#   output_binary_name: base name for output (optional)


def compile_all_kernels(cache, config, seq_len, cpu_attn=True):
    """Pre-compile all unique kernel configs to cache.

    Args:
        cache: KernelCache instance
        config: LlamaConfig
        seq_len: Sequence length (e.g. 2048)
        cpu_attn: If True, skip flash attention compilation (use CPU fallback)
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling unique kernels (seq_len={seq_len})...")
    print(f"{'='*60}\n")

    # External-GEMM mm.o variants — compile FIRST (before any compile_and_cache, so
    # prepare_air_project copies them into air_project/ for every ELF that links
    # them). The per-GEMM-method builders reference SUFFIXED symbols + filenames so
    # drain (_m32 / mm_m32.o, tile_m=32) and fused-cast (_m64 / mm_m64.o, tile_m=64)
    # can co-link in ONE ELF (rms mixes them; o_ffn is all-fused).
    from llama_kernel_builder.external_kernels import compile_gemm_mm

    compile_gemm_mm(
        tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )

    # 1. RMSNorm + QKV GEMMs + RoPE Q+K: one ELF (registry-driven per-GEMM method).
    from multi_launch_builder.rms_gemms_rope_multi import (
        build_rms_gemms_rope_module,
    )

    cache.compile_and_cache(
        "rms_gemms_rope",
        build_rms_gemms_rope_module(
            seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
        ),
        {"verbose": cache.verbose, **_rms_gemms_rope_run_backend()},
    )

    # 3. O GEMM + Residual Add + FFN (registry-driven fused-cast GEMMs).
    from multi_launch_builder.o_ffn_multi import build_o_ffn_module

    o_ffn_backend = {
        "verbose": cache.verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }
    _o_ffn_mod = build_o_ffn_module(seq_len, emb_dim, hidden_dim)
    cache.compile_and_cache("o_ffn", _o_ffn_mod, o_ffn_backend)

    # 8. Flash Attention GQA (skip if using CPU attention fallback)
    if not cpu_attn:
        from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
            build_module as build_attn,
        )

        lkp = head_dim  # 64
        lqp = 256
        enable_shared_buffers = lkp == head_dim
        cache.compile_and_cache(
            "flash_attn",
            build_attn(
                lk=seq_len,
                lkp=lkp,
                lq=seq_len,
                lqp=lqp,
                dk=head_dim,
                dv=head_dim,
                num_q_tiles=4,
                num_cascade_stages=4,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                causal=True,
            ),
            {
                "verbose": cache.verbose,
                "omit_while_true_loop": not enable_shared_buffers,
                "omit_pingpong": "all",
                "runtime_loop_tiling_sizes": [1, 1],
                "output_format": "elf",
                "instance_name": "attention_bf16",
            },
        )
    else:
        print("  Skipping flash_attn compilation (using CPU attention fallback)")

    # Note: no separate LM Head GEMM ELF in prefill — the unified inference
    # flow reuses the decode-side `lm_head_gemv.elf` for the single-position
    # last-token projection (autoregressive generation only needs that row).

    # Save manifest for --run-only
    cache._save_manifest()

    print(
        f"\nAll {len(cache.artifacts)} kernels compiled and cached to {cache.cache_dir}/"
    )
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Transformer block execution
# ---------------------------------------------------------------------------


_ATTN_BACKEND_KWARGS = {
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


def run_transformer_block(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=True,
    verbose=False,
):
    """Execute a single transformer block on NPU using cached kernels.

    Args:
        x_bf16: (seq_len, emb_dim) bfloat16 input
        layer_weights: LayerWeights for this layer
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT
        config: LlamaConfig
        cache: KernelCache instance (kernels must be pre-compiled)
        layer_idx: Layer index for logging
        cpu_attn: If True, use CPU attention fallback instead of NPU kernel
        verbose: If True, print per-step progress

    Returns:
        (output_bf16, npu_intermediates_dict)
    """
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    n_total = seq_len * emb_dim

    intermediates = {}

    # Cache key for pre-built input arrays (avoids reconstructing weight/buffer
    # arrays on every call — the BO data is already pre-loaded via static_input_indices)
    _arg_cache = getattr(run_transformer_block, "_arg_cache", {})
    run_transformer_block._arg_cache = _arg_cache

    if verbose:
        print(f"  Layer {layer_idx}: Running transformer block...")

    # 1-6. RMSNorm + Q/K/V Projection + RoPE Q+K [6-launch multi-launch ELF]
    kv_dim = n_kv_heads * head_dim  # 512
    if verbose:
        print(
            f"    Steps 1-6: RMSNorm + QKV + RoPE [6-launch ELF] "
            f"(Q: {seq_len}x{emb_dim}, K/V: {seq_len}x{kv_dim})"
        )
    _rms_key = f"rms_gemms_rope_L{layer_idx}"
    if _rms_key not in _arg_cache:
        # First call: build all arrays and cache static/intermediate ones
        _rms_args = [
            None,  # arg0: x_in (dynamic, replaced each call)
            np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # normed_buf
            np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # q_buf
            np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # k_buf
            np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # v_buf
            np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
            np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # q_roped_buf
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # k_roped_buf
        ]
        # Registry-driven f32 C-scratch args, in builder order (Q,K,V). One per
        # fused-cast GEMM (indices from 13). GQA (kv_dim<emb_dim) → Q only (arg13);
        # MHA (kv_dim==emb_dim) → Q,K,V (arg13,14,15). Mirrors the builder's own
        # gemm_registry_config so the args match the ELF's declared scratch slots.
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
        static_input_indices={1, 3, 5, 7, 9, 10},  # weights + LUTs
        intermediate_indices=_rms_inter,
        bo_key=_rms_key,
    )
    v = results[8].reshape(seq_len, kv_dim)
    q_roped = results[11].reshape(seq_len, n_heads * head_dim)
    k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)
    # Store per-probe intermediates — used by KV-cache extraction (v, k_roped)
    # AND by verify/runners/npu_runner.py to capture per-probe NPU outputs.
    intermediates["v"] = v
    intermediates["k_roped"] = k_roped
    intermediates["q_roped"] = q_roped

    # 7. Flash Attention GQA
    if cpu_attn:
        if verbose:
            print(
                f"    Step 7: Attention GQA [CPU fallback] ({n_heads}Q/{n_kv_heads}KV heads)"
            )
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)
    else:
        if verbose:
            print(
                f"    Step 7: Flash Attention GQA [NPU, seq-first] ({n_heads}Q/{n_kv_heads}KV heads)"
            )
        q_attn = np.ascontiguousarray(q_roped)
        k_attn = np.ascontiguousarray(k_roped)
        v_attn = np.ascontiguousarray(v)
        attn_output = np.zeros((seq_len, n_heads * head_dim), dtype=bfloat16)
        results = cache.load_and_run(
            "flash_attn",
            _ATTN_BACKEND_KWARGS,
            q_attn,
            k_attn,
            v_attn,
            attn_output,
        )
        attn_out = results[-1].reshape(seq_len, n_heads * head_dim)
    intermediates["attn_out"] = attn_out

    # 8-15. O GEMM + Residual Add + FFN [8-launch multi-launch ELF]
    if verbose:
        print(
            f"    Steps 8-15: O+FFN [8-launch ELF] "
            f"(O: {seq_len}x{emb_dim}, FFN: {seq_len}x{emb_dim}x{hidden_dim})"
        )
    _offn_key = f"o_ffn_L{layer_idx}"
    # Fused-cast o_ffn layout: all outward buffers bf16 (the fused-cast GEMM's own
    # cast launch writes bf16); 4 f32 C-scratch args (15..18); output arg14.
    if _offn_key not in _arg_cache:
        # First call: build all arrays and cache static/intermediate ones
        offn_args = [
            None,  # arg0: attn_out (dynamic)
            np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # proj_buf
            None,  # arg3: x_residual (dynamic)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # res1_buf
            np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # normed2_buf
            np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
                emb_dim, hidden_dim
            ),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # gate_buf
            np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # up_buf
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # swiglu_buf
            np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(
                hidden_dim, emb_dim
            ),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # down_buf
            np.zeros(n_total, dtype=bfloat16),  # output_buf (arg14)
            # arg15..18: per-GEMM f32 C-scratch (proj, gate, up, down).
            np.zeros((seq_len, emb_dim), dtype=np.float32),
            np.zeros((seq_len, hidden_dim), dtype=np.float32),
            np.zeros((seq_len, hidden_dim), dtype=np.float32),
            np.zeros((seq_len, emb_dim), dtype=np.float32),
        ]
        _arg_cache[_offn_key] = offn_args
    cached_args = _arg_cache[_offn_key]
    cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
    # `copy=False` makes .astype a no-op when x_bf16 is already bf16 (which
    # it always is in production: layer 0 starts as cast embed table, then
    # each subsequent layer is the previous o_ffn output). Saves ~12 MB
    # heap alloc per layer call → ~190 MB churn per inference run on
    # llama3.2-1B's 16 layers, eliminating the bulk of the trial-1 perf gradient.
    cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

    _out_idx = 14
    _inter = {2, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18}
    results = cache.load_and_run(
        "o_ffn",
        _o_ffn_run_backend(),
        *cached_args,
        output_indices=[_out_idx],
        static_input_indices={1, 5, 7, 9, 12},  # wo, ffn_norm_w, w_gate, w_up, w_down
        intermediate_indices=_inter,
        bo_key=_offn_key,
    )
    output_bf16 = results[_out_idx].reshape(seq_len, emb_dim)
    intermediates["ffn_out"] = output_bf16

    return output_bf16, intermediates


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------


def preload_prefill_weights(weights, config, cache, seq_len, rope_lut_bf16):
    """Pre-load all transformer block weights into per-layer BOs.

    Like IRON's prepare_runtime(): writes all weight data once before timing
    starts. During inference, static_input_indices skips weight writes.
    """
    if hasattr(weights, "_prefill_weights_preloaded"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim

    print("Pre-loading transformer block weights (per-layer BOs)...")
    profiler_enabled = cache.profiler.enabled
    cache.profiler.enabled = False

    # Pre-compute LUTs (same for all layers)
    rope_lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    rope_lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # Also populate run_transformer_block's arg cache so the inference pass
    # can reuse these arrays instead of reconstructing them (~500ms saved).
    _arg_cache = getattr(run_transformer_block, "_arg_cache", {})
    run_transformer_block._arg_cache = _arg_cache

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # rms_gemms_rope: warmup to allocate + write weights
        rms_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg0: x_in (dynamic)
            np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),  # arg1
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg2
            np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),  # arg3
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg4
            np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),  # arg5
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # arg6
            np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),  # arg7
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # arg8
            rope_lut_q,  # arg9
            rope_lut_k,  # arg10
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg11
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # arg12
        ]
        # Registry-driven f32 C-scratch (Q,K,V order). Allocates the per-layer BO
        # set with the SAME arg count the block uses at inference time — must match
        # run_transformer_block above or the bo_key-reused BO set would be missing
        # MHA's K/V scratch (the original GQA-hardcoded bug). GQA → 1; MHA → 3.
        scratch_arrays, scratch_inter = _rms_scratch_specs(seq_len, emb_dim, kv_dim)
        rms_args.extend(scratch_arrays)
        _arg_cache[f"rms_gemms_rope_L{layer_idx}"] = (rms_args, scratch_inter)
        _rms_inter_warm = {2, 4, 6, 8, 11, 12} | scratch_inter
        cache.load_and_run(
            "rms_gemms_rope",
            _rms_gemms_rope_run_backend(),
            *rms_args,
            output_indices=[8, 11, 12],
            static_input_indices={1, 3, 5, 7, 9, 10},
            intermediate_indices=_rms_inter_warm,
            bo_key=f"rms_gemms_rope_L{layer_idx}",
        )

        # o_ffn warmup (allocate + write weights). Fused-cast layout: all-bf16
        # buffers + output arg14 + 4 f32 C-scratch args (15..18). Matches
        # run_transformer_block's _offn_key cache.
        offn_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg0: attn_out (dynamic)
            np.asarray(lw.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),  # arg1: wo
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg2: proj
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg3: x_residual (dynamic)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg4
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),  # arg5
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg6
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # arg7
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # arg8: gate
            np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # arg9
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # arg10: up
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # arg11: swiglu
            np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),  # arg12
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg13: down
            np.zeros(n_total, dtype=bfloat16),  # arg14: output
            np.zeros((seq_len, emb_dim), dtype=np.float32),  # arg15 proj_f32
            np.zeros((seq_len, hidden_dim), dtype=np.float32),  # arg16 gate_f32
            np.zeros((seq_len, hidden_dim), dtype=np.float32),  # arg17 up_f32
            np.zeros((seq_len, emb_dim), dtype=np.float32),  # arg18 down_f32
        ]
        _arg_cache[f"o_ffn_L{layer_idx}"] = offn_args
        cache.load_and_run(
            "o_ffn",
            _o_ffn_run_backend(),
            *offn_args,
            output_indices=[14],
            static_input_indices={1, 5, 7, 9, 12},
            intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18},
            bo_key=f"o_ffn_L{layer_idx}",
        )

    cache.profiler.enabled = profiler_enabled
    weights._prefill_weights_preloaded = True
    total_mb = (
        config.n_layers
        * (
            emb_dim * emb_dim * 2  # wq
            + emb_dim * kv_dim * 2 * 2  # wk + wv
            + emb_dim * emb_dim * 2  # wo
            + emb_dim * hidden_dim * 2 * 2  # w_gate + w_up
            + hidden_dim * emb_dim * 2  # w_down
        )
        // 1024
        // 1024
    )
    print(f"  Pre-loaded {config.n_layers} layers ({total_mb}MB)")
