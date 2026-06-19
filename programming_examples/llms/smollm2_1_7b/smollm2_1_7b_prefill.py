# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B Prefill on MLIR-AIR (NPU2) — kernel-first fork.

SmolLM2 is a bit-for-bit llama kernel sequence EXCEPT on one axis: it is pure
MHA (n_kv_heads == n_heads == 32, so kv_dim == emb_dim == 2048). The reference
`llama32_1b_prefill.run_transformer_block` hardcodes a GQA assumption in its
`rms_gemms_rope` argument construction: it appends exactly ONE f32 C-scratch
arg (for Q's fused-cast GEMM) and treats K/V as drain (no scratch). That is
correct only while kv_dim < emb_dim (llama's 512).

At kv_dim == 2048 the registry resolves K and V to fused-cast too, so the
`build_rms_gemms_rope_module` builder emits THREE scratch args (Q=13, K=14,
V=15) and a 16-arg func. The reference caller still passes only 14 args, so
K/V's fused-cast GEMMs read unallocated scratch and produce NaN.

This module forks ONLY `run_transformer_block`, making the scratch-arg
construction registry-driven (mirroring the builder's own per-shape
gemm_registry_config lookup) so it supplies exactly the scratch args the ELF
declares — correct for any kv_dim (GQA or MHA). Everything else (kernel
compilation, weight preloading, the o_ffn path) is imported unchanged from
the llama32_1b reference.
"""

from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

# Resolve shared llms/ packages + the llama32_1b reference we fork from.
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG_EXAMPLES = _LLMS_DIR.parent
_LLAMA_REF = _LLMS_DIR / "llama32_1b"
for _p in (str(_PROG_EXAMPLES), str(_LLMS_DIR), str(_LLAMA_REF), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the reference's compilation plumbing unchanged. preload is FORKED below
# (the reference's preload hardcodes the same GQA scratch assumption as the
# reference block, pre-populating the BO set with 14 args before inference runs).
from llama32_1b_prefill import (  # noqa: E402
    compile_all_kernels,
    _rms_gemms_rope_run_backend,
    _o_ffn_run_backend,
)
from llama32_1b_cpu_helpers import attention_reference  # noqa: E402
from llama32_1b.gemm_builder import gemm_registry_config  # noqa: E402


def _rms_scratch_specs(seq_len, emb_dim, kv_dim):
    """Registry-driven f32 C-scratch args for the Q/K/V GEMMs of rms_gemms_rope,
    in builder order (Q, K, V). Returns (list_of_scratch_arrays, set_of_indices).
    One scratch per GEMM whose registry method is fused-cast; index starts at 13.
    For GQA (kv_dim<emb_dim) this is Q only; for MHA (kv_dim==emb_dim) it's Q,K,V.
    """
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
    """Single transformer block on NPU — registry-driven scratch args (MHA-safe).

    Drop-in replacement for llama32_1b_prefill.run_transformer_block; identical
    behaviour for GQA (kv_dim<emb_dim → 1 scratch), correct for MHA
    (kv_dim==emb_dim → 3 scratch).
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

    # 1-6. RMSNorm + Q/K/V Projection + RoPE Q+K [multi-launch ELF]
    _rms_key = f"rms_gemms_rope_L{layer_idx}"
    if _rms_key not in _arg_cache:
        _rms_args = [
            None,  # arg0: x_in (dynamic)
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
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # q_roped_buf (arg11)
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # k_roped_buf (arg12)
        ]
        # Registry-driven f32 C-scratch args, in builder order (Q,K,V). One per
        # fused-cast GEMM. GQA → Q only (arg13); MHA → Q,K,V (arg13,14,15).
        _scratch_arrays, _scratch_inter = _rms_scratch_specs(seq_len, emb_dim, kv_dim)
        _rms_args.extend(_scratch_arrays)
        _arg_cache[_rms_key] = (_rms_args, _scratch_inter)
    cached_args, _scratch_inter = _arg_cache[_rms_key]
    cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

    # Intermediates: builder's fixed buffers {2,4,6,8,11,12} + the scratch args.
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
    intermediates["v"] = v
    intermediates["k_roped"] = k_roped
    intermediates["q_roped"] = q_roped

    # 7. Attention (CPU fallback or NPU FlashAttention) — unchanged from reference.
    if cpu_attn:
        attn_out = attention_reference(
            q_roped.astype(np.float32),
            k_roped.astype(np.float32),
            v.astype(np.float32),
            n_heads,
            n_kv_heads,
        ).astype(bfloat16)
    else:
        from llama32_1b_prefill import _ATTN_BACKEND_KWARGS

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

    # 8-15. O GEMM + Residual + FFN [multi-launch ELF] — unchanged from reference
    # (o_ffn's GEMMs are all emb_dim/hidden_dim shapes, identical for GQA and MHA,
    # so the reference's scratch handling there is already correct).
    _offn_key = f"o_ffn_L{layer_idx}"
    if _offn_key not in _arg_cache:
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


def preload_prefill_weights(weights, config, cache, seq_len, rope_lut_bf16):
    """MHA-safe fork of llama32_1b_prefill.preload_prefill_weights.

    The reference preload pre-populates run_transformer_block._arg_cache AND
    allocates the per-layer BO sets with a hardcoded 14-arg (GQA, 1-scratch)
    rms_gemms_rope layout. Since preload runs (via prepare_runtime) BEFORE any
    inference, that 14-BO set is what every later block call reuses by bo_key —
    so even an MHA-correct block ends up running the 16-arg ELF against 14 BOs,
    and K/V's fused-cast scratch args (14,15) are missing → K/V output stays
    zero. This fork builds the rms args + BO set registry-driven (16 for MHA),
    matching run_transformer_block above. o_ffn is unchanged (its shapes are
    GQA/MHA-identical).
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

    print("Pre-loading transformer block weights (per-layer BOs, MHA-safe)...")
    profiler_enabled = cache.profiler.enabled
    cache.profiler.enabled = False

    rope_lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    rope_lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # Populate OUR run_transformer_block's arg cache (tuple format).
    _arg_cache = getattr(run_transformer_block, "_arg_cache", {})
    run_transformer_block._arg_cache = _arg_cache

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        rms_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg0 x_in
            np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg2
            np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg4
            np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # arg6
            np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # arg8
            rope_lut_q,
            rope_lut_k,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg11
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # arg12
        ]
        scratch_arrays, scratch_inter = _rms_scratch_specs(seq_len, emb_dim, kv_dim)
        rms_args.extend(scratch_arrays)
        _arg_cache[f"rms_gemms_rope_L{layer_idx}"] = (rms_args, scratch_inter)
        cache.load_and_run(
            "rms_gemms_rope",
            _rms_gemms_rope_run_backend(),
            *rms_args,
            output_indices=[8, 11, 12],
            static_input_indices={1, 3, 5, 7, 9, 10},
            intermediate_indices={2, 4, 6, 8, 11, 12} | scratch_inter,
            bo_key=f"rms_gemms_rope_L{layer_idx}",
        )

        offn_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg0 attn_out
            np.asarray(lw.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg2
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg3 x_residual
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg4
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg6
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # arg8
            np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # arg10
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # arg11
            np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg13
            np.zeros(n_total, dtype=bfloat16),  # arg14 output
            np.zeros((seq_len, emb_dim), dtype=np.float32),
            np.zeros((seq_len, hidden_dim), dtype=np.float32),
            np.zeros((seq_len, hidden_dim), dtype=np.float32),
            np.zeros((seq_len, emb_dim), dtype=np.float32),
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
    print(f"  Pre-loaded {config.n_layers} layers (MHA-safe)")
