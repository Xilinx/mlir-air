# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B BF16 Inference on MLIR-AIR (NPU2).

Unified script: NPU prefill + NPU decode with NPU LM Head.
- Prefill: runs full prompt through 16 transformer layers on NPU
- Decode: generates tokens one at a time using GEMV kernels on NPU
- LM Head: NPU-accelerated for both prefill (8-partition GEMM) and decode (8-partition GEMV)

Usage:
    cd build_peano

    # Compile both prefill and decode kernels:
    python3 ../llama32_1b_inference.py --compile-only

    # Run inference with cached kernels:
    python3 ../llama32_1b_inference.py --run-only --n-tokens 10 --profile
    python3 ../llama32_1b_inference.py --run-only --n-tokens 100 --profile
    python3 ../llama32_1b_inference.py --run-only --n-tokens 20 --prompt "Once upon a time"
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama32_1b_weights import (
    LlamaConfig,
    load_weights,
    generate_rope_lut,
)
from kernel_builder.cache import KernelCache, Profiler
from kernel_builder.external_kernels import compile_all_external_kernels
from kernel_builder.backend_presets import (
    LM_GEMV_BACKEND,
    RGR_BACKEND,
    OGF_BACKEND,
)
from llama32_1b_prefill import (
    compile_all_kernels,
    run_transformer_block,
    preload_prefill_weights,
)
from llama32_1b_decode import (
    compile_decode_kernels,
    run_decode_block,
)

# ---------------------------------------------------------------------------
# Streaming-decode helpers (BPE-safe incremental output)
# ---------------------------------------------------------------------------


class _StreamState:
    """Tracks how many characters of the running decoded text have been emitted.

    BPE tokens may decode to '' in isolation but combine into characters when
    paired with later tokens. The safest streaming pattern is to decode the
    full id list each call and emit only the suffix we have not printed yet.
    """

    def __init__(self) -> None:
        self.printed_len: int = 0


def _delta_text(tokenizer: Any, ids: list[int], state: _StreamState) -> str:
    """Return the new text fragment since the last call, advancing state."""
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


# ---------------------------------------------------------------------------
# Session: long-lived state created once per process
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """Everything `run_once` needs that should not be rebuilt per turn."""

    config: Any  # LlamaConfig
    seq_len: int  # padded prompt length (today: 2048)
    weights: Any  # LlamaWeights, mutated by prepare_runtime()
    tokenizer: Any  # transformers AutoTokenizer
    prefill_cache: Any  # KernelCache
    decode_cache: Any  # KernelCache
    rope_lut_bf16: np.ndarray  # (max_seq, head_dim) bfloat16
    model_variant: str  # "base" | "instruct"


# Decode LM Head constants
_LM_N_PART = 16384
_LM_N_PARTITIONS = 8


# ---------------------------------------------------------------------------
# Runtime preparation (all one-time init, outside profiling scope)
# ---------------------------------------------------------------------------


def prepare_runtime(
    prefill_cache,
    decode_cache,
    weights,
    config,
    seq_len,
    rope_lut_bf16,
):
    """One-time runtime initialization. Called before any timed inference.

    Does:
        1. Compile external C++ kernels from source
        2. Pre-transpose decode GEMV weights
        3. Pre-load prefill weights into per-layer BOs
        4. Pre-load prefill LM Head weights
        5. Pre-load decode weights into per-layer BOs
        6. Pre-load decode LM Head GEMV weights
        7. NPU warmup pass (run one decode token to wake NPU)

    Args:
        prefill_cache: KernelCache with prefill kernels loaded
        decode_cache: KernelCache with decode kernels loaded
        weights: LlamaWeights (modified in-place)
        config: LlamaConfig
        seq_len: prompt sequence length (for LM Head BO sizing)
        rope_lut_bf16: (max_seq, head_dim) bfloat16 RoPE LUT
    """
    print(f"\n{'='*60}")
    print("Preparing runtime (one-time init, outside profiling scope)...")
    print(f"{'='*60}")
    t0 = time.time()

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    # 1. Compile external C++ kernels from source
    compile_all_external_kernels(head_dim=head_dim)

    # 2. Pre-transpose all decode GEMV weights
    #    GEMV kernel expects A[M,K] but HuggingFace stores (out_features, in_features)
    if not hasattr(weights, "_decode_weights_transposed"):
        print("  Pre-transposing weights for GEMV...")
        for lw in weights.layers:
            lw._wq_t = np.ascontiguousarray(
                lw.wq.astype(bfloat16).reshape(emb_dim, emb_dim).T
            )
            lw._wk_t = np.ascontiguousarray(
                lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )
            lw._wv_t = np.ascontiguousarray(
                lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )
            lw._wo_t = np.ascontiguousarray(
                lw.wo.astype(bfloat16).reshape(emb_dim, emb_dim).T
            )
            lw._wgate_t = np.ascontiguousarray(
                lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )
            lw._wup_t = np.ascontiguousarray(
                lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )
            lw._wdown_t = np.ascontiguousarray(
                lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T
            )
        weights._decode_weights_transposed = True

    # 3. Tag layers with index for per-layer BO isolation
    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    # 4. Pre-load prefill weights into per-layer BOs
    preload_prefill_weights(weights, config, prefill_cache, seq_len, rope_lut_bf16)

    # 5. Pre-load decode weights into per-layer BOs
    #    (lm_head_gemv 8-partition weights here are also reused by prefill's
    #    last-token projection — refactored from full-seq GEMM for ~150 ms savings)
    _preload_decode_weights(decode_cache, weights, config)

    # Note: NPU warmup pass not needed here — the NPU prefill keeps
    # the NPU active. Only needed in llama32_1b_decode.py where CPU prefill
    # leaves the NPU idle for ~17s (triggering power-save).

    t_prep = time.time() - t0
    print(f"  Runtime prepared in {t_prep:.1f}s")
    # Stash on both profilers for the dataflow summary (one-time cost,
    # outside per-query wall but useful context).
    prefill_cache.profiler.preprocessing_s = t_prep
    decode_cache.profiler.preprocessing_s = t_prep


def _preload_decode_weights(decode_cache, weights, config):
    """Pre-load all decode transformer block weights into per-layer BOs.

    Mirrors the preloading pattern from llama32_1b_decode.py: writes all weight
    data once before timing starts. During inference, static_input_indices
    skips weight re-writes.
    """
    if hasattr(weights, "_decode_weights_preloaded_to_bos"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim
    vocab_size = weights.lm_head.shape[0]

    print("  Pre-loading decode weights into per-layer BOs...")

    # Suppress profiling during warmup — these BO-allocate / weight-write
    # calls happen in prepare_runtime (outside the user-visible wall time
    # for prefill / decode). Mirrors the same guard in preload_prefill_weights.
    _was_enabled = decode_cache.profiler.enabled
    decode_cache.profiler.enabled = False

    rope_lut_q_dummy = np.zeros(n_heads * head_dim, dtype=bfloat16)
    rope_lut_k_dummy = np.zeros(n_kv_heads * head_dim, dtype=bfloat16)

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # rms_gemv_rope: allocate + write weights
        decode_cache.load_and_run(
            "rms_gemv_rope",
            RGR_BACKEND,
            np.zeros(emb_dim, dtype=bfloat16),  # x_in
            lw.attn_norm.reshape(emb_dim).astype(bfloat16),  # norm_w
            np.zeros(emb_dim, dtype=bfloat16),  # normed
            lw._wq_t,  # wq
            np.zeros(emb_dim, dtype=bfloat16),  # q
            lw._wk_t,  # wk
            np.zeros(kv_dim, dtype=bfloat16),  # k
            lw._wv_t,  # wv
            np.zeros(kv_dim, dtype=bfloat16),  # v
            rope_lut_q_dummy,  # lut_q
            rope_lut_k_dummy,  # lut_k
            np.zeros(emb_dim, dtype=bfloat16),  # q_roped
            np.zeros(kv_dim, dtype=bfloat16),  # k_roped
            output_indices=[8, 11, 12],
            static_input_indices={1, 3, 5, 7},
            intermediate_indices={2, 4, 6, 8, 11, 12},
            bo_key=f"rms_gemv_rope_L{layer_idx}",
        )

        # o_gemv_ffn (3-stage): build the interleaved w_gateup [2*hidden, emb]
        # and the packed [2, emb] RMSNorm-input buffer (row 1 = ffn_norm_w,
        # row 0 left zero for stage 1 to overwrite per token). Stashed on
        # LayerWeights for reuse across all decode tokens. Frees the original
        # _wgate_t/_wup_t once the interleaved copy is in place — they're
        # otherwise unused after this preload (~1 GB host RAM saved).
        wgate = lw._wgate_t
        wup = lw._wup_t
        wgateup = np.empty((2 * hidden_dim, emb_dim), dtype=bfloat16)
        wgateup[0::2] = wgate
        wgateup[1::2] = wup
        lw._wgateup_t = wgateup
        del lw._wgate_t
        del lw._wup_t

        packed = np.empty((2, emb_dim), dtype=bfloat16)
        packed[0] = 0.0
        packed[1] = lw.ffn_norm.reshape(emb_dim).astype(bfloat16)
        lw._packed_rms_buf = packed

        z_emb = np.zeros(emb_dim, dtype=bfloat16)
        z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
        z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)

        decode_cache.load_and_run(
            "o_gemv_ffn",
            OGF_BACKEND,
            lw._wo_t,  # arg0 wo (static)
            z_emb,  # arg1 attn_out
            z_emb,  # arg2 (dead)
            z_emb,  # arg3 x_residual
            z_emb,  # arg4 (dead)
            z_emb,  # arg5 (dead)
            lw._packed_rms_buf,  # arg6 packed (static)
            lw._wgateup_t,  # arg7 w_gateup (static)
            z_hidden,  # arg8 (dead)
            z_hidden_emb,  # arg9 (dead)
            z_hidden,  # arg10 (dead)
            z_hidden,  # arg11 swiglu
            lw._wdown_t,  # arg12 wdown (static)
            z_emb,  # arg13 (dead)
            z_emb,  # arg14 output
            output_indices=[14],
            static_input_indices={0, 6, 7, 12},
            intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
            bo_key=f"o_gemv_ffn_L{layer_idx}",
        )

    # LM Head GEMV weights (8 partitions)
    weights._lm_weight_parts_gemv = []
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        w = np.zeros((_LM_N_PART, emb_dim), dtype=bfloat16)
        w[: n_end - n_start, :] = np.ascontiguousarray(
            weights.lm_head[n_start:n_end, :]
        ).astype(bfloat16)
        weights._lm_weight_parts_gemv.append(w)

    # Pre-load LM Head GEMV BOs
    lm_inputs = [np.zeros(emb_dim, dtype=bfloat16)]
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
    decode_cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )

    # Restore profiler state — subsequent decode_cache.load_and_run calls
    # (from prefill end + decode loop) record timing as intended.
    decode_cache.profiler.enabled = _was_enabled

    weights._decode_weights_preloaded_to_bos = True
    total_mb = (
        config.n_layers
        * (
            emb_dim * emb_dim * 2  # wq
            + kv_dim * emb_dim * 2 * 2  # wk, wv
            + emb_dim * emb_dim * 2  # wo
            + hidden_dim * emb_dim * 2 * 2  # w_gate, w_up
            + emb_dim * hidden_dim * 2  # w_down
        )
        // 1024
        // 1024
    )
    print(
        f"  Pre-loaded {config.n_layers} decode layers + LM Head ({total_mb + 512}MB)"
    )


# ---------------------------------------------------------------------------
# NPU Prefill with KV cache extraction
# ---------------------------------------------------------------------------


def run_npu_prefill(
    token_ids,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    max_seq,
    tokenizer,
    cpu_attn=True,
    profile=False,
    quiet=False,
):
    """Run NPU prefill and extract KV cache for decode.

    Returns:
        prefill_token: int -- first predicted token ID (= argmax(logits_row))
        logits_row: (vocab_size,) f32 -- raw NPU LM-head logits at the
                    prediction position (before argmax). Production
                    callers can discard with `_`; the verify subsystem
                    reads this for top-k extraction.
        k_cache: (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        v_cache: (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        prompt_len: actual prompt length (before padding)
    """
    seq_len = len(token_ids)
    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    # Pre-allocate KV cache
    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    # Token embedding (CPU gather + dtype casts)
    with prefill_cache.profiler.time_cpu("embed_lookup"):
        embed_f32 = weights.embed_table[token_ids].astype(np.float32)
        x_bf16 = embed_f32.astype(bfloat16)

    # ---- TIMED SECTION START ----
    if not quiet:
        print(f"Running NPU prefill ({config.n_layers} layers, seq_len={seq_len})...")
    t_prefill_start = time.time()

    # Run 16 transformer layers on NPU, collecting KV cache
    for layer_idx in range(config.n_layers):
        t0 = prefill_cache.profiler.start_layer()

        x_bf16, intermediates = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            cpu_attn=cpu_attn,
            verbose=profile,
        )

        # Extract KV cache from intermediates (CPU: reshape + transpose +
        # cast + slice-assign). 16 invocations per prefill, one per layer.
        with prefill_cache.profiler.time_cpu("kv_cache_extract"):
            k_roped = intermediates["k_roped"]
            v_raw = intermediates["v"]
            k_cache[layer_idx, :, :seq_len, :] = (
                k_roped.astype(bfloat16)
                .reshape(seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
            v_cache[layer_idx, :, :seq_len, :] = (
                v_raw.astype(bfloat16)
                .reshape(seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )

        prefill_cache.profiler.end_layer(layer_idx, t0)

    # Final RMSNorm + LM Head — single-position only.
    # Autoregressive generation only needs logits at the last real-token row;
    # computing the full (seq_len, vocab) projection wastes ~2047x compute.
    # Do CPU RMSNorm on just that row (<1 ms) and reuse the decode-side
    # 8-partition GEMV ELF (~14 ms NPU) instead of the full-seq GEMM ELF.
    vocab_size = weights.lm_head.shape[0]
    prompt_len = len([t for t in token_ids if t != tokenizer.eos_token_id])
    pred_pos = prompt_len - 1

    from llama32_1b_cpu_helpers import rms_norm

    # Final RMSNorm on the single prediction-position row (CPU; <1 ms).
    with prefill_cache.profiler.time_cpu("final_rms_norm"):
        last_hidden = np.asarray(x_bf16, dtype=np.float32)[pred_pos : pred_pos + 1]
        last_normed_bf16 = (
            rms_norm(last_hidden, weights.final_norm).flatten().astype(bfloat16)
        )

    # NPU LM Head GEMV — reuse the decode-cache 8-partition GEMV ELF
    lm_inputs = [last_normed_bf16]
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
    results = decode_cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )
    logits_row = np.concatenate(results, axis=0)[:vocab_size]
    prefill_token = int(np.argmax(logits_row))

    t_prefill = time.time() - t_prefill_start
    # ---- TIMED SECTION END ----
    if not quiet:
        print(f"NPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}")

    return prefill_token, logits_row, k_cache, v_cache, prompt_len


# ---------------------------------------------------------------------------
# Single decode step (one transformer block traversal + LM head)
# ---------------------------------------------------------------------------
#
# Extracted from generate()'s decode loop so the verify subsystem can call
# the exact same code path production uses, instead of reimplementing the
# loop body in NpuRunner. Pure compute — no print / timing / streaming
# state. Caller is responsible for KV-cache positioning (current_pos), for
# feeding next_token's embedding back as x_decode_bf16 on the next step,
# and for any per-token bookkeeping (timing, EOS check, streaming).


def run_npu_decode_step(
    x_decode_bf16,
    weights,
    config,
    decode_cache,
    rope_lut_bf16,
    k_cache,
    v_cache,
    current_pos,
):
    """Run one NPU decode step: 16 transformer blocks + final RMSNorm + LM head.

    Args:
        x_decode_bf16: (emb_dim,) bfloat16 — input embedding for this step.
        weights, config, decode_cache, rope_lut_bf16: passed through to
            run_decode_block + the LM-head GEMV.
        k_cache, v_cache: shape (n_layers, n_kv_heads, max_seq, head_dim).
            run_decode_block writes into [layer_idx, :, current_pos, :].
        current_pos: position to write the new K/V at (and to read prior
            K/V from for attention).

    Returns:
        next_token: int — argmax of the LM-head logits.
        logits: (vocab_size,) f32 — raw LM-head logits (production
            discards with `_`; verify reads for top-k extraction).
    """
    from llama32_1b_cpu_helpers import rms_norm

    vocab_size = weights.lm_head.shape[0]

    # 16 transformer blocks on NPU.
    x = x_decode_bf16.copy()
    for layer_idx in range(config.n_layers):
        t0 = decode_cache.profiler.start_layer()
        x = run_decode_block(
            x,
            weights.layers[layer_idx],
            decode_cache,
            config,
            k_cache[layer_idx],
            v_cache[layer_idx],
            current_pos,
            rope_lut_bf16,
        )
        decode_cache.profiler.end_layer(layer_idx, t0)

    # Final RMSNorm (CPU, single row — cheap).
    with decode_cache.profiler.time_cpu("final_rms_norm"):
        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, config.emb_dim),
            weights.final_norm.astype(np.float32),
        )

    # NPU LM Head: 8-partition GEMV, single XRT call.
    x_lm = x_normed.flatten().astype(bfloat16)
    lm_inputs = [x_lm]
    lm_output_indices = []
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
        lm_output_indices.append(2 + 2 * p)
    lm_results = decode_cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=lm_output_indices,
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )

    # Assemble logits from 8 partitions.
    logits = np.zeros(vocab_size, dtype=np.float32)
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        logits[n_start:n_end] = lm_results[2 + 2 * p][: n_end - n_start].astype(
            np.float32
        )
    next_token = int(np.argmax(logits))
    return next_token, logits


# ---------------------------------------------------------------------------
# Full inference: NPU prefill + NPU decode
# ---------------------------------------------------------------------------


def generate(
    prompt_tokens,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    tokenizer,
    n_tokens=10,
    profile=False,
    cpu_attn=True,
    on_token=None,
    ttft_start=None,
):
    """Run NPU prefill + NPU decode generation.

    Token 0 = from prefill, tokens 1+ = from decode.
    Both prefill and decode use NPU LM Head.

    `ttft_start`, if provided, is the perf_counter() reading from the
    caller before tokenization. The Time-To-First-Token (TTFT) message
    measures from that point to when the first token is decoded — i.e.
    tokenize + EOS-pad + NPU prefill + LM head. This matches the
    standard vLLM/TGI/TRT-LLM TTFT definition (end-to-end submit →
    first token). If not provided, TTFT is measured from the start
    of NPU prefill only.
    """
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens
    streaming = on_token is not None
    ttft_includes_tokenize = ttft_start is not None
    if ttft_start is None:
        ttft_start = time.perf_counter()

    if not streaming:
        print(f"\n{'='*60}")
        print(f"LLAMA Inference: prompt_len={seq_len}, n_tokens={n_tokens}")
        print(f"{'='*60}\n")

    # --- Phase 1: NPU Prefill ---
    # logits_row is unused in production; verify reads it via run_npu_prefill directly.
    # quiet=True: the unified TTFT line below covers the user-visible timing;
    # run_npu_prefill's own "NPU prefill done in X.XXs" would be redundant.
    prefill_token, _logits_row, k_cache, v_cache, prompt_len = run_npu_prefill(
        prompt_tokens,
        weights,
        config,
        prefill_cache,
        decode_cache,
        rope_lut_bf16,
        max_seq,
        tokenizer=tokenizer,
        cpu_attn=cpu_attn,
        profile=profile,
        quiet=True,
    )

    ttft = time.perf_counter() - ttft_start
    if not streaming:
        scope = (
            "tokenize + EOS-pad + NPU prefill + LM head"
            if ttft_includes_tokenize
            else "NPU prefill + LM head"
        )
        print(
            f"Time to first token (TTFT): {ttft:.2f}s ({scope}). "
            f"First token: {prefill_token}"
        )

    # --- Phase 2: NPU Decode ---
    generated_tokens = [prefill_token]  # Token 0 = from prefill
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    # Streaming state — only used when on_token is provided.
    stream_state = _StreamState() if streaming else None
    if streaming:
        on_token(prefill_token, _delta_text(tokenizer, generated_tokens, stream_state))

    if not streaming:
        print(f"\nDecoding {n_tokens} tokens (token 1 to {n_tokens})...")
    t_decode_start = time.time()

    for token_idx in range(n_tokens):
        # One decode step (16 transformer blocks + final RMSNorm + LM head).
        # Verify subsystem calls the same function — keeps "what we test" and
        # "what we deploy" identical. Per-layer / per-call timings are
        # recorded automatically inside cache.load_and_run when the
        # decode_cache's Profiler is enabled (--profile).
        next_token, _logits = run_npu_decode_step(
            x_decode,
            weights,
            config,
            decode_cache,
            rope_lut_bf16,
            k_cache,
            v_cache,
            current_pos,
        )

        generated_tokens.append(next_token)
        current_pos += 1
        # Embed lookup for next iteration's input (CPU).
        with decode_cache.profiler.time_cpu("embed_lookup"):
            x_decode = weights.embed_table[next_token].astype(bfloat16)

        if streaming:
            on_token(next_token, _delta_text(tokenizer, generated_tokens, stream_state))

        # Stop on EOS or EOT (instruct model emits <|eot_id|> = 128009)
        if next_token in (tokenizer.eos_token_id, 128009):
            break

    t_decode = time.time() - t_decode_start
    n_generated = len(generated_tokens) - 1  # exclude prefill token

    if not streaming:
        print(f"\nGenerated {n_generated} tokens in {t_decode:.2f}s")
        print(f"Tokens/second: {n_generated / t_decode:.2f}")
        print(f"Time/token: {t_decode / n_generated * 1000:.0f}ms")

    # Fine-grained profiling report. Each Profiler is a noop unless
    # build_session enabled it for --profile (production path is identical
    # to make run; verify path also leaves these disabled).
    if prefill_cache.profiler.enabled or decode_cache.profiler.enabled:
        _print_dataflow_summary(
            prefill_cache, decode_cache, config.n_layers, n_generated
        )
    if prefill_cache.profiler.enabled:
        print(f"\n{'='*60}\nPREFILL — detail tables")
        prefill_cache.profiler.report()
    if decode_cache.profiler.enabled:
        print(f"\n{'='*60}\nDECODE ({n_generated} tokens) — detail tables")
        decode_cache.profiler.report()

    return generated_tokens


def _avg(times):
    return sum(times) / len(times) if times else 0.0


def _print_dataflow_summary(prefill_cache, decode_cache, n_layers, n_decode_tokens):
    """Architecture-aware dataflow-ordered summary that mirrors the SVG in
    docs/PROFILE.html. Generic detail tables (Per-Layer / NPU XRT / CPU Op
    / Fine-Grained) print after this from each Profiler.report()."""
    pp = prefill_cache.profiler
    dp = decode_cache.profiler

    # Convert kernel_times / cpu_times entries to ms averages.
    def k_avg(prof, name):
        ts = prof.kernel_times.get(name, [])
        return _avg(ts) * 1000.0

    def c_avg(prof, name):
        ts = prof.cpu_times.get(name, [])
        return _avg(ts) * 1000.0

    def k_count(prof, name):
        return len(prof.kernel_times.get(name, []))

    def c_count(prof, name):
        return len(prof.cpu_times.get(name, []))

    print(f"\n{'='*68}")
    print("END-TO-END DATAFLOW (per make profile, dataflow order)")
    print(f"{'='*68}")

    # Preprocessing reminder (one-time setup, not per-query).
    prep_s = getattr(pp, "preprocessing_s", None)
    if prep_s is not None:
        print(
            f"\n  Preprocessing (one-time, prepare_runtime): {prep_s:.1f} s"
            f"   ← not counted in per-query wall below"
        )

    # ---- PREFILL ----
    if pp.enabled:
        print(f"\n--- PREFILL (per query, seq_len padded) ---")
        rms_p = k_avg(pp, "rms_gemms_rope")
        fa_p = k_avg(pp, "flash_attn")
        offn_p = k_avg(pp, "o_ffn")
        kv_extract = c_avg(pp, "kv_cache_extract")
        layer_avg = (
            sum(t for _, t in pp.layer_times) * 1000.0 / n_layers
            if pp.layer_times
            else 0
        )
        layer_npu_cpu = rms_p + fa_p + offn_p + kv_extract
        layer_sched = max(0.0, layer_avg - layer_npu_cpu)
        tok = c_avg(pp, "tokenize") * c_count(pp, "tokenize")
        pad = c_avg(pp, "eos_pad") * c_count(pp, "eos_pad")
        embed = c_avg(pp, "embed_lookup") * c_count(pp, "embed_lookup")
        final_n = c_avg(pp, "final_rms_norm") * c_count(pp, "final_rms_norm")
        # LM head is recorded in decode_cache (production runs the prefill-end
        # LM head through the same 8-partition ELF).
        lm_total = sum(dp.kernel_times.get("lm_head_gemv", [])) * 1000.0
        n_lm = k_count(dp, "lm_head_gemv")
        # Per-token tracking: out of N lm_head calls, 1 is the prefill end
        # and N-1 are decode tokens. Approximate prefill LM head as the avg.
        lm_prefill = lm_total / n_lm if n_lm else 0.0
        layer_total = layer_avg * n_layers
        e2e = tok + pad + embed + layer_total + final_n + lm_prefill

        col = 38  # label column width

        def row(label, kind, ms, note=""):
            print(f"  {label:<{col}}{kind:<6}{ms:>8.2f} ms  {note}")

        row("tokenize", "CPU", tok)
        row("eos_pad", "CPU", pad)
        row("embed_lookup", "CPU", embed)
        print(
            f"  ┌─ Decoder block × {n_layers} (per layer) ─────────────────────────────┐"
        )
        row("  rms_gemms_rope.elf", "NPU", rms_p)
        row("  flash_attn.elf", "NPU", fa_p)
        row("  o_ffn.elf", "NPU", offn_p)
        row("  kv_cache_extract", "CPU", kv_extract)
        row("  python/numpy scheduling", "—", layer_sched)
        print(f"  │  {'─'*52}")
        print(f"  │  {'per-layer wall':<{col-3}}{'':<6}{layer_avg:>8.2f} ms")
        print(f"  └──────────────────────────────────────────────────────────┘")
        print(
            f"  {'× ' + str(n_layers) + ' layers':<{col}}{'':<6}{layer_total:>8.2f} ms"
        )
        row("final_rms_norm", "CPU", final_n)
        row("lm_head_gemv.elf", "NPU", lm_prefill)
        print(f"  {'─'*60}")
        print(f"  {'End-to-end (prefill, per query)':<{col}}{'':<6}{e2e:>8.2f} ms")

    # ---- DECODE ----
    if dp.enabled and n_decode_tokens > 0:
        print(f"\n--- DECODE (avg per token, {n_decode_tokens} tokens) ---")
        rms_d = k_avg(dp, "rms_gemv_rope")
        ogf_d = k_avg(dp, "o_gemv_ffn")
        dec_attn = c_avg(dp, "decode_attention_cpu")
        embed_d = c_avg(dp, "embed_lookup")
        final_d = c_avg(dp, "final_rms_norm")
        lm_d = k_avg(dp, "lm_head_gemv")
        layer_d = (
            sum(t for _, t in dp.layer_times) * 1000.0 / (n_layers * n_decode_tokens)
            if dp.layer_times
            else 0
        )
        layer_d_sub = rms_d + ogf_d + dec_attn
        layer_d_sched = max(0.0, layer_d - layer_d_sub)
        e2e_d = embed_d + layer_d * n_layers + final_d + lm_d

        col = 38

        def row(label, kind, ms, note=""):
            print(f"  {label:<{col}}{kind:<6}{ms:>8.2f} ms  {note}")

        row("embed_lookup", "CPU", embed_d)
        print(
            f"  ┌─ Decoder block × {n_layers} (per layer, per token) ─────────────────┐"
        )
        row("  rms_gemv_rope.elf", "NPU", rms_d)
        row("  decode_attention_cpu", "CPU", dec_attn)
        row("  o_gemv_ffn.elf", "NPU", ogf_d)
        row("  python/numpy scheduling", "—", layer_d_sched)
        print(f"  │  {'─'*52}")
        print(f"  │  {'per-layer wall':<{col-3}}{'':<6}{layer_d:>8.2f} ms")
        print(f"  └──────────────────────────────────────────────────────────┘")
        print(
            f"  {'× ' + str(n_layers) + ' layers':<{col}}{'':<6}{layer_d * n_layers:>8.2f} ms"
        )
        row("final_rms_norm", "CPU", final_d)
        row("lm_head_gemv.elf", "NPU", lm_d)
        print(f"  {'─'*60}")
        print(f"  {'End-to-end (per token)':<{col}}{'':<6}{e2e_d:>8.2f} ms")

        # Per-token trend: did wall time grow with token index? (decode CPU
        # attention is O(current_pos), but with 2048-token prompt the slope
        # is usually invisible for short generations.)
        walls = dp.per_token_walls_ms(n_layers)
        if len(walls) >= 3:
            avg_w = sum(walls) / len(walls)
            mn = min(walls)
            mx = max(walls)
            # Show first/middle/last samples for the slope.
            first = walls[0]
            mid = walls[len(walls) // 2]
            last = walls[-1]
            slope = last - first
            slope_pct = (slope / first * 100.0) if first else 0
            print(
                f"\n  Per-token layer-loop wall trend (decode-attention CPU scales with KV cache size):"
            )
            print(
                f"    token  1 = {first:6.2f} ms   token {len(walls)//2 + 1:2d} = {mid:6.2f} ms   "
                f"token {len(walls):2d} = {last:6.2f} ms"
            )
            print(
                f"    min = {mn:6.2f} ms   max = {mx:6.2f} ms   avg = {avg_w:6.2f} ms   "
                f"first→last drift = {slope:+.2f} ms ({slope_pct:+.1f}%)"
            )


# ---------------------------------------------------------------------------
# Session lifecycle and per-turn execution
# ---------------------------------------------------------------------------


def build_session(args) -> Session:
    """One-time setup: load kernel caches, weights, tokenizer, RoPE LUT,
    and run prepare_runtime(). Safe to call once per process; do not call
    twice (prepare_runtime mutates `weights` with idempotency guards but the
    intent is one-shot)."""
    config = LlamaConfig()
    seq_len = 2048

    # Each cache gets its own Profiler so the final report can separate
    # prefill from decode phases. Profilers are enabled only under
    # --profile; otherwise every record_* call is a noop (production
    # path is identical to make run).
    prefill_cache = KernelCache(
        "prefill_kernel_cache",
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )
    decode_cache = KernelCache(
        "decode_kernel_cache",
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )

    if not args.run_only:
        print("Compiling prefill kernels...")
        compile_all_kernels(prefill_cache, config, seq_len, cpu_attn=args.cpu_attn)
        print("\nCompiling decode kernels...")
        compile_decode_kernels(decode_cache, config)

    if args.compile_only:
        # Stable end-of-compile marker for CI (run_npu2_compile.lit FileCheck).
        # Do not change without also updating the lit CHECK line.
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    model_id = (
        "meta-llama/Llama-3.2-1B-Instruct"
        if args.model == "instruct"
        else "meta-llama/Llama-3.2-1B"
    )
    print(f"\nLoading weights ({model_id})...")
    weights = load_weights(model_id)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=seq_len + args.n_tokens,
    ).astype(bfloat16)

    prepare_runtime(
        prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
    )

    return Session(
        config=config,
        seq_len=seq_len,
        weights=weights,
        tokenizer=tokenizer,
        prefill_cache=prefill_cache,
        decode_cache=decode_cache,
        rope_lut_bf16=rope_lut_bf16,
        model_variant=args.model,
    )


def _tokenize_prompt(session: Session, prompt_text: str) -> list:
    """Apply chat template if instruct model, then tokenize. Does NOT pad."""
    if session.model_variant == "instruct":
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = session.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return session.tokenizer.encode(chat_text)
    return session.tokenizer.encode(prompt_text)


def run_once(
    session: Session,
    prompt_text: str,
    *,
    n_tokens: int,
    profile: bool = False,
    cpu_attn: bool = True,
    on_token: Optional[Callable[[int, str], None]] = None,
) -> tuple[list, int]:
    """Tokenize, pad to seq_len, and call generate(). Returns
    (generated_token_ids, prompt_len_actual)."""
    # Tokenize + EOS-pad are part of the per-query critical path (standard
    # TTFT scope per vLLM/TGI/TRT-LLM), so we time them with the rest of
    # prefill: ttft_start is captured BEFORE tokenize, then handed to
    # generate(), which prints the unified "Time to first token (TTFT)"
    # line covering tokenize + EOS-pad + NPU prefill + LM head.
    ttft_start = time.perf_counter()
    with session.prefill_cache.profiler.time_cpu("tokenize"):
        tokens = _tokenize_prompt(session, prompt_text)
    prompt_len_actual = len(tokens)
    with session.prefill_cache.profiler.time_cpu("eos_pad"):
        if len(tokens) < session.seq_len:
            tokens = tokens + [session.tokenizer.eos_token_id] * (
                session.seq_len - len(tokens)
            )

    generated = generate(
        tokens,
        session.weights,
        session.config,
        session.prefill_cache,
        session.decode_cache,
        session.rope_lut_bf16,
        tokenizer=session.tokenizer,
        n_tokens=n_tokens,
        profile=profile,
        cpu_attn=cpu_attn,
        on_token=on_token,
        ttft_start=ttft_start,
    )
    return generated, prompt_len_actual


def _print_one_shot_output(
    session: Session,
    prompt_text: str,
    generated: list,
    prompt_len_actual: int,
) -> None:
    """Format and print the final output for non-interactive mode."""
    print(f"\n{'='*60}")
    if session.model_variant == "instruct":
        response = session.tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"Q: {prompt_text}")
        print(f"A: {response}")
    else:
        # Reconstruct the unpadded prompt + generated tokens.
        prompt_tokens = _tokenize_prompt(session, prompt_text)
        print(f"Generated text:")
        print(f"{'='*60}")
        all_tokens = prompt_tokens[:prompt_len_actual] + generated
        print(session.tokenizer.decode(all_tokens))


def repl_loop(session: Session, args) -> None:
    """Interactive REPL: prompt-> stream-> repeat. Each turn is independent."""
    print("\nInteractive mode — Ctrl-D or /quit to exit.")
    print("Each prompt is independent (no chat memory).\n")

    def _stream_cb(_token_id: int, delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return

        # Length guard.
        check_ids = _tokenize_prompt(session, prompt)
        if len(check_ids) > session.seq_len:
            print(
                f"Prompt too long ({len(check_ids)} > {session.seq_len} tokens). "
                "Skipped."
            )
            continue

        sys.stdout.write("\nResponse: ")
        sys.stdout.flush()
        try:
            run_once(
                session,
                prompt,
                n_tokens=args.n_tokens,
                # profile is forced to False by the --interactive mutex
                # block in __main__; pass through as the single source of truth.
                profile=args.profile,
                cpu_attn=args.cpu_attn,
                on_token=_stream_cb,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue

        # Blank line before next "Prompt>".
        print()
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLAMA-3.2-1B Inference (NPU)")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile both prefill and decode kernels, then exit",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Use cached kernels (skip compilation)",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number of decode tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-token timing instrumentation",
    )
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Use CPU attention for prefill (default: NPU flash attention)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "instruct"],
        default="instruct",
        help="Model variant: instruct (Q&A, default) or base (completion)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drop into a REPL after runtime prep. Loops on prompts; each is independent.",
    )
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if not args.run_only:
            parser.error("--interactive requires --run-only")
        if args.prompt != parser.get_default("prompt"):
            print(
                "WARNING: --prompt is ignored in --interactive mode.",
                file=sys.stderr,
            )
        if args.profile:
            print(
                "WARNING: --profile is ignored in --interactive mode.",
                file=sys.stderr,
            )
            args.profile = False

    session = build_session(args)

    if args.interactive:
        repl_loop(session, args)
    else:
        generated, prompt_len_actual = run_once(
            session,
            args.prompt,
            n_tokens=args.n_tokens,
            profile=args.profile,
            cpu_attn=args.cpu_attn,
        )
        _print_one_shot_output(session, args.prompt, generated, prompt_len_actual)
