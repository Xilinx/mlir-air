# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B int4-AWQ end-to-end inference on MLIR-AIR (NPU2).

Loads a HuggingFace AutoAWQ checkpoint, runs NPU prefill (bf16 stitchers
on dequantized AWQ weights), then NPU int4 decode token-by-token.

This is the e2e entry. Building blocks:
- bf16 prefill stitchers (`llama32_1b_int4_prefill.py` --prefill-dtype=bf16
  compiles them; we re-use the same kernel builders directly from the bf16
  sibling dir)
- int4 decode kernels (`llama32_1b_int4_decode.py`)

Prefill weight path: AWQ → dense bf16 dequant on each LayerWeights field
(wq/wk/wv/wo/w_gate/w_up/w_down) → existing bf16 NPU prefill stitchers.
Decode weight path: AWQ → packed BO on `_wq_packed`/... → int4 NPU decode
ELFs (`rms_qkv_int4_rope`, `o_gemv_ffn_int4`).

Both load in a single safetensors scan via `load_weights_awq`.

Usage:
    cd build_peano
    python3 ../llama32_1b_int4_inference.py --compile-only \
        --model-path amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead
    python3 ../llama32_1b_int4_inference.py --run-only --n-tokens 30 \
        --prompt "Once upon a time"
    python3 ../llama32_1b_int4_inference.py --run-only --interactive
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROG_EXAMPLES = os.path.dirname(_THIS_DIR)
_LLAMA_BF16 = os.path.join(_PROG_EXAMPLES, "llama32_1b")
for _p in (_PROG_EXAMPLES, _LLAMA_BF16, _THIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from llama32_1b_weights import LlamaConfig, generate_rope_lut  # noqa: E402
from shared.infra.cache import KernelCache, Profiler  # noqa: E402
from shared.infra.external_kernels import (  # noqa: E402
    compile_all_external_kernels,
)
from shared.infra.backend_presets import (  # noqa: E402
    LM_GEMV_BACKEND,
    RGR_INT4_BACKEND,
    OGF_INT4_BACKEND,
)
from llama32_1b_int4_weights import load_weights_awq  # noqa: E402
from llama32_1b_int4_decode import (  # noqa: E402
    compile_decode_kernels,
    run_decode_block,
)

# bf16 prefill kernel builders + per-layer runner — re-used as-is from the
# bf16 sibling dir. The bf16 fields (`_wq_t`, `wq` etc.) that these consume
# are populated by `load_weights_awq`'s dequant-to-bf16 pass below.
from llama32_1b_prefill import (  # noqa: E402
    compile_all_kernels as compile_prefill_kernels,
    run_transformer_block as run_prefill_block,
    preload_prefill_weights,
)

import contextlib  # noqa: E402

# Decode LM Head constants
_LM_N_PART = 16384
_LM_N_PARTITIONS = 8


@contextlib.contextmanager
def _multi_launch_dir(dir_path: str):
    """Make `multi_launch_builder` resolve to `dir_path/multi_launch_builder`.

    Both the bf16 and int4 dirs ship a `multi_launch_builder/__init__.py`
    package. Once Python caches one of them as `sys.modules[
    "multi_launch_builder"]`, all later `from multi_launch_builder.X` lookups
    resolve against that cached package — so the second dir's modules become
    invisible. Flush the cache and re-pin sys.path[0] inside this context.
    Caller should wrap the prefill (bf16) and decode (int4) compile phases.
    """
    for mod_name in list(sys.modules.keys()):
        if mod_name == "multi_launch_builder" or mod_name.startswith(
            "multi_launch_builder."
        ):
            del sys.modules[mod_name]
    saved = list(sys.path)
    while dir_path in sys.path:
        sys.path.remove(dir_path)
    sys.path.insert(0, dir_path)
    try:
        yield
    finally:
        sys.path[:] = saved
        for mod_name in list(sys.modules.keys()):
            if mod_name == "multi_launch_builder" or mod_name.startswith(
                "multi_launch_builder."
            ):
                del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Streaming helpers (BPE-safe incremental decode output)
# ---------------------------------------------------------------------------


class _StreamState:
    def __init__(self) -> None:
        self.printed_len: int = 0


def _delta_text(tokenizer: Any, ids: list[int], state: _StreamState) -> str:
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    config: Any
    seq_len: int
    weights: Any
    tokenizer: Any
    prefill_cache: Any
    decode_cache: Any
    rope_lut_bf16: np.ndarray
    model_path: str


# ---------------------------------------------------------------------------
# Runtime preparation
# ---------------------------------------------------------------------------


def prepare_runtime(
    prefill_cache,
    decode_cache,
    weights,
    config,
    seq_len,
    rope_lut_bf16,
):
    """One-time setup: external kernels, weight transposes, BO preloads.

    Touches three weight surfaces:
    - bf16 dequant fields (wq/wk/wv/wo/w_gate/w_up/w_down) for NPU bf16 prefill
    - packed BO attrs (_wq_packed/...) already set by load_weights_awq for decode
    - decode-side `_wgateup_t` / `_packed_rms_buf` shims for the int4 FFN ELF
    """
    print(f"\n{'='*60}")
    print("Preparing runtime (one-time init)...")
    print(f"{'='*60}")
    t0 = time.time()

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    compile_all_external_kernels(head_dim=head_dim, quant="awq")

    if not hasattr(weights, "_decode_weights_transposed"):
        print("  Pre-transposing bf16 weights for prefill GEMV...")
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

    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    # bf16 prefill weight + RoPE LUT BO preload (same path as bf16 sibling).
    preload_prefill_weights(weights, config, prefill_cache, seq_len, rope_lut_bf16)

    # Decode-side preload: packed BOs into the int4 decode ELFs.
    _preload_decode_weights(decode_cache, weights, config)

    t_prep = time.time() - t0
    print(f"  Runtime prepared in {t_prep:.1f}s")
    prefill_cache.profiler.preprocessing_s = t_prep
    decode_cache.profiler.preprocessing_s = t_prep


def _preload_decode_weights(decode_cache, weights, config):
    """Preload int4 decode weights into per-layer BOs.

    Same shape as the bf16 sibling's preload but with packed-uint8 slots in
    the static positions (3/5/7 for rms_qkv, 0/7/12 for o_gemv_ffn). The
    `_packed_rms_buf` shim (row 0 = scratch, row 1 = ffn_norm_w) is built
    here once per layer and stashed on LayerWeights for the decode loop.
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

    print("  Pre-loading int4 decode weights into per-layer BOs...")
    _was_enabled = decode_cache.profiler.enabled
    decode_cache.profiler.enabled = False

    rope_lut_q_dummy = np.zeros(n_heads * head_dim, dtype=bfloat16)
    rope_lut_k_dummy = np.zeros(n_kv_heads * head_dim, dtype=bfloat16)

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        decode_cache.load_and_run(
            "rms_qkv_int4_rope",
            RGR_INT4_BACKEND,
            np.zeros(emb_dim, dtype=bfloat16),  # x_in
            lw.attn_norm.reshape(emb_dim).astype(bfloat16),  # norm_w
            np.zeros(emb_dim, dtype=bfloat16),  # normed
            lw._wq_packed,  # wq (packed-i8)
            np.zeros(emb_dim, dtype=bfloat16),  # q
            lw._wk_packed,  # wk (packed-i8)
            np.zeros(kv_dim, dtype=bfloat16),  # k
            lw._wv_packed,  # wv (packed-i8)
            np.zeros(kv_dim, dtype=bfloat16),  # v
            rope_lut_q_dummy,
            rope_lut_k_dummy,
            np.zeros(emb_dim, dtype=bfloat16),  # q_roped
            np.zeros(kv_dim, dtype=bfloat16),  # k_roped
            output_indices=[8, 11, 12],
            static_input_indices={1, 3, 5, 7},
            intermediate_indices={2, 4, 6, 8, 11, 12},
            bo_key=f"rms_qkv_int4_rope_L{layer_idx}",
        )

        # Build the [2, emb_dim] packed RMS-input buffer for o_gemv_ffn_int4:
        # row 0 = stage-1 in-kernel scratch (left zero), row 1 = ffn_norm_w.
        packed = np.empty((2, emb_dim), dtype=bfloat16)
        packed[0] = 0.0
        packed[1] = lw.ffn_norm.reshape(emb_dim).astype(bfloat16)
        lw._packed_rms_buf = packed

        z_emb = np.zeros(emb_dim, dtype=bfloat16)
        z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
        z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)

        decode_cache.load_and_run(
            "o_gemv_ffn_int4",
            OGF_INT4_BACKEND,
            lw._wo_packed,  # arg0 wo (static, packed-i8)
            z_emb,  # arg1 attn_out
            z_emb,  # arg2 (dead)
            z_emb,  # arg3 x_residual
            z_emb,  # arg4 (dead)
            z_emb,  # arg5 (dead)
            lw._packed_rms_buf,  # arg6 packed (static)
            lw._wgateup_packed,  # arg7 w_gateup (static, packed-i8)
            z_hidden,  # arg8 (dead)
            z_hidden_emb,  # arg9 (dead)
            z_hidden,  # arg10 (dead)
            z_hidden,  # arg11 swiglu
            lw._wdown_packed,  # arg12 wdown (static, packed-i8)
            z_emb,  # arg13 (dead)
            z_emb,  # arg14 output
            output_indices=[14],
            static_input_indices={0, 6, 7, 12},
            intermediate_indices={2, 4, 5, 8, 9, 10, 11, 13, 14},
            bo_key=f"o_gemv_ffn_int4_L{layer_idx}",
        )

    # LM Head GEMV weights (8 partitions) — same bf16 lm_head as bf16 dir.
    weights._lm_weight_parts_gemv = []
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        w = np.zeros((_LM_N_PART, emb_dim), dtype=bfloat16)
        w[: n_end - n_start, :] = np.ascontiguousarray(
            weights.lm_head[n_start:n_end, :]
        ).astype(bfloat16)
        weights._lm_weight_parts_gemv.append(w)

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

    decode_cache.profiler.enabled = _was_enabled
    weights._decode_weights_preloaded_to_bos = True
    print(f"  Pre-loaded {config.n_layers} decode layers + LM Head")


# ---------------------------------------------------------------------------
# Prefill + decode step entries
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
    quiet=False,
):
    """NPU bf16 prefill on dequantized AWQ weights → first token + KV cache."""
    from llama32_1b_cpu_helpers import rms_norm

    seq_len = len(token_ids)
    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    vocab_size = weights.lm_head.shape[0]

    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    with prefill_cache.profiler.time_cpu("embed_lookup"):
        x_bf16 = weights.embed_table[token_ids].astype(np.float32).astype(bfloat16)

    if not quiet:
        print(f"Running NPU prefill ({config.n_layers} layers, seq_len={seq_len})...")
    t_prefill_start = time.time()

    for layer_idx in range(config.n_layers):
        t0 = prefill_cache.profiler.start_layer()
        x_bf16, intermediates = run_prefill_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            cpu_attn=cpu_attn,
            verbose=False,
        )
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

    prompt_len = len([t for t in token_ids if t != tokenizer.eos_token_id])
    pred_pos = prompt_len - 1

    with prefill_cache.profiler.time_cpu("final_rms_norm"):
        last_hidden = np.asarray(x_bf16, dtype=np.float32)[pred_pos : pred_pos + 1]
        last_normed_bf16 = (
            rms_norm(last_hidden, weights.final_norm).flatten().astype(bfloat16)
        )

    lm_inputs = [last_normed_bf16]
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
    lm_results = decode_cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )
    logits_row = np.concatenate(lm_results, axis=0)[:vocab_size]
    prefill_token = int(np.argmax(logits_row))

    t_prefill = time.time() - t_prefill_start
    if not quiet:
        print(f"NPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}")
    return prefill_token, logits_row, k_cache, v_cache, prompt_len


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
    """One int4 decode token: 16 transformer blocks + final RMSNorm + LM head."""
    from llama32_1b_cpu_helpers import rms_norm

    vocab_size = weights.lm_head.shape[0]

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

    with decode_cache.profiler.time_cpu("final_rms_norm"):
        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, config.emb_dim),
            weights.final_norm.astype(np.float32),
        )

    x_lm = x_normed.flatten().astype(bfloat16)
    lm_inputs = [x_lm]
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
    lm_results = decode_cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )

    logits = np.zeros(vocab_size, dtype=np.float32)
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        logits[n_start:n_end] = lm_results[2 + 2 * p][: n_end - n_start].astype(
            np.float32
        )
    return int(np.argmax(logits)), logits


# ---------------------------------------------------------------------------
# Full inference
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
    cpu_attn=True,
    on_token=None,
    ttft_start=None,
):
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens
    streaming = on_token is not None
    if ttft_start is None:
        ttft_start = time.perf_counter()

    if not streaming:
        print(f"\n{'='*60}")
        print(f"LLAMA int4 Inference: prompt_len={seq_len}, n_tokens={n_tokens}")
        print(f"{'='*60}\n")

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
        quiet=streaming,
    )

    ttft = time.perf_counter() - ttft_start
    if not streaming:
        print(f"Time to first token (TTFT): {ttft:.2f}s. First token: {prefill_token}")

    generated_tokens = [prefill_token]
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    stream_state = _StreamState() if streaming else None
    if streaming:
        on_token(prefill_token, _delta_text(tokenizer, generated_tokens, stream_state))

    if not streaming:
        print(f"\nDecoding {n_tokens} tokens...")
    t_decode_start = time.time()

    for _ in range(n_tokens):
        next_token, _ = run_npu_decode_step(
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
        with decode_cache.profiler.time_cpu("embed_lookup"):
            x_decode = weights.embed_table[next_token].astype(bfloat16)
        if streaming:
            on_token(next_token, _delta_text(tokenizer, generated_tokens, stream_state))
        if next_token in (tokenizer.eos_token_id, 128009):
            break

    t_decode = time.time() - t_decode_start
    n_generated = len(generated_tokens) - 1
    if not streaming:
        print(f"\nGenerated {n_generated} tokens in {t_decode:.2f}s")
        if n_generated > 0:
            print(f"Tokens/second: {n_generated / t_decode:.2f}")
            print(f"Time/token: {t_decode / n_generated * 1000:.0f}ms")

    return generated_tokens


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def build_session(args) -> Session:
    config = LlamaConfig()
    seq_len = 2048

    prefill_cache = KernelCache(
        cache_dir=args.prefill_cache_dir,
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )
    decode_cache = KernelCache(
        cache_dir=args.decode_cache_dir,
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )

    if not args.run_only:
        # Prefill: bf16 NPU stitchers. Reuses the bf16 sibling's
        # `compile_all_kernels` — same kernels as `llama32_1b` baseline.
        # Both dirs ship a `multi_launch_builder` namesake package, so we
        # have to flush the cached package + put bf16 at sys.path[0]
        # around the prefill compile, then swap back for the int4 decode.
        print("Compiling bf16 prefill kernels...")
        with _multi_launch_dir(_LLAMA_BF16):
            compile_prefill_kernels(
                prefill_cache, config, seq_len, cpu_attn=args.cpu_attn
            )
        print("\nCompiling int4 decode kernels...")
        with _multi_launch_dir(_THIS_DIR):
            compile_decode_kernels(decode_cache, config)

    if args.compile_only:
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    print(f"\nLoading AWQ weights ({args.model_path})...")
    weights = load_weights_awq(args.model_path, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=seq_len + args.n_tokens
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
        model_path=args.model_path,
    )


def _tokenize_prompt(session: Session, prompt_text: str) -> list:
    """Apply Llama-3 instruct chat template, then tokenize. Does NOT pad."""
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = session.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return session.tokenizer.encode(chat_text)


def run_once(
    session: Session,
    prompt_text: str,
    *,
    n_tokens: int,
    cpu_attn: bool = True,
    on_token: Optional[Callable[[int, str], None]] = None,
) -> tuple[list, int]:
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
        cpu_attn=cpu_attn,
        on_token=on_token,
        ttft_start=ttft_start,
    )
    return generated, prompt_len_actual


def _print_one_shot_output(session: Session, prompt: str, generated: list) -> None:
    print(f"\n{'='*60}")
    response = session.tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(f"Q: {prompt}")
    print(f"A: {response}")


def repl_loop(session: Session, args) -> None:
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
                cpu_attn=args.cpu_attn,
                on_token=_stream_cb,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        print()
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLAMA-3.2-1B int4-AWQ end-to-end inference (NPU2)"
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Use CPU attention for prefill (default: NPU flash attention)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument(
        "--model-path",
        type=str,
        default="amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead",
        help="HF model id or local dir of an AutoAWQ int4 Llama checkpoint.",
    )
    parser.add_argument(
        "--prefill-cache-dir",
        type=str,
        default=None,
        help="Disk dir for prefill kernel cache (default: shared/infra/kernel_cache).",
    )
    parser.add_argument(
        "--decode-cache-dir",
        type=str,
        default=None,
        help="Disk dir for decode kernel cache.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drop into a REPL after runtime prep. Each prompt is independent.",
    )
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if not args.run_only:
            parser.error("--interactive requires --run-only")

    session = build_session(args)
    if args.interactive:
        repl_loop(session, args)
    else:
        generated, _ = run_once(
            session,
            args.prompt,
            n_tokens=args.n_tokens,
            cpu_attn=args.cpu_attn,
        )
        _print_one_shot_output(session, args.prompt, generated)
