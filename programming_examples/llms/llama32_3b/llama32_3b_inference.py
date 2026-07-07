# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-3B BF16 Inference on MLIR-AIR (NPU2).

Unified driver: NPU prefill (28 layers) + decode (NPU Q/K/V + CPU attn + CPU
FFN) + NPU LM-head.

Pure-Llama. Prefill reuses llama32_1b's config-driven pipeline verbatim. Decode
forks llama32_1b only where emb_dim=3072 forces the FFN/O SwiGLU stage off the
NPU (the fused o_gemv_ffn ELF's swiglu_rms stage overflows the AIE2P BD size
limit at K=3072 — see llama32_3b_decode.py). The NPU still runs decode Q/K/V
projections + RoPE and the LM-head GEMV.

Usage:
    cd build_peano
    python3 ../llama32_3b_inference.py --compile-only
    python3 ../llama32_3b_inference.py --run-only --n-tokens 32 --prompt "..."
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG = _LLMS_DIR.parent
_LLAMA1B = _LLMS_DIR / "llama32_1b"
for _p in (str(_PROG), str(_LLMS_DIR), str(_LLAMA1B), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama32_3b_weights import LlamaConfig, load_weights, generate_rope_lut
from llama32_3b_cpu_helpers import rms_norm
from shared.infra.cache import KernelCache, Profiler
from shared.infra.external_kernels import compile_all_external_kernels
from shared.infra.backend_presets import LM_GEMV_BACKEND, RGR_BACKEND

# Prefill is bit-for-bit llama32_1b (config-driven). Reuse verbatim.
from llama32_3b_prefill import (
    compile_all_kernels,
    run_transformer_block,
    preload_prefill_weights,
)
from llama32_3b_decode import compile_decode_kernels, run_decode_block
import llama32_3b_decode as _decode_mod

MODEL_CHOICES = {
    "base": "meta-llama/Llama-3.2-3B",
    "instruct": "meta-llama/Llama-3.2-3B-Instruct",
}

# LM-head: vocab 128256, 8 partitions x 16384 = 131072 >= 128256.
_LM_N_PART = 16384
_LM_N_PARTITIONS = 8


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


class _StreamState:
    def __init__(self) -> None:
        self.printed_len: int = 0


def _delta_text(tokenizer, ids, state):
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


@dataclass
class Session:
    config: Any
    seq_len: int
    weights: Any
    tokenizer: Any
    prefill_cache: Any
    decode_cache: Any
    rope_lut_bf16: np.ndarray
    model_variant: str


# ---------------------------------------------------------------------------
# Runtime preparation
# ---------------------------------------------------------------------------


def prepare_runtime(
    prefill_cache, decode_cache, weights, config, seq_len, rope_lut_bf16
):
    print(f"\n{'='*60}")
    print("Preparing runtime (one-time init, outside profiling scope)...")
    print(f"{'='*60}")
    t0 = time.time()

    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    compile_all_external_kernels(head_dim=head_dim)

    # Pre-transpose decode weights. KEEP _wgate_t/_wup_t (CPU FFN needs them;
    # unlike <=2048-emb models we do NOT build the interleaved w_gateup / run
    # the NPU o_gemv_ffn preload).
    if not hasattr(weights, "_decode_weights_transposed"):
        print("  Pre-transposing weights for decode GEMV...")
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

    # Prefill BOs (reuse 1b).
    preload_prefill_weights(weights, config, prefill_cache, seq_len, rope_lut_bf16)

    # Decode BOs: rms_gemv_rope per-layer + LM-head GEMV. NO o_gemv_ffn.
    _preload_decode_weights(decode_cache, weights, config)

    t_prep = time.time() - t0
    print(f"  Runtime prepared in {t_prep:.1f}s")
    prefill_cache.profiler.preprocessing_s = t_prep
    decode_cache.profiler.preprocessing_s = t_prep


def _preload_decode_weights(decode_cache, weights, config):
    if hasattr(weights, "_decode_weights_preloaded_to_bos"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim
    vocab_size = weights.lm_head.shape[0]

    print(
        "  Pre-loading decode weights into per-layer BOs (rms_gemv_rope + LM-head)..."
    )
    _was = decode_cache.profiler.enabled
    decode_cache.profiler.enabled = False

    lut_q_dummy = np.zeros(n_heads * head_dim, dtype=bfloat16)
    lut_k_dummy = np.zeros(n_kv_heads * head_dim, dtype=bfloat16)

    for li in range(config.n_layers):
        lw = weights.layers[li]
        decode_cache.load_and_run(
            "rms_gemv_rope",
            RGR_BACKEND,
            np.zeros(emb_dim, dtype=bfloat16),
            lw.attn_norm.reshape(emb_dim).astype(bfloat16),
            np.zeros(emb_dim, dtype=bfloat16),
            lw._wq_t,
            np.zeros(emb_dim, dtype=bfloat16),
            lw._wk_t,
            np.zeros(kv_dim, dtype=bfloat16),
            lw._wv_t,
            np.zeros(kv_dim, dtype=bfloat16),
            lut_q_dummy,
            lut_k_dummy,
            np.zeros(emb_dim, dtype=bfloat16),
            np.zeros(kv_dim, dtype=bfloat16),
            output_indices=[8, 11, 12],
            static_input_indices={1, 3, 5, 7},
            intermediate_indices={2, 4, 6, 8, 11, 12},
            bo_key=f"rms_gemv_rope_L{li}",
        )
        # o_gemv / gate_gemv / up_gemv: standalone GEMV ELFs on NPU. Weight
        # static {0}.
        q_dim = n_heads * head_dim
        decode_cache.load_and_run(
            "o_gemv",
            _decode_mod._gemv_backend(False, "o_gemv"),
            lw._wo_t,
            np.zeros(q_dim, dtype=bfloat16),
            np.zeros(emb_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"o_gemv_L{li}",
        )
        decode_cache.load_and_run(
            "gate_gemv",
            _decode_mod._gemv_backend(False, "gate_gemv"),
            lw._wgate_t,
            np.zeros(emb_dim, dtype=bfloat16),
            np.zeros(hidden_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"gate_gemv_L{li}",
        )
        decode_cache.load_and_run(
            "up_gemv",
            _decode_mod._gemv_backend(False, "up_gemv"),
            lw._wup_t,
            np.zeros(emb_dim, dtype=bfloat16),
            np.zeros(hidden_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"up_gemv_L{li}",
        )

        # down_gemv: M=emb=3072, K=hidden=8192. Standalone GEMV ELF on NPU
        # (the fused cascade overflowed L1, the standalone does not). Weight
        # static {0}.
        decode_cache.load_and_run(
            "down_gemv",
            _decode_mod._gemv_backend(False, "down_gemv", omit_pingpong=True),
            lw._wdown_t,
            np.zeros(hidden_dim, dtype=bfloat16),
            np.zeros(emb_dim, dtype=bfloat16),
            output_indices=[2],
            static_input_indices={0},
            intermediate_indices={2},
            bo_key=f"down_gemv_L{li}",
        )

    # LM-head GEMV (8 partitions).
    weights._lm_weight_parts_gemv = []
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        w = np.zeros((_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
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

    decode_cache.profiler.enabled = _was
    weights._decode_weights_preloaded_to_bos = True
    print(f"  Pre-loaded {config.n_layers} decode layers + LM Head.")


def _run_lm_head(decode_cache, weights, x_normed_bf16, vocab_size):
    lm_inputs = [x_normed_bf16.flatten().astype(bfloat16)]
    out_idx = []
    for p in range(_LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
        out_idx.append(2 + 2 * p)
    res = decode_cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=out_idx,
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )
    logits = np.zeros(vocab_size, dtype=np.float32)
    for p in range(_LM_N_PARTITIONS):
        n_start = p * _LM_N_PART
        n_end = min(n_start + _LM_N_PART, vocab_size)
        logits[n_start:n_end] = res[2 + 2 * p][: n_end - n_start].astype(np.float32)
    return logits


# ---------------------------------------------------------------------------
# NPU Prefill (reuses 1b run_transformer_block) + KV cache extraction
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
    seq_len = len(token_ids)
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    vocab_size = weights.lm_head.shape[0]

    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    with prefill_cache.profiler.time_cpu("embed_lookup"):
        x_bf16 = weights.embed_table[token_ids].astype(np.float32).astype(bfloat16)

    if not quiet:
        print(f"Running NPU prefill ({config.n_layers} layers, seq_len={seq_len})...")
    t_start = time.time()

    for layer_idx in range(config.n_layers):
        t0 = prefill_cache.profiler.start_layer()
        x_bf16, inter = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            cpu_attn=cpu_attn,
            verbose=profile,
        )
        with prefill_cache.profiler.time_cpu("kv_cache_extract"):
            k_roped = inter["k_roped"]
            v_raw = inter["v"]
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
        last_normed = (
            rms_norm(last_hidden, weights.final_norm).flatten().astype(bfloat16)
        )

    logits_row = _run_lm_head(decode_cache, weights, last_normed, vocab_size)
    prefill_token = int(np.argmax(logits_row))

    t_prefill = time.time() - t_start
    if not quiet:
        print(f"NPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}")
    return prefill_token, logits_row, k_cache, v_cache, prompt_len


# ---------------------------------------------------------------------------
# Single decode step (NPU Q/K/V + CPU attn + CPU FFN + NPU LM-head)
# ---------------------------------------------------------------------------


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
            x.astype(np.float32).reshape(1, config.emb_dim), weights.final_norm
        )
    logits = _run_lm_head(
        decode_cache, weights, x_normed.flatten().astype(bfloat16), vocab_size
    )
    next_token = int(np.argmax(logits))
    return next_token, logits


# ---------------------------------------------------------------------------
# Full generation
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
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens
    streaming = on_token is not None
    if ttft_start is None:
        ttft_start = time.perf_counter()

    if not streaming:
        print(f"\n{'='*60}")
        print(f"LLAMA-3.2-3B Inference: prompt_len={seq_len}, n_tokens={n_tokens}")
        print(f"{'='*60}\n")

    prefill_token, _logits, k_cache, v_cache, prompt_len = run_npu_prefill(
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
        print(f"Time to first token (TTFT): {ttft:.2f}s. First token: {prefill_token}")

    generated_tokens = [prefill_token]
    current_pos = prompt_len
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    stream_state = _StreamState() if streaming else None
    if streaming:
        on_token(prefill_token, _delta_text(tokenizer, generated_tokens, stream_state))

    if not streaming:
        print(f"\nDecoding {n_tokens} tokens...")
    t_dec = time.time()

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

    t_decode = time.time() - t_dec
    n_gen = len(generated_tokens) - 1
    if not streaming and n_gen > 0:
        print(
            f"\nGenerated {n_gen} tokens in {t_decode:.2f}s ({n_gen / t_decode:.2f} tok/s)"
        )

    if prefill_cache.profiler.enabled:
        print(f"\n{'='*60}\nPREFILL detail")
        prefill_cache.profiler.report()
    if decode_cache.profiler.enabled:
        print(f"\n{'='*60}\nDECODE detail")
        decode_cache.profiler.report()

    return generated_tokens


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def build_session(args) -> Session:
    config = LlamaConfig()
    seq_len = 2048

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
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    model_id = MODEL_CHOICES.get(args.model, args.model)
    print(f"\nLoading weights ({model_id})...")
    weights = load_weights(model_id, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        model_variant=args.model,
    )


def _tokenize_prompt(session, prompt_text):
    if session.model_variant == "instruct":
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = session.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return session.tokenizer.encode(chat_text)
    return session.tokenizer.encode(prompt_text)


def run_once(
    session, prompt_text, *, n_tokens, profile=False, cpu_attn=True, on_token=None
):
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


def _print_one_shot_output(session, prompt_text, generated, prompt_len_actual):
    print(f"\n{'='*60}")
    if session.model_variant == "instruct":
        response = session.tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"Q: {prompt_text}")
        print(f"A: {response}")
    else:
        prompt_tokens = _tokenize_prompt(session, prompt_text)
        all_tokens = prompt_tokens[:prompt_len_actual] + generated
        print("Generated text:")
        print(session.tokenizer.decode(all_tokens))


def repl_loop(session, args):
    print("\nInteractive mode — Ctrl-D or /quit to exit.\n")

    def _cb(_tid, delta):
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
        sys.stdout.write("\nResponse: ")
        sys.stdout.flush()
        try:
            run_once(
                session,
                prompt,
                n_tokens=args.n_tokens,
                profile=False,
                cpu_attn=args.cpu_attn,
                on_token=_cb,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLAMA-3.2-3B Inference (NPU)")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    # Default: NPU head-first FlashAttention. Pass --cpu-attn to fall back to
    # the FP32 host attention reference.
    parser.add_argument("--cpu-attn", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument(
        "--model", type=str, choices=["base", "instruct"], default="instruct"
    )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if not args.run_only:
            parser.error("--interactive requires --run-only")
        args.profile = False

    session = build_session(args)

    if args.interactive:
        repl_loop(session, args)
    else:
        generated, plen = run_once(
            session,
            args.prompt,
            n_tokens=args.n_tokens,
            profile=args.profile,
            cpu_attn=args.cpu_attn,
        )
        _print_one_shot_output(session, args.prompt, generated, plen)
