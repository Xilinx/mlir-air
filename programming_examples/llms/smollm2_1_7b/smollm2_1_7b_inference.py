# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B Inference on MLIR-AIR (NPU2) — thin inheritance entry point.

SmolLM2 is a bit-for-bit llama kernel sequence, so the heavy machinery (Session,
prepare_runtime, run_npu_prefill, run_npu_decode_step, generate, REPL, profiling
printers) is reused verbatim from `llama32_1b_inference`. SmolLM2 is pure MHA
(n_kv_heads == n_heads, kv_dim == emb_dim); the shared
llama32_1b_prefill.run_transformer_block / preload_prefill_weights are
registry-driven and MHA-safe (they query gemm_registry_config per shape for the
fused-cast f32 C-scratch args), so no fork/monkeypatch is needed. This module
only:

  1. Supplies SmolLM2's config / weights / RoPE / model IDs.
  2. Provides a `build_session` + CLI mirroring the reference's, using (1).

Usage:
  python3 smollm2_1_7b_inference.py --compile-only
  python3 smollm2_1_7b_inference.py --run-only --n-tokens 100 --model base
  python3 smollm2_1_7b_inference.py --run-only --profile --n-tokens 20
"""

from pathlib import Path
import sys

from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG_EXAMPLES = _LLMS_DIR.parent
_LLAMA_REF = _LLMS_DIR / "llama32_1b"
for _p in (str(_PROG_EXAMPLES), str(_LLMS_DIR), str(_LLAMA_REF), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- (1) SmolLM2 config / weights / RoPE / model IDs. ---
from smollm2_1_7b_weights import (  # noqa: E402
    LlamaConfig,
    load_weights,
    generate_rope_lut,
)
from shared.infra.cache import KernelCache, Profiler  # noqa: E402
from llama32_1b_prefill import compile_all_kernels  # noqa: E402
from llama32_1b_decode import compile_decode_kernels  # noqa: E402

# Reuse the reference's Session machinery + run loops verbatim.
from llama32_1b_inference import (  # noqa: E402
    Session,
    prepare_runtime,
    run_once,
    repl_loop,
    _print_one_shot_output,
)

MODEL_CHOICES = {
    "base": "HuggingFaceTB/SmolLM2-1.7B",
    "instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
DEFAULT_MODEL = "base"
SEQ_LEN = 2048


def build_session(args) -> Session:
    """One-time setup: compile/load caches, weights, tokenizer, RoPE LUT, and
    prepare_runtime(). Mirrors llama32_1b_inference.build_session with SmolLM2
    config + the MHA-safe prefill patched in at import time."""
    config = LlamaConfig()
    seq_len = SEQ_LEN

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
        # Stable end-of-compile marker for CI (mirrors reference lit CHECK).
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmolLM2-1.7B Inference (NPU)")
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--n-tokens", type=int, default=100)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--cpu-attn", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        if args.compile_only:
            parser.error("--interactive cannot be combined with --compile-only")
        if not args.run_only:
            parser.error("--interactive requires --run-only")
        if args.profile:
            print(
                "WARNING: --profile is ignored in --interactive mode.", file=sys.stderr
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
