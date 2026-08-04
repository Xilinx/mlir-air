# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-0.6B Q4NX Inference on MLIR-AIR (NPU2).

Same on-device pipeline as the bf16 qwen3_0_6b example (NPU prefill + decode +
LM head, QK-norm, tied embeddings), but weights come from FastFlowLM's 4-bit
`model.q4nx` bundle (I8/GGUF-Q4_1-derived codec), dequantized to bf16 on the host
at load. Everything downstream of weight loading is the bf16 qwen3_0_6b driver,
reused verbatim.

Usage:
    cd build_peano
    python3 ../qwen3_0_6b_q4nx_inference.py --compile-only
    python3 ../qwen3_0_6b_q4nx_inference.py --run-only --n-tokens 32 --prompt "..."
"""

import argparse
import os
import sys
from pathlib import Path

from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG = _LLMS_DIR.parent
_QWEN3 = _LLMS_DIR / "qwen3_0_6b"
for _p in (str(_PROG), str(_LLMS_DIR), str(_QWEN3), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.infra.cache import KernelCache, Profiler  # noqa: E402
from qwen3_0_6b_weights import LlamaConfig, generate_rope_lut  # noqa: E402
from qwen3_0_6b_prefill import compile_all_kernels  # noqa: E402
from qwen3_0_6b_decode import compile_decode_kernels  # noqa: E402

# Reuse the bf16 driver's runtime + generation loop unchanged.
from qwen3_0_6b_inference import (  # noqa: E402
    Session,
    prepare_runtime,
    run_once,
    repl_loop,
    _print_one_shot_output,
)
from qwen3_0_6b_q4nx_weights import load_q4nx_weights  # noqa: E402

MODEL_SOURCE_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-0.6B-NPU2")
# The Q4NX bundle ships its own tokenizer; default to the HF checkpoint (also the
# bf16 reference the verify gate compares against). Overridable.
TOKENIZER_DEFAULT = os.environ.get("Q4NX_TOKENIZER", "Qwen/Qwen3-0.6B")


def build_session(args) -> Session:
    """Same as qwen3_0_6b.build_session, but weights come from model.q4nx."""
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
        compile_all_kernels(
            prefill_cache, config, seq_len, verbose=args.verbose, cpu_attn=args.cpu_attn
        )
        print("\nCompiling decode kernels...")
        compile_decode_kernels(decode_cache, config, verbose=args.verbose)

    if args.compile_only:
        print("\nCompilation passed.")
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    print(f"\nLoading Q4NX weights ({args.model_source})...")
    weights = load_q4nx_weights(args.model_source, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Q4NX Inference (NPU)")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu-attn", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument(
        "--model", type=str, choices=["base", "instruct"], default="instruct"
    )
    parser.add_argument(
        "--model-source",
        dest="model_source",
        type=str,
        default=MODEL_SOURCE_DEFAULT,
        help=f"Q4NX weight source (HF repo id or local dir/file; default {MODEL_SOURCE_DEFAULT})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=TOKENIZER_DEFAULT,
        help=f"HF tokenizer id (default {TOKENIZER_DEFAULT})",
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
