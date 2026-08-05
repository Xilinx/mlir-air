# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-3B Q4NX Inference on MLIR-AIR (NPU2) — FLM-faithful decode.

The whole decoder layer runs on the AIE array: proj -> RoPE -> flash attention
with an on-device KV cache -> o-proj -> FFN, x28 layers, then the LM head, all in
ONE dispatch per token via the `fused_decode` superkernel. There is no CPU
attention and no host-side KV cache.

Weights come from FastFlowLM's 4-bit `model.q4nx` bundle, re-quantized once into
the decode's q4k-cascade layout and cached (see q4nx_decode_3b).

The prompt runs as ONE batched NPU prefill (llama32_3b_q4nx_prefill) whose per-layer
roped-K / raw-V seed the decode's device KV cache, then generation continues one
fused dispatch per token. `--no-prefill` replays the prompt through the decode
instead (slower TTFT, useful for isolating the decode path).

Usage:
    python3 llama32_3b_q4nx_inference.py --compile-only     # build prefill ELFs
    python3 llama32_3b_q4nx_inference.py --run-only --n-tokens 32 --prompt "..."
    python3 llama32_3b_q4nx_inference.py --run-only --interactive
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG = _LLMS_DIR.parent
_FUSED = _PROG / "fused_decode"
for _p in (str(_PROG), str(_LLMS_DIR), str(_FUSED), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from q4nx_decode_3b import FusedDecode3B, EOS_IDS  # noqa: E402

MODEL_TYPE = "LLAMA_3_2_3B"
# NPU weight source: the self-contained model.q4nx bundle (HF repo id or local
# dir/file). Overridable with --model-source / Q4NX_MODEL_SOURCE.
MODEL_SOURCE_DEFAULT = os.environ.get(
    "Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-3B-NPU2"
)
# The Q4NX bundle carries no chat template; take the tokenizer from the HF
# checkpoint (also the bf16 reference the verify gate compares against).
TOKENIZER_DEFAULT = os.environ.get("Q4NX_TOKENIZER", "meta-llama/Llama-3.2-3B-Instruct")
# The fused decode templates (decode_L<N>.{xclbin,insts.bin}) are built INTO THIS
# EXAMPLE'S directory rather than the shared fused_decode one (same pattern as
# gemma3_4b_q4nx): every model emits the same file names, so per-example output is
# what keeps a 1B build from silently satisfying a 3B run.
TEMPLATES_DEFAULT = os.environ.get("DECODE_TEMPLATES", str(_THIS_DIR))


def compile_prefill(seq_len: int) -> None:
    """Build/cache the prefill ELFs (weight-free, CI-runnable). The fused decode
    templates are built separately by `make compile-decode` (~15 min)."""
    from llama32_3b_q4nx_prefill import LlamaQ4nxPrefill

    LlamaQ4nxPrefill(seq_len=seq_len, n_layers=28)  # constructing compiles + caches


def _flat_ids(encoded) -> list:
    """Normalize tokenizer output to a flat list of ints. apply_chat_template may
    return a BatchEncoding (a Mapping, not a dict subclass) and/or batch-nested ids."""
    from collections.abc import Mapping

    if hasattr(encoded, "input_ids"):  # BatchEncoding
        encoded = encoded.input_ids
    elif isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return [int(t) for t in np.asarray(encoded).reshape(-1)]


def format_prompt(tokenizer, prompt: str, model_variant: str) -> list:
    """Instruct variants get the chat template; base variants get raw text."""
    if model_variant == "instruct" and getattr(tokenizer, "chat_template", None):
        return _flat_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return _flat_ids(tokenizer(prompt)["input_ids"])


def generate_stream(
    dec,
    tokenizer,
    prompt_ids,
    n_tokens,
    stream=True,
    prefiller=None,
    min_prefill=None,
):
    """Consume the prompt, then generate greedily.

    A prompt of at least `min_prefill` tokens runs as ONE batched NPU prefill whose
    per-layer roped-K / raw-V seed the decode's device KV cache (the warm-start
    handoff); shorter prompts are replayed token-by-token through the decode, which
    is faster than a full padded prefill. Both paths produce identical tokens.
    Returns (generated_ids, prompt_seconds, decode_seconds)."""
    min_prefill = PREFILL_MIN_TOKENS if min_prefill is None else min_prefill
    P = len(prompt_ids)
    if P >= dec.ATTN_MAXL:
        raise RuntimeError(
            f"prompt length {P} >= decode ATTN_MAXL {dec.ATTN_MAXL}; build a larger template"
        )
    use_prefill = prefiller is not None and P >= min_prefill
    t0 = time.perf_counter()
    if use_prefill:
        prefiller.clear_context()
        logits = np.asarray(prefiller.prefill(list(prompt_ids)), np.float32)
        kvs = [prefiller.kv_view(L) for L in range(dec.N_LAYERS)]
        dec.seed_kv([k for k, _ in kvs], [v for _, v in kvs])
    else:
        dec.reset_kv()
        logits = None
        for p, t in enumerate(prompt_ids):
            logits = dec.dispatch(t, p)
    t_prompt = time.perf_counter() - t0

    gen = []
    t1 = time.perf_counter()
    for step in range(n_tokens):
        nxt = int(np.argmax(logits))
        if nxt in EOS_IDS:
            break
        gen.append(nxt)
        if stream:
            print(tokenizer.decode([nxt]), end="", flush=True)
        p = P + step
        if step == n_tokens - 1 or p + 1 >= dec.ATTN_MAXL:
            break
        logits = dec.dispatch(nxt, p)
    t_gen = time.perf_counter() - t1
    if stream:
        print(flush=True)
    return gen, t_prompt, t_gen


# The prefill engine always runs at the padded seq_len (the registry GEMM shapes are
# built for it), so its cost is ~flat in the real prompt length: ~5 s at seq_len=2048
# on NPU2. Replaying the prompt through the decode instead costs ~60 ms/token. Short
# prompts are therefore FASTER token-by-token; only past the crossover does the
# batched prefill win (and then by a lot: a 512-token prompt is ~5 s vs ~31 s).
PREFILL_MIN_TOKENS = int(os.environ.get("Q4NX_PREFILL_MIN", "96"))


def build_prefiller(args, prompt_len=None):
    """The NPU prefill engine (None if disabled or not worth it for this prompt).
    Compiles/caches its ELFs on first use and pre-loads the dequantized weights
    into resident BOs."""
    if args.no_prefill:
        return None
    if prompt_len is not None and prompt_len < args.prefill_min:
        print(
            f"\nPrompt is {prompt_len} tokens (< --prefill-min {args.prefill_min}); "
            f"replaying it through the decode is faster than a padded "
            f"seq_len={args.seq_len} prefill. Skipping the prefill engine."
        )
        return None
    from llama32_3b_q4nx_prefill import LlamaQ4nxPrefill

    print(f"\nBuilding NPU prefill engine (seq_len={args.seq_len}) ...")
    pf = LlamaQ4nxPrefill(seq_len=args.seq_len, n_layers=28)
    pf.load_weights(model=args.model_source)
    return pf


def build_decoder(args) -> FusedDecode3B:
    if not os.path.isdir(args.templates) or not any(
        f.startswith("decode_L") for f in os.listdir(args.templates)
    ):
        raise SystemExit(
            f"No decode templates in {args.templates}. Run `make compile-decode` "
            f"first (or pass --templates)."
        )
    print(f"\nLoading Q4NX weights ({args.model_source}) + decode templates ...")
    return FusedDecode3B(args.model_source, args.templates, model_type=MODEL_TYPE)


def repl(dec, tokenizer, args, prefiller=None):
    print("\nInteractive chat (Ctrl-D or 'exit' to quit).")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line in ("exit", "quit"):
            break
        ids = format_prompt(tokenizer, line, args.model)
        pf = prefiller if len(ids) >= args.prefill_min else None
        _, tp, tg = generate_stream(dec, tokenizer, ids, args.n_tokens, prefiller=pf)
        print(f"\n[{len(ids)} prompt tok in {tp:.2f}s | decode {tg:.2f}s]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLAMA-3.2-3B Q4NX Inference (NPU2, FLM-faithful on-device decode)"
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
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
    parser.add_argument(
        "--templates",
        type=str,
        default=TEMPLATES_DEFAULT,
        help="directory holding decode_L<N>.{xclbin,insts.bin} builds",
    )
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="replay the prompt token-by-token through the decode instead of the "
        "batched NPU prefill (slower TTFT; useful for isolating the decode path)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=int(os.environ.get("Q4NX_SEQ_LEN", "2048")),
        help="padded prefill length (prefill ELFs are compiled for this)",
    )
    parser.add_argument(
        "--prefill-min",
        dest="prefill_min",
        type=int,
        default=PREFILL_MIN_TOKENS,
        help="use the batched NPU prefill only for prompts at least this long "
        f"(default {PREFILL_MIN_TOKENS}; shorter prompts decode faster token-by-token)",
    )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive and args.compile_only:
        parser.error("--interactive cannot be combined with --compile-only")

    if not args.run_only:
        compile_prefill(args.seq_len)
        if args.compile_only:
            print("\nCompilation passed.")
            sys.exit(0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dec = build_decoder(args)

    if args.interactive:
        # Prompt lengths vary per turn, so build the prefill engine up front and let
        # generate_stream fall back per turn for short prompts.
        prefiller = build_prefiller(args)
        repl(dec, tokenizer, args, prefiller)
        sys.exit(0)

    ids = format_prompt(tokenizer, args.prompt, args.model)
    prefiller = build_prefiller(args, prompt_len=len(ids))
    print(f"\nPrompt: {args.prompt}\n")
    gen, t_prompt, t_gen = generate_stream(
        dec, tokenizer, ids, args.n_tokens, prefiller=prefiller
    )
    if args.profile:
        npt, ngt = max(len(ids), 1), max(len(gen), 1)
        print(
            f"\n[profile] prompt {len(ids)} tok in {t_prompt:.3f}s "
            f"({1000 * t_prompt / npt:.1f} ms/tok) | "
            f"decode {len(gen)} tok in {t_gen:.3f}s "
            f"({1000 * t_gen / ngt:.1f} ms/tok, {ngt / max(t_gen, 1e-9):.1f} tok/s) | "
            f"28 layers + LM head on-device, one dispatch/token"
        )
