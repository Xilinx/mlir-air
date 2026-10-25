# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Streaming greedy generation and resident, thinking-disabled chat on NPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from common.model import DFlashModel
from common.runtime import (
    PrefillWeights,
    encode_prompt,
    load_pair,
    load_pipeline,
    prepare_prompt,
)
from common.streaming import TextStream


def check_capacity(loop, input_ids, tokens):
    if len(input_ids) + tokens - 1 + loop.batch > loop.capacity:
        available = max(0, loop.capacity - len(input_ids) - loop.batch + 1)
        raise ValueError(
            f"Prompt ({len(input_ids)} tokens) plus output and final B{loop.batch} "
            f"verification exceed C{loop.capacity}; at most {available} output tokens fit."
        )


def run_prompt(
    model: DFlashModel,
    loop,
    tokenizer,
    prompt,
    args,
    *,
    prefill_model=None,
    npu_prefill=None,
):
    started = time.perf_counter()
    input_ids = encode_prompt(model, tokenizer, prompt)
    check_capacity(loop, input_ids, args.tokens)
    if npu_prefill is not None and len(input_ids) > npu_prefill.seq:
        raise ValueError(
            f"Prompt exceeds the NPU prefill capacity {npu_prefill.seq}; use --prefill-context 4096"
        )
    print(
        f"{args.prefill.upper()} prefill: {len(input_ids)} input tokens...", flush=True
    )
    if npu_prefill is None:
        prefill, prefill_seconds = prepare_prompt(
            loop.target, input_ids, loop.tap_slots, prefill_model=prefill_model
        )
    else:
        prefill, prefill_seconds = npu_prefill.prepare(input_ids)
    reset_started = time.perf_counter()
    loop.reset(prefill)
    reset_seconds = time.perf_counter() - reset_started
    first_token_seconds = time.perf_counter() - started

    stream = None if args.no_stream else TextStream(tokenizer)
    if stream is not None:
        print("Response: ", end="", flush=True)
        # Prefill supplies the first output token. Subsequent callbacks
        # receive only the accepted prefix + verifier token after each round.
        stream.put([prefill["first"]])
    try:
        generated = loop.generate(
            args.tokens,
            eos_ids=model.eos_ids,
            observer=(
                None
                if stream is None
                else lambda row, _: stream.put(row["emitted_ids"])
            ),
        )
    finally:
        if stream is not None:
            stream.finish()
            print(flush=True)

    rounds = len(generated.rounds)
    accepted = sum(row["accepted_drafts"] for row in generated.rounds)
    proposed = sum(row["proposed"] for row in generated.rounds)
    result = {
        "prompt": prompt,
        "input_tokens": len(input_ids),
        "output_ids": generated.output_ids,
        "output_text": tokenizer.decode(
            generated.output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "stopped_on_eos": generated.stopped_on_eos,
        "prefill_backend": args.prefill,
        "prefill_seconds": prefill_seconds,
        "cpu_prefill_seconds": prefill_seconds if npu_prefill is None else None,
        "npu_prefill_seconds": prefill_seconds if npu_prefill is not None else None,
        "prefill_padded_length": npu_prefill.seq if npu_prefill is not None else None,
        "kv_reset_seconds": reset_seconds,
        "time_to_first_token_seconds": first_token_seconds,
        "request_seconds": time.perf_counter() - started,
        "decode_seconds": generated.wall_seconds,
        "decode_tokens": max(0, len(generated.output_ids) - 1),
        "decode_tps": (
            max(0, len(generated.output_ids) - 1) / generated.wall_seconds
            if generated.wall_seconds
            else None
        ),
        "verification_rounds": rounds,
        "mean_tau": 1 + accepted / rounds if rounds else None,
        "accepted_proposed_fraction": accepted / proposed if proposed else None,
        "stage_seconds": generated.stages,
        "streaming": stream is not None,
        "cached_cpu_prefill_weights": prefill_model is not None,
    }
    if stream is not None:
        assert (
            stream.ids == generated.output_ids
        ), "Stream must contain exactly the committed output"
    else:
        print(result["output_text"], flush=True)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "input_tokens",
                    "prefill_backend",
                    "prefill_seconds",
                    "time_to_first_token_seconds",
                    "decode_tokens",
                    "decode_seconds",
                    "decode_tps",
                    "verification_rounds",
                    "mean_tau",
                    "accepted_proposed_fraction",
                    "stopped_on_eos",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def save_result(result, output, *, append=False):
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a" if append else "w") as handle:
            handle.write(json.dumps(result, indent=None if append else 2) + "\n")


def chat(model: DFlashModel, loop, tokenizer, args, prefill_model, npu_prefill):
    print("\nReady. Each prompt is independent (no conversation history).", flush=True)
    print("Enter /quit or /exit, or press Ctrl-D / Ctrl-C to exit.\n", flush=True)
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt in ("/quit", "/exit"):
            return
        if not prompt:
            continue
        # Reject an oversized request before touching device state and keep
        # the resident session available for the next prompt.
        input_ids = encode_prompt(model, tokenizer, prompt)
        try:
            check_capacity(loop, input_ids, args.tokens)
            if npu_prefill is not None and len(input_ids) > npu_prefill.seq:
                raise ValueError(
                    f"Prompt exceeds NPU prefill capacity {npu_prefill.seq}; restart with PREFILL_CTX=4096"
                )
        except ValueError as error:
            print(error, flush=True)
            continue
        result = run_prompt(
            model,
            loop,
            tokenizer,
            prompt,
            args,
            prefill_model=prefill_model,
            npu_prefill=npu_prefill,
        )
        save_result(result, args.output, append=True)
        print(flush=True)


def _run(model: DFlashModel) -> None:
    here = model.directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=os.environ.get(
            "DFLASH_PROMPT", "Explain why speculative decoding can improve latency."
        ),
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--prefill",
        choices=("npu", "cpu") if model.cpu_prefill else ("npu",),
        default="npu",
    )
    parser.add_argument(
        "--prefill-context", type=int, choices=(2048, 4096), default=2048
    )
    parser.add_argument(
        "--prefill-artifacts", type=Path, help="Sequence-specific NPU prefill cache"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Load once, then stream independent prompts",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print the complete response after decoding",
    )
    parser.add_argument("--target-artifacts", type=Path, default=here / "build/target")
    parser.add_argument("--draft-artifacts", type=Path, default=here / "build/draft")
    parser.add_argument("--target-model", default=model.target_model)
    parser.add_argument("--tokenizer-model", default=model.tokenizer_model)
    parser.add_argument(
        "--target-cache", type=Path, default=model.default_target_cache()
    )
    parser.add_argument("--draft-cache", type=Path, default=model.default_draft_cache())
    parser.add_argument(
        "--draft-snapshot", type=Path, default=model.default_draft_snapshot()
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON result; in interactive mode append each turn as JSONL",
    )
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.interactive and args.no_stream:
        parser.error("--interactive streams output; omit --no-stream")

    started = time.perf_counter()
    print(
        "Loading tokenizer, target, drafter, and NPU context preparation...", flush=True
    )
    tokenizer = model.load_tokenizer(args.tokenizer_model)
    prefill_model = npu_prefill = None
    if args.prefill == "npu":
        print("Preparing NPU prefill and resident model weights...", flush=True)
        loop, npu_prefill = load_pipeline(
            model,
            args.target_artifacts,
            args.draft_artifacts,
            target_model=args.target_model,
            target_cache=args.target_cache,
            draft_cache=args.draft_cache,
            draft_snapshot=args.draft_snapshot,
            prefill_context=args.prefill_context,
            prefill_artifacts=args.prefill_artifacts,
        )
    else:
        loop = load_pair(
            model,
            args.target_artifacts,
            args.draft_artifacts,
            target_model=args.target_model,
            target_cache=args.target_cache,
            draft_cache=args.draft_cache,
            draft_snapshot=args.draft_snapshot,
        )
        if args.interactive:
            prefill_model = PrefillWeights(loop.target)
            print(
                f"Resident CPU prefill weights: {prefill_model.nbytes / 2**30:.2f} GiB",
                flush=True,
            )
    print(f"Initialization: {time.perf_counter() - started:.2f} s", flush=True)
    if args.interactive:
        chat(model, loop, tokenizer, args, prefill_model, npu_prefill)
    else:
        result = run_prompt(
            model, loop, tokenizer, args.prompt, args, npu_prefill=npu_prefill
        )
        save_result(result, args.output)


def main(model: DFlashModel) -> None:
    try:
        _run(model)
    except KeyboardInterrupt:
        # Exit on interruption during generation rather than reuse a context
        # whose device dispatch may still be outstanding.
        print("\n[Interrupted]", flush=True)
        raise SystemExit(130)
