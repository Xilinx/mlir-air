#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Confirms (or refutes) whether dflash_phase2_upstream_check.py's single-prompt
reversal (thinking mode disabled: tau 2.79 -> 6.16 on one GSM8K prompt,
docs/DFlashFeasibility.md's "## 8") holds up across a real spread of tasks --
the same discipline this session already applied once (the 10-prompt raw-
completion sweep that ruled out the enumerative-prompt hypothesis), now
needed for the opposite reason: confirming a promising result isn't a
one-prompt fluke before recommending the item-11 (on-device batch=16)
investment.

Chat-templated, `enable_thinking=False` (matching the paper's Table 1
caption: "Qwen3 models with thinking mode disabled"), greedy, block_size=16,
via the REAL unmodified upstream `dflash_generate` (dflash/model.py, pulled
from github.com/z-lab/dflash -- see dflash_phase2_upstream_check.py's
docstring for why this, not the older trust_remote_code files, is used).

CPU-only, both models loaded once and reused across all prompts (no XRT
involved at all in this script, so no subprocess-per-prompt segfault
workaround is needed here, unlike the NPU-involving sweep).
"""

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"
MAX_NEW_TOKENS = 200

PROMPTS = [
    (
        "math",
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April "
        "and May?\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ),
    (
        "math",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?\nPlease reason step by step, "
        "and put your final answer within \\boxed{}.",
    ),
    (
        "math",
        "If a train travels 60 miles in 1.5 hours, what is its average speed in "
        "miles per hour? Please reason step by step, and put your final answer "
        "within \\boxed{}.",
    ),
    (
        "code",
        "Write a solution to the following problem and make sure that it passes "
        'the tests:\n```python\ndef fibonacci(n: int) -> int:\n    """Return '
        'the n-th Fibonacci number."""\n```',
    ),
    (
        "code",
        "Write a solution to the following problem and make sure that it passes "
        'the tests:\n```python\ndef is_palindrome(s: str) -> bool:\n    """Return '
        'True if s reads the same forwards and backwards."""\n```',
    ),
    (
        "code",
        "Write a solution to the following problem and make sure that it passes "
        "the tests:\n```python\ndef merge_sorted(a: list, b: list) -> list:\n    "
        '"""Merge two sorted lists into one sorted list."""\n```',
    ),
    ("chat", "What are the main benefits of regular exercise?"),
    ("chat", "Explain the water cycle in simple terms."),
    ("chat", "Write a short poem about the ocean."),
]


def _load_upstream_model_module():
    spec = importlib.util.spec_from_file_location(
        "dflash_upstream_model", str(_UPSTREAM / "model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import time as _time
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = _load_upstream_model_module()
    m._cuda_time = lambda: _time.perf_counter()  # CPU-only

    print(f"[sweep] loading target {TARGET_ID} (bf16, CPU, sdpa)...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

    print(f"[sweep] loading draft {DRAFT_ID} via upstream model.py...", flush=True)
    config = AutoConfig.from_pretrained(DRAFT_ID)
    draft = m.DFlashDraftModel.from_pretrained(
        DRAFT_ID, config=config, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    draft.eval()
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

    results = {}
    for i, (category, question) in enumerate(PROMPTS):
        messages = [{"role": "user", "content": question}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = torch.tensor(
            [tokenizer.encode(chat_text, add_special_tokens=False)], dtype=torch.long
        )
        print(f"\n[sweep] === {i} [{category}]: {question[:60]!r}... ===", flush=True)
        t0 = _time.perf_counter()
        result = m.dflash_generate(
            draft,
            target,
            input_ids,
            MAX_NEW_TOKENS,
            stop_ids,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            block_size=16,
            return_stats=True,
        )
        lens = result.acceptance_lengths
        mean = sum(lens) / len(lens) if lens else 0.0
        print(
            f"[sweep] {i} [{category}]: {len(lens)} blocks, mean={mean:.2f}/16 "
            f"({100*mean/16:.1f}%), lens={lens}, wall={_time.perf_counter()-t0:.1f}s",
            flush=True,
        )
        results[(category, i)] = lens

    print("\n" + "=" * 70)
    print("[sweep] SUMMARY (thinking mode disabled, block_size=16, greedy)")
    print("=" * 70)
    by_cat = {}
    grand = []
    for (category, i), lens in results.items():
        by_cat.setdefault(category, []).extend(lens)
        grand.extend(lens)
    for category, lens in by_cat.items():
        n = len(lens)
        mean = sum(lens) / n if n else 0.0
        print(
            f"  {category:<8} n_blocks={n:>3}  mean={mean:.2f}/16 ({100*mean/16:.1f}%)"
        )
    if grand:
        n = len(grand)
        mean = sum(grand) / n
        print("-" * 70)
        print(
            f"  {'OVERALL':<8} n_blocks={n:>3}  mean={mean:.2f}/16 ({100*mean/16:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
