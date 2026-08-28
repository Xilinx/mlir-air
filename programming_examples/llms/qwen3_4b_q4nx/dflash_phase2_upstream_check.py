#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validates this session's Phase 2 replay methodology against the REAL,
current, unmodified z-lab/dflash `dflash_generate` function (downloaded from
github.com/z-lab/dflash, `dflash/model.py` -- NOT the older `dflash.py`/
`modeling_dflash.py` bundled with the HF-hosted checkpoint's trust_remote_code
files, which this session already found to be written against an
incompatible old transformers Cache API).

Two things this checks in one shot, both real open questions from
docs/DFlashFeasibility.md's "## 8":

1. Does the REAL reference implementation (not this session's
   reimplementation) also show low acceptance on a raw, non-chat-templated
   sentence-completion prompt at temperature=0 (matching what was measured:
   ~2.2/16 mean over 10 prompts)?
2. Does it show HIGHER acceptance on a chat-templated, task-oriented prompt
   (GSM8K-style), matching how `dflash/benchmark.py`'s own default eval
   methodology actually drives it (`apply_chat_template` over gsm8k/math500/
   humaneval/mbpp/mt-bench, `--temperature` DEFAULT 0.0 per `cli.py` --
   greedy is the default, not an edge case)? If so, the gap is explained by
   prompt/task style, not a bug in this session's harness.

CPU-only (`return_stats=False` to avoid `dflash_generate`'s CUDA-only
`_cuda_time()` timing path; only accept/reject correctness is being checked
here, not throughput).
"""

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"


def _load_upstream_model_module():
    spec = importlib.util.spec_from_file_location("dflash_upstream_model", str(_UPSTREAM / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = _load_upstream_model_module()
    import time as _time

    m._cuda_time = lambda: _time.perf_counter()  # CPU-only: skip torch.cuda.synchronize()

    print(f"[upstream_check] loading target {TARGET_ID} (bf16, CPU, sdpa)...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_ID, attn_implementation="sdpa", dtype=torch.bfloat16)
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

    print(f"[upstream_check] loading draft {DRAFT_ID} via upstream model.py's DFlashDraftModel...", flush=True)
    config = AutoConfig.from_pretrained(DRAFT_ID)
    draft = m.DFlashDraftModel.from_pretrained(DRAFT_ID, config=config, attn_implementation="sdpa", dtype=torch.bfloat16)
    draft.eval()

    def run(label, prompt_ids, n_tokens):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
        result = m.dflash_generate(
            draft, target, input_ids, n_tokens, stop_ids,
            temperature=0.0, top_p=1.0, top_k=0, block_size=16, return_stats=True,
        )
        lens = result.acceptance_lengths
        mean = sum(lens) / len(lens) if lens else 0.0
        print(f"[upstream_check] {label}: {len(lens)} blocks, accepted lengths (produced tokens/block) = {lens}")
        print(f"[upstream_check] {label}: mean produced/block = {mean:.2f} of 16 ({100*mean/16:.1f}%)")
        return lens

    print("\n=== 1) raw sentence completion (matches this session's Phase 2 prompts) ===", flush=True)
    raw_prompt = tokenizer.encode("The capital of France is", add_special_tokens=False)
    run("raw completion", raw_prompt, 96)

    print("\n=== 2) chat-templated GSM8K-style task (matches dflash/benchmark.py's real eval) ===", flush=True)
    gsm8k_question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April "
        "and May?\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    messages = [{"role": "user", "content": gsm8k_question}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_prompt = tokenizer.encode(chat_text, add_special_tokens=False)
    run("gsm8k chat-templated (thinking default)", chat_prompt, 160)

    print(
        "\n=== 3) SAME gsm8k prompt, thinking mode explicitly disabled "
        "(the paper's Table 1 caption: 'Qwen3 models with thinking mode disabled') ===",
        flush=True,
    )
    chat_text_nothink = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    chat_prompt_nothink = tokenizer.encode(chat_text_nothink, add_special_tokens=False)
    run("gsm8k chat-templated (no-think)", chat_prompt_nothink, 160)

    return 0


if __name__ == "__main__":
    sys.exit(main())
