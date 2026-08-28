#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Firms up dflash_phase2_upstream_sweep.py's 3-prompts-per-category result
(math tau=6.30, code tau=7.56, both close to the paper's reported 6.53/7.84 --
docs/DFlashFeasibility.md's "## 8") with a real sample from the actual
datasets `dflash/benchmark.py` uses (gsm8k, humaneval), not hand-picked
prompts. Same seeded shuffle (`random.Random(42)`) and dataset loading code
as the real upstream benchmark.py, imported directly from it -- reusing their
selection logic rather than re-deriving it, so this sample is drawn exactly
the way their own benchmark draws it.

Chat-templated, `enable_thinking=False`, greedy, block_size=16, real
unmodified upstream `dflash_generate`. CPU-only, no XRT.
"""

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"
MAX_NEW_TOKENS = 200
N_PER_DATASET = 20
DATASETS_TO_RUN = ["gsm8k", "humaneval"]


def _load_upstream(name):
    spec = importlib.util.spec_from_file_location(
        f"dflash_upstream_{name}", str(_UPSTREAM / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import time as _time
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = _load_upstream("model")
    m._cuda_time = lambda: _time.perf_counter()  # CPU-only
    b = _load_upstream("benchmark")  # real dataset loading + selection logic

    print(f"[sweep_large] loading target {TARGET_ID} (bf16, CPU, sdpa)...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

    print(
        f"[sweep_large] loading draft {DRAFT_ID} via upstream model.py...", flush=True
    )
    config = AutoConfig.from_pretrained(DRAFT_ID)
    draft = m.DFlashDraftModel.from_pretrained(
        DRAFT_ID, config=config, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    draft.eval()
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

    by_dataset = {}
    for ds_name in DATASETS_TO_RUN:
        print(
            f"\n[sweep_large] loading dataset '{ds_name}' (real, HF hub)...", flush=True
        )
        full = b.load_and_process_dataset(ds_name)
        sample = b._select_dataset(full, N_PER_DATASET)
        print(
            f"[sweep_large] {ds_name}: {len(sample)} prompts selected (seed 42, matches upstream benchmark.py)",
            flush=True,
        )

        lens_all = []
        for i, instance in enumerate(sample):
            question = instance["turns"][0]
            messages = [{"role": "user", "content": question}]
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = torch.tensor(
                [tokenizer.encode(chat_text, add_special_tokens=False)],
                dtype=torch.long,
            )
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
                f"[sweep_large] {ds_name} {i}/{len(sample)}: {len(lens)} blocks, "
                f"mean={mean:.2f}/16, wall={_time.perf_counter()-t0:.1f}s",
                flush=True,
            )
            lens_all.extend(lens)
        by_dataset[ds_name] = lens_all

    print("\n" + "=" * 70)
    print(
        f"[sweep_large] SUMMARY ({N_PER_DATASET} real prompts/dataset, thinking disabled, block16, greedy)"
    )
    print("=" * 70)
    grand = []
    for ds_name, lens in by_dataset.items():
        grand.extend(lens)
        n = len(lens)
        mean = sum(lens) / n if n else 0.0
        print(
            f"  {ds_name:<12} n_blocks={n:>4}  mean={mean:.2f}/16 ({100*mean/16:.1f}%)"
        )
    if grand:
        n = len(grand)
        mean = sum(grand) / n
        print("-" * 70)
        print(
            f"  {'OVERALL':<12} n_blocks={n:>4}  mean={mean:.2f}/16 ({100*mean/16:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
