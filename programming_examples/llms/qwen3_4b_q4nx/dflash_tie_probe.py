#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Why does the block size change the output, when the algorithm says it cannot?

`dflash_block_equiv.py` compared greedy generations at blocks 1/4/8/16 and found
them NOT identical: gsm8k matched everywhere, humaneval diverged at token 141
(block 8 only -- block 16 matched), mt-bench at 175 (blocks 4 and 8) and 103
(block 16). Non-monotone in the block size, and late in the sequence.

The algorithm forbids this. A drafted token is kept only where it equals the
target's own `argmax`, and the bonus token IS the target's `argmax`, so in exact
arithmetic every block size emits the target's greedy stream. The obvious
suspect is therefore not the algorithm but the ARITHMETIC: a verify pass runs the
target over B tokens at once, a block-1 pass runs it over one, and the two take
different reduction orders through sdpa and the projections. In bf16 that moves
logits by a fraction of an ulp -- normally invisible, but where the top two
logits are nearly tied it flips the argmax, and one flipped token changes every
token after it.

This tests that. It teacher-forces the reference generation through the target in
one pass and reports the top-2 logit GAP at each position, so the divergence
positions can be read against the distribution of gaps everywhere else. If those
positions sit in the bottom percentiles of the gap distribution, near-ties are
the explanation and the block size is numerically -- not algorithmically -- free.

The distinction decides what the on-device loop can be gated on: an exact token
match is the wrong gate if even the CPU reference cannot hold it.

    python3 dflash_tie_probe.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"

# (dataset, first divergent token index) as dflash_block_equiv.py reported them.
# The index is into output_ids, which starts at the prompt.
CASES = [("humaneval", 141), ("mt-bench", 175)]


def _load_upstream(name):
    spec = importlib.util.spec_from_file_location(
        f"dflash_upstream_{name}", str(_UPSTREAM / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = _load_upstream("model")
    import time

    m._cuda_time = lambda: time.perf_counter()
    b = _load_upstream("benchmark")

    print("[tie] loading models (bf16, CPU)...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    config = AutoConfig.from_pretrained(DRAFT_ID)
    draft = m.DFlashDraftModel.from_pretrained(
        DRAFT_ID, config=config, attn_implementation="sdpa", dtype=torch.bfloat16
    )
    draft.eval()
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

    for ds_name, div in CASES:
        inst = b._select_dataset(b.load_and_process_dataset(ds_name), 1)[0]
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": inst["turns"][0]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = torch.tensor(
            [tokenizer.encode(chat_text, add_special_tokens=False)], dtype=torch.long
        )
        n_prompt = input_ids.shape[1]

        # The block-1 reference stream: model.py's verify_size==1 path, i.e.
        # plain greedy decode.
        ref = m.dflash_generate(
            draft,
            target,
            input_ids,
            args.max_new_tokens,
            stop_ids,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            block_size=1,
            return_stats=True,
        ).output_ids

        # One teacher-forced pass over the whole reference stream. Position j's
        # logits predict token j+1, so the decision that produced output token
        # `div` was taken at position div-1.
        with torch.no_grad():
            logits = target(ref).logits[0].float()
        top2 = logits.topk(2, dim=-1).values
        gap = (top2[:, 0] - top2[:, 1]).numpy()

        gen = gap[
            n_prompt - 1 : len(ref[0]) - 1
        ]  # decisions that made generated tokens
        d = gap[div - 1]
        pct = 100.0 * (gen < d).mean()
        import numpy as np

        print(
            f"\n[tie] {ds_name}: prompt {n_prompt}, generated {len(ref[0]) - n_prompt}"
        )
        print(f"    top-2 logit gap at the divergence (token {div}): {d:.5f}")
        print(
            f"    that is the {pct:.1f}th percentile of the {len(gen)} generated "
            f"decisions"
        )
        print(
            f"    gap distribution: min {gen.min():.5f}  p1 {np.percentile(gen,1):.5f}  "
            f"p10 {np.percentile(gen,10):.5f}  median {np.percentile(gen,50):.5f}"
        )
        n_tiny = int((gen < 0.05).sum())
        print(
            f"    {n_tiny} of {len(gen)} decisions have a gap < 0.05 "
            f"({100.0*n_tiny/len(gen):.1f}%) -- every one is a flip candidate"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
