#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does the block size change what DFlash GENERATES, or only how fast?

docs/DFlashFeasibility.md's section 3.1 recommends running the b16 checkpoint at
block 8, which is smaller than it was trained at. That raises a fair question the
acceptance numbers do not answer: is a block-8 run still a faithful DFlash
inference, or has something been traded away for the speed?

The answer should be that nothing is traded, and it should be exact rather than
approximate. Greedy speculative decoding keeps a drafted token only where it
equals the target's own `argmax`, and the bonus token IS the target's `argmax`
(model.py: `acceptance_length` is a cumprod over `block_output_ids[:, 1:] ==
posterior[:, :-1]`, then `bonus = posterior[:, acceptance_length]`). So every
emitted token is a token plain autoregressive decoding would have emitted, and
the output must be token-identical at every block size.

That is an argument from reading the code. This checks it: the same prompts
generated at block 1 (which takes model.py's `verify_size > 1` false path and is
therefore plain greedy decode), 4, 8 and 16, compared token for token.

A mismatch would mean the block size is NOT free -- that speculation is changing
the output and the block-8 recommendation is buying speed with fidelity. CPU
only, no XRT, a few minutes.

    python3 dflash_block_equiv.py
    python3 dflash_block_equiv.py --blocks 8 16 --max-new-tokens 100
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "_dflash_upstream"

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"

# One per category, so a divergence that only shows up on a particular kind of
# text has somewhere to show up. Drawn the way section 3.1's sweep draws them.
DATASETS = ["gsm8k", "humaneval", "mt-bench"]


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
    ap.add_argument("--blocks", nargs="*", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--n", type=int, default=1, help="prompts per dataset")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    m = _load_upstream("model")
    m._cuda_time = lambda: time.perf_counter()
    b = _load_upstream("benchmark")

    print(f"[equiv] loading {TARGET_ID} + {DRAFT_ID} (bf16, CPU)...", flush=True)
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

    ok = True
    for ds_name in DATASETS:
        for i, instance in enumerate(
            b._select_dataset(b.load_and_process_dataset(ds_name), args.n)
        ):
            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": instance["turns"][0]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = torch.tensor(
                [tokenizer.encode(chat_text, add_special_tokens=False)],
                dtype=torch.long,
            )
            print(f"\n[equiv] {ds_name} #{i}  ({input_ids.shape[1]} prompt tokens)")
            ref = None
            for blk in args.blocks:
                t0 = time.perf_counter()
                out = m.dflash_generate(
                    draft,
                    target,
                    input_ids,
                    args.max_new_tokens,
                    stop_ids,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    block_size=blk,
                    return_stats=True,
                )
                ids = out.output_ids[0].tolist()
                wall = time.perf_counter() - t0
                if ref is None:
                    ref, ref_blk = ids, blk
                    print(
                        f"    block {blk:>2}: {len(ids)} tokens  "
                        f"[reference, {'plain greedy' if blk == 1 else 'speculative'}]"
                        f"  wall={wall:.1f}s"
                    )
                    continue
                same = ids == ref
                ok &= same
                if same:
                    verdict = f"IDENTICAL to block {ref_blk}"
                else:
                    n = min(len(ids), len(ref))
                    first = next((j for j in range(n) if ids[j] != ref[j]), n)
                    verdict = (
                        f"*** DIFFERS *** first at token {first} "
                        f"(len {len(ids)} vs {len(ref)})"
                    )
                print(
                    f"    block {blk:>2}: {len(ids)} tokens  {verdict}  wall={wall:.1f}s"
                )

    print("\n" + "=" * 70)
    print(
        "ALL BLOCK SIZES PRODUCE IDENTICAL OUTPUT -- block size is a speed knob"
        if ok
        else "MISMATCH: the block size changes the output. Do not treat it as free."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
