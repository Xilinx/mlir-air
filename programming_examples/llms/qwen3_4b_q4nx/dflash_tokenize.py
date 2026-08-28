#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tokenize (and de-tokenize) in a SEPARATE PROCESS, because transformers and
XRT cannot share one.

`hidden_taps_verify.py` documents the rule and `dflash_draft_oracle.py` is
built around it: importing torch/transformers into a process that later opens
an XRT device segfaults. Anything on the device side that needs a tokenizer
therefore shells out to this.

    python3 dflash_tokenize.py --encode "The capital of France is"
    python3 dflash_tokenize.py --decode 12095,13,576
    python3 dflash_tokenize.py --dataset gsm8k --n 20    # the sweep's prompts

Prints JSON on stdout and nothing else, so a caller can `json.loads` it.
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

MODEL = "Qwen/Qwen3-4B"


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--encode")
    ap.add_argument("--decode", help="comma-separated ids")
    ap.add_argument("--dataset", choices=("gsm8k", "humaneval", "mtbench"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-prompt", type=int, default=180, help="drop longer prompts")
    args = ap.parse_args()

    tk = _tok()
    if args.encode is not None:
        print(json.dumps(tk.encode(args.encode, add_special_tokens=False)))
        return 0
    if args.decode is not None:
        ids = [int(x) for x in args.decode.split(",") if x.strip()]
        print(json.dumps(tk.decode(ids)))
        return 0

    # THE UPSTREAM'S OWN LOADER, SELECTION AND CHAT TEMPLATE -- not a
    # re-derivation. Section 3.1 measured the bf16 drafter's acceptance
    # distribution through `benchmark.load_and_process_dataset` +
    # `_select_dataset` (seed 42, upstream order) and `apply_chat_template`, so
    # going through the same three functions is what makes the device number
    # comparable to it. A hand-rolled `load_dataset` here would silently be a
    # different prompt set, differently formatted, and the comparison would be
    # meaningless in a way nothing downstream could detect.
    import importlib.util

    _up = _HERE / "_dflash_upstream"
    spec = importlib.util.spec_from_file_location(
        "dflash_upstream_benchmark", str(_up / "benchmark.py")
    )
    b = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = b
    spec.loader.exec_module(b)

    name = {"gsm8k": "gsm8k", "humaneval": "humaneval", "mtbench": "mt-bench"}[
        args.dataset
    ]
    sample = b._select_dataset(b.load_and_process_dataset(name), args.n * 3)

    out = []
    for inst in sample:
        # First turn only (the device loop is single-turn), and formatted the
        # way dflash_acceptance_hist.py:137 formats it -- add_generation_prompt,
        # enable_thinking=False. Not the upstream `apply_chat_template` wrapper:
        # section 3.1 called the tokenizer directly, and matching 3.1 is the
        # point.
        text = tk.apply_chat_template(
            [{"role": "user", "content": inst["turns"][0]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tk.encode(text, add_special_tokens=False)
        if len(ids) <= args.max_prompt:
            out.append(ids)
        if len(out) >= args.n:
            break
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
