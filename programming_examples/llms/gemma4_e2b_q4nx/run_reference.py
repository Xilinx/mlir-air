# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Driver for the Gemma4-E2B Q4NX CPU reference (no NPU involved).
#
# `paris` and `gen` are SMOKE TESTS, not correctness gates. Both a doubled
# embedding scale and a per-layer-embedding input read from the wrong tensor
# produced fluent, sometimes-correct text while the model was numerically broken.
# The gate is check_vs_flm_reference.py (`make check`).
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_e2b_q4nx_weights as W  # noqa: E402

BUNDLE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")


def _tok(bundle):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(bundle)


def cmd_paris(a):
    tok, m = _tok(a.bundle), W.Q4nxModel(a.bundle)
    ids = [2] + tok("The capital of France is")["input_ids"]
    logits, _ = W.forward_prompt(m, ids)
    top = int(np.argmax(logits))
    got = tok.decode([top]).strip()
    print(f"top-1: {tok.decode([top])!r}")
    if got != "Paris":
        raise SystemExit(f"FAIL: expected 'Paris', got {got!r}")
    print("PASS")


def cmd_gen(a):
    tok, m = _tok(a.bundle), W.Q4nxModel(a.bundle)
    enc = tok.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": a.prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    ids = [int(t) for t in np.asarray(enc["input_ids"]).reshape(-1)]
    # No KV reuse across tokens: each step re-runs the whole prompt. Fine for a
    # smoke test (~35 s/token at short context); the NPU path is what gets a
    # real incremental decode.
    out = []
    for _ in range(a.n_tokens):
        logits, _ = W.forward_prompt(m, ids + out)
        nxt = int(np.argmax(logits))
        out.append(nxt)
        if nxt == tok.eos_token_id or nxt == 106:  # 106 = <turn|>
            break
    print(repr(tok.decode(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=BUNDLE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paris", help="one forward; PASS iff top-1 is ' Paris'")
    p.set_defaults(fn=cmd_paris)
    p = sub.add_parser("gen", help="greedy generation via the chat template")
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--n-tokens", type=int, default=12)
    p.set_defaults(fn=cmd_gen)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
