#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The eight-token decoder at long contexts, against the NPU prefill.

For each context N, the prefill runs N + 16 tokens of real text and gives
the logits of the last 16, teacher forced. Its cache's first N positions
seed the decoder, which runs the same 16 tokens as two calls, at N and at
N + 8 (the second reads the K/V the first appended). Past 512 tokens the
sliding layers see only their window.

A row passes if the prefill scores the decoder's top-1 within TIE logits of
its own top-1 and the two rows' logits have a cosine of at least MIN_COS.
Each path moves logits by 1 to 1.5 from reference_b8's float32 forward at a
few thousand tokens, and each misses near ties the other gets, so the two
can disagree by about 2; a broken sliding window misses by 10 or more.

  python3 tests/check_long_context.py --target DIR --prefill-artifacts DIR
      --output FILE [--contexts N ...]
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_paths import load_tokenizer  # noqa: E402
from npu_prefill import NpuPrefill  # noqa: E402
from target_runtime import GemmaTarget, pack_prefill_kv  # noqa: E402

DOCS = Path(__file__).resolve().parents[5] / "docs"
B = 8
TIE = 3.0
MIN_COS = 0.98


def text_ids(tokenizer, n):
    """n tokens of real text: the repository's documentation."""
    text = "".join(p.read_text() for p in sorted(DOCS.glob("*.md")))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + ids
    return [2] + ids[: n - 1]


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--prefill-artifacts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--contexts", type=int, nargs="+", default=[64, 505, 600, 1100, 2000, 4064]
    )
    a = ap.parse_args()
    tokenizer = load_tokenizer()
    longest = max(a.contexts) + 2 * B
    prefill = NpuPrefill(-(-longest // 32) * 32, a.prefill_artifacts)
    prefill.load_weights()
    target = GemmaTarget(a.target, prefill.engine.model)
    result = dict(tie=TIE, min_cos=MIN_COS, contexts=[])
    for n in a.contexts:
        ids = text_ids(tokenizer, n + 2 * B)
        e = prefill.engine
        e.clear_context()
        e.keep_kv, e.kv = True, {}
        try:
            ref, _ = e.forward(ids, 0, prefill.taps, 2 * B)
        finally:
            e.keep_kv = False
        target.seed_kv(*pack_prefill_kv(*e.kv_stack(), n), n)
        got = np.concatenate(
            [target.verify(ids[n + i : n + i + B], n + i).logits for i in (0, B)]
        )
        rows = []
        for r in range(2 * B):
            pick = int(got[r].argmax())
            rows.append(
                dict(
                    top1=pick,
                    reference_top1=int(ref[r].argmax()),
                    shortfall=float(ref[r].max() - ref[r][pick]),
                    cos=cos(ref[r], got[r]),
                )
            )
        passed = all(x["shortfall"] < TIE and x["cos"] >= MIN_COS for x in rows)
        result["contexts"].append(dict(context=n, passed=passed, rows=rows))
        print(
            f"N={n}: top-1 {sum(x['top1'] == x['reference_top1'] for x in rows)}"
            f"/{2 * B}, logit cos min {min(x['cos'] for x in rows):.4f}"
            + ("" if passed else " FAIL")
        )
    result["passed"] = all(c["passed"] for c in result["contexts"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
