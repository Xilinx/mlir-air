#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The two-sequence decoder (build_target.py --batch 2 --sequences 2) against
reference_b8's float32 forward of each sequence on its own.

Sequence A takes a token every step. Sequence B starts two steps later and
sits out some steps on the way, so the two run at different positions and B's
cache must come through its idle steps untouched. Every step's logits gate on
cosine against the reference.

  python3 tests/check_sequences.py --artifact DIR --output FILE
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference_b8  # noqa: E402
from model_paths import load_tokenizer  # noqa: E402
from target_runtime import GemmaTarget  # noqa: E402

TEXT_A = "The quick brown fox jumps over the lazy dog while the cat sleeps"
TEXT_B = "Adaptive cruise control keeps a safe distance from the car ahead"
# per step: does B take a token
B_ACTIVE = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
MIN_LOGIT_COS = 0.99


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    tok = load_tokenizer()
    A = [2] + tok(TEXT_A, add_special_tokens=False)["input_ids"]
    B = [2] + tok(TEXT_B, add_special_tokens=False)["input_ids"]
    A = A[: len(B_ACTIVE)]
    target = GemmaTarget(a.artifact)
    if target.sequences != 2:
        raise SystemExit("the artifact is not a two-sequence build")
    got = {"A": [], "B": []}
    j = 0
    for i, b_on in enumerate(B_ACTIVE):
        on = [True, bool(b_on)]
        r = target.step([A[i], B[j] if b_on else 0], [i, j if b_on else 0], on)
        got["A"].append(r.logits[0])
        if b_on:
            got["B"].append(r.logits[1])
            j += 1
    result = dict(min_logit_cos=MIN_LOGIT_COS, sequences={})
    for name, ids in (("A", A), ("B", B[:j])):
        _, ref = reference_b8.forward(target.model, ids, target.layers)
        lc = [cos(x, y) for x, y in zip(got[name], ref)]
        top = [int(x.argmax()) for x in got[name]]
        want = ref.argmax(1).tolist()
        result["sequences"][name] = dict(
            input_ids=ids, logit_cos=lc, top1=top, reference_top1=want
        )
        print(
            f"sequence {name}: {len(ids)} steps, logit cos min {min(lc):.4f}, "
            f"top-1 {sum(x == y for x, y in zip(top, want))}/{len(ids)}"
        )
    result["passed"] = all(
        min(s["logit_cos"]) >= MIN_LOGIT_COS for s in result["sequences"].values()
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
