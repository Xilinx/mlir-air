#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The eight-token decoder against reference_b8's float32 forward of the
same weights.

Two calls: eight tokens at positions 0..7 on an empty cache, then the next
eight over the K/V the first appended. Every layer boundary is compared by
cosine and every token's logits by cosine and top-1; the logits gate.

  python3 tests/check_target.py --artifact DIR --output FILE
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

TEXT = (
    "The quick brown fox jumps over the lazy dog while the cat sleeps on the warm"
    " windowsill of the old house"
)
MIN_LOGIT_COS = 0.99


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    ids = [2] + load_tokenizer()(TEXT, add_special_tokens=False)["input_ids"]
    ids = ids[:16]
    target = GemmaTarget(a.artifact)
    n = target.layers
    xs_ref, lg_ref = reference_b8.forward(target.model, ids, n)
    result = dict(input_ids=ids, min_logit_cos=MIN_LOGIT_COS, blocks=[])
    for p in (0, 8):
        lg = target.verify(ids[p : p + 8], p).logits
        xs = target.boundaries(range(1, n + 1))
        bounds = [
            min(cos(xs[b][t], xs_ref[b][p + t]) for t in range(8))
            for b in range(1, n + 1)
        ]
        lc = [cos(lg[t], lg_ref[p + t]) for t in range(8)]
        top = lg.argmax(1).tolist()
        ref = lg_ref[p : p + 8].argmax(1).tolist()
        result["blocks"].append(
            dict(
                position=p,
                boundary_cos_min=bounds,
                logit_cos=lc,
                top1=top,
                reference_top1=ref,
            )
        )
        print(
            f"block at {p}: logit cos min {min(lc):.4f}, top-1 "
            f"{sum(x == y for x, y in zip(top, ref))}/8, boundary cos min "
            f"{min(bounds):.4f}"
        )
    result["passed"] = all(
        min(b["logit_cos"]) >= MIN_LOGIT_COS for b in result["blocks"]
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
