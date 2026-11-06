#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The two-request decoder (build_target.py --requests 2 --tokens-per-request
1) against reference_b8's float32 forward of each request on its own.

Over the schedule below each request decodes some steps alone, both together
on others, and each sits out stretches and resumes, so the two run at
different positions, every mask is used, and an idle request's cache must
come through untouched. Each step names the next one as `upcoming`, wrongly
on some steps, so the instruction stream built during a dispatch is both used
and discarded. Every step's logits gate on cosine against the reference.

  python3 tests/check_requests.py --artifact DIR --output FILE
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
# per step, which requests take a token: A, B or both (X)
SCHEDULE = "AAXXXBBXXAXXBX"
# steps whose `upcoming` misnames the next step's requests
MISPREDICT = (4, 9)
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
    target = GemmaTarget(a.artifact)
    if target.requests != 2:
        raise SystemExit("the artifact is not a two-request build")
    ids = (A, B)
    on = [[c in (name, "X") for name in "AB"] for c in SCHEDULE]
    got = ([], [])
    pos = [0, 0]
    for i, act in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else act
        if i in MISPREDICT:
            nxt = [not x for x in nxt] if not all(nxt) else [True, False]
        upcoming = ([p + a for p, a in zip(pos, act)], nxt)
        r = target.step([ids[k][pos[k]] for k in range(2)], pos, act, upcoming)
        for k in range(2):
            if act[k]:
                got[k].append(r.logits[k])
                pos[k] += 1
    got = dict(A=got[0], B=got[1])
    A, B = A[: pos[0]], B[: pos[1]]
    result = dict(min_logit_cos=MIN_LOGIT_COS, requests={})
    for name, ids in (("A", A), ("B", B)):
        _, ref = reference_b8.forward(target.model, ids, target.layers)
        lc = [cos(x, y) for x, y in zip(got[name], ref)]
        top = [int(x.argmax()) for x in got[name]]
        want = ref.argmax(1).tolist()
        result["requests"][name] = dict(
            input_ids=ids, logit_cos=lc, top1=top, reference_top1=want
        )
        print(
            f"request {name}: {len(ids)} steps, logit cos min {min(lc):.4f}, "
            f"top-1 {sum(x == y for x, y in zip(top, want))}/{len(ids)}"
        )
    result["passed"] = all(
        min(s["logit_cos"]) >= MIN_LOGIT_COS for s in result["requests"].values()
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
