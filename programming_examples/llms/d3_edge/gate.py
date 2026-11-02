# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The d3 decision gate, the models' correctness gate (`make verify`): the
production NPU path on a frozen request set against a saved CPU reference
(reference.py's format).

A question passes when its option logits are all finite, the NPU answer is in
the reference's top 2 and the reference answer in the NPU's top 2, and, when
the reference's top-1 and top-2 probabilities differ by at least 0.05, the
answers are equal. The gate passes when every question passes and the capture
is complete (every question of the set, input ids equal to the reference's).
Exits 1 when it fails.

  python3 gate.py --build BUILD --model edge [--wform hilo] \
      --reference reference.json --out gate_report.json
"""

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

from air_examples.llms.d3_edge import d3_weights as W3  # noqa: E402
from air_examples.llms.d3_edge.d3_npu import D3Npu, add_model_args  # noqa: E402
from air_examples.llms.d3_edge.reference import requests_sha256  # noqa: E402

MARGIN = 0.05


def top2(p):
    return [int(i) for i in np.argsort(-np.asarray(p), kind="stable")[:2]]


def check(ref_p, npu_logits, npu_p):
    ref_p, npu_p = np.asarray(ref_p), np.asarray(npu_p)
    finite = bool(np.isfinite(npu_logits).all())
    ra, na = int(np.argmax(ref_p)), int(np.argmax(npu_p))
    rt, nt = top2(ref_p), top2(npu_p)
    srt = np.sort(ref_p)[::-1]
    margin = float(srt[0] - srt[1]) if len(srt) > 1 else 1.0
    mutual = na in rt and ra in nt
    decisive_ok = margin < MARGIN or na == ra
    return dict(
        finite=finite,
        ref_answer=ra,
        npu_answer=na,
        ref_top2=rt,
        npu_top2=nt,
        ref_margin=margin,
        mutual_top2=mutual,
        decisive_ok=decisive_ok,
        agree=na == ra,
        max_abs_dp=float(np.abs(ref_p - npu_p).max()),
        passed=finite and mutual and decisive_ok,
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--requests", default=str(HERE / "requests.json"))
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    reqs = json.loads(Path(a.requests).read_text())
    ref = json.loads(Path(a.reference).read_text())
    assert ref["identity"]["requests_sha256"] == requests_sha256(
        a.requests
    ), "reference is for another request set"
    # both models share the request set: the reference must be of this
    # model, and for lite of the pinned bundle's values
    idn = ref["identity"]
    assert (idn["repo"], idn["revision"]) == W3.MODELS[
        a.model
    ], "reference is for another model"
    if a.model == "lite":
        pinned = (W3.LITE_REPO, W3.LITE_REVISION, W3.LITE_SHA256)
        assert (
            idn.get("package_repo"),
            idn.get("package_revision"),
            idn.get("bundle_sha256"),
        ) == pinned, "reference is not of the pinned d3-lite bundle"
    refq = {(q["request"], q["question"]): q for q in ref["questions"]}
    rows, t0 = [], time.time()
    m = D3Npu(a.build, a.model, a.wform)
    for req in reqs["requests"]:
        for key, nq, ids in m.prepare(req["state"], req["questions"]):
            r = refq[(req["id"], key)]
            count = len(nq.keys)
            z = m.logits(ids)[:count]
            p = m.softmax(z)
            row = dict(
                request=req["id"],
                question=key,
                type=r["type"],
                bucket=req["bucket"],
                n_tokens=len(ids),
                ids_match=ids == r["input_ids"],
                logits=z.tolist(),
                probabilities=p.tolist(),
                ref_probabilities=r["probabilities"],
            )
            row.update(check(r["probabilities"], z, p))
            row["passed"] = row["passed"] and row["ids_match"]
            rows.append(row)
            print(
                f"{req['id']}/{key} n={len(ids)} ref={row['ref_answer']} "
                f"npu={row['npu_answer']} margin={row['ref_margin']:.3f} "
                f"dp={row['max_abs_dp']:.4f} {'PASS' if row['passed'] else 'FAIL'}",
                flush=True,
            )
    complete = len(rows) == len(ref["questions"]) == reqs["n_questions"]
    dps = [
        abs(x - y)
        for r in rows
        for x, y in zip(r["probabilities"], r["ref_probabilities"])
    ]
    summary = dict(
        gate="PASS" if complete and all(r["passed"] for r in rows) else "FAIL",
        model=a.model,
        wform=m.wform,
        complete=complete,
        questions=len(rows),
        passed=sum(r["passed"] for r in rows),
        agreement=sum(r["agree"] for r in rows) / len(rows),
        max_abs_dp=max(dps),
        mean_abs_dp=float(np.mean(dps)),
        by_bucket={
            str(b): dict(
                questions=sum(r["bucket"] == b for r in rows),
                passed=sum(r["passed"] for r in rows if r["bucket"] == b),
                agree=sum(r["agree"] for r in rows if r["bucket"] == b),
            )
            for b in reqs["buckets"]
        },
        by_type={
            t: dict(
                questions=sum(r["type"] == t for r in rows),
                passed=sum(r["passed"] for r in rows if r["type"] == t),
            )
            for t in ("choice", "noul", "score")
        },
        requests_sha256=requests_sha256(a.requests),
        reference_sha256=sha256(a.reference),
        reference_identity=ref["identity"],
        seconds=time.time() - t0,
    )
    Path(a.out).write_text(json.dumps(dict(summary=summary, questions=rows), indent=1))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "reference_identity"}, indent=1
        )
    )
    print(f"[d3 gate] {summary['gate']} {summary['passed']}/{reqs['n_questions']}")
    if summary["gate"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
