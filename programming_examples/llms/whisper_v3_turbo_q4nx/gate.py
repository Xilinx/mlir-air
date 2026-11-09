# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Model gate: AIR's top-5 token check of the production path
(whisper_npu.transcribe) against the CPU BF16 reference (reference.py).

Per clip the NPU decodes greedily until EOS (at most whisper_npu.MAX_NEW
tokens) with every step's full-vocabulary logits checked finite. Walking both
token streams in lockstep, at the first step where the chosen tokens differ
(or one stream ends) each side's token must be in the other side's top-5;
comparison stops there. Every clip must pass.

  python3 gate.py --build BUILD --reference REF_DIR --out OUT_DIR [--clips N]
"""

import argparse
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

TOPK = 5


def compare(npu_toks, npu_top, ref_toks, ref_top):
    """(verdict, first divergence step or None). A stream that ends first
    diverges where its EOS meets the other's token."""
    n = min(len(npu_toks), len(ref_toks))
    for i in range(n):
        if npu_toks[i] != ref_toks[i]:
            ok = npu_toks[i] in ref_top[i][:TOPK] and ref_toks[i] in npu_top[i][:TOPK]
            return ("PASS" if ok else "FAIL"), i
    if len(npu_toks) != len(ref_toks):
        return "FAIL", n
    return "PASS", None


def main():
    from air_examples.llms.whisper_v3_turbo_q4nx import inputs
    from air_examples.llms.whisper_v3_turbo_q4nx.whisper_npu import (
        GEN_REPO,
        GEN_REVISION,
        Decoding,
        WhisperNpu,
        npu_lock,
        tokenizer,
    )
    from air_examples.llms.whisper_v3_turbo_q4nx.whisper_weights import Q4nx

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=None)
    a = ap.parse_args()
    ref = json.loads((Path(a.reference) / "reference.json").read_text())
    # the reference must be of the pinned checkpoint, in BF16, on the pinned
    # clips: a stale reference.json is not reused silently
    want = dict(
        model=GEN_REPO,
        revision=GEN_REVISION,
        precision="bf16",
        dataset=inputs.DATASET,
        dataset_revision=inputs.REVISION,
    )
    stale = {k: ref.get(k) for k, v in want.items() if ref.get(k) != v}
    if stale:
        sys.exit(f"[verify] FAIL: reference is not {want}: {stale}; rerun reference.py")
    refs = {r["id"]: r for r in ref["clips"]}
    clips = inputs.clips(a.clips)
    if not clips:
        sys.exit("[verify] FAIL: no clips selected")
    w = Q4nx()
    tok, dec = tokenizer(w), Decoding()
    with npu_lock():
        results = run(WhisperNpu(a.build, w), clips, refs, tok, dec)
    fails = sum(r["verdict"] != "PASS" for r in results)
    matched = sum(r["divergence"] is None for r in results)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "gate.json").write_text(
        json.dumps(dict(reference=ref["model"], results=results))
    )
    n = len(results)
    print(f"[gate] {n - fails}/{n} clips pass, {matched}/{n} token-exact")
    print("[verify] PASS" if fails == 0 else "[verify] FAIL")
    sys.exit(1 if fails else 0)


def run(npu, clips, refs, tok, dec):
    from air_examples.llms.whisper_v3_turbo_q4nx.whisper_npu import transcribe

    results = []
    for cid, audio, _ in clips:
        if cid not in refs:
            sys.exit(f"[verify] FAIL: no reference for {cid}; rerun reference.py")
        r = refs[cid]
        tops = []

        def on_step(i, z):
            tops.append(np.argsort(-z, kind="stable")[:TOPK].tolist())

        t0 = time.perf_counter()
        out = transcribe(npu, audio, dec, on_step=on_step)
        dt = time.perf_counter() - t0
        verdict, d = compare(out, tops, r["tokens"], r["top"])
        results.append(
            dict(id=cid, verdict=verdict, divergence=d, tokens=out, ms=dt * 1e3)
        )
        div = "exact" if d is None else f"first divergence at {d}"
        print(f"[gate] {cid}: {verdict} ({len(out)} tokens, {div}, {dt * 1e3:.0f} ms)")
        if d is not None:
            print(f"    npu: {tok.decode([t for t in out if t < dec.eos])}")
            print(f"    ref: {tok.decode([t for t in r['tokens'] if t < dec.eos])}")
    return results


if __name__ == "__main__":
    main()
