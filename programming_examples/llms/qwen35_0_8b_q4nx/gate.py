# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Model gate: AIR's vLLM-derived top-5 token check of the production path
(Qwen35Npu.generate) against the CPU BF16 reference (reference.py).

Per prompt the NPU greedily decodes 32 tokens (EOS ignored) from one
configuration's builds (prefill bucket P, decode capacity C), with every
step's full-vocabulary logits checked finite. Walking both token streams in
lockstep, at the first step where the chosen tokens differ each side's token
must be in the other side's top-5; comparison stops there. Every prompt must
pass.

  python3 gate.py --build BUILD --reference REF_DIR --out OUT_DIR --prefill P --capacity C
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
    """(verdict, first divergence step or None, steps compared)."""
    n = min(len(npu_toks), len(ref_toks))
    for i in range(n):
        if npu_toks[i] != ref_toks[i]:
            ok = npu_toks[i] in ref_top[i][:TOPK] and ref_toks[i] in npu_top[i][:TOPK]
            return ("PASS" if ok else "FAIL"), i, i + 1
    return "PASS", None, n


def main():
    from ml_dtypes import bfloat16

    from air_examples.llms.qwen35_0_8b_q4nx.q35_npu import Qwen35Npu, tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill", type=int, required=True)
    ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument(
        "--fit",
        action="store_true",
        help="only the reference prompts that fit the bucket (the timing-prompt set)",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((Path(a.reference) / "reference.json").read_text())
    tok = tokenizer()
    rows = []
    m = Qwen35Npu(a.build, a.capacity)
    for i, r in enumerate(ref["prompts"]):
        if a.fit and len(r["ids"]) > a.prefill:
            continue
        if r.get("from_ids"):  # a timing prompt, given as ids
            ids = r["ids"]
        else:
            ids = tok(r["prompt"])["input_ids"]
            assert ids == r["ids"], "tokenizer differs from the reference capture"
        toks, logits, ttft, dt = m.generate(ids, len(r["tokens"]))
        lg = np.stack(logits)
        finite = bool(np.isfinite(lg).all())
        np.save(out / f"npu_logits_{i}.npy", lg.astype(bfloat16).view(np.int16))
        top = np.argsort(-lg, 1)[:, :TOPK].tolist()
        verdict, div, cmp = compare(toks, top, r["tokens"], r["top5"])
        if not finite or len(toks) != len(r["tokens"]):
            verdict = "FAIL"
        rows.append(
            dict(
                prompt=r["prompt"],
                n_prompt=len(ids),
                tokens=toks,
                top5=top,
                ref_tokens=r["tokens"],
                finite=finite,
                captured=len(toks),
                requested=len(r["tokens"]),
                first_divergence=div,
                steps_compared=cmp,
                verdict=verdict,
                ttft_s=ttft,
                decode_s=dt,
                text=tok.decode(toks),
            )
        )
        print(
            f"[{i}] {verdict} div={div} finite={finite} {tok.decode(toks)[:60]!r}",
            flush=True,
        )
    rep = dict(
        prefill=a.prefill,
        capacity=a.capacity,
        reference=ref["meta"],
        decode="fused_decode_gdn",
        method="greedy 32, EOS ignored, mutual top-5 at first divergence, full finite logits",
        # an empty selection compared nothing and must not pass
        verdict=(
            "PASS" if rows and all(r["verdict"] == "PASS" for r in rows) else "FAIL"
        ),
        prompts=rows,
        time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (out / "gate_report.json").write_text(json.dumps(rep, indent=1))
    print(
        f"P={a.prefill} C={a.capacity}: {rep['verdict']} "
        f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})"
    )
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
