# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gate and time the fused prefill against the CPU reference.

  python3 verify.py BUILD_DIR [--prompt TEXT | --text-file F --text-tokens N |
                              --n-tokens N] [--repeat R]

Prints the logit cosine and first token against forward_prompt, the lowest
per-layer K/V cosine against the reference's cache, and per-run prefill times.
GATE PASS needs logit cosine >= --tol and the same first token.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np  # noqa: E402

import gemma4_e2b_q4nx_weights as gw  # noqa: E402
from runtime import FusedPrefill  # noqa: E402


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument(
        "--n-tokens", type=int, default=0, help="synthetic prompt of N tokens"
    )
    ap.add_argument(
        "--text-file",
        default=None,
        help="prompt from a text file, cut to --text-tokens",
    )
    ap.add_argument("--text-tokens", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    model = gw.Q4nxModel(a.model)
    if a.n_tokens:
        ids = [2] + list(
            np.random.default_rng(0).integers(1000, 200000, a.n_tokens - 1)
        )
        tok = None
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model.path.parent)
        prompt = a.prompt
        if a.text_file:
            body = tok(Path(a.text_file).read_text())["input_ids"][: a.text_tokens]
            prompt = "Summarize this:\n" + tok.decode(body)
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        ids = list(ids if isinstance(ids, list) else ids["input_ids"])
    n = len(ids)
    print(f"prompt {n} tokens ({-(-n // 128)} chunk(s))", flush=True)

    pf = FusedPrefill(a.build_dir, max_len=max(2048, n))
    t0 = time.perf_counter()
    pf.load_weights(model)
    print(f"weights loaded in {time.perf_counter() - t0:.0f} s", flush=True)

    logits = pf.prefill(ids)
    fails = []
    for r in range(a.repeat):
        pf.dev_t, pf.by_op = 0.0, {}
        t0 = time.perf_counter()
        lg = pf.prefill(ids)
        wall = time.perf_counter() - t0
        print(
            f"run {r}: prefill {wall * 1e3:.1f} ms (device {pf.dev_t * 1e3:.1f} ms)",
            flush=True,
        )
        if int(lg.argmax()) != int(logits.argmax()):
            fails.append(f"run {r} first token differs from the first run")
    ks, vs = pf.kv_stack()

    ref, kv = gw.forward_prompt(model, ids)
    c = cos(logits, ref)
    print(f"logit cosine vs CPU reference = {c:.5f}")
    first, rfirst = int(logits.argmax()), int(ref.argmax())
    name = (lambda t: repr(tok.decode([t]))) if tok else str
    print(
        f"first token: device {first} {name(first)}  reference {rfirst} {name(rfirst)}"
    )
    worst = min(
        min(
            cos(ks[L], kv[gw.kv_source_layer(L)][0][:, 0]),
            cos(vs[L], kv[gw.kv_source_layer(L)][1][:, 0]),
        )
        for L in range(gw.NUM_LAYERS)
    )
    print(f"worst per-layer K/V cosine vs reference = {worst:.5f}")
    if c < a.tol:
        fails.append(f"logit cosine {c:.5f} < {a.tol}")
    if first != rfirst:
        fails.append("first token differs from the reference")
    if fails:
        print("GATE FAIL: " + "; ".join(fails))
        sys.exit(1)
    print("GATE PASS")


if __name__ == "__main__":
    main()
