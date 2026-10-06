# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build, or gate and time, the fused prefill of a dense model (models.py).

  python3 dense_main.py build MODEL BUILD_DIR [-j N]
  python3 dense_main.py verify MODEL BUILD_DIR [--prompt TEXT |
      --text-file F --text-tokens N] [--repeat R] [--tol T]

verify prints the logit cosine and first token against the fp32 numpy
reference on the same 4-bit weights, and per-run prefill times. GATE PASS needs logit
cosine >= --tol and the same first token.
"""

import argparse
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
# This directory's module names are too generic to publish: the package is
# reached as air_examples.*.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[2])
]

import numpy as np  # noqa: E402

from air_examples.llms.shared.fused_prefill.build import main as build  # noqa: E402
from air_examples.llms.shared.fused_prefill.models import MODELS  # noqa: E402


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def verify(name, argv):
    from air_examples.llms.shared.fused_prefill import dense

    desc = MODELS[name]
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--text-tokens", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tol", type=float, default=desc.tol)
    ap.add_argument("--model", default=None, help="bundle file or directory")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    if desc.family == "lfm2":
        from air_examples.llms.shared.fused_prefill import lfm2 as fam
    else:
        fam = dense
    path = fam.resolve(desc, a.model)
    tok = AutoTokenizer.from_pretrained(path.parent)
    prompt = a.prompt
    if a.text_file:
        body = tok(Path(a.text_file).read_text())["input_ids"][: a.text_tokens]
        prompt = "Summarize this:\n" + tok.decode(body)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
        user_system_prompt="",
        date_string="26 Jul 2024",
    )
    ids = list(ids if isinstance(ids, list) else ids["input_ids"])
    n = len(ids)
    print(f"prompt {n} tokens ({-(-n // 128)} chunk(s))", flush=True)

    t0 = time.perf_counter()
    pf = dense.load(name, a.build_dir, path)
    print(f"weights loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    logits = pf.prefill(ids)
    fails = []
    for r in range(a.repeat):
        t0 = time.perf_counter()
        lg = pf.prefill(ids)
        wall = time.perf_counter() - t0
        print(f"run {r}: prefill {wall * 1e3:.1f} ms", flush=True)
        if int(lg.argmax()) != int(logits.argmax()):
            fails.append(f"run {r} first token differs from the first run")

    ref = fam.reference(desc, path, ids)
    c = cos(logits, ref)
    print(f"logit cosine vs CPU reference = {c:.5f}")
    first, rfirst = int(logits.argmax()), int(ref.argmax())
    print(
        f"first token: device {first} {tok.decode([first])!r}  "
        f"reference {rfirst} {tok.decode([rfirst])!r}"
    )
    if not c >= a.tol:  # also catches NaN
        fails.append(f"logit cosine {c:.5f} < {a.tol}")

    # two tokens within bf16 resolution of each other, in either the
    # device's logits or the reference's, are the same first token
    def tied(v, a, b):
        return v[b] >= float(v[a]) - abs(float(v[a])) * 2**-7

    if first != rfirst and not (
        tied(logits, first, rfirst) or tied(ref, rfirst, first)
    ):
        fails.append("first token differs from the reference")
    if fails:
        print("GATE FAIL: " + "; ".join(fails))
        sys.exit(1)
    print("GATE PASS")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "verify"):
        sys.exit(__doc__)
    cmd, name, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if name not in MODELS:
        sys.exit(f"unknown model {name}; one of {', '.join(MODELS)}")
    if cmd == "build":
        build(f"dense:{name}", rest)
    else:
        verify(name, rest)


if __name__ == "__main__":
    main()
