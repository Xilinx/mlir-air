# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B's LM head and a single-run check against the host reference.
import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

import filelock
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import common  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.decode import lm_head as LH  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.015

D, VOCAB = W.D, W.VOCAB
NCOL, NROW, RB, BB, KB = LH.NCOL, LH.NROW, LH.RB, LH.BB, D // 256
rounds = LH.rounds
ROUNDS = rounds(VOCAB)


def build_module(rows=1, vocab=VOCAB):
    return LH.build_module(D, vocab, rows)


def pack_weights(w, vocab=VOCAB):
    return LH.pack_weights(w.raw("lm_head.weight"), D, vocab)


def unpack_logits(y, rows=1, vocab=VOCAB):
    return LH.unpack_logits(y, vocab, rows)


def compile_lm_head(rows=1, vocab=VOCAB):
    return LH.compile_lm_head(common.compile_kernel, D, vocab, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--tokens", default="9338", help="one token, or two (rows=2)")
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    toks = [int(v) for v in a.tokens.split(",")]
    rows = len(toks)
    if rows not in (1, 2):
        ap.error("--tokens: one token, or two")
    if a.print_module_only:
        print(build_module(rows))
        return 0
    import mlp_block

    w = W.Qwen38Q4nx()
    art = compile_lm_head(rows)
    x = w.embed(toks).astype(bfloat16)
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    y0 = np.zeros((NCOL, ROUNDS, NROW * rows * RB), bfloat16)
    args = [x.ravel(), nw, pack_weights(w), y0]
    if a.bench:
        return mlp_block.bench(art, args, a.bench, (1, 2))
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        be = XRTBackend(instance_name="lm_head", output_format="elf")
        run = be.load(art)
        res = run(*args)
        be.unload()
    got = unpack_logits(res[3], rows).astype(np.float32).reshape(rows, VOCAB)
    xf = x.astype(np.float32)
    xn = (
        xf
        / np.sqrt(np.mean(xf * xf, -1, keepdims=True) + W.EPS)
        * nw.astype(np.float32)
    )
    xnb = xn.astype(bfloat16).astype(np.float32)
    want = np.concatenate(
        [
            xnb @ w.lm_head_rows(r, min(r + 8192, VOCAB)).T
            for r in range(0, VOCAB, 8192)
        ],
        axis=1,
    )
    ok = True
    for t in range(rows):
        c, l2, e = common.compare(got[t], want[t])
        print(f"row {t} logits: corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
        print(
            "top5 dev",
            np.argsort(-got[t])[:5].tolist(),
            "ref",
            np.argsort(-want[t])[:5].tolist(),
        )
        ok &= c > 0.9999 and l2 < TOL and np.argmax(got[t]) == np.argmax(want[t])
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
