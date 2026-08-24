#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does row t of the batched projection actually get TOKEN t? Asked on device.

THE BLIND SPOT THIS EXISTS FOR. batch_equiv.py dispatches B copies of ONE token,
because that is what makes token t comparable to a batch-1 run at position P+t.
It is the right premise for everything downstream of rope -- and it makes any
TOKEN PERMUTATION inside the projection invisible, because permuting identical
rows is the identity. batch_path_check.py cannot see one either: it models ONE
core, and the fault is in a cascade PAIR. proj_qmm_gate.py the same.

So this asks the question the other three structurally cannot.

HOW. Build with RMS_CHUNK_PROBE=1 (see rms_residual.cc): the rms core stops
normalising and feeds row t the CONSTANT (t+1)/8. The projection is linear, so
every output row then comes out proportional to whatever X its row of the mmul
actually saw, and the ratio of row t's output to row 0's IS that token index,
read straight off the DDR KV cache. V is copied through rope unrotated, so the
V region carries it; K is rotated and does not.

WHAT IT FOUND, on its first run [measured, llama-3.2-1b batch 8]:

    role 1:  1.000  2.000  3.000  4.000  5.000  6.000  7.000  8.000   exact
    role 0:  1.000  1.500  2.000  2.498  3.000  3.500  4.000  0.000

Role 0 -- the LEAD of every cascade pair, so half the output rows of every
projection -- reads `(1 + t/2)/8`, which is exactly `(X[0] + X[t])/2`. Its row t
is the MEAN of true row 0 and true row t. Row 7 is zero-or-garbage on top.
128 of 128 output rows agree on the ratio, so it is a clean structural map and
not noise.

    ./build_template.sh 8 1           # under RMS_CHUNK_PROBE=1
    python3 batch_row_probe.py --batch 8 --L 1 --prefix x1

Exit code is the gate: 0 when every role's row t scales as t+1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llms" / "bench"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument(
        "--prefix", default="x1", help="template prefix for the probe build"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()

    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("pyxrt not importable: this gate needs the NPU")

    import batch_equiv as be

    g = be.geom(args.model, args.vocab_chunk_i2, args.L, args.batch)
    g1 = be.geom(args.model, args.vocab_chunk_i2, args.L, 1)
    rng = np.random.default_rng(args.seed)
    row = be.bf16(rng.uniform(-1.0, 1.0, size=g1["k"]))
    xb, ib = be.template(args.prefix, args.batch, args.L)
    if not xb.exists():
        sys.exit(
            f"{xb.name} not found. Build it with\n"
            f"    RMS_CHUNK_PROBE=1 UNI_WAVE_HI=1 ./build_template.sh "
            f"{args.batch} {args.L}\n"
            f"and rename decode_b{args.batch}_L{args.L}.* to {xb.stem}.*"
        )
    _, _, kv = be.dispatch(xb, ib, g, args.batch, row, args.seed, xrt)

    w, stride, ngrp, L = g["kv_region"]
    base = ngrp * stride + (L - 1) * w
    V = [
        be.as_f32(kv[base + t * w : base + (t + 1) * w]).astype(np.float64)
        for t in range(args.batch)
    ]
    # A 64-element emitter block is [lead's ROW_BLOCK | partner's ROW_BLOCK],
    # so the two roles are the two halves and have to be scored separately --
    # scoring them together averages a broken role into a working one, which is
    # how this went unnoticed.
    import proj_qmm_pack as pk

    rb, pay = pk.ROW_BLOCK, 2 * pk.ROW_BLOCK
    if w % pay:
        sys.exit(f"V region {w} is not a whole number of {pay}-element pairs")
    print(f"\nprojection row map  [{args.model}, batch {args.batch}, L {args.L}]")
    print("  X row t is the constant (t+1)/8, so the ratio IS the token index\n")
    ok = True
    for role in range(2):
        idx = np.concatenate(
            [
                np.arange(b * pay + role * rb, b * pay + role * rb + rb)
                for b in range(w // pay)
            ]
        )
        v0 = V[0][idx]
        nz = v0 != 0
        if not nz.any():
            print(f"  role {role}: token 0's row is all zero -- nothing to divide by")
            ok = False
            continue
        line, bad = [], False
        for t in range(args.batch):
            r = V[t][idx][nz] / v0[nz]
            med = float(np.median(r))
            agree = int(np.sum(np.isclose(r, med, rtol=1e-2)))
            line.append(f"{med:7.3f}")
            if abs(med - (t + 1)) > args.tol * (t + 1) or agree < 0.9 * nz.sum():
                bad = True
        print(f"  role {role}: " + " ".join(line) + ("   <-- WRONG" if bad else ""))
        ok = ok and not bad
    print("\n  want " + " ".join(f"{t + 1:7.3f}" for t in range(args.batch)))
    if not ok:
        print(
            "\n  A role whose row t does not scale as t+1 is reading the wrong\n"
            "  token. Nothing else in this directory can see that: batch_equiv\n"
            "  dispatches B copies of ONE token, so a permutation of identical\n"
            "  rows is the identity, and batch_path_check and proj_qmm_gate both\n"
            "  model a single core rather than a cascade PAIR.\n"
            "  FAIL"
        )
        return 1
    print("\n  every role's row t scales as t+1 -- GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
