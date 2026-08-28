#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Dispatch ONE template and report, per PAYLOAD-wide chunk of the X buffer,
how many elements came back zero and how many still hold the host's fill.

Written for one question: qwen3-4b's batch-1 layer-output drain leaves the LAST
512 of K=2560 as zeros (see docs/DFlashFeasibility.md 5.3), and llama-3.2-1b
(K=2048) does not. Three outcomes are distinguishable here and are not by a
two-template compare:

  chunk is ZERO          something wrote zeros over it
  chunk is UNTOUCHED     the drain never covered it (still the host's fill)
  chunk is WRITTEN       normal

    TEMPLATE=decode_b1_L128 python3 _x_hole.py
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_equiv as BE

MODEL = os.environ.get("DECODE_MODEL", "qwen3-4b")
VOCAB_I2 = int(os.environ.get("VOCAB_CHUNK_I2", "30"))
BATCH = int(os.environ.get("HOLE_BATCH", "1"))
L = int(os.environ.get("HOLE_L", "128"))
SEED = 0


def main():
    import pyxrt as xrt

    g = BE.geom(MODEL, VOCAB_I2, L, BATCH)
    K = BE.geom(MODEL, VOCAB_I2, L, 1)["k"]
    rng = np.random.default_rng(SEED)
    row = BE.bf16(rng.uniform(-1.0, 1.0, size=K))

    xc, xi = BE.template("decode", BATCH, L)
    if not Path(xc).exists():
        sys.exit(f"missing {xc}")

    y, _, _ = BE.dispatch(xc, xi, g, BATCH, row, SEED, xrt, 60000)

    PAY = 512
    nch = K // PAY
    print(f"\nX-buffer hole probe  [{MODEL}, batch {BATCH}, L {L}, {Path(xc).name}]")
    print(f"  K={K}, {nch} chunks of {PAY}\n")
    print(f"  {'chunk':>5} {'zeros':>7} {'== host fill':>13} {'rms':>10}  verdict")
    for c in range(nch):
        got = y[c * PAY : (c + 1) * PAY]
        src = row[c * PAY : (c + 1) * PAY]
        nz = int((got == 0).sum())
        same = int((got == src).sum())
        r = float(np.sqrt(np.mean(BE.as_f32(got).astype(np.float64) ** 2)))
        verdict = (
            "ALL ZERO" if nz == PAY
            else "UNTOUCHED (host fill)" if same == PAY
            else "written"
        )
        print(f"  {c:>5} {nz:>7} {same:>13} {r:>10.4g}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
