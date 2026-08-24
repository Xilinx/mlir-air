#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What does aie::mmul<8,8,8,bf16> mean by an A, B and C tile? Asked on device.

q4k_mm.h's whole layout derivation rests on one assumption: that a 64-element
mmul operand vector is a row-major 8x8 tile -- A as [r][s], B as [s][t], C as
[r][t]. Everything else in q4k_mmul is explicit in the source (the pointer walk
is written down; the strides are not in doubt), so when a whole-block result
disagrees with numpy, this assumption is what is left.

It is also the one thing that can be settled without any strides at all. Feed
ONE multiply an operand pair where the answer names its own layout:

    probe 0   A = 1..64,   B = I     ->  C == A, in A's own order
    probe 1   A = I,       B = 1..64 ->  C == B, in B's own order
    probe 2   A = 1..64,   B = 1..64 ->  the full product, as a cross-check
    probe 3   A = e_0 e_0^T          ->  C picks out B's first row

Values 1..64 are integers, so bf16 holds them exactly and so does the bfp16
emulation the build turns on; the comparison is ==. Reading C back and asking
which reshape of 1..64 it equals answers the question directly instead of
inferring it from a 512-deep contraction that has many ways to be wrong.

Reuses q4k_mm_gate.py's design verbatim -- same four buffers, same herd -- with
the kernel body swapped by -DGATE_MMUL_PROBE. Nothing here is a gate; it is the
instrument the gate needed.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

import q4k_mm_gate as gate

NP = 4  # probes per run
TILE = 64
EYE = np.eye(8, dtype=np.float32)


def probes():
    """-> (A [NP,8,8], B [NP,8,8]) as plain row-major maths matrices."""
    seq = (np.arange(TILE, dtype=np.float32) + 1).reshape(8, 8)
    A = np.stack([seq, EYE, seq, np.zeros((8, 8), np.float32)])
    B = np.stack([EYE, seq, seq, seq])
    A[3, 0, 0] = 1.0
    return A, B


def name_layout(C, seq):
    """Which simple reshape of `seq` does this 64-vector equal?"""
    cands = {
        "row-major [r][s]": seq.ravel(),
        "transposed [s][r]": seq.T.ravel(),
    }
    for nm, v in cands.items():
        if np.array_equal(C, v):
            return nm
    return "NEITHER row-major nor transposed"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--device", default="npu2")
    args = ap.parse_args()

    from air.ir import Context, Location
    from air.backend.xrt import XRTBackend

    # Same shapes as the gate, so the same design and the same DMA path.
    mrows, kcol, batch, nblk = gate.ROW_BLOCK, gate.COL_BLOCK, 16, 1
    with Context(), Location.unknown():
        module = gate.build_module(mrows, kcol, batch, nblk)

    build, obj = gate.prepare_build()
    gate.compile_kernel(
        obj, mrows, kcol, batch, nblk, extra=[f"-DGATE_MMUL_PROBE={NP}"]
    )
    gate.stage(build, obj)

    A, B = probes()
    a_bo = np.zeros(nblk * batch * kcol, bfloat16)
    b_bo = np.zeros(nblk * gate.BLOCK_BF16, bfloat16)
    for p in range(NP):
        a_bo[p * TILE : (p + 1) * TILE] = A[p].ravel().astype(bfloat16)
        b_bo[p * TILE : (p + 1) * TILE] = B[p].ravel().astype(bfloat16)
    y_bo = np.zeros(batch * mrows, np.float32)
    w_bo = np.zeros(mrows * kcol, bfloat16)

    backend = XRTBackend(
        verbose=args.verbose,
        omit_pingpong=True,
        target_device=args.device,
        stack_size=4096,
    )
    fn = backend.load(backend.compile(module))
    outs = fn(b_bo, a_bo, y_bo, w_bo)
    backend.unload()
    C = np.asarray(outs[-2], np.float32).ravel()
    W = np.asarray(outs[-1]).ravel().astype(np.float32)

    seq = (np.arange(TILE, dtype=np.float32) + 1).reshape(8, 8)
    print("\naie::mmul<8,8,8,bf16> tile semantics, measured")
    b_ok = np.array_equal(W[: NP * TILE], b_bo[: NP * TILE].astype(np.float32))
    print(f"  B operand reached L1 intact: {b_ok}")
    for p in range(NP):
        c = C[p * TILE : (p + 1) * TILE]
        want = (A[p] @ B[p]).ravel()
        tag = "== A@B row-major" if np.array_equal(c, want) else "!= A@B row-major"
        print(f"\n  probe {p}: {tag}")
        if p == 0:
            print(f"    C vs A=1..64 -> {name_layout(c, seq)}")
        if p == 1:
            print(f"    C vs B=1..64 -> {name_layout(c, seq)}")
        print(f"    C = {np.array2string(c.reshape(8, 8).astype(int))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
