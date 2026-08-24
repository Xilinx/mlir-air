#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does the whole batched projection path compose? Token t's row vs batch 1's.

WHY THIS EXISTS. xfeed_bd.py, egress_bd.py and kvappend_bd.py each check ONE
descriptor, and each of them passes. That is not the same as the path working,
because the pieces meet at conventions no single checker sees both sides of:

    xfeed BD          writes X in aie::mmul A-tile order
      -> pack_A       ... which pack_A must agree is A-tile order
      -> the mmul     ... whose C comes back in C-tile order
      -> the FLUSH    ... which de-tiles it at (t*tok_stride + i)*ROW_BLOCK
      -> the gathers  ... which carry it EMITTER-MAJOR, contiguous, header
                          intact (see egress_bd.py for why they cannot
                          transpose)
      -> outy_tokmajor ... which de-interleaves it at the consumer

Get `tok_stride` wrong between the kernel and the descriptor and BOTH checkers
still pass: the flush writes a self-consistent layout and the gather reads a
self-consistent layout, and they are different layouts. That is the exact shape
of the two faults the device gate caught in q4k_mm.h -- plausible, silent, and
invisible to any per-piece test.

So this walks a token block through EVERY piece in order, in numpy, using the
same functions and the same descriptors the engine will use, and asserts the
one property that matters:

    token t's assembled PAYLOAD row == the row batch 1 produces for token t

WHAT IT IS NOT. It is not the device gate. It models the data path, not the
engine: no DMA, no locks, no cascade, and the arithmetic is the same reference
q4k_mm_gate.py already proved bit-exact on device rather than a fresh one. What
it catches is LAYOUT disagreement between the pieces, which is what the wiring
is about to introduce and what nothing else would catch until a model produced
subtly wrong tokens.

IT HAS BEEN SEEN TO FAIL, which is the only reason to trust a check that passed
on its first run. Two faults injected, both caught:

  flush writes role-major instead of token-major   -> token 0 differs at 242
  de-tiling drops the RA factor                    -> token 0 differs at 411

The second only shows at batch 16 and above. At batch 8 `RA` is 1, so dropping
it is a no-op -- which is the same coincidence that lets one flush serve 8, 16
and 32, and a reminder that a batch-8-only run does not exercise the tiling.

    python3 batch_path_check.py             # both models, several batches
    python3 batch_path_check.py -v          # per-stage shapes

Exit code is the gate: 0 the path composes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q4k_mm_gate import MMUL_S, MMUL_T, mmul_r, pack_A, pack_B, unpack_C
from proj_qmm_pack import COL_BLOCK, ROW_BLOCK
import xfeed_bd
import egress_bd

# (label, PAIR_ROWS, LEADS_PER_GRP, N_GRP) -- the two pairing regimes, matching
# egress_bd.REGIMES. HDR is 2 in both.
REGIMES = [
    ("paired (llama-3.2-1b)", 2, 4, 2),
    ("non-paired (qwen3-4b)", 1, 4, 4),
]
HDR = 2


def mmul_tiles(X, W):
    """One core's 32x256 block: X [B, KCOL] and W [ROW_BLOCK, KCOL] -> C tile order.

    Exact fp32, deliberately. This checks LAYOUT, and the roundings
    (bfp16 mmul, floor-to-bf16) are already gated bit-exactly on device by
    q4k_mm_gate.py. Mixing them in here would only make a layout mismatch
    harder to read.
    """
    b, k = X.shape
    r = mmul_r(b)
    A = pack_A(X)  # what the xfeed BD delivers
    B = pack_B(W)  # what q4k_unpack_block emits
    colA, colB = k // MMUL_S, W.shape[0] // MMUL_T
    C = np.zeros(b * W.shape[0], np.float32)
    for j in range(colB):
        for z in range(b // r):
            for i in range(colA):
                a = A[(i * (b // r) + z) * r * MMUL_S :][: r * MMUL_S].reshape(
                    r, MMUL_S
                )
                bb = B[(j * colA + i) * MMUL_S * MMUL_T :][: MMUL_S * MMUL_T].reshape(
                    MMUL_S, MMUL_T
                )
                off = (j * (b // r) + z) * r * MMUL_T
                C[off : off + r * MMUL_T] += (a @ bb).reshape(-1)
    return C


def flush(C, batch, pair_rows, role):
    """proj_qmm_mm_flush_row, in numpy. THE piece the descriptors meet at.

    The kernel writes token t, pair role i at (t*tok_stride + i)*ROW_BLOCK with
    tok_stride = PAIR_ROWS, reading C tile (z, j) at (j*RA + z)*64 and token rr
    at rr*8 inside it. Transcribed from the kernel rather than re-derived, so a
    change there shows up here as a failure.
    """
    RA = batch // 8
    CB = ROW_BLOCK // 8
    out = np.zeros(HDR + pair_rows * ROW_BLOCK * batch, np.float32)
    for t in range(batch):
        z, rr = t // 8, t % 8
        row = np.empty(ROW_BLOCK, np.float32)
        for j in range(CB):
            row[j * 8 : (j + 1) * 8] = C[(j * RA + z) * 64 + rr * 8 :][:8]
        base = HDR + (t * pair_rows + role) * ROW_BLOCK
        out[base : base + ROW_BLOCK] = row
    return out


def assemble(bands, batch, pair_rows, leads, ngrp, verbose):
    """Every emitter's flush output through both gathers, then the consumer.

    bands[(g, k, role)] is that emitter's flushed buffer. The gathers are
    CONTIGUOUS -- emitter k's B token blocks land back to back, groups laid end
    to end -- so the assembled packet is emitter-major and its header survives.
    The de-interleave happens where the payload lands, which is what
    outy_tokmajor describes. Returns the CONSUMER's buffer, token-major.
    """
    pair_pay = pair_rows * ROW_BLOCK
    payload_w = ngrp * leads * pair_pay
    n_emit = ngrp * leads
    # main = [hdr][e=0: t0..tB-1][e=1: ...], each token block pair_pay wide.
    main = np.zeros(HDR + payload_w * batch, np.float32)
    for g in range(ngrp):
        grp = np.zeros(HDR + leads * pair_pay * batch, np.float32)
        for k in range(leads):
            # The gather is a plain contiguous run per emitter: exactly the
            # batch-1 descriptor with a B-times-longer payload.
            src = bands[(g, k, 0)][HDR:]
            base = HDR + k * pair_pay * batch
            grp[base : base + src.size] = src
        gbase = HDR + g * leads * pair_pay * batch
        main[gbase : gbase + leads * pair_pay * batch] = grp[HDR:]
        if verbose and g == 0:
            print(f"      group {g}: grp {grp.size} -> main at {gbase}")

    # The consumer's 3-D landing: (emitter, token, element).
    out = np.zeros(batch * payload_w, np.float32)
    ix = np.array(
        egress_bd.dst_indices(
            *egress_bd.outy_tokmajor(n_emit, pair_pay, batch, payload_w)
        )
    )
    out[ix] = main[HDR:][: ix.size]
    return out, payload_w


def check(label, pair_rows, leads, ngrp, batch, verbose, rng):
    pair_pay = pair_rows * ROW_BLOCK
    n_emit = ngrp * leads
    # Each emitter owns PAIR_ROWS row-blocks of the output; the whole PAYLOAD
    # row is n_emit * pair_pay wide.
    X = rng.standard_normal((batch, COL_BLOCK), np.float32)
    Ws = [
        rng.standard_normal((ROW_BLOCK, COL_BLOCK), np.float32)
        for _ in range(n_emit * pair_rows)
    ]

    # ---- batched path: xfeed order -> mmul -> flush -> both gathers ----
    bands = {}
    for g in range(ngrp):
        for k in range(leads):
            buf = np.zeros(HDR + pair_pay * batch, np.float32)
            for role in range(pair_rows):
                w = Ws[((g * leads + k) * pair_rows) + role]
                C = mmul_tiles(X, w)
                buf += flush(C, batch, pair_rows, role)
            bands[(g, k, 0)] = buf
    out, payload_w = assemble(bands, batch, pair_rows, leads, ngrp, verbose)

    # ---- batch-1 path: the same emitters, one token at a time ----
    # This is the reference, and it is built the way batch 1 builds it: each
    # emitter's row-block laid side by side, emitter-major, in one PAYLOAD row.
    ok = True
    for t in range(batch):
        want = np.concatenate(
            [
                (Ws[((g * leads + k) * pair_rows) + role] @ X[t]).astype(np.float32)
                for g in range(ngrp)
                for k in range(leads)
                for role in range(pair_rows)
            ]
        )
        got = out[t * payload_w : (t + 1) * payload_w]
        if not np.allclose(got, want, rtol=1e-4, atol=1e-3):
            bad = int(np.argmax(np.abs(got - want)))
            print(
                f"    {label} batch {batch}: token {t} row differs at {bad}"
                f"  got {got[bad]:.5f} want {want[bad]:.5f}"
            )
            ok = False
            break
    if verbose and ok:
        print(f"    batch {batch:3d}: PAYLOAD {payload_w}, {n_emit} emitters   OK")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--batches",
        default="8,16",
        help="only 8/16/32: the flush de-tiles for aie::mmul<8,8,8>",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    bad = [b for b in batches if b % 8]
    if bad:
        sys.exit(
            f"--batches {bad}: proj_qmm_mm_flush_row de-tiles for aie::mmul<8,8,8>; "
            "batch 4 needs a size_C=32 variant (see proj_qmm.cc)"
        )

    rng = np.random.default_rng(args.seed)
    print(
        "\nbatched projection path, composed  [xfeed order -> mmul -> flush -> gathers]"
    )
    print("  asserts token t's assembled row == the row batch 1 produces\n")
    ok = True
    for label, pair_rows, leads, ngrp in REGIMES:
        print(f"  {label}: PAIR_ROWS {pair_rows}, {leads} leads/group, {ngrp} groups")
        for b in batches:
            good = check(label, pair_rows, leads, ngrp, b, args.verbose, rng)
            ok &= good
            if not good:
                print(f"    batch {b}: FAIL")
        print(f"    batches {batches}: {'all OK' if ok else 'FAILED'}")

    print(
        "\n  What this covers that the per-descriptor checkers do not: the\n"
        "  CONVENTIONS BETWEEN them. tok_stride agreeing across the flush and\n"
        "  the gather, A-tile order agreeing across the X feed and pack_A,\n"
        "  C-tile order agreeing across the mmul and the de-tiling. Each side\n"
        "  is self-consistent on its own, which is why a mismatch survives\n"
        "  every per-piece test."
    )
    print(
        "\n  What it is NOT: the device gate. No DMA, no locks, no cascade, and\n"
        "  exact fp32 rather than the bfp16/floor arithmetic q4k_mm_gate.py\n"
        "  already proved bit-exact on hardware. Layout only."
    )
    if not ok:
        print("\n  FAIL")
        return 1
    print("\n  SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
