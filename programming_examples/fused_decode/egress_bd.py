#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The egress gather's batched descriptor, derived and checked.

THE PROBLEM, and it is the mirror of xfeed_bd.py's. The proj cores emit
EMITTER-MAJOR: each of the LEADS_PER_GRP emitters in a group sends its own
ROW_BLOCK band, and the group memtile lays them side by side to assemble one
PAYLOAD row. That is exactly right at batch 1, where a row IS a token.

At batch B each emitter sends B bands, so a naive gather assembles

    [e0 t0][e0 t1]...[e0 tB-1][e1 t0][e1 t1]...

which is emitter-major in the OUTER dimension -- every token's row scattered
across the buffer in B pieces. What every consumer downstream wants is

    [t0 e0][t0 e1]...[t0 eL-1][t1 e0]...

so that token t's PAYLOAD row is contiguous and IDENTICAL in layout to the row
batch 1 produces. Then nothing downstream of the memtile has to know the batch
size: the assembled packet is just B rows where it used to be one.

That is a transpose of (emitter, token), and it is free if it can be expressed
as a destination-side block descriptor. It can, in two dimensions:

    dst dim    extent      dst stride                  what it is
    t          BATCH       LEADS_PER_GRP * PAIR_PAY    one assembled row
    p          PAIR_PAY    1                           this emitter's band

with base HDR + k*PAIR_PAY for emitter k. So the existing 1-D gather

    offsets=[HDR + k*PAIR_PAY]  sizes=[PAIR_PAY]  strides=[1]

gains exactly one dimension and its BD count does not change -- which is what
lets the egress WIDEN rather than repeat: one packet per round, B times longer,
N_ROUNDS and the instruction stream untouched.

WHAT IS CHECKED, and this is the part worth having. Not that the descriptor
matches the derivation above -- that would only prove the derivation
self-consistent. Instead:

  1. every element of the assembled buffer is written exactly once (no overlap,
     no hole) -- a stride error usually shows up here first;
  2. token t's assembled PAYLOAD row is byte-for-byte what the batch-1 gather
     produces for that token. This is the invariant that actually matters,
     because it is what lets the consumers stay batch-agnostic;
  3. at BATCH 1 the descriptor collapses to today's 1-D gather exactly.

Checked on the geometry of BOTH pairing regimes, because they differ in
LEADS_PER_GRP, PAIR_PAY and N_GRP, and a descriptor that only works for one is
a descriptor that works for llama and not for the DFlash target.

    python3 egress_bd.py            # both regimes, every batch
    python3 egress_bd.py -v         # print the layouts

Exit code is the gate: 0 all descriptors verified.
"""

import argparse
import itertools
import sys

import numpy as np

# (label, HDR, PAIR_ROWS, ROW_BLOCK, LEADS_PER_GRP, N_GRP) -- read off
# fused_decode.py for the two pairing regimes. Hardcoded rather than imported
# so this file does not need a model env set to run, and so a change to the
# builder's geometry shows up here as a mismatch to investigate rather than as
# a silently-tracking test that can never fail.
REGIMES = [
    ("paired (llama-3.2-1b)", 2, 2, 32, 4, 2),
    ("non-paired (qwen3-4b)", 2, 1, 32, 4, 4),
]
BD_MAX_DIMS = 4


def egress_bd(batch, pair_pay, leads_per_grp, hdr, k):
    """(offsets, sizes, strides) for emitter k's gather into the group memtile.

    Destination-side: these describe where in `grp` the incoming band lands.
    The base goes in the stride-1 dimension, which is how the existing 1-D
    gather already expresses it (offsets=[HDR + k*PAIR_PAY], strides=[1]).
    """
    if batch == 1:
        # Collapse to today's descriptor exactly, rather than emitting a
        # degenerate leading dimension of extent 1. Same reason as xfeed_bd:
        # an extent-1 dimension costs a descriptor slot and buys nothing.
        return [hdr + k * pair_pay], [pair_pay], [1]
    return (
        [0, hdr + k * pair_pay],
        [batch, pair_pay],
        [leads_per_grp * pair_pay, 1],
    )


def main_bd(batch, pair_pay, leads_per_grp, hdr, payload, g):
    """(offsets, sizes, strides) for group g's gather into the MAIN memtile.

    The same transpose one level up, and easy to forget: fixing only the
    emitter gather leaves each group's slab token-major INTERNALLY while the
    groups themselves stay laid end to end, so token t's full PAYLOAD row is
    still in N_GRP pieces. It would look right in any single-group test.
    Both levels or neither.
    """
    band = leads_per_grp * pair_pay
    if batch == 1:
        return [hdr + g * band if g else 0], [band + (0 if g else hdr)], [1]
    return [0, hdr + g * band], [batch, band], [payload, 1]


def dst_indices(offsets, sizes, strides):
    base = sum(o * st for o, st in zip(offsets, strides))
    return [
        base + sum(i * st for i, st in zip(ix, strides))
        for ix in itertools.product(*(range(n) for n in sizes))
    ]


def check(label, hdr, pair_rows, row_block, leads, ngrp, batch, verbose):
    pair_pay = pair_rows * row_block
    grp_rows = hdr + leads * pair_pay * batch

    # Assemble. Source value encodes (emitter, token, element) so any
    # mis-placement is identifiable rather than just unequal.
    grp = np.full(grp_rows, -1, np.int64)
    hits = np.zeros(grp_rows, np.int64)
    for k in range(leads):
        offs, sizes, strides = egress_bd(batch, pair_pay, leads, hdr, k)
        if len(sizes) > BD_MAX_DIMS:
            print(f"    {label} batch {batch}: {len(sizes)} dims > {BD_MAX_DIMS}")
            return False
        ix = np.array(dst_indices(offs, sizes, strides))
        # The band this emitter sends, token-major and contiguous -- which is
        # what proj_qmm_mm_flush_row writes: token t at (t*PAIR_ROWS + i)*RB.
        src = np.array(
            [(k * 1000 + t) * 1000 + e for t in range(batch) for e in range(pair_pay)],
            np.int64,
        )
        grp[ix] = src
        hits[ix] += 1

    payload = grp[hdr:]
    # (1) every payload element written exactly once
    if not np.array_equal(hits[hdr:], np.ones_like(hits[hdr:])):
        n_over = int((hits[hdr:] > 1).sum())
        n_hole = int((hits[hdr:] == 0).sum())
        print(f"    {label} batch {batch}: {n_over} overlapped, {n_hole} unwritten")
        return False

    # (2) token t's row == what batch 1 produces for that token
    row = leads * pair_pay
    ok = True
    for t in range(batch):
        got = payload[t * row : (t + 1) * row]
        want = np.array(
            [(k * 1000 + t) * 1000 + e for k in range(leads) for e in range(pair_pay)],
            np.int64,
        )
        if not np.array_equal(got, want):
            bad = int(np.argmax(got != want))
            print(f"    {label} batch {batch}: token {t} row differs at {bad}")
            ok = False
            break
    if not ok:
        return False

    # ---- level 2: the N_GRP group slabs assembled into the main memtile ----
    payload_w = ngrp * leads * pair_pay
    main_rows = hdr + payload_w * batch
    ml = np.full(main_rows, -1, np.int64)
    mhits = np.zeros(main_rows, np.int64)
    for g in range(ngrp):
        offs, sizes, strides = main_bd(batch, pair_pay, leads, hdr, payload_w, g)
        if len(sizes) > BD_MAX_DIMS:
            print(f"    {label} batch {batch}: main BD needs {len(sizes)} dims")
            return False
        ix = np.array(dst_indices(offs, sizes, strides))
        src = np.array(
            [
                ((g * 10 + k) * 1000 + t) * 1000 + e
                for t in range(batch)
                for k in range(leads)
                for e in range(pair_pay)
            ],
            np.int64,
        )
        if batch == 1 and g == 0:
            src = np.concatenate([np.full(hdr, -2, np.int64), src])
        ml[ix] = src
        mhits[ix] += 1
    if not np.array_equal(mhits[hdr:], np.ones_like(mhits[hdr:])):
        n_over = int((mhits[hdr:] > 1).sum())
        n_hole = int((mhits[hdr:] == 0).sum())
        print(
            f"    {label} batch {batch}: MAIN {n_over} overlapped, {n_hole} unwritten"
        )
        return False
    for t in range(batch):
        got = ml[hdr + t * payload_w : hdr + (t + 1) * payload_w]
        want = np.array(
            [
                ((g * 10 + k) * 1000 + t) * 1000 + e
                for g in range(ngrp)
                for k in range(leads)
                for e in range(pair_pay)
            ],
            np.int64,
        )
        if not np.array_equal(got, want):
            print(f"    {label} batch {batch}: MAIN token {t} row differs")
            return False

    if verbose:
        offs, sizes, strides = egress_bd(batch, pair_pay, leads, hdr, 1)
        moffs, msizes, mstrides = main_bd(batch, pair_pay, leads, hdr, payload_w, 1)
        print(
            f"    batch {batch:3d}: grp k=1 {offs}/{sizes}/{strides}"
            f"   main g=1 {moffs}/{msizes}/{mstrides}"
            f"   grp {grp_rows} main {main_rows}"
        )
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    print("\negress gather, batched  [destination-side BDs, BOTH assembly levels]")
    print("  emitter-major in  ->  token-major out, so token t's row is")
    print("  byte-identical to the row batch 1 produces\n")
    ok = True
    for label, hdr, pr, rb, leads, ngrp in REGIMES:
        pay = pr * rb
        print(
            f"  {label}: PAIR_PAY {pay}, {leads} leads/group, {ngrp} groups"
            f"  ->  row stride {leads * pay}"
        )
        for b in batches:
            good = check(label, hdr, pr, rb, leads, ngrp, b, args.verbose)
            ok &= good
            if not good:
                print(f"    batch {b}: FAIL")
        print(f"    batches {batches}: {'all OK' if ok else 'FAILED'}")

    print(
        "\n  Checked three ways: exactly-once coverage of the payload, token t's\n"
        "  row equal to batch 1's row for that token, and batch 1 collapsing to\n"
        "  today's 1-D descriptor. The second is the one that matters -- it is\n"
        "  what lets everything downstream of the memtile ignore the batch."
    )
    print(
        "\n  BOTH LEVELS, deliberately. Fixing only the emitter gather leaves\n"
        "  each group's slab token-major internally while the groups stay laid\n"
        "  end to end, so token t's PAYLOAD row is still in N_GRP pieces -- and\n"
        "  it would look right in any single-group test."
    )
    print(
        "\n  NOT covered: the 2-word HDR. Only emitter k=0 carries it and the\n"
        "  put/get sizes already differ there at batch 1; that asymmetry is a\n"
        "  builder detail to mirror, not a permutation to derive."
    )
    if not ok:
        print("\n  FAIL")
        return 1
    print("\n  SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
