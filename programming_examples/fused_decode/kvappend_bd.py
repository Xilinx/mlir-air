#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The KV append's batched descriptor, derived and checked.

THE THIRD of the three permutations batching needs, after xfeed_bd.py (feed)
and egress_bd.py (drain). It was not on step 2's list at all -- it surfaced
from reading the shipping q4nx decode driver, which appends exactly one slot
per dispatch.

THE LAYOUT. The KV cache is REGION-MAJOR: per layer, NGRP K regions then NGRP V
regions, each ATTN_MAXL*REGION_W elements, with position p at p*REGION_W inside
its region. So one token's K is NGRP disjoint REGION_W runs, REGION_STRIDE
apart -- which is why the append is already 2-D today:

    offsets=[(L-1)*REGION_W]  sizes=[NGRP, REGION_W]  strides=[REGION_STRIDE, 1]

THE CHANGE. A block of B tokens appends B CONSECUTIVE positions. rope emits per
token, so the channel arrives token-major -- [t0: g0..gN-1][t1: g0..gN-1]... --
and walking it linearly walks the destination in (t, g, w):

    dim    extent      dst stride       what it is
    t      BATCH       REGION_W         next position, inside every region
    g      NGRP        REGION_STRIDE    next region
    w      REGION_W    1                this group's slice of this token

3-D, one more than today, with a dimension still spare.

THE HAZARD THIS FILE EXISTS FOR. `p*REGION_W` is only a valid position slot
while `p < ATTN_MAXL`. At batch 1 the cache can overrun by at most one position
and the driver's own window bookkeeping prevents it. At batch B it can overrun
by B, and the overrun does not fault: position ATTN_MAXL of region g is exactly
position 0 of region g+1, so a block that crosses the end SILENTLY CORRUPTS THE
NEXT GROUP'S CACHE -- and the next group is a real KV region being read by a
real attention CU. Wrong logits, no error. `check_bounds` is the guard, and the
builder should carry the same one.

WHAT IS CHECKED:
  1. exactly-once coverage of the B*NGRP*REGION_W cells the block writes;
  2. token t's K lands at position P+t of EVERY region -- which is what the
     readback will later stream, so it is the property that actually matters;
  3. nothing outside the intended positions is touched (the corruption case);
  4. batch 1 collapses to today's 2-D descriptor exactly.

    python3 kvappend_bd.py              # both models, several batches
    python3 kvappend_bd.py --overrun    # show the hazard the guard catches

Exit code is the gate: 0 all descriptors verified.
"""

import argparse
import itertools
import sys

import numpy as np

# (label, NGRP, REGION_W, ATTN_MAXL) read off fused_decode.py. REGION_STRIDE is
# derived rather than listed, because it IS ATTN_MAXL*REGION_W and writing it
# separately is how the two drift apart.
MODELS = [
    ("llama-3.2-1b", 2, 256, 2048),
    ("qwen3-4b", 2, 512, 2048),
]
BD_MAX_DIMS = 4


def kvappend_bd(batch, ngrp, region_w, region_stride, p):
    """(offsets, sizes, strides) for a B-token append starting at position p."""
    if batch == 1:
        # Today's descriptor. The builder writes its offsets list RANK-DEFICIENT
        # here -- one entry against two sizes -- and that is not sloppiness: AIR
        # LEFT-PADS a short offsets list with zeros
        # (air::canonicalizeWrapAndStrideList, mlir/lib/Util/Util.cpp), so a
        # single offset is right-aligned onto the stride-1 dimension and behaves
        # as a flat element offset. Written out in full here, because a reader
        # who assumes LEFT alignment gets p*REGION_W*REGION_STRIDE and a reader
        # who assumes a flat base for a MULTI-entry list gets something else
        # again.
        return [0, p * region_w], [ngrp, region_w], [region_stride, 1]
    # Base goes in the dimension whose stride is REGION_W, so offsets[0] = p
    # lands the window on position p. AIR offsets are memref.subview -- the
    # address is base + SUM(offsets[d]*strides[d]), so a flat p*REGION_W written
    # into offsets[0] here would be multiplied by REGION_W a second time.
    return [p, 0, 0], [batch, ngrp, region_w], [region_w, region_stride, 1]


def check_bounds(batch, attn_maxl, p):
    """Does this block fit inside the window? The guard the builder needs too.

    Not >= : position p..p+batch-1 must all be < ATTN_MAXL.
    """
    return p >= 0 and p + batch <= attn_maxl


def dst_indices(offsets, sizes, strides):
    base = sum(o * st for o, st in zip(offsets, strides))
    return [
        base + sum(i * st for i, st in zip(ix, strides))
        for ix in itertools.product(*(range(n) for n in sizes))
    ]


def check(label, ngrp, region_w, attn_maxl, batch, p, verbose):
    region_stride = attn_maxl * region_w
    offs, sizes, strides = kvappend_bd(batch, ngrp, region_w, region_stride, p)
    if len(sizes) > BD_MAX_DIMS:
        print(f"    {label} batch {batch}: {len(sizes)} dims > {BD_MAX_DIMS}")
        return False

    # One K half of one layer: NGRP regions laid end to end.
    cache = np.full(ngrp * region_stride, -1, np.int64)
    hits = np.zeros(ngrp * region_stride, np.int64)
    ix = np.array(dst_indices(offs, sizes, strides))
    # Channel order: token-major, group-minor -- what rope emits, one token at a
    # time, NGRP groups per token.
    src = np.array(
        [
            (t * 100 + g) * 10000 + w
            for t in range(batch)
            for g in range(ngrp)
            for w in range(region_w)
        ],
        np.int64,
    )
    if ix.max() >= cache.size:
        print(f"    {label} batch {batch} p={p}: writes past the cache end")
        return False
    cache[ix] = src
    hits[ix] += 1

    # (1) exactly once
    if int(hits.sum()) != batch * ngrp * region_w or hits.max() > 1:
        print(f"    {label} batch {batch}: coverage {int(hits.sum())} max {hits.max()}")
        return False

    # (2) token t at position p+t of EVERY region, and (3) nothing else touched
    for g in range(ngrp):
        for t in range(batch):
            slot = g * region_stride + (p + t) * region_w
            got = cache[slot : slot + region_w]
            want = np.array(
                [(t * 100 + g) * 10000 + w for w in range(region_w)], np.int64
            )
            if not np.array_equal(got, want):
                print(f"    {label} batch {batch}: group {g} token {t} misplaced")
                return False
        touched = np.nonzero(hits[g * region_stride : (g + 1) * region_stride])[0]
        lo, hi = p * region_w, (p + batch) * region_w
        if touched.min() < lo or touched.max() >= hi:
            print(f"    {label} batch {batch}: group {g} wrote outside [{lo},{hi})")
            return False

    if verbose:
        print(
            f"    batch {batch:3d} p={p:5d}: offsets={offs} sizes={sizes} "
            f"strides={strides}   {len(sizes)}D"
        )
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument(
        "--pos", type=int, default=None, help="start position (default: a few)"
    )
    ap.add_argument(
        "--overrun",
        action="store_true",
        help="demonstrate the end-of-window hazard the bounds guard catches",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    print("\nKV append, batched  [region-major cache, destination-side BD]")
    print("  channel is token-major; the cache is region-major, so the append")
    print("  transposes (token, group) -- 3-D where batch 1 is 2-D\n")
    ok = True
    for label, ngrp, region_w, attn_maxl in MODELS:
        print(
            f"  {label}: NGRP {ngrp}, REGION_W {region_w}, ATTN_MAXL {attn_maxl}"
            f"  ->  REGION_STRIDE {attn_maxl * region_w}"
        )
        for b in batches:
            positions = (
                [args.pos] if args.pos is not None else [0, 1, 17, attn_maxl - b]
            )
            for p in positions:
                if not check_bounds(b, attn_maxl, p):
                    print(f"    batch {b} p={p}: out of window (guard would refuse)")
                    ok = False
                    continue
                good = check(label, ngrp, region_w, attn_maxl, b, p, args.verbose)
                ok &= good
                if not good:
                    print(f"    batch {b} p={p}: FAIL")
        print(f"    batches {batches}: {'all OK' if ok else 'FAILED'}")

    if args.overrun:
        print("\n  the hazard, with the guard removed:")
        label, ngrp, region_w, attn_maxl = MODELS[0]
        b, p = 8, attn_maxl - 3  # 3 slots left, 8 tokens
        rs = attn_maxl * region_w
        offs, sizes, strides = kvappend_bd(b, ngrp, region_w, rs, p)
        ix = np.array(dst_indices(offs, sizes, strides))
        spill = [i for i in ix if (i % rs) < p * region_w]
        print(
            f"    {label} batch {b} at p={p} (only {attn_maxl - p} slots left):"
            f"\n    {len(spill)} of {len(ix)} elements land in the NEXT group's"
            f" region, at its\n    positions 0..{b - (attn_maxl - p) - 1} -- live KV"
            " for a real attention CU."
            "\n    No fault, no error: just wrong logits. check_bounds() refuses"
            f" it ({check_bounds(b, attn_maxl, p)})."
        )

    print(
        "\n  Checked: exactly-once coverage, token t at position p+t of EVERY\n"
        "  region (what the readback later streams), nothing written outside\n"
        "  the intended slots, and batch 1 collapsing to today's 2-D descriptor."
    )
    if not ok:
        print("\n  FAIL")
        return 1
    print("\n  SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
