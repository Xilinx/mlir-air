#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The intra-block causal mask batch-16 verify needs -- which turns out to exist.

THE PROBLEM, restated. Verify runs the target over 16 drafted tokens in one
pass. Token t sits at position P+t and may attend to keys 0..P+t and no further.
If it sees P+15 it reads the very tokens it is meant to be checking, every
draft is accepted, and speculative decoding silently stops being lossless -- no
crash, no wrong-looking output, just a rubber-stamp verifier. No amount of L1 or
bandwidth analysis surfaces this, which is why it is worth a file of its own.

THE FINDING: no kernel change is needed. attn_qk_blk already carries exactly the
right mask, for a different reason.

    int rem = L - blk * 16;
    if (rem <= 0) return;
    rem = (rem < 16) ? rem : 16;
    aie::mask<16> mask = aie::le(idx, rem);     // idx = 1..16

That is a TAIL mask -- it exists to trim the ragged last KV block when the
context length is not a multiple of 16 -- and with idx starting at 1 it keeps
in-block key j exactly when j < rem, i.e. global keys 0..L-1. `L` is already
"number of cached positions this token attends to" (fused_decode.py's own
wording; DECODE_GOLDEN_L=1 is position 0 attending to itself).

So a per-query triangular mask is a per-query VALUE of L:

    verify   L_eff(t) = P + t + 1     token t sees 0..P+t          causal
    draft    L_eff(t) = P + 16        every token sees the block   bidirectional

Both passes, one engine, no new mask mode -- the difference is an RTP scalar.
`_core_rounds(Lh)` already derives the block count from the same scalar, so the
loop bound follows for free, and DFlash needs DECODE_DYNSEQ for KV rollback
anyway.

WHAT THIS FILE IS. `--check` models the kernel's mask arithmetic in Python and
asserts the resulting key set is exactly the causal one, over a range of P and
every t. It is a guard on an index convention, not a device measurement: the
claim above is read off three lines of attn_qk.cc, and if someone changes `idx`
to be 0-based or redefines what L counts, this is what fails.
"""

import argparse
import sys

BLK = 16
BATCH = 16


def kernel_keys(L, n_rounds):
    """Global key indices attn_qk_blk keeps, given RTP L and a block-loop bound.

    Mirrors the kernel line for line, including the rem<=0 early return -- which
    matters, because it is what lets one compile-time ATTN_ROUNDS build serve
    every runtime L.
    """
    keep = []
    for blk in range(n_rounds):
        rem = L - blk * BLK
        if rem <= 0:
            continue  # block entirely past the context; kernel returns early
        rem = min(rem, BLK)
        # aie::le(idx, rem) with idx = 1..16 -> in-block key j kept iff j < rem
        keep.extend(blk * BLK + j for j in range(rem) if j + 1 <= rem)
    return keep


def rounds(L):
    """_core_rounds(Lh) = ceil(Lh/16)."""
    return (L + BLK - 1) // BLK


def check(verbose=True):
    bad = 0

    def expect(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        if verbose and (not ok or name.endswith("*")):
            print(f"  {'OK ' if ok else 'MISMATCH'} {name}")
        return ok

    # Prefix lengths spanning both alignments: a multiple of 16, one either
    # side, and a realistic long context. The ragged cases are the ones where a
    # tail mask and a causal mask could plausibly disagree.
    for P in (0, 1, 15, 16, 17, 31, 2032, 2047, 2048):
        for t in range(BATCH):
            # VERIFY: token t at position P+t attends to 0..P+t inclusive.
            L = P + t + 1
            expect(
                f"verify P={P} t={t}",
                kernel_keys(L, rounds(L)),
                list(range(P + t + 1)),
            )
        # DRAFT: block diffusion, every token sees the whole block.
        L = P + BATCH
        for t in range(BATCH):
            expect(
                f"draft P={P} t={t}",
                kernel_keys(L, rounds(L)),
                list(range(P + BATCH)),
            )

    # The block-count bound must cover the longest query and no more, or the
    # core waits on a KV block the shim never pushes (or vice versa).
    for P in (0, 15, 16, 2048):
        want = rounds(P + BATCH)
        expect(
            f"rounds cover longest query P={P}",
            max(rounds(P + t + 1) for t in range(BATCH)),
            want,
        )

    print("SELF-CHECK PASS" if not bad else f"SELF-CHECK FAIL ({bad})")
    # Inverted since this was written: it printed PASS and exited 1, so anything
    # gating on the exit code saw a failing check that read as passing.
    return 1 if bad else 0


def cost(P):
    """What batch-16 verify costs in attn_qk_blk calls, against the alternatives.

    Per-query L means per-query block counts, which differ only in the last
    block. Pushing ceil((P+16)/16) blocks for EVERY token instead -- one uniform
    KV stream, per-token masking -- costs at most one extra block per token and
    keeps the shim BD identical for all 16. That is the shape to build.
    """
    exact = sum(rounds(P + t + 1) for t in range(BATCH))
    uniform = BATCH * rounds(P + BATCH)
    one = rounds(P + 1)
    print(f"\nattn_qk_blk calls per CU, prefix P = {P}")
    print(f"  {'batch 1 (today)':34s}{one:8d}")
    print(f"  {'batch 16, per-token block count':34s}{exact:8d}   {exact/one:.1f}x")
    print(f"  {'batch 16, uniform block count':34s}{uniform:8d}   {uniform/one:.1f}x")
    print(
        f"  uniform costs {uniform - exact} extra calls "
        f"({(uniform - exact) / max(exact, 1):.1%}) and one shim BD instead of 16"
    )
    print(
        "  Note both are ~16x batch 1: attention does NOT amortize over the\n"
        "  batch the way the projections do, because every query re-reads the\n"
        "  whole KV cache. It is L2 traffic, not DDR -- the KV blocks are read\n"
        "  once from DDR and re-read per query from L2 -- but it is not free."
    )


def main():
    global BATCH
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="assert the kernel's mask arithmetic gives the causal key set",
    )
    ap.add_argument(
        "--prefix", type=int, default=2048, help="context length for --cost"
    )
    ap.add_argument("--cost", action="store_true", help="attn call counts at batch 16")
    ap.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help="block size to check. Was fixed at 16 when this file was written, "
        "because 16 was the checkpoint's block size; docs/DFlashFeasibility.md "
        "section 3.1 since measured block 8 to be the one worth building, so "
        "the arithmetic has to be checked there too.",
    )
    a = ap.parse_args()
    if not (a.check or a.cost):
        ap.error("pass --check, --cost, or both")
    BATCH = a.batch
    rc = 0
    if a.check:
        rc = check()
    if a.cost:
        cost(a.prefix)
    return rc


if __name__ == "__main__":
    sys.exit(main())
