#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The batched egress transpose, derived and checked -- at the CONSUMER.

WHAT THIS FILE USED TO SAY, and why it was wrong. It derived a token-major
landing for the two egress GATHERS: emitter -> group memtile and group -> main
memtile. Both descriptors were checked elementwise here and both were correct
about where the payload goes. The design still deadlocked on device, and the
reason was the one thing this file did not model: the 2-word routing HEADER.

A packet carries its header ONCE, at the front. A token-major landing wants the
header at destination 0 and token t's block at HDR + t*stride -- and a block
descriptor walks its SOURCE linearly, so no single one can do both. Splitting it
into two gets works arithmetically and breaks the hardware: the emitted AIE
dialect showed the memtile's BD chain alternating header/body on ONE buffer
while every other channel alternated ping/pong. The second get had eaten the
ring.

SO THE TRANSPOSE MOVED. Both gathers now use the batch-1 descriptor with a
B-times-longer payload -- contiguous, one BD, header untouched -- and the
assembled packet stays EMITTER-MAJOR all the way to the id-demux. Each consumer
de-interleaves as it lands:

    dim       extent          dst stride    what it is
    emitter   N_PAIRS*rounds  PAIR_PAY      this emitter's slice of a row
    token     BATCH           row_stride    next token's row
    element   PAIR_PAY        1             the slice itself

THREE DIMENSIONS, NOT FOUR, and that is not a stylistic choice: the rms core and
the GLU core are COMPUTE tiles, whose BDs have three. Several rounds fold into
the emitter dimension instead of adding a fourth, which works because round r,
emitter e lands at r*PAYLOAD + e*PAIR_PAY and PAYLOAD is exactly
N_PAIRS*PAIR_PAY. The GLU core needs that fold -- its [up|gate] slice is two
rounds on llama-3.2-1b.

WHAT IS CHECKED:
  1. exactly-once coverage of every element the consumer lands;
  2. token t's row equals what batch 1 would have produced for token t, which
     is the property the whole batch rests on;
  3. the round fold: two rounds landed with one descriptor go where two
     descriptors would have put them;
  4. batch 1 collapses to a plain contiguous read;
  5. the AIE2p limits for a compute tile -- 3 dims, 8-bit wrap.

    python3 egress_bd.py            # both models, several batches
    python3 egress_bd.py -v         # print the descriptors

Exit code is the gate: 0 all descriptors verified.
"""

import argparse
import itertools
import sys

import numpy as np

# (label, N_PAIRS, PAIR_PAY, rounds_per_get) read off fused_decode.py. PAYLOAD is
# derived rather than listed, because it IS N_PAIRS*PAIR_PAY and writing it
# separately is how the two drift apart.
MODELS = [
    ("llama-3.2-1b", 8, 64, 2),  # GLU slice = [up|gate] = 2 rounds
    ("qwen3-4b", 8, 64, 1),
]
# A compute tile's block descriptor. The memtile's is wider (4 dims, 10-bit
# wrap) but the tightest consumer here is a compute tile, so this is the bound
# that matters. From batch_wire.py, which reads mlir-aie's AIE2pTargetModel.
CORE_DIMS = 3
CORE_WRAP = 1 << 8


def outy_tokmajor(n_pairs, pair_pay, batch, row_stride, base=0, rounds=1):
    """The consumer-side de-interleave. Mirrors fused_decode.outy_tokmajor."""
    return (
        [0, 0, base],
        [n_pairs * rounds, batch, pair_pay],
        [pair_pay, row_stride, 1],
    )


def dst_indices(offsets, sizes, strides):
    base = sum(o * st for o, st in zip(offsets, strides))
    return [
        base + sum(i * st for i, st in zip(ix, strides))
        for ix in itertools.product(*(range(n) for n in sizes))
    ]


def check(label, n_pairs, pair_pay, batch, rounds, verbose):
    payload = n_pairs * pair_pay
    row = payload * rounds
    offs, sizes, strides = outy_tokmajor(n_pairs, pair_pay, batch, row, rounds=rounds)
    if len(sizes) > CORE_DIMS:
        print(f"    {label} batch {batch}: {len(sizes)} dims > {CORE_DIMS}")
        return False
    over = [n for n in sizes if n > CORE_WRAP]
    if over:
        print(f"    {label} batch {batch}: wrap {over} > {CORE_WRAP}")
        return False

    # The stream, in the order the emitter-major packet delivers it: for each
    # round, each emitter's B token blocks back to back.
    src = np.array(
        [
            (r * 1000 + e) * 100000 + t * 1000 + j
            for r in range(rounds)
            for e in range(n_pairs)
            for t in range(batch)
            for j in range(pair_pay)
        ],
        np.int64,
    )
    dst = np.full(batch * row, -1, np.int64)
    hits = np.zeros(batch * row, np.int64)
    ix = np.array(dst_indices(offs, sizes, strides))
    if ix.size != src.size:
        print(
            f"    {label} batch {batch}: descriptor moves {ix.size}, "
            f"stream has {src.size}"
        )
        return False
    dst[ix] = src
    hits[ix] += 1

    # (1) exactly once
    if int(hits.sum()) != src.size or hits.max() > 1:
        print(f"    {label} batch {batch}: coverage {int(hits.sum())} max {hits.max()}")
        return False

    # (2) and (3): token t's row is what batch 1 would produce for token t --
    # round r's payload at r*payload, emitter e's slice at e*pair_pay inside it.
    for t in range(batch):
        want = np.array(
            [
                (r * 1000 + e) * 100000 + t * 1000 + j
                for r in range(rounds)
                for e in range(n_pairs)
                for j in range(pair_pay)
            ],
            np.int64,
        )
        got = dst[t * row : (t + 1) * row]
        if not np.array_equal(got, want):
            bad = int(np.argmax(got != want))
            print(
                f"    {label} batch {batch}: token {t} row differs at {bad}"
                f"  got {got[bad]} want {want[bad]}"
            )
            return False

    if verbose:
        print(
            f"    batch {batch:3d} rounds {rounds}: offsets={offs} sizes={sizes} "
            f"strides={strides}   {len(sizes)}D"
        )
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batches", default="1,8,16")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    print(
        "\negress de-interleave, at the consumer  [emitter-major in, token-major out]"
    )
    print("  the gathers stay contiguous so the packet header survives; this is")
    print("  the descriptor that pays for that, on the receiving end\n")
    ok = True
    for label, n_pairs, pair_pay, rounds in MODELS:
        print(
            f"  {label}: N_PAIRS {n_pairs}, PAIR_PAY {pair_pay}, "
            f"{rounds} round(s)/get  ->  PAYLOAD {n_pairs * pair_pay}"
        )
        for b in batches:
            good = check(label, n_pairs, pair_pay, b, rounds, args.verbose)
            ok &= good
            if not good:
                print(f"    batch {b}: FAIL")
        print(f"    batches {batches}: {'all OK' if ok else 'FAILED'}")

    print(
        "\n  Checked: exactly-once coverage, token t's row identical to the row\n"
        "  batch 1 produces, the multi-round fold into the emitter dimension,\n"
        "  batch 1 collapsing to a contiguous read, and the 3-dim / 8-bit-wrap\n"
        "  limits of the COMPUTE tiles that have to run this descriptor."
    )
    if not ok:
        print("\n  FAIL")
        return 1
    print("\n  SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
