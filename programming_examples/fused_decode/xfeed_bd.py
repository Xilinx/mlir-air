#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The X feed's tile-blocking descriptor, derived and checked against pack_A.

THE PROBLEM. The batched projection kernel does not want the activations the
way the engine currently delivers them. q4k_mm_block feeds aie::mmul, and
aie::mmul wants its A operand in TILE order -- tile (z, i) at a fixed stride,
tokens interleaved inside each tile -- where the memtile holds plain
[BATCH][COL_BLOCK] row-major chunks. Something has to permute.

Doing it on the core would cost more than the batching saves: it is a full
BATCH*COL_BLOCK shuffle per j-step, per core, 16 cores. Doing it in the DMA
costs nothing, IF the permutation is expressible as a strided block descriptor
-- at most 4 dimensions, one constant stride each. This file answers whether it
is, and if so what the descriptor's numbers are.

THE DERIVATION. pack_A does, for X of shape [b, k]:

    X.reshape(b//r, r, k//s, s).transpose(2, 0, 1, 3).reshape(-1)
             z    rr   i       ss          i  z  rr  ss

so walking the DESTINATION linearly walks the source in the nested order
(i, z, rr, ss) with constant strides:

    dim     extent     source stride     what it is
    i       k // s     s                 contraction tile
    z       b // r     r * tok           token block (mmul rowA)
    rr      r          tok               token within the block
    ss      s          1                 contraction within the tile

`tok` is the source's per-token stride, which is NOT k: the X memtile holds
X_CHUNKS chunks per token, and one put covers one chunk, so consecutive tokens
sit X_CHUNKS*COL_BLOCK apart. Getting that wrong is the kind of error that
produces a plausible wrong answer rather than a crash -- see the 9216-vs-5120
block-stride fault in docs/DFlashFeasibility.md.

At BATCH 8 the z dimension is degenerate (r == 8 == BATCH, so b//r == 1) and
the descriptor drops to 3D, which leaves a dimension spare. At BATCH 16 it is
exactly 4D, with none spare.

WHAT IS CHECKED. The descriptor is expanded into the index list it would
generate and compared elementwise against pack_A's own output on the same
data -- not against a restatement of the derivation above, which would only
prove the derivation self-consistent. pack_A is the function whose output is
bit-exact on device (q4k_mm_gate.py), so agreeing with it is the claim worth
making.

    python3 xfeed_bd.py               # emit + check every supported batch
    python3 xfeed_bd.py --batch 8     # just one

Exit code is the gate: 0 all descriptors verified.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q4k_mm_gate import MMUL_S, mmul_r, pack_A
from proj_qmm_pack import COL_BLOCK

# Hardware block descriptors carry at most 4 dimensions. A permutation that
# needs 5 is not a descriptor problem, it is a layout problem, and the answer
# would be to change the layout rather than to split the transfer.
BD_MAX_DIMS = 4


def xfeed_bd(batch, kcol=COL_BLOCK, chunks_per_token=2):
    """(sizes, strides) for one chunk's put, outermost dimension first.

    chunks_per_token is what makes the token stride bigger than kcol: the X
    memtile stages X_CHUNKS*COL_BLOCK per token and each put covers one chunk.
    """
    r, s = mmul_r(batch), MMUL_S
    tok = chunks_per_token * kcol
    dims = [
        (kcol // s, s),  # i   contraction tile
        (batch // r, r * tok),  # z   token block (mmul rowA)
        (r, tok),  # rr  token within the block
        (s, 1),  # ss  contraction within the tile
    ]
    # Drop degenerate dimensions: extent 1 contributes nothing and spending a
    # descriptor dimension on it is what pushes BATCH 8 from 3D to 4D for no
    # reason.
    dims = [(n, st) for n, st in dims if n > 1]
    return [n for n, _ in dims], [st for _, st in dims]


def chunk_offsets(sizes, strides, chunk, kcol):
    """Per-dimension offsets that place the window on chunk `chunk`.

    NOT a flat byte offset. AIR's [offsets][sizes][strides] follow the
    memref.subview convention (docs/AIRComputeModel.md): the element address is
    base + SUM(offsets[d] * strides[d]). Writing the chunk offset into
    offsets[0] and expecting it to add flatly would silently multiply it by
    strides[0] -- 8x here -- and read the wrong activations while transferring
    exactly the right NUMBER of them. Another plausible-wrong-answer failure.

    The chunk stride is kcol, so it has to be expressed against some dimension
    whose stride divides it. Dimension 0 has stride s and extent kcol//s, so
    offsets[0] = chunk * (kcol // s) lands exactly one window on.

    A RANK-DEFICIENT offsets list is a different thing again, and the builder
    uses one (the KV append: one offset against two sizes). AIR LEFT-PADS a
    short list with zeros -- air::canonicalizeWrapAndStrideList in
    mlir/lib/Util/Util.cpp -- so the entries are RIGHT-aligned and a single
    offset lands on the stride-1 dimension, behaving as a flat element offset.
    Assume left alignment instead and a flat offset silently picks up the
    OUTERMOST stride. Full-rank lists here, so this file is unaffected; noted
    because the two conventions coexist in the same builder.
    """
    off = [0] * len(sizes)
    off[0] = chunk * (kcol // strides[0])
    got = sum(o * st for o, st in zip(off, strides))
    assert got == chunk * kcol, f"offsets {off} give {got}, want {chunk * kcol}"
    return off


def expand(sizes, strides, offsets=None):
    """The source index list a BD with these sizes/strides/offsets generates."""
    base = 0 if offsets is None else sum(o * st for o, st in zip(offsets, strides))
    return [
        base + sum(i * st for i, st in zip(ix, strides))
        for ix in itertools.product(*(range(n) for n in sizes))
    ]


def check(batch, kcol, chunks_per_token, verbose):
    sizes, strides = xfeed_bd(batch, kcol, chunks_per_token)
    ok_dims = len(sizes) <= BD_MAX_DIMS

    # Source: the memtile's view -- [batch][chunks_per_token][kcol], token
    # major. Fill with distinct values so any mis-indexing shows up.
    src = np.arange(batch * chunks_per_token * kcol, dtype=np.int64).reshape(
        batch, chunks_per_token, kcol
    )
    ok_all = True
    for chunk in range(chunks_per_token):
        offs = chunk_offsets(sizes, strides, chunk, kcol)
        got = src.reshape(-1)[np.array(expand(sizes, strides, offs))]
        # What pack_A makes of THIS chunk's [batch, kcol] slice. pack_A is the
        # reference because it is the ordering the device gate proved bit-exact.
        want = pack_A(src[:, chunk, :].astype(np.float32)).astype(np.int64)
        ok = np.array_equal(got, want)
        ok_all &= ok
        if verbose or not ok:
            print(f"    chunk {chunk}: {'match' if ok else 'MISMATCH'}")
            if not ok:
                bad = int(np.argmax(got != want))
                print(f"      first at {bad}: bd {got[bad]} vs pack_A {want[bad]}")

    print(
        f"  batch {batch:3d}  r={mmul_r(batch)}  {len(sizes)}D"
        f"  sizes={sizes} strides={strides}"
        f"   {'OK' if ok_all and ok_dims else 'FAIL'}"
    )
    o0 = kcol // strides[0]
    print(
        f"            offsets=[chunk*{o0}" + ", 0" * (len(sizes) - 1) + "]"
        f"   one put per chunk, {chunks_per_token} chunks/token"
    )
    if not ok_dims:
        print(f"      needs {len(sizes)} dimensions, hardware BDs carry {BD_MAX_DIMS}")
    return ok_all and ok_dims


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=None, help="just this batch")
    ap.add_argument("--kcol", type=int, default=COL_BLOCK)
    ap.add_argument(
        "--chunks",
        type=int,
        default=2,
        help="X chunks staged per token in the memtile (xmt_l2 / COL_BLOCK)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # 4 and 8 use aie::mmul<r,8,8> with rowA 1; 16 and 32 use the 2x2 kernel.
    # Both shapes have to produce a legal descriptor or the block size is
    # constrained by the DMA rather than by the arithmetic.
    batches = [args.batch] if args.batch else [4, 8, 16, 32]
    print(
        f"\nX feed tile-blocking descriptor  [kcol {args.kcol}, {args.chunks} chunks/token]"
    )
    print("  source: memtile [BATCH][chunks][kcol], token-major")
    print("  dest:   aie::mmul A tile order, written linearly\n")
    ok = all(check(b, args.kcol, args.chunks, args.verbose) for b in batches)
    print(
        "\n  Checked against pack_A itself, not against a restatement of the\n"
        "  derivation -- pack_A is the ordering q4k_mm_gate.py proved bit-exact\n"
        "  on device, so agreeing with it is the claim worth making."
    )
    if not ok:
        print("\n  FAIL")
        return 1
    print("\n  SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
