#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Is every compute-tile buffer aligned for a 512-bit access? Asked of the build.

WHY THIS EXISTS. mlir-aie aligns compute-tile buffers to the tile's LOAD/STORE
BUS width and packs the rest end to end (AIEAssignBuffers.cpp: `aligned`
defaults true, the width comes from getComputeTileLoadStoreBusWidth). On AIE2p
that width is 256 bits -- THIRTY-TWO bytes. It is not 64.

aie::mmul<8,8,8> wants 64. Its C tile is 64 floats = 256 bytes and Peano moves
it in 512-bit chunks, so a 32-byte-aligned accumulator is a 32-byte-MISALIGNED
512-bit access, and AIE2 does not fault on that -- it masks the low address bits
and the whole accumulator lands 32 bytes low. Every value comes out shifted by 8
floats and the last 8 are never written.

So the alignment the kernel needs is the CALLER's job, and the way a caller
loses it is by declaring a buffer whose SIZE is not a multiple of 64 bytes:
everything the allocator packs after it inherits the odd 32.

WHAT IT COST. `ypair_mm_l1` was 16 + PAIR_ROWS*ROW_BLOCK*BATCH = 528 bf16 =
1056 bytes = 16.5 x 64. The proj LEAD tile is the only tile that hosts those
shared egress buffers, so only lead tiles misplaced the buffer packed next --
their SECOND accumulator, at ...820 instead of ...800. Round _e=1 uses that
accumulator and round _e=0 uses the other, and the QKV phase's 6 rounds put K on
round 4 and V on round 5. Result: K correct, V wrong, and wrong on exactly the
lead half of every emitter block -- which reads as "half of every projection's
output rows are computed against the wrong token" and sent the search into the
X feed, the descriptors and the transpose, all of which were fine.

    python3 l1_align.py                    # after a build, on air_project/
    python3 l1_align.py --dir some/other/air_project

Exit code is the gate: 0 when every compute-tile buffer ADDRESS is 64-byte
aligned. Odd SIZES are listed but do not fail -- a size is only a hazard, and
several in this design (the attention softmax state, 32 bytes each) have never
had anything packed after them. Failing on those would fail the shipping batch-1
build for a risk it does not run.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 512-bit vector move. Not the bus width the allocator uses (256 bits on AIE2p)
# -- the width the mmul's C tile is actually stored in.
VEC_BYTES = 64

_BUF_DECL = re.compile(
    r"%(\w+)\s*=\s*aie\.buffer\(%tile_(\d+)_(\d+)\)"
    r".*?:\s*memref<(\d+)x(\w+),\s*2\s*:\s*i32>"
)
_LD_ENTRY = re.compile(r"\.\s*=\s*(0x[0-9A-Fa-f]+);\s*\n\s*(\w+)\s*=\s*\.;")
_ELT_BYTES = {"bf16": 2, "f32": 4, "i32": 4, "i8": 1, "i16": 2, "f16": 2}


def parse_buffers(mlir):
    """name -> (tile, n_elems, elt, bytes) for every COMPUTE-tile buffer."""
    out = {}
    for name, col, row, n, elt in _BUF_DECL.findall(mlir):
        if elt not in _ELT_BYTES:
            continue
        out[name] = ((int(col), int(row)), int(n), elt, int(n) * _ELT_BYTES[elt])
    return out


def parse_addresses(scripts):
    """name -> address, from the emitted linker scripts.

    A script carries its core's own buffers AND its neighbours', at different
    bases -- but the bases differ by whole 0x10000 tile strides, so every view
    of a buffer agrees modulo 64 and taking the first is enough.
    """
    addr = {}
    for p in scripts:
        for a, name in _LD_ENTRY.findall(p.read_text()):
            addr.setdefault(name, int(a, 16))
    return addr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", default=str(HERE / "air_project"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    mlir = d / "aie.air.mlir"
    scripts = sorted(d.glob("ldScripts_*.ld.script"))
    if not mlir.exists() or not scripts:
        sys.exit(
            f"{d} has no aie.air.mlir + ldScripts_*.ld.script.\n"
            "Build a template first -- the addresses this checks are assigned by "
            "aiecc, not derivable from the builder."
        )

    bufs = parse_buffers(mlir.read_text())
    addrs = parse_addresses(scripts)

    print(f"\ncompute-tile L1 alignment  [{len(bufs)} buffers, {VEC_BYTES}-byte rule]")
    print(f"  {d}\n")

    bad_size, bad_addr = [], []
    for name, (tile, n, elt, nbytes) in sorted(bufs.items()):
        a = addrs.get(name)
        if nbytes % VEC_BYTES:
            bad_size.append((name, tile, n, elt, nbytes))
        if a is not None and a % VEC_BYTES:
            bad_addr.append((name, tile, n, elt, nbytes, a))
        if args.verbose:
            print(
                f"  {name:14s} tile {tile[0]},{tile[1]}  {n:6d}x{elt:4s}"
                f"  {nbytes:6d}B  @ {'0x%05X' % a if a is not None else '?':>8s}"
            )

    # The ADDRESS is the gate. An odd SIZE is only the usual cause, and it is a
    # hazard rather than a defect: it misaligns the successor if the allocator
    # puts one there, and several odd-sized buffers in this design (the
    # attention softmax state, 32 bytes each) have never had one. Failing on
    # size would fail the shipping batch-1 build for a risk it does not run.
    if bad_addr:
        print(f"  ADDRESS not {VEC_BYTES}-byte aligned -- a 512-bit access on this")
        print("  buffer lands 32 bytes low and the tail is never written:")
        by_tile = defaultdict(list)
        for name, tile, n, elt, nbytes, a in bad_addr:
            by_tile[tile].append((name, n, elt, a))
        for tile in sorted(by_tile):
            for name, n, elt, a in by_tile[tile]:
                print(
                    f"    {name:14s} tile {tile[0]},{tile[1]}  {n}x{elt}"
                    f"  @ 0x{a:05X}  (0x{a % VEC_BYTES:02X} past a {VEC_BYTES}B line)"
                )
        print()
    if bad_size:
        print(
            f"  {'cause' if bad_addr else 'note'}: SIZE not a multiple of "
            f"{VEC_BYTES} bytes, so whatever is packed after inherits the odd 32:"
        )
        for name, tile, n, elt, nbytes in bad_size:
            pad = -n % (VEC_BYTES // _ELT_BYTES[elt])
            print(
                f"    {name:14s} tile {tile[0]},{tile[1]}  {n}x{elt} = {nbytes}B"
                f"   pad by {pad} {elt} -> {n + pad}"
            )
        print()

    if bad_addr:
        print(
            "  A misaligned f32 accumulator is the failure mode worth naming: it\n"
            "  does not fault, it shifts. See the header, and batch_row_probe.py\n"
            "  for the on-device symptom.\n  FAIL"
        )
        return 1
    print(f"  every compute-tile buffer is {VEC_BYTES}-byte aligned -- GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
