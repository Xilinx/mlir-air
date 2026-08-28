#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What is on each tile, against the three limits that bite at high batch.

None of them is visible in AIR, and mlir-aie reports each as a bare error naming
at most one buffer:

    512 KB of L2 per memtile        'aie.tile' op allocated buffers exceeded
                                    available memory
    24 BD ids per DMA channel,      'aie.dma_bd' op Allocator exhausted
    SHARED between that channel's   available BD IDs (maximum 24 available
    MM2S and S2MM halves            for channel N)
    64 KB of L1 per compute tile,   the same 'exceeded available memory' -- and
    stack and the per-herd RTP      the RTP word is placed AFTER every buffer,
    word included                   so a tile landing on exactly 65,536 is over
                                    by four bytes and nothing says so

Each of those cost a build cycle to diagnose from the error alone. This prints
the map instead. Everything it reports except the weight fan scales with
DECODE_BATCH, so the fastest way to see what a batch doubling will cost is to
run it at the batch that already works and double the numbers.

    python3 tile_budget.py                          # air_project/npu.air.mlir
    python3 tile_budget.py path/to/npu.air.mlir --stack 6080
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

L1 = 65536
L2 = 524288
BD_PER_CHANNEL = 24


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", nargs="?", default="air_project/npu.air.mlir")
    ap.add_argument(
        "--stack",
        type=int,
        default=0,
        help="DECODE_STACK, added to every COMPUTE tile's total: it is allocated "
        "there but is not an aie.buffer, so the IR does not show it",
    )
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        sys.exit(f"missing {p} -- build a template first, aircc writes it")
    lines = p.read_text(errors="replace").split("\n")

    def width(ty):
        return 4 if ty in ("f32", "i32") else 2

    cores = defaultdict(list)
    mems = defaultdict(list)
    for ln in lines:
        m = re.search(
            r"%(\w+) = aie\.buffer\(%(\w+)\).*?: memref<(\d+)x(\w+), (\d) :", ln
        )
        if not m:
            continue
        entry = (m.group(1), int(m.group(3)) * width(m.group(4)))
        (mems if m.group(5) == "1" else cores)[m.group(2)].append(entry)

    # BDs per (memtile, direction, channel). Track brace depth so the walk stops
    # at the end of the region instead of counting the next tile's BDs too.
    chans = {}
    tile = chan = None
    depth = 0
    for ln in lines:
        m = re.match(r"\s*%\w+ = aie\.memtile_dma\(%(\w+)\)", ln)
        if m:
            tile, depth, chan = m.group(1), 1, None
            continue
        if tile is None:
            continue
        depth += ln.count("{") - ln.count("}")
        if depth <= 0:
            tile = chan = None
            continue
        m = re.search(r"aie\.dma_start\((MM2S|S2MM), (\d+)", ln)
        if m:
            chan = (tile, m.group(1), int(m.group(2)))
            chans.setdefault(chan, 0)
        elif "aie.dma_bd(" in ln and chan:
            chans[chan] += 1

    if cores:
        print(f"\ncompute tiles  [{p}]   {L1} B of L1, stack {args.stack} + 4 B of RTP")
        for t in sorted(cores, key=lambda t: -sum(b for _, b in cores[t]))[:6]:
            tot = sum(b for _, b in cores[t]) + args.stack
            free = L1 - tot
            flag = (
                "  <== OVER"
                if free < 4
                else ("  <== only the RTP fits" if free < 8 else "")
            )
            named = ", ".join(
                f"{n}:{b}" for n, b in sorted(cores[t], key=lambda x: -x[1])[:4]
            )
            print(f"  {t:<12} {tot:>6} B, {free:>6} free{flag}   {named}")

    print(f"\nmemtiles  [{p}]   {L2} B of L2, {BD_PER_CHANNEL} BD ids per channel")
    for t in sorted(set(list(mems) + [c[0] for c in chans])):
        tot = sum(b for _, b in mems.get(t, []))
        flag = "  <== OVER L2" if tot > L2 else ""
        print(f"\n  {t}   {tot:>7} B ({tot * 100 // L2:>3}%){flag}")
        for n, b in sorted(mems.get(t, []), key=lambda x: -x[1])[:4]:
            print(f"      {n:<10} {b:>7}")
        # MM2S n and S2MM n draw from ONE pool of 24 -- report them together.
        per_index = defaultdict(int)
        for c in (k for k in chans if k[0] == t):
            per_index[c[2]] += chans[c]
        for i in sorted(per_index):
            n = per_index[i]
            over = "  <== OVER 24" if n > BD_PER_CHANNEL else ""
            print(f"      channel {i}   {n:>3} BDs{over}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
