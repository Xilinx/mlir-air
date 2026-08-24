#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Did the batch move any tile's DMA channels? The other half of the deadlock.

check_channel_balance.py asks whether both sides of a channel move the same
number of elements. That is necessary and it is not sufficient: the batched
build passed it and still hung. What it cannot see is where AIR PUT each flow.

This reads the emitted AIE dialect (`air_project/aie.air.mlir`, which every
build leaves behind) for two builds and reports every tile whose DMA channel
allocation changed. Batch 1 is the reference because it is the design that runs.

WHAT IT CAUGHT, on its first use. The batched rms core had

    batch 1:  MM2S0, MM2S1, S2MM0, S2MM1
    batch 8:  MM2S0, MM2S1, S2MM0, S2MM0      <-- two chains on one channel

A DMA channel has ONE BD chain. The second aie.dma_start(S2MM, 0) also stopped
the first chain cycling, so layer 1 never received its X and the dispatch timed
out with no message. The trigger was buffer aliasing that the batched rms body
does on purpose -- its staging buffer is both the outY destination and the
@xnorm source -- which sends the allocator's packet-flow reuse to the wrong
port. Fixed with an air.tile_dma_channel pin.

REPEATED CHANNELS ARE THE HARD FAILURE. A tile that merely moved a flow from
S2MM1 to S2MM0 is reported but does not fail: that happens legitimately when a
tile has one flow in a direction.

    ./build_template.sh 1 128 && cp air_project/aie.air.mlir aie_b1.mlir
    ./build_template.sh 8 128 && cp air_project/aie.air.mlir aie_b8.mlir
    python3 check_dma_alloc.py aie_b1.mlir aie_b8.mlir

Exit code is the gate: 0 no tile has two chains on one channel.
"""

import argparse
import re
import sys
from collections import OrderedDict

TILE_RE = re.compile(r"aie\.(?:mem|memtile_dma)\((%tile_\d+_\d+|%mem_tile_\d+_\d+)\)")
START_RE = re.compile(r"aie\.dma_start\((MM2S|S2MM),\s*(\d+)")
END_RE = re.compile(r"^    \}")


def allocs(path):
    """{tile: [(dir, channel), ...]} in emission order."""
    out, cur = OrderedDict(), None
    for ln in open(path):
        m = TILE_RE.search(ln)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
            continue
        if cur is None:
            continue
        m = START_RE.search(ln)
        if m:
            out[cur].append((m.group(1), int(m.group(2))))
        elif END_RE.match(ln):
            cur = None
    return out


def dupes(chans):
    seen, bad = set(), []
    for c in chans:
        if c in seen:
            bad.append(c)
        seen.add(c)
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ref", help="reference AIE dump (the build that runs)")
    ap.add_argument("new", help="AIE dump under test")
    ap.add_argument("-v", "--verbose", action="store_true", help="every tile")
    args = ap.parse_args()

    a, b = allocs(args.ref), allocs(args.new)
    print(f"\nDMA channel allocation  [{args.new} vs {args.ref}]")
    print(f"  {'tile':16s}{'reference':30s}{'under test'}")
    bad = []
    for t in sorted(set(a) | set(b), key=lambda s: (len(s), s)):
        ca, cb = a.get(t, []), b.get(t, [])
        fa = ",".join(f"{d}{c}" for d, c in ca)
        fb = ",".join(f"{d}{c}" for d, c in cb)
        rep = dupes(cb)
        if rep:
            bad.append((t, rep))
        if rep or ca != cb or args.verbose:
            note = (
                "   TWO CHAINS ON " + ",".join(f"{d}{c}" for d, c in rep)
                if rep
                else ("   moved" if ca != cb else "")
            )
            print(f"  {t:16s}{fa:30s}{fb}{note}")

    if bad:
        print(
            "\n  A DMA channel has ONE BD chain. Where AIR emitted two, the\n"
            "  second one wins and the first stops cycling -- the design runs\n"
            "  once and then waits forever. Pin the flow with\n"
            "  air.tile_dma_channel (compute tiles) or\n"
            "  air.memtile_dma_channel_min (memtiles)."
        )
        return 1
    print("\n  no tile has two chains on one channel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
