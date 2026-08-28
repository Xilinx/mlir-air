#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""How many elements does the SHIM actually move on each channel, after lowering?

`check_channel_balance.py` compares the two sides of a channel in the AIR, which
is the right place to catch a scaling mistake. It cannot catch a descriptor that
is correct AIR and does not survive to the hardware -- and that is a real class:
a shim BD holds three dimensions plus a repeat count, and a fourth is dropped in
AIRRtToNpuPass with no diagnostic. A 5-D `air.channel.put` of 13x5 bands came
out the other side as 5 bands, and the only symptom was a wave-0 timeout.

So this reads air_project/npu.air.mlir -- the last IR before the instruction
stream -- and totals what each shim task really transfers:

    elements = len * (repeat_count + 1)

Per channel, with the task count, so it can be divided by whatever the consumer
is expected to take. It is a ONE-SIDED tool by design: it reports what the host
pushes, and the number it is compared against comes from the core-side loop
counts. Both sides in one place is what check_channel_balance.py does.

    python3 shim_volume.py                       # air_project/npu.air.mlir
    python3 shim_volume.py path/to/npu.air.mlir --per-wave 36
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

TASK = re.compile(r"aiex\.dma_configure_task_for\s+@(\w+)")
BD = re.compile(r"aie\.dma_bd\([^)]*?len\s*=\s*(\d+)")
REPEAT = re.compile(r"repeat_count\s*=\s*(\d+)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", nargs="?", default="air_project/npu.air.mlir")
    ap.add_argument(
        "--per-wave",
        type=int,
        default=0,
        help="divide by this many waves (UNI_DEC) to get a per-layer figure",
    )
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        sys.exit(f"missing {p} -- build a template first, aircc writes it")
    lines = p.read_text(errors="replace").split("\n")

    elems = defaultdict(int)
    tasks = defaultdict(int)
    bds = defaultdict(int)
    chan = None
    # A task runs from its `dma_configure_task_for` to its closing `}` line,
    # which is where the repeat_count attribute sits. Track the open task and
    # attribute every dma_bd inside it to that channel.
    pending = []
    for ln in lines:
        m = TASK.search(ln)
        if m:
            chan, pending = m.group(1), []
            continue
        if chan is None:
            continue
        b = BD.search(ln)
        if b:
            pending.append(int(b.group(1)))
            continue
        if "aie.end" in ln:
            continue
        # The line carrying the task's trailing attribute dict closes it.
        if "}" in ln and pending is not None and pending:
            r = REPEAT.search(ln)
            rep = int(r.group(1)) + 1 if r else 1
            for n in pending:
                elems[chan] += n * rep
                bds[chan] += 1
            tasks[chan] += 1
            chan, pending = None, []

    if not elems:
        sys.exit("no shim tasks found -- is this an npu.air.mlir?")
    w = max(len(c) for c in elems)
    print(f"\nshim volume  [{p}]")
    print(f"  {'channel':<{w}} {'tasks':>6} {'BDs':>5} {'elements':>12}", end="")
    print(f" {'per wave':>10}" if args.per_wave else "")
    for c in sorted(elems, key=lambda c: -elems[c]):
        print(f"  {c:<{w}} {tasks[c]:>6} {bds[c]:>5} {elems[c]:>12}", end="")
        print(f" {elems[c]/args.per_wave:>10.1f}" if args.per_wave else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
