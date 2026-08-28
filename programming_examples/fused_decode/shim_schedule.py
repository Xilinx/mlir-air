#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""In what ORDER does the host issue and await its shim transfers?

`shim_volume.py` answers how much each channel moves; this answers when. The
distinction matters because most of what looks like a device hang in this design
is the HOST blocked on a `dma_await_task` it cannot satisfy, and the await
positions are not written anywhere in the Python -- they are synthesized during
lowering (AIRRtToNpuPass: `generateAwaitsFromWaitAllOps` for the dependency
joins, `synthesizeDoubleBufferedAwaits` for the depth-2 pacing and the
per-segment tail drain, plus the coalesced-feed phase barrier).

The rule the output is read against: an await is only safe if everything the
awaited transfer's consumer needs has already been STARTED. So

    start  %45  air_rmsX[...]        <- residual2's band feed
    await  %45
    start  %46  air_inW0c0[...]      <- the down projection's weights

is a deadlock -- the core takes those bands only after the down projection has
produced its output, and the weights that let it are issued one line too late.
Reading the emitted order is how that was found; it is not visible in the AIR.

    python3 shim_schedule.py                          # air_project/npu.air.mlir
    python3 shim_schedule.py path/to/npu.air.mlir --channels rmsX layerOut
"""

import argparse
import re
import sys
from pathlib import Path

CFG = re.compile(r"\s*%(\d+) = aiex\.dma_configure_task_for @(\w+)")
BD = re.compile(r"aie\.dma_bd\((%\w+)[^)]*offset = (\d+) len = (\d+)")
REPEAT = re.compile(r"repeat_count = (\d+)")
OP = re.compile(r"\s*aiex\.dma_(start|await|free)_task\(%(\d+)\)")


def parse(path):
    """(ops, labels): the start/await/free sequence, and each task's shape."""
    labels, cur, ops = {}, None, []
    for ln in path.read_text(errors="replace").split("\n"):
        m = CFG.match(ln)
        if m:
            cur, labels[cur := m.group(1)] = m.group(1), m.group(2)
            continue
        if cur:
            b = BD.search(ln)
            if b:
                labels[cur] += f"[{b.group(1)}+{b.group(2)}, {b.group(3)}"
                continue
            if "}" in ln and "repeat_count" in ln:
                r = REPEAT.search(ln)
                labels[cur] += f" x{int(r.group(1)) + 1}]"
                cur = None
                continue
            if re.match(r"\s*\} \{", ln):
                labels[cur] += "]"
                cur = None
                continue
        m = OP.match(ln)
        if m:
            ops.append((m.group(1), m.group(2)))
    return ops, labels


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", nargs="?", default="air_project/npu.air.mlir")
    ap.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="substrings to keep (default: every channel). The weight feed is "
        "eight identical channels, so `--channels rmsX layerOut inW0c0` is "
        "usually what you want.",
    )
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        sys.exit(f"missing {p} -- build a template first, aircc writes it")
    ops, labels = parse(p)
    if not ops:
        sys.exit("no dma tasks found -- is this an npu.air.mlir?")

    print(f"\nshim schedule  [{p}]")
    for op, t in ops:
        lab = labels.get(t, "?")
        chan = lab.split("[")[0]
        if args.channels and not any(c in chan for c in args.channels):
            continue
        print(f"  {op:<6} %{t:<4} {lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
