# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A core starts one L3 transfer from a loop whose trip count it reads at run
# time, under a condition.
#
# The router core scans idx[0:idx[15]] and puts W[idx[i], :] for each entry
# that is not negative, waiting for the forwarding core's report before the
# next start. The host can count neither the iterations nor the starts, so it
# waits on none of them. The forwarding core scans the same entries, lands
# each row in its slot and sends the slots out once, so the run ends the same
# way however many rows were sent. One loaded program runs several index
# sets in turn, including ones that start nothing.

import argparse

import numpy as np

from air import api as air
from air.api.types import i32
from air.backend.xrt import XRTBackend

NE, E, K = 8, 256, 4


def build_module():
    W = air.tensor([NE, E], i32)
    IDX = air.tensor([16], i32)
    Y = air.tensor([NE, E], i32)
    idx_in = air.channel("IdxIn")
    wexp = air.channel("Wexp")
    yout = air.channel("YOut")

    with air.launch(name="device_issue") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    idx_in.put(IDX)
                    ack = air.alloc([16], i32, scope=seg.shared())
                    rows = air.alloc([16], i32, scope=seg.shared())

                    with air.herd(
                        [range(1)], name="router", shape=(1,), at=(0, 2)
                    ) as r:

                        @r.body
                        def _(tx):
                            # the indices, then room for the report
                            topk = air.alloc([32], i32, scope=r.private())
                            idx_in.get(topk[0:16])
                            rows[:] = topk[0:16]
                            for i in air.sequential(0, topk[15].as_index()):
                                with air.ops.branch(topk[i] >= 0):
                                    wexp.put(W[topk[i], :])
                                    topk[16:32] = ack[:]

                    with air.herd([range(1)], name="fwd", shape=(1,), at=(0, 3)) as f:

                        @f.body
                        def _(tx):
                            slots = air.alloc([NE, E], i32, scope=f.private())
                            o = air.alloc([NE, E], i32, scope=f.private())
                            a = air.alloc([16], i32, scope=f.private())
                            n = air.alloc([16], i32, scope=f.private())
                            air.ops.fill(slots, 0)
                            n[:] = rows[:]
                            for i in air.sequential(0, n[15].as_index()):
                                with air.ops.branch(n[i] >= 0):
                                    wexp.get(slots[n[i], :])
                                    ack[:] = a[:]
                            o[:] = slots[:]
                            yout.put(o)

                    yout.get(Y)

    return launch.build(target="npu2")


# (entries, how many of them the router scans)
CASES = [
    ([3, 0, 7, 5], 4),
    ([3, 0, 7, 5], 0),
    ([-1, -1, -1, -1], 4),
    ([2, -1, 6, -1], 4),
    ([1, 4, -1, 6], 3),
    ([3, 0, 7, 5], 4),
]


def expected(w, entries, n):
    y = np.zeros((NE, E), np.int32)
    for r in entries[:n]:
        if r >= 0:
            y[r] = w[r]
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    w = (np.arange(NE)[:, None] * 1000 + np.arange(E)[None, :] + 1).astype(np.int32)
    backend = XRTBackend(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="device_issue",
    )
    run = backend.load(backend.compile(mlir_module))
    failed = 0
    for entries, n in CASES:
        idx = np.zeros(16, np.int32)
        idx[:K] = entries
        idx[15] = n
        y = np.asarray(run(w, idx, np.zeros((NE, E), np.int32))[-1])
        ok = np.array_equal(y.reshape(NE, E), expected(w, entries, n))
        print(f"{entries} scan {n}: {'ok' if ok else 'MISMATCH'}")
        failed += not ok
    backend.unload()
    print("PASS!" if not failed else "failed.")
    exit(1 if failed else 0)
