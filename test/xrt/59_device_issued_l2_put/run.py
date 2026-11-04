# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A core picks which rows of an L2 buffer the memtile sends it.
#
# The host fills an L2 buffer with every row of W. The fwd core reads K row
# indices and gets W[topk[i], :] from L2: the offset of the L2 slice is a value
# the core loaded from L1. The compiler leaves the memtile's buffer descriptor
# out of its static program; the core patches its offset and starts it through
# control packets, then forwards each row it receives to the host.

import argparse

import numpy as np

from air import api as air
from air.api.types import i32
from air.backend.xrt_runner import XRTRunner

NE, E, K = 8, 256, 4


def build_module():
    W = air.tensor([NE, E], i32)
    IDX = air.tensor([16], i32)
    Y = air.tensor([K, E], i32)
    w_in = air.channel("WIn")
    idx_in = air.channel("IdxIn")
    row = air.channel("Row")
    yout = air.channel("YOut")

    with air.launch(name="device_issue_l2") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    wl2 = air.alloc([NE, E], i32, scope=seg.private())
                    w_in.put(W)
                    w_in.get(wl2)
                    idx_in.put(IDX)

                    with air.herd([range(1)], name="fwd", shape=(1,)) as f:

                        @f.body
                        def _(tx):
                            topk = air.alloc([16], i32, scope=f.private())
                            w = air.alloc([E], i32, scope=f.private())
                            o = air.alloc([E], i32, scope=f.private())
                            idx_in.get(topk)
                            for i in air.sequential(0, K):
                                row.put(wl2[topk[i], :])
                                row.get(w)
                                o[:] = w[:]
                                yout.put(o)

                    yout.get(Y)

    return launch.build(target="npu2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--idx", type=str, default="3,0,7,5")
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

    rows = [int(v) for v in args.idx.split(",")]
    idx = np.zeros(16, np.int32)
    idx[:K] = rows
    w = (np.arange(NE)[:, None] * 1000 + np.arange(E)[None, :]).astype(np.int32)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="device_issue_l2",
    )
    exit(runner.run_test(mlir_module, inputs=[w, idx], expected_outputs=[w[rows]]))
