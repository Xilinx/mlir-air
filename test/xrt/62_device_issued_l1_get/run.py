# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A core picks where in its own L1 each incoming block lands.
#
# The host sends K rows. The fwd core reads K slot indices and receives row i
# into slots[topk[i], :]: the offset of the L1 destination is a value the core
# loaded. The compiler leaves the core's own receive descriptor out of its
# static program; the core patches its address and starts it through control
# packets. The core then sends all its slots to the host.

import argparse

import numpy as np

from air import api as air
from air.api.types import i32
from air.backend.xrt_runner import XRTRunner

NE, E, K = 8, 256, 4  # NE slots, K rows


def build_module():
    W = air.tensor([K, E], i32)
    IDX = air.tensor([16], i32)
    Y = air.tensor([NE, E], i32)
    w_in = air.channel("WIn")
    idx_in = air.channel("IdxIn")
    yout = air.channel("YOut")

    with air.launch(name="device_issue_l1") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    idx_in.put(IDX)
                    for i in range(K):
                        w_in.put(W[i : i + 1, :])

                    with air.herd([range(1)], name="fwd", shape=(1,)) as f:

                        @f.body
                        def _(tx):
                            topk = air.alloc([16], i32, scope=f.private())
                            slots = air.alloc([NE, E], i32, scope=f.private())
                            o = air.alloc([NE, E], i32, scope=f.private())
                            air.ops.fill(slots, 0)
                            idx_in.get(topk)
                            for i in air.sequential(0, K):
                                w_in.get(slots[topk[i], :])
                            o[:] = slots[:]
                            yout.put(o)

                    yout.get(Y)

    return launch.build(target="npu2")


def expected_slots(w, rows):
    y = np.zeros((NE, E), np.int32)
    for i, s in enumerate(rows):
        y[s] = w[i]
    return y


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
    assert len(set(rows)) == K, "each row needs a slot of its own"
    idx = np.zeros(16, np.int32)
    idx[:K] = rows
    w = (np.arange(K)[:, None] * 1000 + np.arange(E)[None, :] + 1).astype(np.int32)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="device_issue_l1",
    )
    exit(
        runner.run_test(
            mlir_module, inputs=[w, idx], expected_outputs=[expected_slots(w, rows)]
        )
    )
