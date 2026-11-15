# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A core starts one L3 transfer from a loop whose trip count it reads at run
# time.
#
# As in 64_device_issued_l3_put_in_loop, the router core puts W[idx[i], :]
# and waits for the forwarding core's report before the next start, but the
# number of rows is idx[15], read on the core. The host cannot count those
# starts, so it waits on none of them: the row the forwarding core sends out
# for each is what the run waits for.

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
                                wexp.put(W[topk[i], :])
                                topk[16:32] = ack[:]

                    with air.herd([range(1)], name="fwd", shape=(1,), at=(0, 3)) as f:

                        @f.body
                        def _(tx):
                            w = air.alloc([E], i32, scope=f.private())
                            o = air.alloc([E], i32, scope=f.private())
                            a = air.alloc([16], i32, scope=f.private())
                            n = air.alloc([16], i32, scope=f.private())
                            n[:] = rows[:]
                            for _i in air.sequential(0, n[15].as_index()):
                                wexp.get(w)
                                o[:] = w[:]
                                yout.put(o)
                                ack[:] = a[:]

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
    idx[15] = K
    w = (np.arange(NE)[:, None] * 1000 + np.arange(E)[None, :]).astype(np.int32)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="device_issue",
    )
    exit(runner.run_test(mlir_module, inputs=[w, idx], expected_outputs=[w[rows]]))
