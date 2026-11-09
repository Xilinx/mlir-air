# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A core starts one L3 transfer from a loop.
#
# As in 58_device_issued_l3_put, the router core puts W[idx[i], :], but from a
# loop: the host arms the shim BD once, and the core patches and starts it on
# every iteration. A start must not land while the last one is queued, so the
# router waits for the forwarding core to report each row before it starts the
# next. The router's two receive channels carry the indices and the replies to
# its control packets, so the report comes through L1 the two cores share.

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

                    with air.herd(
                        [range(1)], name="router", shape=(1,), at=(0, 2)
                    ) as r:

                        @r.body
                        def _(tx):
                            # the indices, then room for the report
                            topk = air.alloc([32], i32, scope=r.private())
                            idx_in.get(topk[0:16])
                            for i in air.sequential(0, K):
                                wexp.put(W[topk[i], :])
                                topk[16:32] = ack[:]

                    with air.herd([range(1)], name="fwd", shape=(1,), at=(0, 3)) as f:

                        @f.body
                        def _(tx):
                            w = air.alloc([E], i32, scope=f.private())
                            o = air.alloc([E], i32, scope=f.private())
                            a = air.alloc([16], i32, scope=f.private())
                            for _i in air.sequential(0, K):
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
    w = (np.arange(NE)[:, None] * 1000 + np.arange(E)[None, :]).astype(np.int32)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="device_issue",
    )
    exit(runner.run_test(mlir_module, inputs=[w, idx], expected_outputs=[w[rows]]))
