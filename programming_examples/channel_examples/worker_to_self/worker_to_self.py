# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A worker that sends data to itself through a channel, on air.api.

    air.channel @ChanIn     L3 -> L2
    air.channel @ToSelf     L2 -> L1, both ends inside the same herd body
    air.channel @ChanOut    L1 -> L3

``ToSelf`` is the point of the example: its put and its get are both in the herd
body, one core talking to itself. That is legal precisely because a channel is
not a value -- it is a module-level symbol, so the two ends need no common
scope, not even a common operand. The staged L2 buffer is what the core puts,
and air.api passes it into the herd for free.

Unchanged from the raw-bindings version this replaces, except for two things:

* The L3-side put and get sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The copy is written ``out[:] = in[:]`` rather than as a scalar loop nest over
  every (i, j). Same transfer, one line. ``vector=0`` keeps it scalar: the image
  is 32 i32 wide and a <8 x i32> add is 256-bit, which does not legalize.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    B = air.tensor(IMAGE_SIZE, i32)

    chan_in = air.channel("ChanIn")
    chan_out = air.channel("ChanOut")
    to_self = air.channel("ToSelf")

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2_in = air.alloc(IMAGE_SIZE, i32, scope=seg.private())

                    chan_in.put(A)
                    chan_in.get(l2_in)

                    with air.herd([range(1)], name="copyherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            l1_in = air.alloc(
                                IMAGE_SIZE, i32, scope=h.private(), vector=0
                            )
                            l1_out = air.alloc(
                                IMAGE_SIZE, i32, scope=h.private(), vector=0
                            )

                            # Both ends of @ToSelf, in one body: the core sends
                            # the staged L2 tile and receives it into L1.
                            to_self.put(l2_in)
                            to_self.get(l1_in)

                            l1_out[:] = l1_in[:]

                            chan_out.put(l1_out)

                    chan_out.get(B)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="worker_to_self.py",
        description="Builds, runs, and tests the channel worker_to_self example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    launch = build_module()
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_matrix = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )
    output_matrix = input_matrix.copy()

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module, inputs=[input_matrix], expected_outputs=[output_matrix]
        )
    )
