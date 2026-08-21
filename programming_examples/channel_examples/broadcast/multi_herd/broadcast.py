# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One image broadcast to three separate herds, on air.api.

    air.channel @ChanIn [1, 1] {broadcast_shape = [3, 1]}
    air.channel @ChanOutB / @ChanOutC / @ChanOutD

The sibling under ``broadcast/single_herd`` fans out to the three *cores* of one
herd; this one fans out to three *herds*, each a 1x1 grid of its own. One put,
three gets, each naming its own destination with ``indices=[n, 0]`` -- and each
herd drains through a channel of its own, which is why there are three output
channels rather than one array.

Nothing about the fan-out changes between the two: ``broadcast_shape`` is a
declaration-level attribute describing the grid of destinations, and it does not
care whether that grid is cores or herds.

Each herd adds its own index plus one, so the three outputs differ, which is
what makes the broadcast observable rather than merely asserted.

Unchanged from the raw-bindings version this replaces, except for two things:

* The L3-side put and gets sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The increment is written ``out[:] = in[:] + n`` rather than as a scalar loop
  nest over every (i, j). ``vector=0`` keeps it scalar, as the predecessor was:
  the image is 32 i32 wide and a <8 x i32> add is 256-bit, which does not
  legalize.
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

HERDS = 3
OUTPUT_HERD_NAMES = ["ChanOutB", "ChanOutC", "ChanOutD"]


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    outs = [air.tensor(IMAGE_SIZE, i32) for _ in range(HERDS)]

    chan_in = air.channel("ChanIn", size=[1, 1], broadcast_shape=[HERDS, 1])
    chan_out = [air.channel(name) for name in OUTPUT_HERD_NAMES]

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    chan_in.put(A)

                    # A closure per herd rather than `def _(tx, n=n)`: the
                    # body's positional arity is how air.api knows the grid's
                    # rank, so a defaulted capture would read as a second
                    # coordinate and raise.
                    def one_herd(n):
                        with air.herd(
                            [range(1)], name=f"broadcastherd{n}", shape=(1,)
                        ) as h:

                            @h.body
                            def _(tx):
                                image_in = air.alloc(
                                    IMAGE_SIZE, i32, scope=h.private(), vector=0
                                )
                                image_out = air.alloc(
                                    IMAGE_SIZE, i32, scope=h.private(), vector=0
                                )

                                chan_in.get(image_in, indices=[n, 0])
                                image_out[:] = image_in[:] + (n + 1)
                                chan_out[n].put(image_out)

                    for n in range(HERDS):
                        one_herd(n)

                    for n, out in enumerate(outs):
                        chan_out[n].get(out)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="broadcast.py",
        description="Builds, runs, and tests the channel broadcast multi herd example",
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

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.arange(1, np.prod(IMAGE_SIZE) + 1, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )
    output_c = np.arange(2, np.prod(IMAGE_SIZE) + 2, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )
    output_d = np.arange(3, np.prod(IMAGE_SIZE) + 3, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a],
            expected_outputs=[output_b, output_c, output_d],
        )
    )
