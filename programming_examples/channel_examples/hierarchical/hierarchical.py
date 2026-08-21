# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A four-channel hierarchy, one hop per level of memory, on air.api.

    air.channel @ChanInL2    L3 -> L2
    air.channel @ChanInL1    L2 -> L1
    air.channel @ChanOutL1   L1 -> L2
    air.channel @ChanOutL2   L2 -> L3

Every hop is a channel, so there is not one ``air.dma_memcpy_nd`` in the
example: it is the same journey ``ops.load``/``ops.store`` would make, written
as four independent producer/consumer pairs instead of four copies. Each level
stages into its own buffer and forwards, and the herd names none of the
channels as an operand -- a channel is a module-level symbol and resolves by
name from any depth.

Unchanged from the raw-bindings version this replaces, except for two things:

* The L3-side put and get sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The increment is written ``out[:] = in[:] + 1`` rather than as a scalar loop
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


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    B = air.tensor(IMAGE_SIZE, i32)

    in_l2 = air.channel("ChanInL2")
    out_l2 = air.channel("ChanOutL2")
    in_l1 = air.channel("ChanInL1")
    out_l1 = air.channel("ChanOutL1")

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    image_in_l2 = air.alloc(IMAGE_SIZE, i32, scope=seg.private())
                    image_out_l2 = air.alloc(IMAGE_SIZE, i32, scope=seg.private())

                    in_l2.put(A)
                    in_l2.get(image_in_l2)
                    in_l1.put(image_in_l2)

                    with air.herd([range(1)], name="addherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            image_in = air.alloc(
                                IMAGE_SIZE, i32, scope=h.private(), vector=0
                            )
                            image_out = air.alloc(
                                IMAGE_SIZE, i32, scope=h.private(), vector=0
                            )

                            in_l1.get(image_in)
                            image_out[:] = image_in[:] + 1
                            out_l1.put(image_out)

                    out_l1.get(image_out_l2)
                    out_l2.put(image_out_l2)
                    out_l2.get(B)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="hierarchical.py",
        description="Builds, runs, and tests the channel hierarchical example",
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
    output_matrix = np.arange(1, np.prod(IMAGE_SIZE) + 1, dtype=INOUT_DATATYPE).reshape(
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
            mlir_module, inputs=[input_matrix], expected_outputs=[output_matrix]
        )
    )
