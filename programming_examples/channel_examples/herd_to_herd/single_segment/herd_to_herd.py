# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Two herds in one segment, wired together by a channel, on air.api.

    air.channel @ChanIn      L3 -> L1, into the producer
    air.channel @Herd2Herd   L1 -> L1, producer herd to consumer herd
    air.channel @ChanOut     L1 -> L3, out of the consumer

``Herd2Herd`` is the point. Its put is in one herd body and its get is in
another, and the two herds share nothing else -- no operand, no buffer, no
enclosing scope beyond the segment they happen to sit in. That works because a
channel is a module-level symbol rather than a value: `air.herd` is
``IsolatedFromAbove``, so anything passed as data would have to be threaded
through as an operand, and a channel simply is not.

The producer squares its tile and the consumer adds one, so the two stages are
distinguishable in the result: an input of 2 leaves as (2*2)+1 = 5.

Unchanged from the raw-bindings version this replaces, except for two things:

* The L3-side put and get sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The two kernels are written as whole-tile expressions rather than, in the
  producer, a hand-declared `linalg_structured_op` standing in for a deprecated
  upstream `elemwise_binary`, and in the consumer a scalar loop nest over every
  (i, j). ``vector=0`` keeps them scalar as the predecessor was: the image is
  32 i32 wide and a <8 x i32> op is 256-bit, which does not legalize.
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
    herd2herd = air.channel("Herd2Herd")

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    chan_in.put(A)

                    with air.herd([range(1)], name="producer_herd", shape=(1,)) as p:

                        @p.body
                        def _(tx):
                            image_in = air.alloc(
                                IMAGE_SIZE, i32, scope=p.private(), vector=0
                            )
                            image_out = air.alloc(
                                IMAGE_SIZE, i32, scope=p.private(), vector=0
                            )

                            chan_in.get(image_in)
                            image_out[:] = image_in[:] * image_in[:]
                            herd2herd.put(image_out)

                    with air.herd([range(1)], name="consumer_herd", shape=(1,)) as c:

                        @c.body
                        def _(tx):
                            image_in = air.alloc(
                                IMAGE_SIZE, i32, scope=c.private(), vector=0
                            )
                            image_out = air.alloc(
                                IMAGE_SIZE, i32, scope=c.private(), vector=0
                            )

                            herd2herd.get(image_in)
                            image_out[:] = image_in[:] + 1
                            chan_out.put(image_out)

                    chan_out.get(B)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="herd_to_herd.py",
        description="Builds, runs, and tests the herd_to_herd channel example",
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

    input_a = np.full(IMAGE_SIZE, 0x2, dtype=INOUT_DATATYPE)
    output_b = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
