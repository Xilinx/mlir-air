# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One image broadcast to every core of a herd, on air.api.

    air.channel @ChanIn  [1, 1] {broadcast_shape = [1, 3]}
    air.channel @ChanOut [1, 3]

``ChanIn`` is declared once and read three times: ``broadcast_shape`` says the
single source fans out to a 1x3 grid of destinations, and each core names its
own with ``indices=[tx, ty]``. ``ChanOut`` is an ordinary channel *array* of the
same shape -- three independent channels, one per core -- which is why the three
gets on the L3 side each carry a constant index.

The schedule is unchanged from the raw-bindings version this replaces. The only
structural difference is that the L3 put and gets sit inside the segment rather
than beside it: reaching L3 needs a shim DMA allocation, and outside a segment
there is none to link to.

Each core adds its own column index plus one, so the three outputs differ, which
is what makes the broadcast observable rather than merely asserted.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

IMAGE_WIDTH = 8
IMAGE_HEIGHT = 6
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32

CORES = 3


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    outs = [air.tensor(IMAGE_SIZE, i32) for _ in range(CORES)]

    chan_in = air.channel("ChanIn", size=[1, 1], broadcast_shape=[1, CORES])
    chan_out = air.channel("ChanOut", size=[1, CORES])

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    chan_in.put(A)

                    with air.herd(
                        [range(1), range(CORES)], name="broadcastherd", shape=(1, CORES)
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            image_in = air.alloc(IMAGE_SIZE, i32, scope=h.private())
                            # vector=0 keeps the add scalar, as the predecessor
                            # wrote it. The image is 8 i32 wide, and a
                            # <8 x i32> add is 256-bit, which does not legalize:
                            # "unable to legalize instruction: <8 x s32> G_ADD".
                            # Nothing here is wide enough to vectorise usefully
                            # anyway.
                            image_out = air.alloc(
                                IMAGE_SIZE, i32, scope=h.private(), vector=0
                            )

                            chan_in.get(image_in, indices=[tx, ty])
                            # ty is the core's own column, broadcast into the
                            # expression as an index_cast -- so each core writes
                            # a different image and the fan-out is observable.
                            image_out[:] = image_in[:] + ty + 1
                            chan_out.put(image_out, indices=[tx, ty])

                    for i, out in enumerate(outs):
                        chan_out.get(out, indices=[0, i])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
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
    expected = [
        np.arange(n, np.prod(IMAGE_SIZE) + n, dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
        for n in range(1, CORES + 1)
    ]

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
            expected_outputs=expected,
        )
    )
