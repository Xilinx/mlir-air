# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A channel *array* -- one channel per worker -- on air.api.

    air.channel @ChanIn  [2, 3]
    air.channel @ChanOut [2, 3]

``size=`` makes a channel an array rather than a single connection, and
``indices=`` picks one of its members. There is no fan-out here and no sharing:
the 2x3 grid of channels is matched one-for-one with the 2x3 herd, so each core
has a private pair and the index is just its own tile coordinate.

The L3 side puts one tile per channel, subscripted out of the image with the
ordinary slice syntax -- ``A[h * TILE_H : ..., w * TILE_W : ...]`` -- and the
offsets, sizes and strides that produces land on the channel op unchanged. The
loops around it are plain Python: they run at trace time and emit one put per
channel, which is what the predecessor does too.

Unchanged from the raw-bindings version except that the L3 puts and gets sit
inside the segment, which is where reaching L3 needs them to be.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_HEIGHT, TILE_WIDTH]

assert IMAGE_HEIGHT % TILE_HEIGHT == 0
assert IMAGE_WIDTH % TILE_WIDTH == 0

INOUT_DATATYPE = np.int32

ROWS = IMAGE_HEIGHT // TILE_HEIGHT
COLS = IMAGE_WIDTH // TILE_WIDTH


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    B = air.tensor(IMAGE_SIZE, i32)

    # One input/output channel per worker.
    chan_in = air.channel("ChanIn", size=[ROWS, COLS])
    chan_out = air.channel("ChanOut", size=[ROWS, COLS])

    def tile(t, h, w):
        return t[
            h * TILE_HEIGHT : (h + 1) * TILE_HEIGHT,
            w * TILE_WIDTH : (w + 1) * TILE_WIDTH,
        ]

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    for h in range(ROWS):
                        for w in range(COLS):
                            chan_in.put(tile(A, h, w), indices=[h, w])

                    with air.herd(
                        [range(ROWS), range(COLS)], name="xaddherd", shape=(ROWS, COLS)
                    ) as herd:

                        @herd.body
                        def _(th, tw):
                            tile_in = air.alloc(TILE_SIZE, i32, scope=herd.private())
                            # vector=0: the tile is 16 i32 wide, and a 256-bit
                            # <8 x i32> operation does not legalize on AIE2.
                            # The predecessor's copy was scalar too.
                            tile_out = air.alloc(
                                TILE_SIZE, i32, scope=herd.private(), vector=0
                            )

                            chan_in.get(tile_in, indices=[th, tw])
                            tile_out[:] = tile_in[:]
                            chan_out.put(tile_out, indices=[th, tw])

                    for h in range(ROWS):
                        for w in range(COLS):
                            chan_out.get(tile(B, h, w), indices=[h, w])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the channel_size example",
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

    input_matrix = np.random.randint(
        low=np.iinfo(INOUT_DATATYPE).min,
        high=np.iinfo(INOUT_DATATYPE).max,
        size=IMAGE_SIZE,
        dtype=INOUT_DATATYPE,
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
