# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Add each tile's index to that tile, streamed through channels, on air.api.

    air.channel @ChanIn     L3 -> L1, one put per tile
    air.channel @ChanOut    L1 -> L3, one put per tile

The image is cut into tiles on the *producer* side: a loop nest at segment scope
puts one tile region per trip, and the single core does one get per trip. So
neither end names a tile index -- the channel carries them in order, and the
core's `tile_num` is just its trip counter. That is the ordering guarantee a
channel gives, used as the whole indexing scheme.

Unchanged from the raw-bindings version this replaces, except for two things:

* The L3-side puts and gets sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The per-tile add is written ``tile_out[:] = tile_in[:] + tile_num`` rather
  than a scalar loop nest over every (i, j); the trip counter broadcasts into
  the expression as an index_cast, which is what the predecessor spelled by
  hand. ``vector=0`` keeps it scalar as it was: a <8 x i32> add is 256-bit and
  does not legalize.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

DTYPE = {np.int32: i32}


def build_module(image_height, image_width, tile_height, tile_width, np_dtype):
    assert image_height % tile_height == 0
    assert image_width % tile_width == 0
    dt = DTYPE[np_dtype]
    tile_size = [tile_height, tile_width]
    rows = image_height // tile_height
    cols = image_width // tile_width

    A = air.tensor([image_height, image_width], dt)
    B = air.tensor([image_height, image_width], dt)

    chan_in = air.channel("ChanIn")
    chan_out = air.channel("ChanOut")

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    # One put per tile, row-major -- the order the core's gets
                    # consume them in.
                    for r in air.sequential(0, rows):
                        for c in air.sequential(0, cols):
                            i0 = r * tile_height
                            j0 = c * tile_width
                            chan_in.put(A[i0 : i0 + tile_height, j0 : j0 + tile_width])

                    with air.herd([range(1)], name="xaddherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tile_in = air.alloc(
                                tile_size, dt, scope=h.private(), vector=0
                            )
                            tile_out = air.alloc(
                                tile_size, dt, scope=h.private(), vector=0
                            )

                            for tile_num in air.sequential(0, rows * cols):
                                chan_in.get(tile_in)
                                tile_out[:] = tile_in[:] + tile_num
                                chan_out.put(tile_out)

                    for r in air.sequential(0, rows):
                        for c in air.sequential(0, cols):
                            i0 = r * tile_height
                            j0 = c * tile_width
                            chan_out.get(B[i0 : i0 + tile_height, j0 : j0 + tile_width])

    return launch


if __name__ == "__main__":
    # Default values.
    IMAGE_WIDTH = 16
    IMAGE_HEIGHT = 32
    TILE_WIDTH = 8
    TILE_HEIGHT = 16
    INOUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
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
        "--image-height",
        type=int,
        default=IMAGE_HEIGHT,
        help="Height of the image data",
    )
    parser.add_argument(
        "--image-width", type=int, default=IMAGE_WIDTH, help="Width of the image data"
    )
    parser.add_argument(
        "--tile-height", type=int, default=TILE_HEIGHT, help="Height of the tile data"
    )
    parser.add_argument(
        "--tile-width", type=int, default=TILE_WIDTH, help="Width of the tile data"
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

    launch = build_module(
        args.image_height,
        args.image_width,
        args.tile_height,
        args.tile_width,
        INOUT_DATATYPE,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.zeros(
        shape=(args.image_height, args.image_width), dtype=INOUT_DATATYPE
    )
    output_b = np.zeros(
        shape=(args.image_height, args.image_width), dtype=INOUT_DATATYPE
    )
    for i in range(args.image_height):
        for j in range(args.image_width):
            input_a[i, j] = i * args.image_height + j
            tile_num = (
                i // args.tile_height * (args.image_width // args.tile_width)
                + j // args.tile_width
            )
            output_b[i, j] = input_a[i, j] + tile_num

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
