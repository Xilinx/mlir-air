# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One worker per tile, each on its own channel pair, on air.api.

    air.channel @ChanInHHWW / @ChanOutHHWW    one pair per tile

The sibling ``single_core_channel`` streams every tile through one channel and
lets ordering do the indexing. This one goes the other way: a channel pair and a
1x1 herd *per tile*, all unrolled at trace time, so each worker has a private
connection and the tile index is baked into the symbol name rather than carried
by arrival order.

Both spellings are worth having -- the first scales in tiles, the second in
cores -- and they are the two ends of the same trade.

Unchanged from the raw-bindings version this replaces, except for two things:

* The L3-side puts and gets sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The per-tile add is written ``tile_out[:] = tile_in[:] + n`` rather than a
  scalar loop nest over every (i, j). ``vector=0`` keeps it scalar as the
  predecessor was: a <8 x i32> add is 256-bit and does not legalize.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

DTYPE = {np.int32: i32}


def format_name(prefix, index_0, index_1):
    return f"{prefix}{index_0:02}{index_1:02}"


def build_module(image_height, image_width, tile_height, tile_width, np_dtype):
    assert image_height % tile_height == 0
    assert image_width % tile_width == 0
    dt = DTYPE[np_dtype]
    tile_size = [tile_height, tile_width]
    rows = image_height // tile_height
    cols = image_width // tile_width

    A = air.tensor([image_height, image_width], dt)
    B = air.tensor([image_height, image_width], dt)

    chan_in = {}
    chan_out = {}
    for h in range(rows):
        for w in range(cols):
            chan_in[h, w] = air.channel(format_name("ChanIn", h, w))
            chan_out[h, w] = air.channel(format_name("ChanOut", h, w))

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    for h in range(rows):
                        for w in range(cols):
                            i0, j0 = tile_height * h, tile_width * w
                            chan_in[h, w].put(
                                A[i0 : i0 + tile_height, j0 : j0 + tile_width]
                            )

                    # A closure per worker rather than a defaulted capture in
                    # the body signature: air.api reads the body's positional
                    # arity to get the grid rank, so `def _(tx, h=h)` would read
                    # as a second coordinate.
                    def one_worker(h, w):
                        with air.herd(
                            [range(1)], name=format_name("xaddherd", h, w), shape=(1,)
                        ) as hd:

                            @hd.body
                            def _(tx):
                                tile_in = air.alloc(
                                    tile_size, dt, scope=hd.private(), vector=0
                                )
                                tile_out = air.alloc(
                                    tile_size, dt, scope=hd.private(), vector=0
                                )

                                chan_in[h, w].get(tile_in)
                                tile_out[:] = tile_in[:] + (rows * h + w)
                                chan_out[h, w].put(tile_out)

                    for h in range(rows):
                        for w in range(cols):
                            one_worker(h, w)

                    for h in range(rows):
                        for w in range(cols):
                            i0, j0 = tile_height * h, tile_width * w
                            chan_out[h, w].get(
                                B[i0 : i0 + tile_height, j0 : j0 + tile_width]
                            )

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
