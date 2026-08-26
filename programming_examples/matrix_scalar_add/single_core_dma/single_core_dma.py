# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Add each tile's index to that tile, one core walking every tile, on air.api.

The image is cut into tiles and a single core visits them in turn, adding the
tile's row-major index to every element it holds:

    for r in air.sequential(0, rows)
      for c in air.sequential(0, cols)
        air.ops.load(tile_in, A[r*th : ..., c*tw : ...])    L3 -> L1, strided
        tile_out[:] = tile_in[:] + (r * cols + c)
        air.ops.store(tile_out, B[r*th : ..., c*tw : ...])  L1 -> L3, strided

This is the DMA counterpart of ``single_core_channel``, and the contrast is the
reason both exist. The channel version puts one tile per trip from the segment
and lets *arrival order* do the indexing, so the core never names a tile. Here
the transfer is addressed, so the loop variables appear twice over: once as the
L3 offset, and once as the value being added.

Both uses go through the same machinery -- ``r`` and ``c`` are index
expressions, so ``r * tile_height`` is a slice bound and ``r * cols + c``
broadcasts into the elementwise expression as an ``index_cast``. The
raw-bindings version this replaces spelled both by hand, with
``arith.MulIOp``/``AddIOp`` for the offsets and an explicit ``arith.index_cast``
for the addend.

Unchanged otherwise, except that the per-tile add is written
``tile_out[:] = tile_in[:] + n`` rather than a scalar loop nest over every
(i, j). ``vector=0`` keeps the emitted loop scalar as the predecessor's was,
whatever ``--tile-width`` is given: a <8 x i32> add is 256-bit and does not
legalize on AIE2.
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

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="xaddherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            # Hoisted above the loop and reused across trips;
                            # the predecessor allocated a fresh pair per tile.
                            tile_in = air.alloc(
                                tile_size, dt, scope=h.private(), vector=0
                            )
                            tile_out = air.alloc(
                                tile_size, dt, scope=h.private(), vector=0
                            )

                            for r in air.sequential(0, rows):
                                for c in air.sequential(0, cols):
                                    i0 = r * tile_height
                                    j0 = c * tile_width
                                    air.ops.load(
                                        tile_in,
                                        A[
                                            i0 : i0 + tile_height,
                                            j0 : j0 + tile_width,
                                        ],
                                    )
                                    tile_out[:] = tile_in[:] + (r * cols + c)
                                    air.ops.store(
                                        tile_out,
                                        B[
                                            i0 : i0 + tile_height,
                                            j0 : j0 + tile_width,
                                        ],
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
        description="Builds, runs, and tests the single_core_dma example",
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
