# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Add each tile's index to that tile, one core per tile, on air.api.

The sibling ``single_core_dma`` walks every tile from one core with a loop nest.
This one hands each tile to its own core: the loop *is* the herd, and the loop
variables are herd coordinates rather than induction variables.

    with air.herd([range(rows), range(cols)], name="xaddherd") as h:
        def _(tx, ty):
            air.ops.load(tile_in, A[tx*th : ..., ty*tw : ...])
            tile_out[:] = tile_in[:] + (tx * cols + ty)
            air.ops.store(tile_out, B[tx*th : ..., ty*tw : ...])

That is the whole difference from the sibling, and it is the one air.api makes
cheapest. In the raw-bindings version this replaces, turning a herd coordinate
into an offset meant building an ``AffineMap`` per expression -- four of them,
each a handful of ``AffineExpr.get_mul``/``get_add`` calls, plus an
``affine_apply`` per use -- because a coordinate is an SSA value and cannot be
multiplied with Python's ``*``. Here it can: ``tx`` is an index expression, so
``tx * tile_height`` is a slice bound and ``tx * cols + ty`` broadcasts into the
elementwise expression as an ``index_cast``. Roughly sixty lines of map
construction become three.

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
                    # shape= is pinned to the tile grid rather than left to the
                    # default. air.api's default 2-D shape on npu1 is (1, 4) --
                    # deliberately conservative, since a 2-D herd can run out of
                    # shim DMA capacity before it runs out of columns -- and
                    # letting it apply here would strip-mine the row axis back
                    # into a temporal loop, giving 2 cores doing 2 tiles each
                    # where the predecessor had 4 cores doing one apiece. One
                    # core per tile is the entire point of this example.
                    with air.herd(
                        [range(rows), range(cols)],
                        name="xaddherd",
                        shape=(rows, cols),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            tile_in = air.alloc(
                                tile_size, dt, scope=h.private(), vector=0
                            )
                            tile_out = air.alloc(
                                tile_size, dt, scope=h.private(), vector=0
                            )

                            i0 = tx * tile_height
                            j0 = ty * tile_width
                            air.ops.load(
                                tile_in,
                                A[i0 : i0 + tile_height, j0 : j0 + tile_width],
                            )
                            tile_out[:] = tile_in[:] + (tx * cols + ty)
                            air.ops.store(
                                tile_out,
                                B[i0 : i0 + tile_height, j0 : j0 + tile_width],
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
        description="Builds, runs, and tests the multi_core_dma example",
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
