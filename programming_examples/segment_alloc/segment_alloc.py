# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A tile staged through L2 on its way to a core, on air.api.

The sibling ``shim_dma_2d`` moves the same 8x16 tile straight from L3 into L1.
This one puts a memtile in between, which is what ``air.segment`` is for:

    l2 = air.alloc(TILE_SIZE, i32, scope=seg.private())   a memtile buffer
      air.ops.load(l2, A[0:8, 0:16])                      L3 -> L2, strided
      air.ops.load(tile_in, l2)                           L2 -> L1, contiguous
      tile_out[:] = tile_in[:]
      air.ops.store(tile_out, B[0:8, 0:16])               L1 -> L3, strided

``seg.private()`` is the whole point of the example: it allocates in memory
space 1, and the herd nested in the segment body carries that buffer in as an
operand automatically. ``air.herd`` is ``IsolatedFromAbove``, so in the
raw-bindings version this replaces, the L2 buffer had to be listed in
``operands=[...]`` by hand and picked back up as a body parameter. Here the
tracer knows which buffers are live in the enclosing segment and threads them
itself.

The strided L3 read is unchanged: the image is 16x32 and the tile 8x16, so the
tile is not contiguous, and the transfer carries sizes [8, 16] strides [32, 1].
Only the top-left tile is touched; the rest of B stays zero.

Unchanged from the raw-bindings version except that the copy is written
``tile_out[:] = tile_in[:]`` rather than a scalar loop nest over every (i, j).
The tile's innermost dimension is 16, so that vectorises to a ``<16 x i32>``
(512-bit) read and write.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_HEIGHT, TILE_WIDTH]

assert IMAGE_HEIGHT % TILE_HEIGHT == 0
assert IMAGE_WIDTH % TILE_WIDTH == 0

INOUT_DATATYPE = np.int32


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    B = air.tensor(IMAGE_SIZE, i32)

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    # L2: a memtile buffer, passed into the herd for us.
                    l2_tile = air.alloc(TILE_SIZE, i32, scope=seg.private())

                    with air.herd([range(1)], name="copyherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tile_in = air.alloc(TILE_SIZE, i32, scope=h.private())
                            tile_out = air.alloc(TILE_SIZE, i32, scope=h.private())

                            air.ops.load(l2_tile, A[0:TILE_HEIGHT, 0:TILE_WIDTH])
                            air.ops.load(tile_in, l2_tile)
                            tile_out[:] = tile_in[:]
                            air.ops.store(tile_out, B[0:TILE_HEIGHT, 0:TILE_WIDTH])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the segment_alloc example",
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

    mlir_module = build_module().build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.zeros(shape=IMAGE_SIZE, dtype=INOUT_DATATYPE)
    for h in range(TILE_HEIGHT):
        for w in range(TILE_WIDTH):
            output_b[h, w] = input_a[h, w]

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
