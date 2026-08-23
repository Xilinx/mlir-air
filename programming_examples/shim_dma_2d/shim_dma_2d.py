# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One tile through L1 with a strided shim DMA, on air.api.

The smallest program that shows a *2-D* shim transfer. The image is 16x32 and
the tile is 8x16, so the tile is not contiguous in L3: reading it means a DMA
with sizes [8, 16] and strides [32, 1], which is what the shim's buffer
descriptor is for. Everything else -- one core, a copy, and the tile written
back to the same corner -- is deliberately trivial, so the transfer is the only
thing under test.

    air.ops.load(tile_in, A[0:8, 0:16])      L3 -> L1, strided
    tile_out[:] = tile_in[:]
    air.ops.store(tile_out, B[0:8, 0:16])    L1 -> L3, strided

Only the top-left tile is touched; the rest of B stays zero, and all three
harnesses in this directory check exactly that.

Unchanged from the raw-bindings version this replaces, except that the copy is
written ``tile_out[:] = tile_in[:]`` rather than a scalar loop nest over every
(i, j). The tile's innermost dimension is 16, so that vectorises to a
``<16 x i32>`` (512-bit) read and write, where the predecessor moved one element
per iteration.

``build_module`` returns the launch rather than a module, matching the other
converted examples; ``.build(target=...)`` produces the module. ``run.py`` and
``test.py`` import the shape constants below, so they stay module-level.
"""

import argparse

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


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    B = air.tensor(IMAGE_SIZE, i32)

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="xaddherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tile_in = air.alloc(TILE_SIZE, i32, scope=h.private())
                            tile_out = air.alloc(TILE_SIZE, i32, scope=h.private())

                            air.ops.load(tile_in, A[0:TILE_HEIGHT, 0:TILE_WIDTH])
                            tile_out[:] = tile_in[:]
                            air.ops.store(tile_out, B[0:TILE_HEIGHT, 0:TILE_WIDTH])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="shim_dma_2d.py",
        description="Prints the AIR module for the shim_dma_2d example",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    args = parser.parse_args()

    print(build_module().build(target=args.target))
