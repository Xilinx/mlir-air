# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A ring of workers passing tiles to their neighbours, on air.api.

    air.channel @ChanIn          L3 -> L1, one bundle member per worker
    air.channel @WorkerToWorker  L1 -> L1, worker (th, tw) -> its successor
    air.channel @ChanOut         L1 -> L3, one bundle member per worker

Each worker stamps its own tile number onto its tile and hands the result not
to itself but to the *next* worker in row-major order, which then writes it
out. So the tile that lands at (th, tw) in the output carries the tile number
of the worker *before* it -- the test's expected output is built from exactly
that permutation, and a kernel that quietly kept its own tile would fail it.

The successor is where the index arithmetic comes in:

    tw_next = (tw + 1) % W
    th_next = (th + (tw + 1) // W) % H

Both must reach the IR as affine expressions rather than a chain of arith ops,
because the AIR dependency analysis and the DMA specialisation passes read
affine offsets directly. The raw-bindings version this replaces built them by
hand -- three ``AffineMap.get`` calls with ``AffineExpr.get_mod`` and
``get_floor_div``, forty lines of them. Written as ordinary Python operators on
the herd coordinates, ``IndexExpr`` produces the same two maps.

Unchanged from the predecessor, except for two things:

* The L3-side put and get sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The two per-tile kernels are whole-tile expressions rather than scalar loop
  nests over every (i, j). ``vector=0`` keeps them scalar as the predecessor
  was: a tile row is 4 i32 wide, well under a vector.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

IMAGE_WIDTH = 12
IMAGE_HEIGHT = 4
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

TILE_WIDTH = 4
TILE_HEIGHT = 2
TILE_SIZE = [TILE_HEIGHT, TILE_WIDTH]

assert IMAGE_HEIGHT % TILE_HEIGHT == 0
assert IMAGE_WIDTH % TILE_WIDTH == 0

# The worker grid: one worker per tile.
GRID_HEIGHT = IMAGE_HEIGHT // TILE_HEIGHT
GRID_WIDTH = IMAGE_WIDTH // TILE_WIDTH

INOUT_DATATYPE = np.int32


def next_worker(th, tw):
    """The successor of worker (th, tw) in row-major order, wrapping around.

    Shared by the kernel and the test's expected output, so the two cannot
    drift: the same expression runs on herd coordinates in the kernel and on
    Python ints in the check below.
    """
    tw_next = (tw + 1) % GRID_WIDTH
    th_next = (th + (tw + 1) // GRID_WIDTH) % GRID_HEIGHT
    return th_next, tw_next


def build_module():
    A = air.tensor(IMAGE_SIZE, i32)
    B = air.tensor(IMAGE_SIZE, i32)

    # An input and an output channel per worker, plus the ring between them.
    chan_in = air.channel("ChanIn", size=[GRID_HEIGHT, GRID_WIDTH])
    chan_out = air.channel("ChanOut", size=[GRID_HEIGHT, GRID_WIDTH])
    ring = air.channel("WorkerToWorker", size=[GRID_HEIGHT, GRID_WIDTH])

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    # One tile of the image into each worker's bundle member.
                    for i in range(GRID_HEIGHT):
                        for j in range(GRID_WIDTH):
                            chan_in.put(
                                A[
                                    i * TILE_HEIGHT : (i + 1) * TILE_HEIGHT,
                                    j * TILE_WIDTH : (j + 1) * TILE_WIDTH,
                                ],
                                indices=[i, j],
                            )

                    with air.herd(
                        [range(GRID_HEIGHT), range(GRID_WIDTH)],
                        name="xaddherd",
                        shape=(GRID_HEIGHT, GRID_WIDTH),
                    ) as h:

                        @h.body
                        def _(th, tw):
                            tile_num = th * GRID_WIDTH + tw
                            th_next, tw_next = next_worker(th, tw)

                            # Stamp this worker's number on its own tile, then
                            # hand it to the next worker rather than keeping it.
                            tile_in = air.alloc(TILE_SIZE, i32, scope=h.private())
                            tile_out = air.alloc(
                                TILE_SIZE, i32, scope=h.private(), vector=0
                            )
                            chan_in.get(tile_in, indices=[th, tw])
                            tile_out[:] = tile_in[:] + tile_num
                            ring.put(tile_out, indices=[th_next, tw_next])

                            # ...and what arrives here came from the previous
                            # worker, carrying that worker's number.
                            tile_in2 = air.alloc(TILE_SIZE, i32, scope=h.private())
                            tile_out2 = air.alloc(
                                TILE_SIZE, i32, scope=h.private(), vector=0
                            )
                            ring.get(tile_in2, indices=[th, tw])
                            tile_out2[:] = tile_in2[:]
                            chan_out.put(tile_out2, indices=[th, tw])

                    for i in range(GRID_HEIGHT):
                        for j in range(GRID_WIDTH):
                            chan_out.get(
                                B[
                                    i * TILE_HEIGHT : (i + 1) * TILE_HEIGHT,
                                    j * TILE_WIDTH : (j + 1) * TILE_WIDTH,
                                ],
                                indices=[i, j],
                            )

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="worker_to_worker.py",
        description="Builds, runs, and tests the channel worker_to_worker example",
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

    input_matrix = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)
    output_matrix = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)

    # The tile that lands at (th, tw) was stamped by that worker's
    # *predecessor*, so invert next_worker to find whose number to expect.
    stamped_by = {}
    for th in range(GRID_HEIGHT):
        for tw in range(GRID_WIDTH):
            stamped_by[next_worker(th, tw)] = th * GRID_WIDTH + tw

    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            output_matrix[i, j] = (
                input_matrix[i, j] + stamped_by[(i // TILE_HEIGHT, j // TILE_WIDTH)]
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
