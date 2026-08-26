# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Selective capture from a broadcast channel, on air.api.

A 1-D array is split into ``NUM_TILES`` tiles and streamed one at a time down a
*broadcast* channel, so every core in the ``[1, NUM_TILES]`` herd receives every
tile. Each core keeps only the tile whose round matches its own column index and
discards the rest, adding its index to the captured tile to prove which core
handled which.

The net effect is a scatter. It is written as a broadcast because DMA channels
are scarce: one broadcast channel and one set of routes serve all four cores,
where four separate streams would not fit.

Two things this example is the smallest case of:

* **Every core must take every tile.** The hardware requires all broadcast
  targets to accept the data, so the ``get`` sits outside the conditional and
  only the *capture* is guarded. Guarding the get instead would hang.
* **The guard is ``ops.branch`` with no ``otherwise``.** A core that is not the
  addressee does nothing this round. That emits an ``scf.if`` whose else region
  holds only its terminator, which ``canonicalize`` removes before anything in
  the AIR pipeline reads the region structure.

The rounds are a Python ``for``: ``NUM_TILES`` is a trace-time constant and each
round names a different broadcast slot, so unrolling them is what the
predecessor did too. The per-round condition ``ty == i`` is what tells the cores
apart, and that has to be a region -- the herd body is traced once for all four.

One difference from the predecessor: the capture is
``out[:] = recv[:] + ty``, replacing a scalar ``memref.load``/``store`` loop over
all 32 elements. The DSL vectorises it, and ``ty`` broadcasts as a scalar.
"""

import argparse

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import i32
from air.backend.xrt_runner import XRTRunner

TILE_SIZE = 32
NUM_TILES = 4  # also the herd width; each core captures exactly one tile

INOUT_DATATYPE = np.int32


def build_module():
    total = TILE_SIZE * NUM_TILES
    src = air.tensor([total], i32)
    dst = air.tensor([total], i32)

    # One put reaches all NUM_TILES cores: size is the producer grid, and
    # broadcast_shape is the consumer grid the get indexes.
    bcast = air.channel("BroadcastIn", size=[1, 1], broadcast_shape=[1, NUM_TILES])
    out = air.channel("ChanOut", size=[1, NUM_TILES])

    with air.launch(name="broadcast_selective_capture") as launch:

        @launch.body
        def _():
            # Stream the tiles, one round each.
            for i in range(NUM_TILES):
                lo = TILE_SIZE * i
                bcast.put(src[lo : lo + TILE_SIZE])

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(1), range(NUM_TILES)],
                        name="compute_herd",
                        shape=(1, NUM_TILES),
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            recv = air.alloc([TILE_SIZE], i32, scope=herd.private())
                            captured = air.alloc([TILE_SIZE], i32, scope=herd.private())

                            for i in range(NUM_TILES):
                                # Outside the branch: every broadcast target has
                                # to accept the data every round or the flow
                                # stalls for all of them.
                                bcast.get(recv, indices=[tx, ty])

                                # Inside: only the addressee keeps it.
                                with ops.branch(ty == i):
                                    captured[:] = recv[:] + ty

                            out.put(captured, indices=[tx, ty])

            for i in range(NUM_TILES):
                lo = TILE_SIZE * i
                out.get(dst[lo : lo + TILE_SIZE], indices=[0, i])

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="broadcast_selective_capture.py",
        description="Selective capture from a broadcast channel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    launch = build_module()
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        return 0

    total = TILE_SIZE * NUM_TILES
    input_a = np.arange(total, dtype=INOUT_DATATYPE)
    # Tile i is captured by core i, which adds its own index.
    expected = np.concatenate(
        [input_a[i * TILE_SIZE : (i + 1) * TILE_SIZE] + i for i in range(NUM_TILES)]
    ).astype(INOUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="broadcast_selective_capture",
        target_device=launch.target,
    )
    return runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[expected])


if __name__ == "__main__":
    exit(main())
