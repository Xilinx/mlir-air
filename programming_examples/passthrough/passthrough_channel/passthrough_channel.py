# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Passthrough over a pair of channels, on air.api.

The vector goes L3 -> channel -> L1, is copied core-side, and comes back
L1 -> channel -> L3. The schedule is unchanged from the raw-bindings version
this replaces, and it is the smallest program that shows what a channel is for:

    air.channel @ChanIn                    declared at module scope
    air.channel @ChanOut
      air.segment
        ChanIn.put(A)                      the whole [n] vector, once
        herd
          for _ in air.sequential(subvectors)
              ChanIn.get(tile_in)          one [n/subvectors] chunk per trip
              tile_out[:] = tile_in[:]
              ChanOut.put(tile_out)
        ChanOut.get(B)

Two things about channels are on display here, and both are why this is not
just ``ops.load``/``ops.store``:

* **The herd carries no channel operand.** ``air.herd`` is ``IsolatedFromAbove``
  and the DSL threads staged L2 buffers in explicitly, but a channel is a
  module-level symbol: the herd's ``get`` finds what the segment's ``put`` sent
  by name alone.
* **Put and get sizes differ, deliberately.** One put of the whole vector feeds
  ``subvectors`` gets of a chunk each -- a channel is a stream, and each get
  takes the next piece. A transfer would have to match shapes; a channel must
  not.

The put and get on the L3 side sit in the segment rather than beside it. That is
load-bearing: reaching L3 needs a shim DMA allocation, and hoisting them out to
function scope fails in air-to-aie with "failed to link to any shim dma
allocation" -- measured on npu1 against this very example. air.api raises at the
call site rather than letting that through.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i8

INOUT_DATATYPE = np.uint8


def build_module(vector_size, num_subvectors):
    assert vector_size % num_subvectors == 0
    chunk = vector_size // num_subvectors

    A = air.tensor([vector_size], i8)
    B = air.tensor([vector_size], i8)

    chan_in = air.channel("ChanIn")
    chan_out = air.channel("ChanOut")

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    chan_in.put(A)

                    with air.herd([range(1)], name="copyherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            # Hoisted above the loop and reused across trips,
                            # which is what the loop is for; the predecessor
                            # allocated a fresh pair per trip.
                            tile_in = air.alloc([chunk], i8, scope=h.private())
                            tile_out = air.alloc([chunk], i8, scope=h.private())

                            for _i in air.sequential(0, num_subvectors):
                                chan_in.get(tile_in)
                                tile_out[:] = tile_in[:]
                                chan_out.put(tile_out)

                    chan_out.get(B)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
    )
    parser.add_argument(
        "-s",
        "--vector_size",
        type=int,
        default=4096,
        help="The size (in bytes) of the data vector to passthrough",
    )
    parser.add_argument(
        "--subvector_size",
        type=int,
        default=4,
        help="The number of sub-vectors to break the vector into",
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

    launch = build_module(args.vector_size, args.subvector_size)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.zeros(shape=args.vector_size, dtype=INOUT_DATATYPE)
    output_b = np.zeros(shape=args.vector_size, dtype=INOUT_DATATYPE)
    for i in range(args.vector_size):
        input_a[i] = i % 0xFF
        output_b[i] = i % 0xFF

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
