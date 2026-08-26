# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Passthrough through a hand-written AIE kernel, on air.api.

Same schedule as ``passthrough_channel`` -- L3 -> channel -> L1, back out
L1 -> channel -> L3 -- with the core-side copy done by ``passThroughLine`` from
``passThrough.cc`` instead of by the DSL:

    air.channel @ChanIn                    declared at module scope
    air.channel @ChanOut
      air.segment
        ChanIn.put(A)                      the whole [n] vector, once
        herd                               link_with = "passThrough.cc.o"
          for _ in air.sequential(subvectors)
              ChanIn.get(tile_in)          one [n/subvectors] chunk per trip
              passThroughLine(tile_in, tile_out, chunk)
              ChanOut.put(tile_out)
        ChanOut.get(B)

The data is ``uint8``, and the tiles say so. That is what ``ui8`` is for here:
``air.extern`` emits

    func.func private @passThroughLine(memref<1024xui8, 2 : i32>,
                                       memref<1024xui8, 2 : i32>, i32)

which is the signature aircc links the object file against, and the object file
was compiled from ``uint8_t *`` (``-DBIT_WIDTH=8`` in the Makefile). Declaring
``i8`` would put a prototype in the IR that ``passThrough.cc`` does not define.

Unsigned types in air.api are movable but not computable -- MLIR's arith and
linalg ops take signless operands -- which costs this example nothing, since
everything done to the data happens inside the external kernel.

The put and get on the L3 side sit in the segment rather than beside it, as in
``passthrough_channel``: reaching L3 needs a shim DMA allocation, and hoisting
them out to function scope fails in air-to-aie with "failed to link to any shim
dma allocation".
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32, ui8

INOUT_DATATYPE = np.uint8


def build_module(vector_size, num_subvectors):
    assert vector_size % num_subvectors == 0
    chunk = vector_size // num_subvectors

    A = air.tensor([vector_size], ui8)
    B = air.tensor([vector_size], ui8)

    chan_in = air.channel("ChanIn")
    chan_out = air.channel("ChanOut")

    # The line width the kernel loops over. Its type has to be declared: a
    # Python int does not say whether the kernel wants i32 or index.
    pass_through_line = air.extern(
        "passThroughLine", link_with="passThrough.cc.o", scalars=[i32]
    )

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
                            tile_in = air.alloc([chunk], ui8, scope=h.private())
                            tile_out = air.alloc([chunk], ui8, scope=h.private())

                            for _i in air.sequential(0, num_subvectors):
                                chan_in.get(tile_in)
                                pass_through_line(tile_in, tile_out, chunk)
                                chan_out.put(tile_out)

                    chan_out.get(B)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_kernel example",
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

    input_a = np.arange(args.vector_size, dtype=INOUT_DATATYPE)
    output_b = np.arange(args.vector_size, dtype=INOUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
