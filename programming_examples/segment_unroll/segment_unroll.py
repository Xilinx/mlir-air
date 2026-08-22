# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Segment unroll, on air.api.

``air.launch``, ``air.segment`` and ``air.herd`` each carry their own iteration
space, and this is the example for the middle one. A grid on ``air.segment``
becomes its ``sizes``, printed as ``unroll(...)``, and it means something
different from the other two:

* a **launch** point replays everything inside it -- temporal;
* a **segment** point is a *spatial copy* of the segment body, which
  ``air-to-aie`` lays out across columns, so N points is N physical herds at
  once rather than one herd run N times;
* a **herd** point is a core.

Each copy is handed its own coordinate and uses it to pick its endpoint of a
channel declared ``size=[SX, SY]``::

    air.segment @segment_with_unroll unroll(%u0, %u1) in (%c2, %c1)
      air.channel.put @ChanIn[%u0, %u1] ...
      air.herd @compute_herd args(%arg=%u0, ...)      <- threaded in explicitly
        air.channel.get @ChanIn[%u0, %u1] ...

That third line is the one worth understanding: ``air.herd`` is
``IsolatedFromAbove``, so the segment's coordinate cannot simply be referenced
from inside the herd body. air.api threads it in as a herd operand, which is
what the raw-bindings predecessor spelled by hand as ``operands=[seg_x, seg_y]``.

The L3 endpoints sit **inside** the segment, where the predecessor had them
beside it at launch scope. air.api has no scope between the two, and this is the
better placement anyway: each copy moves its own chunk, indexed by its own
coordinate, rather than a host-side Python loop stamping out one put per copy
from outside. The offset is ``seg_x * CHUNK`` -- an ordinary expression on a
coordinate.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i32

VECTOR_LEN = 64
SEGMENT_SIZE_X = 2  # Segment unroll factor in X
SEGMENT_SIZE_Y = 1  # Segment unroll factor in Y
INOUT_DATATYPE = np.int32

# CHUNK sizes both the L1 tiles and each copy's slice of L3, so the vector has
# to divide evenly across the copies.
assert VECTOR_LEN % SEGMENT_SIZE_X == 0, (
    f"VECTOR_LEN ({VECTOR_LEN}) must be evenly divisible by "
    f"SEGMENT_SIZE_X ({SEGMENT_SIZE_X})"
)
assert VECTOR_LEN % (SEGMENT_SIZE_X * SEGMENT_SIZE_Y) == 0, (
    f"VECTOR_LEN ({VECTOR_LEN}) must be evenly divisible by "
    f"total segment count ({SEGMENT_SIZE_X * SEGMENT_SIZE_Y})"
)

CHUNK = VECTOR_LEN // SEGMENT_SIZE_X


def build_module():
    """Build the segment-unroll kernel. Returns an ``air.api`` launch."""
    A = air.tensor([VECTOR_LEN], i32)
    B = air.tensor([VECTOR_LEN], i32)

    # One endpoint per unrolled copy, which is why these are sized by the
    # segment grid and not by the herd.
    chan_in = air.channel("ChanIn", size=[SEGMENT_SIZE_X, SEGMENT_SIZE_Y])
    chan_out = air.channel("ChanOut", size=[SEGMENT_SIZE_X, SEGMENT_SIZE_Y])

    with air.launch(name="segment_unroll_test") as launch:

        @launch.body
        def _():
            with air.segment(
                [range(SEGMENT_SIZE_X), range(SEGMENT_SIZE_Y)],
                name="segment_with_unroll",
            ) as seg:

                @seg.body
                def _(seg_x, seg_y):
                    # This copy's slice of the vector.
                    off = seg_x * CHUNK
                    chan_in.put(A[off : off + CHUNK], indices=[seg_x, seg_y])

                    with air.herd([range(1)], name="compute_herd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tile_in = air.alloc([CHUNK], i32, scope=h.private())
                            tile_out = air.alloc([CHUNK], i32, scope=h.private())

                            chan_in.get(tile_in, indices=[seg_x, seg_y])
                            tile_out[:] = tile_in[:] + 10
                            chan_out.put(tile_out, indices=[seg_x, seg_y])

                    chan_out.get(B[off : off + CHUNK], indices=[seg_x, seg_y])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="segment_unroll.py",
        description="Builds, runs, and tests the segment unroll example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print the generated MLIR module and exit",
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

    # Input: [0, 1, ..., 63]; expected output: [10, 11, ..., 73]
    input_a = np.arange(VECTOR_LEN, dtype=INOUT_DATATYPE)
    output_b = input_a + 10

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="segment_unroll_test",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
