# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Segment Unroll Example

This example demonstrates the segment unroll feature in MLIR-AIR. A segment with
a 'sizes' attribute is unrolled, creating multiple copies of the segment body,
each with different segment indices. This allows for efficient parallelization
across multiple segment instances.

The kernel reads a vector, adds 10 to each element across multiple unrolled
segments, and writes the result back. Each segment processes a portion of the
input data using channels indexed by segment coordinates.
"""

import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects import arith
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

# Configuration
VECTOR_LEN = 64
SEGMENT_SIZE_X = 2  # Segment unroll factor in X dimension
SEGMENT_SIZE_Y = 1  # Segment unroll factor in Y dimension
INOUT_DATATYPE = np.int32

# Sanity checks: ensure vector length is evenly divisible by segment dimensions
# This is required because VECTOR_LEN // SEGMENT_SIZE_X is used to size L1 memrefs
# and compute host-side slice sizes. Non-divisible configurations would silently
# truncate and/or mis-size buffers.
assert VECTOR_LEN % SEGMENT_SIZE_X == 0, (
    f"VECTOR_LEN ({VECTOR_LEN}) must be evenly divisible by "
    f"SEGMENT_SIZE_X ({SEGMENT_SIZE_X})"
)
assert VECTOR_LEN % (SEGMENT_SIZE_X * SEGMENT_SIZE_Y) == 0, (
    f"VECTOR_LEN ({VECTOR_LEN}) must be evenly divisible by "
    f"total segment count ({SEGMENT_SIZE_X * SEGMENT_SIZE_Y})"
)


@module_builder
def build_module():
    """Build a kernel with segment unroll to demonstrate its lowering."""
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyInOut = T.memref(VECTOR_LEN, xrt_dtype)

    # L1 memory space for tile data
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_type_l1 = MemRefType.get(
        shape=[VECTOR_LEN // SEGMENT_SIZE_X],
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    # Define channels for data movement with dimensions matching segment unroll
    # Each unrolled segment instance needs its own channel endpoint
    Channel("ChanIn", size=[SEGMENT_SIZE_X, SEGMENT_SIZE_Y])
    Channel("ChanOut", size=[SEGMENT_SIZE_X, SEGMENT_SIZE_Y])

    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def segment_unroll_test(input_buf, output_buf):
        """Test function with segment unroll."""

        @launch(operands=[input_buf, output_buf])
        def launch_body(a, b):
            # Put/Get on each subchannel at launch level
            # Each unrolled segment instance reads from its own subchannel
            chunk_size = VECTOR_LEN // SEGMENT_SIZE_X
            for sx in range(SEGMENT_SIZE_X):
                for sy in range(SEGMENT_SIZE_Y):
                    offset = sx * chunk_size
                    # Use python indices to create the MLIR constants for channel indices
                    sx_idx = arith.constant(T.index(), sx)
                    sy_idx = arith.constant(T.index(), sy)
                    offset_idx = arith.constant(T.index(), offset)
                    size_idx = arith.constant(T.index(), chunk_size)
                    stride_idx = arith.constant(T.index(), 1)
                    # Put a portion of input to each subchannel
                    ChannelPut(
                        "ChanIn",
                        a,
                        indices=[sx_idx, sy_idx],
                        offsets=[offset_idx],
                        sizes=[size_idx],
                        strides=[stride_idx],
                    )
                    # Get a portion of output from each subchannel
                    ChannelGet(
                        "ChanOut",
                        b,
                        indices=[sx_idx, sy_idx],
                        offsets=[offset_idx],
                        sizes=[size_idx],
                        strides=[stride_idx],
                    )

            # Segment with unroll - this creates SEGMENT_SIZE_X * SEGMENT_SIZE_Y
            # copies of the segment body, each with different segment indices
            @segment(name="segment_with_unroll", sizes=[SEGMENT_SIZE_X, SEGMENT_SIZE_Y])
            def segment_body(seg_x, seg_y, seg_sx, seg_sy):
                """Segment body that will be unrolled."""

                # Pass segment unroll indices to herd via operands
                @herd(name="compute_herd", sizes=[1, 1], operands=[seg_x, seg_y])
                def herd_body(tx, ty, sx, sy, herd_seg_x, herd_seg_y):
                    """Simple compute: add 10 to each element."""
                    tile_in = AllocOp(image_type_l1, [], [])
                    tile_out = AllocOp(image_type_l1, [], [])

                    # Use segment indices to select the correct channel endpoint
                    ChannelGet("ChanIn", tile_in, indices=[herd_seg_x, herd_seg_y])

                    for i in range_(VECTOR_LEN // SEGMENT_SIZE_X):
                        val = load(tile_in, [i])
                        val_plus_10 = arith.addi(val, arith.constant(xrt_dtype, 10))
                        store(val_plus_10, tile_out, [i])
                        yield_([])

                    ChannelPut("ChanOut", tile_out, indices=[herd_seg_x, herd_seg_y])

                    DeallocOp(tile_in)
                    DeallocOp(tile_out)


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

    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Prepare test data
    # Input: [0, 1, 2, ..., 63]
    # Expected output: [10, 11, 12, ..., 73]
    input_a = np.arange(VECTOR_LEN, dtype=INOUT_DATATYPE)
    output_b = input_a + 10

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="segment_unroll_test",
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
