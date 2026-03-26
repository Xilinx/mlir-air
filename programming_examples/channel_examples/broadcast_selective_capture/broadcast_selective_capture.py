# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This example demonstrates a common pattern where, due to DMA channel
# limitations, data that is NOT logically broadcast still needs to be
# distributed via a broadcast channel to reuse routing and DMA resources.
#
# Scenario:
#   - A 1-D array is divided into NUM_TILES tiles.
#   - The tiles are streamed one-by-one into a herd of size [1, NUM_TILES]
#     via a broadcast channel (all cores receive every tile).
#   - Each core counts the iteration and only captures the tile whose
#     iteration index matches its own column index (ty). All other tiles
#     are received but discarded.
#   - Each core adds its own index to the captured tile (to prove which
#     core handled which tile) and writes the result out.
#
# The net effect is equivalent to a non-broadcast scatter, but implemented
# over a single broadcast channel to conserve DMA channels.

import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects import arith, scf
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

TILE_SIZE = 32
NUM_TILES = 4  # Also the herd width; each core captures exactly one tile

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)

    total_size = TILE_SIZE * NUM_TILES
    memrefTyIn = MemRefType.get([total_size], xrt_dtype)
    memrefTyOut = MemRefType.get([total_size], xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    tile_type_l1 = MemRefType.get(
        shape=[TILE_SIZE],
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    # Broadcast channel: size [1, 1] broadcast to [1, NUM_TILES]
    # All cores in the herd receive the same data on each put.
    Channel("BroadcastIn", size=[1, 1], broadcast_shape=[1, NUM_TILES])

    # Output channel: one per core (sized to herd)
    Channel("ChanOut", size=[1, NUM_TILES])

    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def broadcast_selective_capture(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def launch_body(l3_in, l3_out):

            # Stream tiles one by one into the broadcast channel
            for i in range(NUM_TILES):
                offset = TILE_SIZE * i
                ChannelPut(
                    "BroadcastIn",
                    l3_in,
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                )

            # Collect output tiles from each core
            for i in range(NUM_TILES):
                offset = TILE_SIZE * i
                ChannelGet(
                    "ChanOut",
                    l3_out,
                    indices=[0, i],
                    offsets=[offset],
                    sizes=[TILE_SIZE],
                    strides=[1],
                )

            @segment(name="seg")
            def segment_body():

                @herd(name="compute_herd", sizes=[1, NUM_TILES])
                def herd_body(tx, ty, _sx, _sy):

                    # Allocate L1 buffers: one for receiving broadcast data,
                    # one for the captured tile that will be written out.
                    recv_buf = AllocOp(tile_type_l1, [], [])
                    out_buf = AllocOp(tile_type_l1, [], [])

                    # Iterate over all broadcast rounds
                    for iter_idx in range(NUM_TILES):
                        # Every core must consume from the broadcast channel
                        # on every iteration (the hardware requires all
                        # broadcast targets to accept the data).
                        ChannelGet("BroadcastIn", recv_buf, indices=[tx, ty])

                        # Only capture when this iteration matches our
                        # core index.
                        cmp_val = arith.cmpi(
                            arith.CmpIPredicate.eq,
                            arith.index_cast(T.i32(), ty),
                            arith.ConstantOp(T.i32(), iter_idx),
                        )
                        if_op = scf.IfOp(cmp_val)
                        with InsertionPoint(if_op.then_block):
                            # Copy received data into the output buffer,
                            # adding the core index to prove ownership.
                            for j in range_(TILE_SIZE):
                                val = load(recv_buf, [j])
                                val_out = arith.addi(val, arith.index_cast(T.i32(), ty))
                                store(val_out, out_buf, [j])
                                yield_([])
                            yield_([])

                    # Write the captured (and modified) tile to the output
                    ChannelPut("ChanOut", out_buf, indices=[tx, ty])

                    DeallocOp(recv_buf)
                    DeallocOp(out_buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the broadcast selective capture example",
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

    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    total_size = TILE_SIZE * NUM_TILES

    # Input: [0, 1, 2, ..., total_size - 1]
    input_a = np.arange(total_size, dtype=INOUT_DATATYPE)

    # Expected output: each core captures tile[ty] and adds ty to each element.
    # Core ty captures input[ty*TILE_SIZE : (ty+1)*TILE_SIZE] and adds ty.
    expected_output = np.zeros(total_size, dtype=INOUT_DATATYPE)
    for ty in range(NUM_TILES):
        start = ty * TILE_SIZE
        end = start + TILE_SIZE
        expected_output[start:end] = input_a[start:end] + ty

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="broadcast_selective_capture",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a],
            expected_outputs=[expected_output],
        )
    )
