# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This test verifies that conditional channel.put/get operations inside a
# single herd body work correctly when using scf.if to branch based on
# herd tile coordinates.
#
# A single herd of size [1, 2] is used:
#   - Tile (0, 0): producer — squares the input, puts result on Tile2Tile
#   - Tile (0, 1): consumer — gets from Tile2Tile, adds 1, outputs result
#
# Input: 0x2 => producer: 0x2*0x2 = 0x4 => consumer: 0x4+1 = 0x5

import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects import arith, scf
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    Channel("ChanIn")
    Channel("ChanOut")
    Channel("Tile2Tile")

    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut", b)

            @segment(name="seg")
            def segment_body():

                @herd(name="herd_0", sizes=[1, 2])
                def herd_body(tx, ty, sx, sy):

                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    # Branch based on herd tile coordinate ty:
                    #   ty == 0: producer (get from ChanIn, square, put to Tile2Tile)
                    #   ty == 1: consumer (get from Tile2Tile, add 1, put to ChanOut)
                    c0 = arith.ConstantOp.create_index(0)
                    cmp = arith.CmpIOp(
                        arith.CmpIPredicate.eq, ty, c0
                    )
                    if_op = scf.IfOp(cmp, hasElse=True)

                    with InsertionPoint(if_op.then_block):
                        # Producer: get input, square each element, put to Tile2Tile
                        ChannelGet("ChanIn", image_in)

                        for i in range_(IMAGE_HEIGHT):
                            for j in range_(IMAGE_WIDTH):
                                val = load(image_in, [i, j])
                                squared = arith.muli(val, val)
                                store(squared, image_out, [i, j])
                                yield_([])
                            yield_([])

                        ChannelPut("Tile2Tile", image_out)
                        yield_([])

                    with InsertionPoint(if_op.else_block):
                        # Consumer: get from Tile2Tile, add 1, put to ChanOut
                        ChannelGet("Tile2Tile", image_in)

                        for i in range_(IMAGE_HEIGHT):
                            for j in range_(IMAGE_WIDTH):
                                val = load(image_in, [i, j])
                                plus_one = arith.addi(
                                    val, ConstantOp(xrt_dtype, 1)
                                )
                                store(plus_one, image_out, [i, j])
                                yield_([])
                            yield_([])

                        ChannelPut("ChanOut", image_out)
                        yield_([])

                    DeallocOp(image_in)
                    DeallocOp(image_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Tests scf.if-based conditional channel access in a single herd",
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

    # Input: all 0x2
    # Expected: producer squares (0x2*0x2 = 0x4), consumer adds 1 (0x4+1 = 0x5)
    input_a = np.full(IMAGE_SIZE, 0x2, dtype=INOUT_DATATYPE)
    output_b = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose, output_format=args.output_format, instance_name="copy"
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
