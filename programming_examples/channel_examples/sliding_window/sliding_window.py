# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects import memref
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

IMAGE_WIDTH = 4
IMAGE_IN_HEIGHT = 8
IMAGE_IN_SIZE = [IMAGE_IN_HEIGHT, IMAGE_WIDTH]

IMAGE_OUT_HEIGHT = IMAGE_IN_HEIGHT - 2  # Window size is 3
IMAGE_OUT_SIZE = [IMAGE_OUT_HEIGHT, IMAGE_WIDTH]
SINGLE_ROW = [1, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32
"""
Goals: Input in one large block
[width, height] = [4, 8]
example:
0 0 0 0
1 1 1 1
2 2 2 2
3 3 3 3
4 4 4 4
5 5 5 5
6 6 6 6
7 7 7 7

Output: 3 rows added to each other
[width, height] = [4, 8 - (window_size - 1)]
tile_size = [1, 4]
exmaple: 
3 3 3 3 (rows: 0 1 2)
6 6 6 6 (rows: 1 2 3)
9 9 9 9 (rows: 2 3 4)
12 12 12 12 (rows: 3 4 5)
15 15 15 15 (rows: 4 5 6)
18 18 18 18 (rows: 5 6 7)
"""


def memref_view(source, shape, dtype, shift):
    byte_width_dtype = dtype.width // 8
    byte_shift = arith.MulIOp(shift, arith.ConstantOp.create_index(byte_width_dtype))
    return memref.subview(T.memref(*shape, dtype), source, byte_shift, [])


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyIn = MemRefType.get(IMAGE_IN_SIZE, xrt_dtype)
    memrefTyOut = MemRefType.get(IMAGE_OUT_SIZE, xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_buffer_l1 = MemRefType.get(
        shape=(np.prod(IMAGE_IN_SIZE) * xrt_dtype.width,),
        element_type=T.ui8(),
    )
    image_row_l1 = MemRefType.get(
        shape=SINGLE_ROW,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def copy(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut", b)

            @segment(name="seg")
            def segment_body():

                @herd(name="sliding_window_herd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    image_in = AllocOp(image_buffer_l1, [], [])
                    ChannelGet("ChanIn", image_in)

                    for i in range_(IMAGE_IN_HEIGHT - 2):
                        row_out = AllocOp(image_row_l1, [], [])

                        row0 = memref_view(
                            image_in.operation,
                            SINGLE_ROW,
                            xrt_dtype,
                            shift=arith.MulIOp(
                                arith.ConstantOp.create_index(IMAGE_WIDTH), i
                            ),
                        )
                        row1 = memref_view(
                            image_in.operation,
                            SINGLE_ROW,
                            xrt_dtype,
                            shift=arith.MulIOp(
                                arith.ConstantOp.create_index(IMAGE_WIDTH),
                                arith.AddIOp(arith.ConstantOp.create_index(1), i),
                            ),
                        )
                        row2 = memref_view(
                            image_in.operation,
                            SINGLE_ROW,
                            xrt_dtype,
                            shift=arith.MulIOp(
                                arith.ConstantOp.create_index(IMAGE_WIDTH),
                                arith.AddIOp(arith.ConstantOp.create_index(2), i),
                            ),
                        )

                        for j in range_(IMAGE_WIDTH):
                            val0 = load(row0, [j])
                            val1 = load(row1, [j])
                            val2 = load(row2, [j])

                            val_out = arith.AddIOp(val0, val1)
                            val_out = arith.AddIOp(val2, val_out)
                            store(row_out, val_out, [j])

                            ChannelPut("ChanOut", row_out)
                            DeallocOp(row_out)
                            yield_([])
                        yield_([])

                    DeallocOp(image_in)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the herd_to_herd channel example",
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
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.full(IMAGE_IN_SIZE, 0x2, dtype=INOUT_DATATYPE)
    output_b = np.full(IMAGE_OUT_SIZE, 0x5, dtype=INOUT_DATATYPE)
    for i in range(IMAGE_IN_HEIGHT):
        for j in range(IMAGE_WIDTH):
            input_a[i, j] = i
    print(input_a)
    for i in range(IMAGE_OUT_HEIGHT):
        for j in range(IMAGE_WIDTH):
            output_b[i, j] = (i + 1) * 3
    print(output_b)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=False)
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
