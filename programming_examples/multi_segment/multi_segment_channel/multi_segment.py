# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

VECTOR_LEN = 32
VECTOR_SIZE = [VECTOR_LEN, 1]

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = INOUT_DATATYPE
    memrefTyInOut = MemRefType.get(VECTOR_SIZE, xrt_dtype)

    # We want to store our data in L1 memory
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the tile
    image_type_l1 = MemRefType.get(
        shape=VECTOR_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    ChannelOp("ChanInA")
    ChannelOp("ChanInB")
    ChannelOp("ChanOutC")
    ChannelOp("ChanOutD")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut, memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1, arg2, arg3):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1, arg2, arg3])
        def launch_body(a, b, c, d):
            ChannelPut("ChanInA", a)
            ChannelPut("ChanInB", b)
            ChannelGet("ChanOutC", c)
            ChannelGet("ChanOutD", d)

            @segment(name="seg1")
            def segment_body():

                @herd(name="addherd1", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    image_in_a = AllocOp(image_type_l1, [], [])
                    image_out_a = AllocOp(image_type_l1, [], [])

                    ChannelGet("ChanInA", image_in_a)

                    # Access every value in the tile
                    c0 = arith.ConstantOp.create_index(0)
                    for j in range_(VECTOR_LEN):
                        val_a = load(image_in_a, [c0, j])
                        val_outa = arith.addi(val_a, arith.constant(xrt_dtype, 10))
                        store(val_outa, image_out_a, [c0, j])
                        yield_([])

                    ChannelPut("ChanOutC", image_out_a)
                    DeallocOp(image_in_a)
                    DeallocOp(image_out_a)

            @segment(name="seg2")
            def segment_body():

                @herd(name="addherd2", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    image_in_b = AllocOp(image_type_l1, [], [])
                    image_out_b = AllocOp(image_type_l1, [], [])

                    ChannelGet("ChanInB", image_in_b)

                    # Access every value in the tile
                    c0 = arith.ConstantOp.create_index(0)
                    for j in range_(VECTOR_LEN):
                        val_b = load(image_in_b, [c0, j])
                        val_outb = arith.addi(arith.constant(xrt_dtype, 10), val_b)
                        store(val_outb, image_out_b, [c0, j])
                        yield_([])

                    ChannelPut("ChanOutD", image_out_b)

                    DeallocOp(image_in_b)
                    DeallocOp(image_out_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the multi segment channel example",
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

    input_a = np.full(VECTOR_SIZE, 2, dtype=INOUT_DATATYPE)
    input_b = np.full(VECTOR_SIZE, 3, dtype=INOUT_DATATYPE)
    output_c = np.full(VECTOR_SIZE, 12, dtype=INOUT_DATATYPE)
    output_d = np.full(VECTOR_SIZE, 13, dtype=INOUT_DATATYPE)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a, input_b],
            expected_outputs=[output_c, output_d],
        )
    )
