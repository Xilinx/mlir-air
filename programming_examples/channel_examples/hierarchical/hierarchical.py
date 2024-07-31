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

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    mem_space_l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)

    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )
    image_type_l2 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l2,
    )

    ChannelOp("ChanInL2")
    ChannelOp("ChanOutL2")
    ChannelOp("ChanInL1")
    ChannelOp("ChanOutL1")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            ChannelPut("ChanInL2", a)
            ChannelGet("ChanOutL2", b)

            @segment(name="seg")
            def segment_body():
                image_in_l2 = AllocOp(image_type_l2, [], [])
                ChannelGet("ChanInL2", image_in_l2)
                ChannelPut("ChanInL1", image_in_l2)
                DeallocOp(image_in_l2)

                image_out_l2 = AllocOp(image_type_l2, [], [])
                ChannelGet("ChanOutL1", image_out_l2)
                ChannelPut("ChanOutL2", image_out_l2)
                DeallocOp(image_out_l2)

                @herd(name="addherd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We must allocate a buffer of image size for the input/output
                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    ChannelGet("ChanInL1", image_in)

                    # Access every value in the image
                    for i in range_(IMAGE_HEIGHT):
                        for j in range_(IMAGE_WIDTH):
                            # Load the input value
                            val_in = load(image_in, [i, j])

                            # Calculate the output value
                            val_out = arith.addi(val_in, arith.ConstantOp(xrt_dtype, 1))

                            # Store the output value
                            store(val_out, image_out, [i, j])
                            yield_([])
                        yield_([])

                    ChannelPut("ChanOutL1", image_out)

                    DeallocOp(image_in)
                    DeallocOp(image_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the channel hierarchical example",
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

    input_matrix = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )
    output_matrix = np.arange(1, np.prod(IMAGE_SIZE) + 1, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module, inputs=[input_matrix], expected_outputs=[output_matrix]
        )
    )
