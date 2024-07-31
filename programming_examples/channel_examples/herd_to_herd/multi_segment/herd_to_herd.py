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

    # We want to store our data in L1 memory
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the tile
    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    # Create two channels which will send/receive the
    # input/output data respectively
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # Create a channel we will use to pass data between works in two herds
    ChannelOp("Herd2Herd")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # Fetch all input data into the channel
            ChannelPut("ChanIn", a)

            # Push all output data out of the channel
            ChannelGet("ChanOut", b)

            @segment(name="producer_segment")
            def segment_body():

                @herd(name="producer_herd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We must allocate a buffer of tilße size for the input/output
                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    ChannelGet("ChanIn", image_in)

                    # Access every value in the image
                    for i in range_(IMAGE_HEIGHT):
                        for j in range_(IMAGE_WIDTH):
                            # Load the input value
                            val_in = load(image_in, [i, j])

                            # Calculate the output value
                            val_out = arith.muli(val_in, val_in)

                            # Store the output value
                            store(val_out, image_out, [i, j])
                            yield_([])
                        yield_([])

                    ChannelPut("Herd2Herd", image_out)

                    DeallocOp(image_in)
                    DeallocOp(image_out)

            @segment(name="consumer_segment")
            def segment_body():

                @herd(name="consumer_herd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We must allocate a buffer of image size for the input/output
                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    ChannelGet("Herd2Herd", image_in)

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

                    ChannelPut("ChanOut", image_out)

                    DeallocOp(image_in)
                    DeallocOp(image_out)


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

    input_a = np.full(IMAGE_SIZE, 0x2, dtype=INOUT_DATATYPE)
    output_b = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
