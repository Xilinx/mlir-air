# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner

range_ = for_

INOUT_DATATYPE = np.uint8
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize


@module_builder
def build_module(vector_size, num_subvectors):
    assert vector_size % num_subvectors == 0

    # chop input in 4 sub-tensors
    lineWidthInBytes = vector_size // num_subvectors

    # Type and method of input/output
    memrefTyInOut = T.memref(vector_size, T.ui8())
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # We want to store our data in L1 memory
    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the image
    tensor_type = MemRefType.get(
        shape=[lineWidthInBytes],
        element_type=T.ui8(),
        memory_space=mem_space,
    )

    passThroughLine = external_func(
        "passThroughLine", inputs=[tensor_type, tensor_type, T.i32()]
    )

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):
            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut", b)

            # The arguments are still the input and the output
            @segment(name="seg")
            def segment_body():

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="copyherd", sizes=[1, 1], link_with="passThrough.cc.o")
                def herd_body(tx, ty, sx, sy):

                    for i in range_(num_subvectors):
                        # We must allocate a buffer of image size for the input/output
                        tensor_in = AllocOp(tensor_type, [], [])
                        tensor_out = AllocOp(tensor_type, [], [])

                        ChannelGet("ChanIn", tensor_in)

                        call(
                            passThroughLine,
                            inputs=[tensor_in, tensor_out, lineWidthInBytes],
                            input_types=[tensor_type, tensor_type, T.i32()],
                        )

                        ChannelPut("ChanOut", tensor_out)

                        # Deallocate our L1 buffers
                        DeallocOp(tensor_in)
                        DeallocOp(tensor_out)
                        yield_([])


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
    args = parser.parse_args()

    mlir_module = build_module(args.vector_size, args.subvector_size)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(1, args.vector_size + 1, dtype=INOUT_DATATYPE)
    output_b = np.arange(1, args.vector_size + 1, dtype=INOUT_DATATYPE)
    for i in range(args.vector_size):
        input_a[i] = i % 0xFF
        output_b[i] = i % 0xFF

    runner = XRTRunner(verbose=args.verbose)
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
