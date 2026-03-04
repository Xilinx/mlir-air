# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

dtype_map = {
    "uint8": np.uint8,
    "int8": np.int8,
    "int16": np.int16,
    "uint16": np.uint16,
    "float32": np.float32,
    "bfloat16": bfloat16,
}
DEFAULT_DTYPE = "uint8"


@module_builder
def build_module(vector_size, num_subvectors, np_dtype):
    assert vector_size % num_subvectors == 0
    xrt_dtype = type_mapper(np_dtype)

    # Type and method of input/output
    memrefTyInOut = T.memref(vector_size, xrt_dtype)

    # The compute core splits input into subvectors for processing
    lineWidthInBytes = vector_size // num_subvectors

    # Memref type definition used by the compute core and external function
    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    tensor_type = MemRefType.get(
        shape=[lineWidthInBytes],
        element_type=xrt_dtype,
        memory_space=mem_space,
    )

    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):

                @herd(name="copyherd", sizes=[1, 1], operands=[arg2, arg3])
                def herd_body(_tx, _ty, _sx, _sy, c, d):

                    # Process each subvector individually
                    for i in range_(
                        0, num_subvectors * lineWidthInBytes, lineWidthInBytes
                    ):
                        # We must allocate a buffer of image size for the input/output
                        tensor_in = AllocOp(tensor_type, [], [])
                        tensor_out = AllocOp(tensor_type, [], [])

                        # Place the input image (a) into the L1 memory region
                        dma_memcpy_nd(
                            tensor_in,
                            c,
                            src_offsets=[i],
                            src_sizes=[lineWidthInBytes],
                            src_strides=[1],
                        )

                        for j in range_(lineWidthInBytes):
                            # Load the input value
                            val = load(tensor_in, [j])

                            # Store the output value
                            store(val, tensor_out, [j])
                            yield_([])

                        dma_memcpy_nd(
                            d,
                            tensor_out,
                            dst_offsets=[i],
                            dst_sizes=[lineWidthInBytes],
                            dst_strides=[1],
                        )

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
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "-t",
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=dtype_map.keys(),
        help="The data type to use (default: uint8)",
    )
    args = parser.parse_args()

    np_dtype = dtype_map[args.dtype]
    mlir_module = build_module(args.vector_size, args.subvector_size, np_dtype)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(args.vector_size, dtype=np_dtype)
    output_b = np.arange(args.vector_size, dtype=np_dtype)

    runner = XRTRunner(
        verbose=args.verbose, output_format=args.output_format, instance_name="copy"
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
