# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import numpy as np

np.random.seed(42)

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import type_mapper, make_air_parser, run_on_npu

dtype_map = {
    "uint32": np.uint32,
    "float32": np.float32,
}
DEFAULT_DTYPE = "uint32"


@module_builder
def build_module(m, k, dtype):
    xrt_dtype = type_mapper(dtype)

    memrefTyIn = MemRefType.get(shape=[m, k], element_type=xrt_dtype)
    memrefTyOut = MemRefType.get(shape=[k, m], element_type=xrt_dtype)

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def transpose(arg0, arg1):
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):
                @herd(name="herd", sizes=[1, 1], operands=[arg2, arg3])
                def herd_body(_tx, _ty, _sx, _sy, a, b):
                    # This is the type definition of the tensor
                    tensor_type = l1_memref_type(
                        [m * k], xrt_dtype
                    )  # Read as one large array

                    # We must allocate a buffer of tile size for the input/output
                    tensor_in = AllocOp(tensor_type, [], [])

                    dma_memcpy_nd(
                        tensor_in,
                        a,
                    )

                    dma_memcpy_nd(
                        b,
                        tensor_in,
                        src_sizes=[1, k, m],
                        src_strides=[1, 1, k],
                    )

                    # Deallocate our L1 buffer
                    DeallocOp(tensor_in)

                    # We must allocate a buffer of tile size for the input/output
                    tensor_in = AllocOp(tensor_type, [], [])

                    DeallocOp(tensor_in)


if __name__ == "__main__":
    parser = make_air_parser(
        "Builds, runs, and tests the matrix_scalar_add/single_core_channel example"
    )
    parser.add_argument(
        "-m",
        type=int,
        default=64,
        help="The matrix to transpose will be of size M x K, this parameter sets the M value",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=32,
        help="The matrix to transpose will be of size M x K, this parameter sets the k value",
    )
    parser.add_argument(
        "-t",
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=dtype_map.keys(),
        help="The data type of the matrix",
    )

    args = parser.parse_args()

    np_dtype = dtype_map[args.dtype]
    mlir_module = build_module(args.m, args.k, np_dtype)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Generate a random matrix
    matrix_shape = (args.m, args.k)
    if np.issubdtype(np_dtype, np.floating):
        for np_type in dtype_map.values():
            if not np.issubdtype(np_type, np.floating):
                if np_type.nbytes == np_dtype.nbytes:
                    int_type_substitution = np_type
        input_matrix = np.random.randint(
            low=np.iinfo(int_type_substitution).min,
            high=np.iinfo(int_type_substitution).max,
            size=matrix_shape,
            dtype=int_type_substitution,
        ).astype(np_dtype)
    else:
        input_matrix = np.random.randint(
            low=np.iinfo(np_dtype).min,
            high=np.iinfo(np_dtype).max,
            size=matrix_shape,
            dtype=np_dtype,
        )
    expected_output_matrix = np.transpose(input_matrix)

    exit(
        run_on_npu(
            args,
            mlir_module,
            inputs=[input_matrix],
            instance_name="transpose",
            expected_outputs=[expected_output_matrix],
        )
    )
