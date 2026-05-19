# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""bf16 matrix transpose using an external kernel.

Transposes an [M, K] bf16 matrix to [K, M] using a C++ kernel compiled
with Peano. The kernel performs scalar element-by-element transpose.

DMA stride-based transpose is not possible for sub-32-bit types on AIE
because the inner-most DMA stride must be 1 for <32b data widths.
Instead, we DMA the matrix into L1 contiguously and let the kernel
perform the transpose.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

np.random.seed(42)

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

INOUT_DATATYPE = bfloat16


@module_builder
def build_module(m, k):
    xrt_dtype = type_mapper(INOUT_DATATYPE)

    memrefTyIn = MemRefType.get(shape=[m * k], element_type=xrt_dtype)
    memrefTyOut = MemRefType.get(shape=[k * m], element_type=xrt_dtype)

    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_type = MemRefType.get(
        shape=[m * k],
        element_type=xrt_dtype,
        memory_space=mem_space,
    )

    transpose_func = external_func("transpose_bf16", inputs=[l1_type, l1_type])

    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def transpose(arg0, arg1):
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):
                @herd(
                    name="herd",
                    sizes=[1, 1],
                    operands=[arg2, arg3],
                    link_with="transpose.o",
                )
                def herd_body(_tx, _ty, _sx, _sy, a, b):
                    l1_in = AllocOp(l1_type, [], [])
                    l1_out = AllocOp(l1_type, [], [])

                    dma_memcpy_nd(l1_in, a)

                    call(
                        transpose_func,
                        inputs=[l1_in, l1_out],
                        input_types=[l1_type, l1_type],
                    )

                    dma_memcpy_nd(b, l1_out)

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)


if __name__ == "__main__":
    M = 64
    K = 32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the bf16 transpose example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-m", type=int, default=M, help="Matrix rows")
    parser.add_argument("-k", type=int, default=K, help="Matrix columns")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(args.m, args.k)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_matrix = np.random.uniform(-1.0, 1.0, (args.m, args.k)).astype(INOUT_DATATYPE)
    expected_output = np.transpose(input_matrix)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            output_format=args.output_format,
            instance_name="transpose",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_matrix.reshape(-1)],
                expected_outputs=[expected_output.reshape(-1)],
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
