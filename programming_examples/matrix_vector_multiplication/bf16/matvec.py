# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Matrix-Vector Multiplication (bf16) — A[M,K] × b[K] = c[M]

Matrix A is pre-transposed on the host to 32-bit-word layout for the
matvec_vectorized kernel (bf16 pairs interleaved column-major).

Uses a single AIE tile with DMA transfers between L3 and L1 memory.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


def transpose_32bit_words(A, M, K):
    """Convert A[M,K] row-major to 32-bit-word-transposed layout.

    The matvec kernel expects columns transposed at 4-byte granularity.
    For bf16 (2 bytes), this means pairs of adjacent rows are interleaved
    column-major.
    """
    return A.reshape(M, K // 2, 2).transpose(1, 0, 2).reshape(-1)


@module_builder
def build_module(M, K, np_dtype):
    xrt_dtype = type_mapper(np_dtype)
    index_type = IndexType.get()

    l3_a_ty = MemRefType.get([M * K], xrt_dtype)
    l3_b_ty = MemRefType.get([K], xrt_dtype)
    l3_c_ty = MemRefType.get([M], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_a_ty = MemRefType.get([M * K], xrt_dtype, memory_space=l1_space)
    l1_b_ty = MemRefType.get([K], xrt_dtype, memory_space=l1_space)
    l1_c_ty = MemRefType.get([M], xrt_dtype, memory_space=l1_space)

    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([l1_a_ty, l1_b_ty, l1_c_ty], []),
        visibility="private",
    )
    zero_func = FuncOp(
        "zero_vectorized_bf16",
        ([l1_c_ty], []),
        visibility="private",
    )
    for func in [matvec_func, zero_func]:
        func.attributes["link_with"] = StringAttr.get("mv.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_a_ty, l3_b_ty, l3_c_ty)
    def matvec(arg_a, arg_b, arg_c):

        @herd(
            name="mv_herd",
            sizes=[1, 1],
            operands=[arg_a, arg_b, arg_c],
        )
        def herd_body(_tx, _ty, _sx, _sy, l3_a, l3_b, l3_c):
            l1_a = AllocOp(l1_a_ty, [], [])
            l1_b = AllocOp(l1_b_ty, [], [])
            l1_c = AllocOp(l1_c_ty, [], [])

            dma_memcpy_nd(l1_a, l3_a)
            dma_memcpy_nd(l1_b, l3_b)

            CallOp(zero_func, [l1_c])
            CallOp(matvec_func, [l1_a, l1_b, l1_c])

            dma_memcpy_nd(l3_c, l1_c)

            DeallocOp(l1_a)
            DeallocOp(l1_b)
            DeallocOp(l1_c)

        herd_body.attributes["link_with"] = StringAttr.get("mv.o")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Matrix-vector multiplication (A in 32-bit-word-transposed layout)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=128, help="Matrix rows (M)")
    parser.add_argument(
        "--k", type=int, default=128, help="Matrix columns / vector length (K)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    M, K = args.m, args.k
    INPUT_DATATYPE = bfloat16

    mlir_module = build_module(M, K, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    A = np.random.randn(M, K).astype(INPUT_DATATYPE)
    b = np.random.randn(K).astype(INPUT_DATATYPE)

    # Host preprocessing: 32-bit-word transpose
    A_transposed = transpose_32bit_words(A, M, K)

    # Reference
    ref_c = np.dot(A.astype(np.float32), b.astype(np.float32)).astype(INPUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="matvec",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[A_transposed, b],
            expected_outputs=[ref_c],
            rtol=1e0,
            atol=0.2,
        )
    )
