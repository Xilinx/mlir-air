# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Weighted RMS Normalization Example

Implements weighted RMS normalization on a 2D input [M, N]:
  1. rms  = sum(x^2, axis=-1) / N
  2. rstd = 1 / sqrt(rms + eps)
  3. y    = x * rstd * weight

The weight vector has shape [N] and is shared across all M rows.

Uses a single AIE tile with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write with
configurable VECTOR_SIZE (default 16 for AIE2).
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

EPS = 1e-5


@module_builder
def build_module(M, N, np_dtype, vector_size=16):
    xrt_dtype = type_mapper(np_dtype)
    assert (
        N % vector_size == 0
    ), f"N ({N}) must be divisible by vector_size ({vector_size})"

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    # L3 types
    l3MemrefTy = MemRefType.get([M, N], xrt_dtype)
    l3WeightTy = MemRefType.get([N], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([N], xrt_dtype, memory_space=l1_mem_space)
    l1VecTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    @FuncOp.from_py_func(l3MemrefTy, l3WeightTy, l3MemrefTy)
    def weighted_rms_norm(arg0, arg1, arg2):

        @herd(name="herd_0", sizes=[1, 1], operands=[arg0, arg1, arg2])
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
            l1_row = AllocOp(l1RowTy, [], [])
            l1_out = AllocOp(l1RowTy, [], [])
            l1_weight = AllocOp(l1RowTy, [], [])
            l1_acc = AllocOp(l1VecTy, [], [])

            c0 = arith.ConstantOp.create_index(0)
            cst0 = arith.ConstantOp(xrt_dtype, 0.0)
            n_f = arith.ConstantOp(xrt_dtype, float(N))
            eps_f = arith.ConstantOp(xrt_dtype, EPS)

            v_zero = BroadcastOp(vecTy, cst0)

            # DMA weight to L1 (shared across all rows)
            dma_memcpy_nd(l1_weight, l3_weight)

            for row in range_(M):
                # DMA: load one row from L3 to L1
                dma_memcpy_nd(
                    l1_row,
                    l3_in,
                    src_offsets=[row, 0],
                    src_sizes=[1, N],
                    src_strides=[N, 1],
                )

                # Step 1: Vectorized sum of x^2
                transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                for j in range_(0, N, vector_size):
                    sub_row = subview(l1_row.result, [j], [vector_size], [1])
                    sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                    v_x = transfer_read(
                        vecTy, sub_row, [c0], identity_map, cst0, [True]
                    )
                    v_sq = arith.mulf(v_x, v_x)
                    # Break mulf→addf chain
                    transfer_write(None, v_sq, sub_tmp, [c0], identity_map, [True])
                    v_sq_rd = transfer_read(
                        vecTy, sub_tmp, [c0], identity_map, cst0, [True]
                    )
                    v_acc = transfer_read(
                        vecTy, l1_acc, [c0], identity_map, cst0, [True]
                    )
                    v_sum = arith.addf(v_acc, v_sq_rd)
                    transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                    yield_([])

                # Horizontal reduce
                v_final = transfer_read(vecTy, l1_acc, [c0], identity_map, cst0, [True])
                total_sum = vector_reduction(xrt_dtype, "add", v_final)
                rms = arith.divf(total_sum, n_f)

                # Step 2: rstd = rsqrt(rms + eps) in f32
                f32 = F32Type.get()
                rms_eps = arith.addf(rms, eps_f)
                rms_eps_f32 = arith.extf(f32, rms_eps)
                rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                rstd = arith.truncf(xrt_dtype, rstd_f32)

                # Step 3: y = x * rstd * weight
                v_rstd = BroadcastOp(vecTy, rstd)
                for j in range_(0, N, vector_size):
                    sub_row = subview(l1_row.result, [j], [vector_size], [1])
                    sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                    sub_out = subview(l1_out.result, [j], [vector_size], [1])
                    v_x = transfer_read(
                        vecTy, sub_row, [c0], identity_map, cst0, [True]
                    )
                    v_w = transfer_read(vecTy, sub_w, [c0], identity_map, cst0, [True])
                    v_normed = arith.mulf(v_x, v_rstd)
                    v_weighted = arith.mulf(v_normed, v_w)
                    transfer_write(
                        None, v_weighted, sub_out, [c0], identity_map, [True]
                    )
                    yield_([])

                # DMA: write result row from L1 to L3
                dma_memcpy_nd(
                    l3_out,
                    l1_out,
                    dst_offsets=[row, 0],
                    dst_sizes=[1, N],
                    dst_strides=[N, 1],
                )

                yield_([])

            DeallocOp(l1_row)
            DeallocOp(l1_out)
            DeallocOp(l1_weight)
            DeallocOp(l1_acc)


if __name__ == "__main__":
    M_DEFAULT = 32
    N_DEFAULT = 64
    VECTOR_SIZE = 16
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the weighted RMS normalization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_DEFAULT, help="M dimension (rows)")
    parser.add_argument("--N", type=int, default=N_DEFAULT, help="N dimension (cols)")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations",
    )
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

    mlir_module = build_module(args.M, args.N, INPUT_DATATYPE, args.vector_size)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x_input = np.random.rand(args.M, args.N).astype(INPUT_DATATYPE)
    weight = np.random.rand(args.N).astype(INPUT_DATATYPE)

    # Reference: weighted RMS normalization
    eps = EPS
    rms = np.sqrt(
        np.mean(x_input.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps
    )
    y_expected = (
        (x_input.astype(np.float32) / rms) * weight.astype(np.float32)
    ).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="weighted_rms_norm",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_input, weight],
                expected_outputs=[y_expected],
                rtol=5e-2,
                atol=5e-1,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
