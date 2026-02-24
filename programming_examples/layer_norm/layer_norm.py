# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Layer Normalization Example

Implements layer normalization on a 2D input [M, N]:
  1. mean = sum(x, axis=-1) / N
  2. var  = sum((x - mean)^2, axis=-1) / N
  3. rstd = 1 / sqrt(var + eps)
  4. y    = (x - mean) * rstd

Uses a single AIE tile with DMA transfers between L3 and L1 memory.
Computation is done element-wise with scalar load/store, using a
1-element L1 buffer as accumulator for reductions.
"""

import argparse

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

EPS = 1e-5


@module_builder
def build_module(M, N, np_dtype):
    xrt_dtype = type_mapper(np_dtype)

    # L3 types
    l3MemrefTy = MemRefType.get([M, N], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([N], xrt_dtype, memory_space=l1_mem_space)
    l1ScalarTy = MemRefType.get([1], xrt_dtype, memory_space=l1_mem_space)

    @FuncOp.from_py_func(l3MemrefTy, l3MemrefTy)
    def layer_norm(arg0, arg1):

        @herd(name="herd_0", sizes=[1, 1], operands=[arg0, arg1])
        def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_out):
            l1_row = AllocOp(l1RowTy, [], [])
            l1_out = AllocOp(l1RowTy, [], [])
            l1_acc = AllocOp(l1ScalarTy, [], [])

            c0 = arith.ConstantOp.create_index(0)

            for row in range_(M):
                # DMA: load one row from L3 to L1
                dma_memcpy_nd(
                    l1_row,
                    l3_in,
                    src_offsets=[row, 0],
                    src_sizes=[1, N],
                    src_strides=[N, 1],
                )

                zero_f = arith.ConstantOp(xrt_dtype, 0.0)
                n_f = arith.ConstantOp(xrt_dtype, float(N))
                eps_f = arith.ConstantOp(xrt_dtype, EPS)
                one_f = arith.ConstantOp(xrt_dtype, 1.0)

                # Step 1: mean = sum(x) / N
                store(zero_f, l1_acc, [c0])
                for i in range_(N):
                    val = load(l1_row, [i])
                    acc = load(l1_acc, [c0])
                    new_acc = arith.addf(acc, val)
                    store(new_acc, l1_acc, [c0])
                    yield_([])
                sum_val = load(l1_acc, [c0])
                mean = arith.divf(sum_val, n_f)

                # Step 2: variance = sum((x - mean)^2) / N
                store(zero_f, l1_acc, [c0])
                for i in range_(N):
                    val = load(l1_row, [i])
                    diff = arith.subf(val, mean)
                    sq = arith.mulf(diff, diff)
                    acc = load(l1_acc, [c0])
                    new_acc = arith.addf(acc, sq)
                    store(new_acc, l1_acc, [c0])
                    yield_([])
                var_sum = load(l1_acc, [c0])
                variance = arith.divf(var_sum, n_f)

                # Step 3: rstd = 1 / sqrt(var + eps)
                var_eps = arith.addf(variance, eps_f)
                std = math_dialect.SqrtOp(var_eps)
                rstd = arith.divf(one_f, std)

                # Step 4: y = (x - mean) * rstd
                for j in range_(N):
                    val = load(l1_row, [j])
                    diff = arith.subf(val, mean)
                    normed = arith.mulf(diff, rstd)
                    store(normed, l1_out, [j])
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
            DeallocOp(l1_acc)


if __name__ == "__main__":
    M_DEFAULT = 32
    N_DEFAULT = 64
    INPUT_DATATYPE = np.float32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the layer normalization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_DEFAULT, help="M dimension (rows)")
    parser.add_argument("--N", type=int, default=N_DEFAULT, help="N dimension (cols)")
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

    mlir_module = build_module(args.M, args.N, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x_input = np.random.rand(args.M, args.N).astype(INPUT_DATATYPE)

    # Reference: layer normalization without weight/bias
    eps = EPS
    mean = np.mean(x_input, axis=-1, keepdims=True)
    variance = np.mean((x_input - mean) ** 2, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(variance + eps)
    y_expected = ((x_input - mean) * rstd).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="layer_norm",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_input],
                expected_outputs=[y_expected],
                rtol=1e-2,
                atol=1e-1,
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
