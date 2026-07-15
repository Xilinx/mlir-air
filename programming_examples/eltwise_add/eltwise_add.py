# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Element-Wise Add

Implements element-wise addition on a 1D input [N]:
  c = a + b

Uses a 1xnum_tiles AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write with
configurable VECTOR_SIZE (default 16 for BF16, 8 for F32).
"""

import argparse

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(
    n, tile_n, np_dtype_in, vector_size=0, num_tiles=2, herd_x=1, herd_y=None
):
    a_size = [n]
    b_size = a_size
    out_size = a_size
    xrt_dtype_in = type_mapper(np_dtype_in)

    # Determine herd shape
    if herd_y is None:
        herd_y = num_tiles
    total_tiles = herd_x * herd_y
    assert (
        n % (tile_n * total_tiles) == 0
    ), f"n ({n}) must be divisible by tile_n*total_tiles ({tile_n}*{total_tiles}={tile_n*total_tiles})"

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    # Vectorization setup
    vectorize = vector_size > 0
    if vectorize:
        assert (
            tile_n % vector_size == 0
        ), f"tile_n ({tile_n}) must be divisible by vector_size ({vector_size})"
        vecTy = VectorType.get([vector_size], xrt_dtype_in)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
        index_type = IndexType.get()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def eltwise_add(arg0, arg1, arg2):
        @herd(
            name="herd_0",
            sizes=[herd_x, herd_y],
            operands=[arg0, arg1, arg2],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_a,
            _l3_b,
            _l3_c,
        ):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
            l1_b_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            chunk_size = n // total_tiles
            for _l_ivx in range_(0, chunk_size, tile_n):

                # Contiguous partitioning: each tile gets a contiguous block.
                # offset = linear_tile_idx * chunk_size + loop_var
                offset_map = AffineMap.get(
                    0,
                    3,
                    [
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(herd_y),
                                    ),
                                    AffineSymbolExpr.get(2),
                                ),
                                AffineConstantExpr.get(chunk_size),
                            ),
                            AffineSymbolExpr.get(0),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [_l_ivx, _tx, _ty])

                dma_memcpy_nd(
                    l1_a_data,
                    _l3_a,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_b_data,
                    _l3_b,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                if vectorize:
                    # Vectorized compute loop
                    c0 = ConstantOp(index_type, 0)
                    cVecSize = ConstantOp(index_type, vector_size)
                    cTileN = ConstantOp(index_type, tile_n)
                    cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)

                    for j in range_(c0, cTileN, cVecSize):
                        sub_a = subview(l1_a_data.result, [j], [vector_size], [1])
                        sub_b = subview(l1_b_data.result, [j], [vector_size], [1])
                        sub_c = subview(l1_out_data.result, [j], [vector_size], [1])
                        v_a = transfer_read(
                            vecTy, sub_a, [c0], identity_map, cst0, [True]
                        )
                        v_b = transfer_read(
                            vecTy, sub_b, [c0], identity_map, cst0, [True]
                        )
                        v_c = arith.AddFOp(v_a, v_b)
                        transfer_write(None, v_c, sub_c, [c0], identity_map, [True])
                        yield_([])
                else:
                    # Scalar compute loop (original)
                    for i in range_(tile_n):
                        val_a = load(l1_a_data, [i])
                        val_b = load(l1_b_data, [i])
                        val_out = arith.addf(val_a, val_b)
                        store(val_out, l1_out_data, [i])
                        yield_([])

                dma_memcpy_nd(
                    _l3_c,
                    l1_out_data,
                    dst_offsets=[
                        offset,
                    ],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    # Default values — optimized BF16 vectorized config for NPU2.
    # For NPU1 (F32 scalar): --dtype f32 --vector-size 0 --herd-x 1 --herd-y 2
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16
    VECTOR_SIZE = 16
    NUM_TILES = 2

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the eltwise_add example",
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
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector width (0 for scalar, 16 for BF16, 8 for F32)",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=NUM_TILES,
        help="Number of herd tiles (parallel cores), used as herd_y when herd-x/herd-y not set",
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Herd x dimension (default: 1)",
    )
    parser.add_argument(
        "--herd-y",
        type=int,
        default=None,
        help="Herd y dimension (default: num-tiles)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "f32"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
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
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after warmup) and "
        "print Latency + bandwidth in addition to the correctness check",
    )
    args = parser.parse_args()

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")

    if args.dtype == "bf16":
        INPUT_DATATYPE = bfloat16
    else:
        INPUT_DATATYPE = np.float32

    mlir_module = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        vector_size=args.vector_size,
        num_tiles=args.num_tiles,
        herd_x=args.herd_x,
        herd_y=args.herd_y,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Use N(0,1) (matching the GPU test standard) so the correctness check sees
    # a realistic signed distribution rather than an all-positive one. Generate
    # in float32 (not the default float64) to avoid a large f64 intermediate.
    rng = np.random.default_rng(0)
    input_a_f32 = rng.standard_normal(args.n, dtype=np.float32)
    input_b_f32 = rng.standard_normal(args.n, dtype=np.float32)
    input_a = input_a_f32.astype(INPUT_DATATYPE)
    input_b = input_b_f32.astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":

        # Reference computed in FP32, then rounded to the output dtype (a
        # bf16-rounded reference when dtype=bf16) — matches how a GPU/HF bf16
        # elementwise op is verified. Add the actual bf16-rounded operands in
        # f32 so the reference sees the same inputs the kernel does.
        ref = (input_a.astype(np.float32) + input_b.astype(np.float32)).astype(
            INPUT_DATATYPE
        )

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="eltwise_add",
            runtime_loop_tiling_sizes=[4, 4],
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        # Two dtype branches only (--dtype choices = bf16 | f32):
        #   bf16: canonical rtol 1.6e-2; add is exact-to-bf16-round
        #         (mean_rel_L1 ~1.9e-3), atol sized to the measured worst-case
        #         single-element bf16 round (~3e-2).
        #   f32:  effectively exact, so tight rtol/atol.
        rtol = 1.6e-2 if INPUT_DATATYPE == bfloat16 else 1e-3
        atol = 5e-2 if INPUT_DATATYPE == bfloat16 else 1e-5
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[ref],
                rtol=rtol,
                atol=atol,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_auto_broadcast=True,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
