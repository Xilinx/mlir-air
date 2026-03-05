# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized AveragePool Example

Implements 1D average pooling on a 2D input [M, N]:
  output[i] = mean(input[i, :]) for each row i

Each row of N elements is scaled by 1/N (vectorized multiply) and then
reduced to a single scalar using vector.reduction with ADD.

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
"""

import argparse
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, store, subview, collapse_shape
from air.dialects.vector import (
    transfer_read,
    reduction,
    CombiningKind,
    broadcast,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(m, n, tile_m, np_dtype_in):
    a_size = [m, n]
    out_size = [m]
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n > 0, "Pool width N must be positive"
    assert m % (tile_m * num_tiles) == 0
    index_type = IndexType.get()

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)
    l3outputMemrefTy = MemRefType.get(out_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = MemRefType.get(
        shape=[tile_m, n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )
    l1outputMemrefTy = MemRefType.get(
        shape=[tile_m, 1],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    @FuncOp.from_py_func(l3memrefTy, l3outputMemrefTy)
    def average_pool(arg0, arg2):
        @herd(
            name="herd_0",
            sizes=[1, num_tiles],
            operands=[arg0, arg2],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_a,
            _l3_c,
        ):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1outputMemrefTy, [], [])

            for _l_ivx in range_(0, m, tile_m * num_tiles):

                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(tile_m),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [_l_ivx, _ty])

                dma_memcpy_nd(
                    l1_a_data,
                    _l3_a,
                    src_offsets=[offset, 0],
                    src_sizes=[tile_m, n],
                    src_strides=[n, 1],
                )
                c0 = ConstantOp(index_type, 0)
                c1 = ConstantOp(index_type, 1)
                cTileN = ConstantOp(index_type, tile_m)
                inv_n = arith.ConstantOp(xrt_dtype_in, 1.0 / n)
                for j in range_(c0, cTileN, c1):
                    sub_a_vec = subview(
                        l1_a_data.result,
                        [j, c0],
                        [1, n],
                        [1, 1],
                    )
                    sub_c_vec = subview(
                        l1_out_data.result,
                        [j, c0],
                        [1, 1],
                        [1, 1],
                    )
                    layout = StridedLayoutAttr.get(
                        ShapedType.get_dynamic_size(),
                        [
                            1,
                        ],
                    )
                    collapsed_type = MemRefType.get(
                        (n,),
                        xrt_dtype_in,
                        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
                        layout=layout,
                    )
                    collapsed_type_2 = MemRefType.get(
                        (1,),
                        xrt_dtype_in,
                        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
                        layout=layout,
                    )
                    collapse_dims = [[0, 1]]
                    collapse_a = collapse_shape(
                        collapsed_type, sub_a_vec, collapse_dims
                    )
                    collapse_c = collapse_shape(
                        collapsed_type_2, sub_c_vec, collapse_dims
                    )
                    cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                    v_a = transfer_read(
                        VectorType.get([n], xrt_dtype_in),
                        collapse_a,
                        [c0],
                        AffineMapAttr.get(AffineMap.get_identity(1)),
                        cst0,
                        [True],
                    )
                    # Multiply by 1/N before reduction to avoid scalar bf16
                    # multiply which can produce corrupted output on AIE2.
                    v_inv_n = broadcast(VectorType.get([n], xrt_dtype_in), inv_n)
                    v_scaled = arith.mulf(v_a, v_inv_n)
                    v_avg = reduction(xrt_dtype_in, CombiningKind.ADD, v_scaled)
                    store(v_avg, collapse_c, [c0])
                    yield_([])

                dma_memcpy_nd(
                    _l3_c,
                    l1_out_data,
                    dst_offsets=[
                        offset,
                    ],
                    dst_sizes=[tile_m],
                    dst_strides=[1],
                )
                DeallocOp(l1_a_data)
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    # Default values.
    M = 65536
    N = 16
    TILE_M = 256
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the AveragePool example",
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
        "--m",
        type=int,
        default=M,
        help="Input size (dimension M)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Input size (dimension N, pool width)",
    )
    parser.add_argument("--tile-m", type=int, default=TILE_M, help="Tile size M")
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

    mlir_module = build_module(
        args.m,
        args.n,
        args.tile_m,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, (args.m * args.n), dtype=INPUT_DATATYPE).reshape(
        args.m, args.n
    )

    if args.compile_mode == "compile-and-run":

        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.m, num_samples)])

        # AveragePool reference: sum of (each element * 1/N) per row
        inv_n_bf16 = INPUT_DATATYPE(1.0 / args.n)
        sampled_values = np.array(
            [np.sum(input_a[i] * inv_n_bf16) for i in zip(*sampled_indices)],
            dtype=INPUT_DATATYPE,
        )

        sampled_data = {
            "shape": (args.m,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="average_pool",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
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
