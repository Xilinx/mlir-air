# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized AXPY Example

Implements the AXPY operation on 1D vectors [N]:
  y = a * x + y

where a is a scalar and x, y are vectors.

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.fma (fused multiply-add)
with configurable VECTOR_SIZE (default 16).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.vector import BroadcastOp, fma
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu
from utils import vec_read, vec_write

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, alpha=2.0, vector_size=16):
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    assert tile_n % vector_size == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get([n], xrt_dtype_in)
    l1MemrefTy = l1_memref_type([tile_n], xrt_dtype_in)
    vecTy = vec_type(VECTOR_SIZE, xrt_dtype_in)
    imap = identity_map_attr()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def axpy(arg0, arg1, arg2):
        # arg0 = x (input), arg1 = y (input), arg2 = output

        @herd(
            name="herd_0",
            sizes=[1, num_tiles],
            operands=[arg0, arg1, arg2],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_x,
            _l3_y,
            _l3_out,
        ):
            l1_x_data = AllocOp(l1MemrefTy, [], [])
            l1_y_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset = tile_offset_1d(_l_ivx, _ty, tile_n)

                dma_memcpy_nd(
                    l1_x_data,
                    _l3_x,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_y_data,
                    _l3_y,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)

                # Broadcast scalar alpha to vector
                a_const = arith.ConstantOp(xrt_dtype_in, alpha)
                v_a = BroadcastOp(vecTy, a_const)

                for j in range_(c0, cTileN, cVecSize):
                    v_x = vec_read(l1_x_data, j, VECTOR_SIZE, c0, vecTy, cst0, imap)
                    v_y = vec_read(l1_y_data, j, VECTOR_SIZE, c0, vecTy, cst0, imap)
                    # a * x + y via vector.fma
                    v_result = fma(v_a, v_x, v_y)
                    vec_write(v_result, l1_out_data, j, VECTOR_SIZE, c0, imap)
                    yield_([])

                # Write result from l1_out back to L3 output buffer
                dma_memcpy_nd(
                    _l3_out,
                    l1_out_data,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_x_data)
                DeallocOp(l1_y_data)
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16
    ALPHA = 2.0

    parser = make_air_parser("Builds, runs, and tests the AXPY example")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--alpha", type=float, default=ALPHA, help="Scalar multiplier a"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=16,
        help="Vector size for SIMD operations",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.n, args.tile_n, INPUT_DATATYPE, args.alpha, args.vector_size
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_x = np.random.randn(args.n).astype(INPUT_DATATYPE)
    input_y = np.random.randn(args.n).astype(INPUT_DATATYPE)

    sampled_indices = np.vstack([np.random.randint(0, args.n, 100)])
    sampled_values = np.array(
        [args.alpha * input_x[i] + input_y[i] for i in zip(*sampled_indices)],
        dtype=INPUT_DATATYPE,
    )
    sampled_data = {"shape": (args.n,), "indices": sampled_indices, "values": sampled_values}

    exit(run_on_npu(
        args, mlir_module,
        inputs=[input_x, input_y],
        instance_name="axpy",
        stochastic_expected_outputs=[sampled_data],
        rtol=1e-2,
    ))
