# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Leaky RELU Example

Implements element-wise Leaky RELU on a 1D input [N]:
  y = x if x >= 0 else alpha * x

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write with
configurable VECTOR_SIZE (default 16).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

np.random.seed(42)
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.vector import BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu
from utils import vec_read, vec_write

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, alpha=0.01, vector_size=16):
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

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def leaky_relu(arg0, arg1):
        @herd(
            name="herd_0",
            sizes=[1, num_tiles],
            operands=[arg0, arg1],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_in,
            _l3_out,
        ):
            l1_in_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset = tile_offset_1d(_l_ivx, _ty, tile_n)

                dma_memcpy_nd(
                    l1_in_data,
                    _l3_in,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)

                v_zero = BroadcastOp(vecTy, cst0)
                alpha_const = arith.ConstantOp(xrt_dtype_in, alpha)
                v_alpha = BroadcastOp(vecTy, alpha_const)

                for j in range_(c0, cTileN, cVecSize):
                    v_x = vec_read(l1_in_data, j, VECTOR_SIZE, c0, vecTy, cst0, imap)
                    v_alpha_x = arith.MulFOp(v_x, v_alpha)
                    # Leaky RELU: x >= 0 ? x : alpha*x
                    cmp = arith.CmpFOp(arith.CmpFPredicate.OGE, v_x, v_zero)
                    v_result = arith.SelectOp(cmp, v_x, v_alpha_x)
                    vec_write(v_result, l1_out_data, j, VECTOR_SIZE, c0, imap)
                    yield_([])

                dma_memcpy_nd(
                    _l3_out,
                    l1_out_data,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_in_data)
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16
    ALPHA = 0.01

    parser = make_air_parser("Builds, runs, and tests the Leaky RELU example")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--alpha", type=float, default=ALPHA, help="Leaky RELU slope for x < 0"
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

    # Mix of positive and negative values for Leaky RELU testing
    input_a = np.random.randn(args.n).astype(INPUT_DATATYPE)

    sampled_indices = np.vstack([np.random.randint(0, args.n, 100)])
    sampled_values = np.array(
        [np.where(input_a[i] >= 0, input_a[i], args.alpha * input_a[i]) for i in zip(*sampled_indices)],
        dtype=INPUT_DATATYPE,
    )
    sampled_data = {"shape": (args.n,), "indices": sampled_indices, "values": sampled_values}

    exit(run_on_npu(
        args, mlir_module,
        inputs=[input_a],
        instance_name="leaky_relu",
        stochastic_expected_outputs=[sampled_data],
        rtol=1e-2,
    ))
