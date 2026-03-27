# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Element-Wise Add

Implements element-wise addition on a 1D input [N]:
  c = a + b

Uses a 1xnum_tiles AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write with
configurable VECTOR_SIZE (default 16 for BF16, 8 for F32).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu

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

    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)
    l1MemrefTy = l1_memref_type([tile_n], xrt_dtype_in)

    # Vectorization setup
    vectorize = vector_size > 0
    if vectorize:
        assert (
            tile_n % vector_size == 0
        ), f"tile_n ({tile_n}) must be divisible by vector_size ({vector_size})"
        vecTy = vec_type(vector_size, xrt_dtype_in)
        imap = identity_map_attr()
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
                            vecTy, sub_a, [c0], imap, cst0, [True]
                        )
                        v_b = transfer_read(
                            vecTy, sub_b, [c0], imap, cst0, [True]
                        )
                        v_c = arith.AddFOp(v_a, v_b)
                        transfer_write(None, v_c, sub_c, [c0], imap, [True])
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

    parser = make_air_parser("Builds, runs, and tests the eltwise_add example")
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
    args = parser.parse_args()

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

    input_a = np.random.uniform(0, 4, args.n).astype(INPUT_DATATYPE)
    input_b = np.random.uniform(0, 4, args.n).astype(INPUT_DATATYPE)

    # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
    num_samples = 100
    sampled_indices = np.vstack(
        [
            np.random.randint(0, args.n, num_samples),  # i indices
        ]
    )

    # Compute reference results for sampled indices
    sampled_values = np.array(
        [input_a[i] + input_b[i] for i in zip(*sampled_indices)],
        dtype=INPUT_DATATYPE,
    )

    # Store as a dictionary
    sampled_data = {
        "shape": (args.n),
        "indices": sampled_indices,
        "values": sampled_values,
    }

    # BF16 has ~0.8% relative precision; use looser tolerance
    rtol = 0.01 if INPUT_DATATYPE == bfloat16 else 1e-3
    exit(run_on_npu(
        args, mlir_module,
        inputs=[input_a, input_b],
        instance_name="eltwise_add",
        stochastic_expected_outputs=[sampled_data],
        rtol=rtol,
    ))
