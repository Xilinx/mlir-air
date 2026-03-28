# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized GELU (Tanh Approximation) Example

Implements element-wise GELU on a 1D input [N] using the standard
tanh approximation:
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Uses the hardware tanh intrinsic (__builtin_aie2p_tanh) directly,
matching the IRON project's GELU implementation. No exp or division
needed.

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.vector import BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper, make_air_parser, run_on_npu
from utils import vec_read, vec_write

range_ = for_

SQRT_2_OVER_PI = 0.7978845608  # sqrt(2/pi)
GELU_BETA = 0.044715


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16):
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
    def gelu(arg0, arg1):

        @herd(name="herd_0", sizes=[1, num_tiles], operands=[arg0, arg1])
        def herd_body(_tx, _ty, _sx, _sy, _l3_in, _l3_out):
            l1_in = AllocOp(l1MemrefTy, [], [])
            l1_out = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset = tile_offset_1d(_l_ivx, _ty, tile_n)

                dma_memcpy_nd(
                    l1_in,
                    _l3_in,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                half_const = arith.ConstantOp(xrt_dtype_in, 0.5)
                one_const = arith.ConstantOp(xrt_dtype_in, 1.0)
                beta_const = arith.ConstantOp(xrt_dtype_in, GELU_BETA)
                s2opi_const = arith.ConstantOp(xrt_dtype_in, SQRT_2_OVER_PI)
                v_half = BroadcastOp(vecTy, half_const)
                v_one = BroadcastOp(vecTy, one_const)
                v_beta = BroadcastOp(vecTy, beta_const)
                v_s2opi = BroadcastOp(vecTy, s2opi_const)

                for j in range_(c0, cTileN, cVecSize):
                    v_x = vec_read(l1_in, j, VECTOR_SIZE, c0, vecTy, cst0, imap)

                    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    # Uses hardware tanh intrinsic — no exp or division needed.
                    v_x2 = arith.mulf(v_x, v_x)
                    v_x3 = arith.mulf(v_x, v_x2)
                    v_beta_x3 = arith.mulf(v_x3, v_beta.result)
                    v_inner = arith.addf(v_x, v_beta_x3)
                    v_scaled = arith.mulf(v_inner, v_s2opi.result)
                    v_tanh = math_dialect.tanh(v_scaled)
                    v_one_plus_tanh = arith.addf(v_tanh, v_one.result)
                    v_half_x = arith.mulf(v_x, v_half.result)
                    v_gelu = arith.mulf(v_half_x, v_one_plus_tanh)

                    vec_write(v_gelu, l1_out, j, VECTOR_SIZE, c0, imap)
                    yield_([])

                dma_memcpy_nd(
                    _l3_out,
                    l1_out,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_in)
                DeallocOp(l1_out)
                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = make_air_parser("Builds, runs, and tests the GELU example")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size", type=int, default=16, help="Vector size for SIMD operations"
    )
    args = parser.parse_args()

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE, args.vector_size)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    input_a = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)

    # Match hardware bf16 computation: each op truncates to bf16
    def gelu_ref(x):
        x_bf = INPUT_DATATYPE(x)
        x2 = INPUT_DATATYPE(np.float32(x_bf) * np.float32(x_bf))
        x3 = INPUT_DATATYPE(np.float32(x_bf) * np.float32(x2))
        beta_x3 = INPUT_DATATYPE(np.float32(x3) * np.float32(INPUT_DATATYPE(GELU_BETA)))
        inner = INPUT_DATATYPE(np.float32(x_bf) + np.float32(beta_x3))
        scaled = INPUT_DATATYPE(
            np.float32(inner) * np.float32(INPUT_DATATYPE(SQRT_2_OVER_PI))
        )
        tanh_val = INPUT_DATATYPE(np.tanh(np.float32(scaled)))
        one_plus_tanh = INPUT_DATATYPE(
            np.float32(tanh_val) + np.float32(INPUT_DATATYPE(1.0))
        )
        half_x = INPUT_DATATYPE(np.float32(x_bf) * np.float32(INPUT_DATATYPE(0.5)))
        return INPUT_DATATYPE(np.float32(half_x) * np.float32(one_plus_tanh))

    sampled_indices = np.vstack([np.random.randint(0, args.n, 100)])
    sampled_values = np.array(
        [gelu_ref(input_a[i]) for i in zip(*sampled_indices)],
        dtype=INPUT_DATATYPE,
    )
    sampled_data = {
        "shape": (args.n,),
        "indices": sampled_indices,
        "values": sampled_values,
    }

    exit(
        run_on_npu(
            args,
            mlir_module,
            inputs=[input_a],
            instance_name="gelu",
            stochastic_expected_outputs=[sampled_data],
            rtol=1e-1,
            atol=5e-2,
        )
    )
