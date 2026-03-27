# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized Tanh Example

Computes element-wise tanh on a 1D bf16 input [N] using the AIE2P
hardware tanh intrinsic (__builtin_aie2p_tanh).

Lowering chain: math.tanh -> aievec.tanh -> xllvm.intr.aie2p.tanh

Uses a 1x2 AIE herd with DMA transfers between L3 and L1 memory.
Computation is vectorized using vector.transfer_read/write.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper
from utils import (
    make_l1_memref,
    make_vec_type,
    identity_map_1d,
    tiled_1d_offset,
    vec_read,
    vec_write,
    make_air_parser,
    make_xrt_runner,
    make_xrt_backend,
    stochastic_check,
    check_print_module,
)

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16):
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    assert tile_n % vector_size == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get([n], xrt_dtype_in)
    l1MemrefTy = make_l1_memref([tile_n], xrt_dtype_in)
    vecTy = make_vec_type(VECTOR_SIZE, xrt_dtype_in)
    imap = identity_map_1d()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def vector_tanh(arg0, arg1):

        @herd(name="herd_0", sizes=[1, num_tiles], operands=[arg0, arg1])
        def herd_body(_tx, _ty, _sx, _sy, _l3_in, _l3_out):
            l1_in = AllocOp(l1MemrefTy, [], [])
            l1_out = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset = tiled_1d_offset(_l_ivx, _ty, tile_n)

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

                for j in range_(c0, cTileN, cVecSize):
                    v_in = vec_read(l1_in, j, VECTOR_SIZE, c0, vecTy, cst0, imap)

                    # Hardware tanh intrinsic on AIE2P
                    v_out = math_dialect.tanh(v_in)

                    vec_write(v_out, l1_out, j, VECTOR_SIZE, c0, imap)
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
    VECTOR_SIZE = 16
    INPUT_DATATYPE = bfloat16

    parser = make_air_parser("Builds, runs, and tests the vectorized tanh example")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2p",
        help="Target AIE architecture (aie2 or aie2p)",
    )

    args = parser.parse_args()

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE, args.vector_size)
    check_print_module(mlir_module, args)

    np.random.seed(42)
    input_a = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        # Reference: compute tanh in f32 precision
        def tanh_ref(x):
            return np.tanh(x.astype(np.float32))

        sampled_data = stochastic_check([input_a], args.n, tanh_ref, INPUT_DATATYPE)
        runner = make_xrt_runner(args, "vector_tanh")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = make_xrt_backend(args)
        module_function = backend.compile(mlir_module)
        backend.unload()
