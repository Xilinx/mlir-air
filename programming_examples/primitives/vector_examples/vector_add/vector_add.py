# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
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

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16):
    a_size = [n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)
    l1MemrefTy = make_l1_memref([tile_n], xrt_dtype_in)
    vecTy = make_vec_type(VECTOR_SIZE, xrt_dtype_in)
    imap = identity_map_1d()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def vector_add(arg0, arg1, arg2):
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
            _l3_a,
            _l3_b,
            _l3_c,
        ):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
            l1_b_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset = tiled_1d_offset(_l_ivx, _ty, tile_n)

                dma_memcpy_nd(
                    l1_a_data,
                    _l3_a,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_b_data,
                    _l3_b,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                for j in range_(c0, cTileN, cVecSize):
                    v_a = vec_read(l1_a_data, j, VECTOR_SIZE, c0, vecTy, cst0, imap)
                    v_b = vec_read(l1_b_data, j, VECTOR_SIZE, c0, vecTy, cst0, imap)
                    v_c = arith.AddFOp(v_a, v_b)
                    vec_write(v_c, l1_out_data, j, VECTOR_SIZE, c0, imap)
                    yield_([])

                dma_memcpy_nd(
                    _l3_c,
                    l1_out_data,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16
    INPUT_DATATYPE = bfloat16

    parser = make_air_parser("Builds, runs, and tests the passthrough_dma example")
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
        help="Vector size for SIMD operations",
    )
    parser.add_argument(
        "--bf16-emulation",
        dest="bf16_emulation",
        default=False,
        action="store_true",
        help="Use f32 input data type and emulate f32 vector arithmetic using bf16 operations.",
    )

    args = parser.parse_args()

    if args.bf16_emulation:
        INPUT_DATATYPE = np.float32
    bf16_emulation = args.bf16_emulation

    mlir_module = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        args.vector_size,
    )
    check_print_module(mlir_module, args)

    input_a = np.arange(0, args.n, dtype=np.int64)
    input_a = input_a.astype(INPUT_DATATYPE)
    input_b = np.arange(0, args.n, dtype=np.int64)
    input_b = input_b.astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        sampled_data = stochastic_check(
            [input_a, input_b], args.n, lambda a, b: a + b, INPUT_DATATYPE
        )
        runner = make_xrt_runner(
            args, "vector_add", bf16_emulation=bf16_emulation
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=5e-2 if bf16_emulation else 1e-3,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = make_xrt_backend(args, bf16_emulation=bf16_emulation)
        module_function = backend.compile(mlir_module)

        backend.unload()
