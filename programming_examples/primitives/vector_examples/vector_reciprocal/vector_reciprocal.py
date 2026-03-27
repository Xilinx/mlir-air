# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.vector import broadcast
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
    check_print_module,
)

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, arch="aie2"):
    a_size = [n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    # Architecture-specific vector size
    arch_vector_sizes = {
        "aie2": 16,
        "aie2p": 32,
    }
    VECTOR_SIZE = arch_vector_sizes.get(arch, 16)  # default to 16 if unknown
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)
    l1MemrefTy = make_l1_memref([tile_n], xrt_dtype_in)
    vecTy = make_vec_type(VECTOR_SIZE, xrt_dtype_in)
    imap = identity_map_1d()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def vector_reciprocal(arg0, arg1):

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
            _l3_a,
            _l3_c,
        ):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
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

                c0 = ConstantOp(index_type, 0)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)

                # Create constant 1.0 scalar and broadcast to vector
                one_scalar = arith.ConstantOp(xrt_dtype_in, 1.0)
                one_vector = broadcast(vecTy, one_scalar)

                for j in range_(c0, cTileN, cVecSize):
                    cst0 = arith.ConstantOp(xrt_dtype_in, 0.0)
                    v_a = vec_read(l1_a_data, j, VECTOR_SIZE, c0, vecTy, cst0, imap)
                    # Compute reciprocal: 1.0 / a
                    v_c = arith.DivFOp(one_vector, v_a)
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
                DeallocOp(l1_out_data)

                yield_([])


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = np.float32

    parser = make_air_parser(
        "Builds, runs, and tests the vector reciprocal (1/x) example"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="Target AIE architecture (aie2 or aie2p)",
    )
    args = parser.parse_args()

    mlir_module = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        args.arch,
    )
    check_print_module(mlir_module, args)

    # Generate random input vector with fixed seed for reproducibility
    np.random.seed(37)
    # Use a safe range [1, 10] to avoid division by zero
    input_a = np.random.uniform(1.0, 10.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.n, num_samples),
            ]
        )
        # Compute reference results for sampled indices: 1.0 / x
        sampled_values = np.array(
            [np.float32(1.0) / np.float32(input_a[i]) for i in sampled_indices[0]],
            dtype=INPUT_DATATYPE,
        )
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }
        runner = make_xrt_runner(args, "vector_reciprocal")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-5,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = make_xrt_backend(args)
        module_function = backend.compile(mlir_module)

        backend.unload()
