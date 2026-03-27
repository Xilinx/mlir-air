# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Version 2: f32 scalar rsqrt in loop
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
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store, subview
from air.dialects.func import FuncOp
from air.dialects.math import rsqrt
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper
from utils import (
    make_l1_memref,
    tiled_1d_offset,
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
    VECTOR_SIZE = 16
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)
    l1MemrefTy = make_l1_memref([tile_n], xrt_dtype_in)

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def vector_rsqrt(arg0, arg2):
        # For aie2, link with external function
        herd_kwargs = {
            "name": "herd_0",
            "sizes": [1, num_tiles],
            "operands": [arg0, arg2],
        }
        if arch == "aie2":
            herd_kwargs["link_with"] = "extern_func.o"

        @herd(**herd_kwargs)
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
                c1 = ConstantOp(index_type, 1)
                cVecSize = ConstantOp(index_type, VECTOR_SIZE)
                cTileN = ConstantOp(index_type, tile_n)
                for j in range_(c0, cTileN, cVecSize):
                    sub_a_vec = subview(
                        l1_a_data.result,
                        [j],
                        [VECTOR_SIZE],
                        [1],
                    )
                    sub_c_vec = subview(
                        l1_out_data.result,
                        [j],
                        [VECTOR_SIZE],
                        [1],
                    )

                    # Scalar loop implementation
                    for elem_i in range_(c0, cVecSize, c1):
                        # Load the input value from tile_in
                        elem = load(sub_a_vec, [elem_i])
                        rsqrt_out = rsqrt(elem)

                        # Store the output value in tile_out
                        store(rsqrt_out, sub_c_vec, [elem_i])
                        yield_([])

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
    N = 512
    TILE_N = 64
    INPUT_DATATYPE = np.float32

    parser = make_air_parser(
        "Builds, runs, and tests the vector_rsqrt example (Version 2: f32 scalar in loop)"
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

    # Version 2 (scalar rsqrt in loop) is not supported on aie2
    if args.arch == "aie2":
        print(
            "Error: Version 2 (scalar rsqrt in loop) is not supported on aie2 architecture."
        )
        print("Please use aie2p architecture: --arch aie2p")
        exit(1)

    mlir_module = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        args.arch,
    )
    check_print_module(mlir_module, args)

    # Generate input values in range [0.1, 3.0] to match working testbench pattern
    np.random.seed(10)
    input_a = np.abs(np.random.uniform(0.1, 3.0, args.n)).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.n, num_samples),
            ]
        )
        sampled_values = np.array(
            [1.0 / np.sqrt(input_a[i]) for i in sampled_indices[0]],
            dtype=INPUT_DATATYPE,
        )
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }
        runner = make_xrt_runner(args, "vector_rsqrt")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = make_xrt_backend(args)
        module_function = backend.compile(mlir_module)

        backend.unload()
