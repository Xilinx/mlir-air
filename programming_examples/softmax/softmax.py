# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from math import cos, sin, sqrt, exp

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper, make_air_parser, run_on_npu
from ml_dtypes import bfloat16

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(n, tile_n, herd_n, np_dtype_in):
    assert n % (tile_n * herd_n) == 0
    a_size = [n]
    out_size = a_size
    xrt_dtype_in = type_mapper(np_dtype_in)

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = l1_memref_type([tile_n], xrt_dtype_in)

    # Function declaration
    softmax_func = FuncOp(
        "softmax_bf16",
        ([l1MemrefTy, T.i32(), l1MemrefTy], []),
        visibility="private",
    )
    for func in [softmax_func]:
        func.attributes["link_with"] = StringAttr.get("softmax.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy)
    def softmax(arg0, arg1):
        @herd(
            name="herd_0",
            sizes=[1, herd_n],
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

            for t in range_(0, n, tile_n * herd_n):

                offset = tile_offset_1d(t, _ty, tile_n)

                dma_memcpy_nd(
                    l1_a_data,
                    _l3_a,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                const_pos = ConstantOp(IntegerAttr.get(T.i32(), tile_n - 1), None)
                softmax_call = CallOp(
                    softmax_func,
                    [l1_a_data, const_pos, l1_out_data],
                )
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
                DeallocOp(l1_out_data)

                yield_([])

        herd_body.attributes["link_with"] = StringAttr.get("softmax.o")


if __name__ == "__main__":
    # Default values.
    N = 1024
    TILE_N = 256
    HERD_N = 4
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
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.n,
        args.tile_n,
        args.herd_n,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Softmax
    num_tiles = args.n // args.tile_n
    inputs = np.random.randn(num_tiles, args.tile_n).astype(INPUT_DATATYPE)
    outputs = np.zeros(shape=(num_tiles, args.tile_n), dtype=INPUT_DATATYPE)

    max_val = np.max(inputs)
    for j in range(num_tiles):
        sum_val = 0.0
        for i in range(args.tile_n):
            outputs[j][i] = exp(inputs[j][i] - max_val)
            sum_val += outputs[j][i]
        for i in range(args.tile_n):
            outputs[j][i] = outputs[j][i] / sum_val

    run_on_npu(
        args,
        mlir_module,
        inputs=[inputs],
        expected_outputs=[outputs],
        instance_name="softmax",
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[4, 4],
        rtol=1e-1,
    )
