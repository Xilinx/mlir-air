# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in):
    a_size = [n]
    b_size = a_size
    out_size = a_size
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)

    # L2 MemRefTypes
    l2MemrefTy = l2_memref_type(a_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = l1_memref_type([tile_n], xrt_dtype_in)

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def eltwise_add(arg0, arg1, arg2):

        launch_size = [1, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            arg0_l,
            arg1_l,
            arg2_l,
        ):
            @segment(name="segment_0", operands=[arg0_l, arg1_l, arg2_l])
            def segment_body(
                arg0_s,
                arg1_s,
                arg2_s,
            ):

                l2_a_data = AllocOp(l2MemrefTy, [], [])

                l2_b_data = AllocOp(l2MemrefTy, [], [])
                l2_out_data = AllocOp(l2MemrefTy, [], [])

                dma_memcpy_nd(
                    l2_a_data,
                    arg0_s,
                )
                dma_memcpy_nd(
                    l2_b_data,
                    arg1_s,
                )

                @herd(
                    name="herd_0",
                    sizes=[1, num_tiles],
                    operands=[l2_a_data, l2_b_data, l2_out_data],
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

                        offset = tile_offset_1d(_l_ivx, _ty, tile_n)

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

                dma_memcpy_nd(
                    arg2_s,
                    l2_out_data,
                )
                DeallocOp(l2_a_data)
                DeallocOp(l2_b_data)
                DeallocOp(l2_out_data)


if __name__ == "__main__":
    # Default values.
    N = 16384
    TILE_N = 1024
    INPUT_DATATYPE = np.float32

    parser = make_air_parser("Builds, runs, and tests the passthrough_dma example")
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")

    args = parser.parse_args()

    mlir_module = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, args.n, dtype=np.int64).reshape(args.n)
    input_a = input_a.astype(INPUT_DATATYPE)
    input_b = np.arange(0, args.n, dtype=np.int64).reshape(args.n)
    input_b = input_b.astype(INPUT_DATATYPE)

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

    run_on_npu(
        args,
        mlir_module,
        inputs=[input_a, input_b],
        stochastic_expected_outputs=[sampled_data],
        instance_name="eltwise_add",
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[4, 4],
        rtol=1e-3,
    )
