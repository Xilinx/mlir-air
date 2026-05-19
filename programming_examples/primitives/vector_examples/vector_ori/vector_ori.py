# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

import numpy as np

np.random.seed(42)

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, vector_size=16):
    a_size = [n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    num_tiles = 2
    assert n % (tile_n * num_tiles) == 0
    assert tile_n % vector_size == 0
    VECTOR_SIZE = vector_size
    index_type = IndexType.get()

    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def vector_ori(arg0, arg1, arg2):
        @herd(
            name="herd_0",
            sizes=[1, num_tiles],
            operands=[arg0, arg1, arg2],
        )
        def herd_body(_tx, _ty, _sx, _sy, _l3_a, _l3_b, _l3_c):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
            l1_b_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            for _l_ivx in range_(0, n, tile_n * num_tiles):
                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(tile_n),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [_l_ivx, _ty])

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
                for j in range_(c0, cTileN, cVecSize):
                    sub_a_vec = subview(l1_a_data.result, [j], [VECTOR_SIZE], [1])
                    sub_b_vec = subview(l1_b_data.result, [j], [VECTOR_SIZE], [1])
                    sub_c_vec = subview(l1_out_data.result, [j], [VECTOR_SIZE], [1])
                    pad = arith.ConstantOp(xrt_dtype_in, 0)
                    v_a = transfer_read(
                        VectorType.get([VECTOR_SIZE], xrt_dtype_in),
                        sub_a_vec,
                        [c0],
                        AffineMapAttr.get(AffineMap.get_identity(1)),
                        pad,
                        [True],
                    )
                    v_b = transfer_read(
                        VectorType.get([VECTOR_SIZE], xrt_dtype_in),
                        sub_b_vec,
                        [c0],
                        AffineMapAttr.get(AffineMap.get_identity(1)),
                        pad,
                        [True],
                    )
                    v_c = arith.OrIOp(v_a, v_b)
                    transfer_write(
                        None,
                        v_c,
                        sub_c_vec,
                        [c0],
                        AffineMapAttr.get(AffineMap.get_identity(1)),
                        [True],
                    )
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
    INPUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Per-element vector arith.ori over int32 tiles on the AIE.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations (16 lanes of i32 = 512 bits)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )

    args = parser.parse_args()

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE, args.vector_size)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    rng = np.random.default_rng(42)
    input_a = rng.integers(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=args.n,
        dtype=INPUT_DATATYPE,
    )
    input_b = rng.integers(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=args.n,
        dtype=INPUT_DATATYPE,
    )

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])
        sampled_values = np.array(
            [input_a[i] | input_b[i] for i in zip(*sampled_indices)],
            dtype=INPUT_DATATYPE,
        )
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="vector_ori",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        backend.compile(mlir_module)
        backend.unload()
