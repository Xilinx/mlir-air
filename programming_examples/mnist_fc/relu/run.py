# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# 2D ReLU: C[i,j] = max(A[i,j], 0)
# Element-wise ReLU on a 2D matrix [M,N]. All data is f32.
#
# MNIST context: op #3 in the GGML MNIST-FC pipeline.
# Default dimensions: M=500, N=500 (hidden layer activation).

import argparse
import math
import numpy as np

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu
from air.extras import types as extrasT

np.random.seed(42)

range_ = for_


@module_builder
def build_module(m, n, tile_m, tile_n, herd_m, herd_n, vector_size=16):
    """Build 2D ReLU module.

    m, n are padded (tile-aligned) dimensions for the launch grid.
    The air.actual_sizes attribute is added after building to handle
    non-tile-aligned actual dimensions.
    """
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert tile_n % vector_size == 0

    xrt_dtype_f32 = type_mapper(np.float32)
    xrt_dtype_bf16 = type_mapper(bfloat16)
    index_type = IndexType.get()
    l1_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)

    # L3 MemRefTypes (f32 for host interface)
    memrefTyA = MemRefType.get([m, n], xrt_dtype_f32)
    memrefTyOut = MemRefType.get([m, n], xrt_dtype_f32)

    # L1 MemRefTypes
    l1TileTy_f32 = MemRefType.get(
        shape=[tile_m, tile_n],
        element_type=xrt_dtype_f32,
        memory_space=l1_mem_space,
    )
    l1TileTy_bf16 = MemRefType.get(
        shape=[tile_m, tile_n],
        element_type=xrt_dtype_bf16,
        memory_space=l1_mem_space,
    )

    vecTy_f32 = VectorType.get([vector_size], xrt_dtype_f32)
    vecTy_bf16 = VectorType.get([vector_size], xrt_dtype_bf16)
    # Rank-reduced subview types for vector transfer_read/write
    l1SubviewTy_f32 = MemRefType.get(
        [vector_size],
        xrt_dtype_f32,
        layout=StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [1]),
        memory_space=l1_mem_space,
    )
    l1SubviewTy_bf16 = MemRefType.get(
        [vector_size],
        xrt_dtype_bf16,
        layout=StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [1]),
        memory_space=l1_mem_space,
    )

    @FuncOp.from_py_func(memrefTyA, memrefTyOut)
    def relu(arg_a, arg_out):
        launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

        @launch(operands=[arg_a, arg_out], sizes=launch_size)
        def launch_body(
            launch_ivx, launch_ivy, launch_sizex, launch_sizey, l3_a, l3_out
        ):

            @segment(
                name="relu_seg",
                operands=[launch_ivx, launch_ivy, l3_a, l3_out],
            )
            def segment_body(launch_ivx_s, launch_ivy_s, l3_a_s, l3_out_s):
                c_tile_m_herd_m = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_m * herd_m), None
                )
                c_tile_n_herd_n = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_n * herd_n), None
                )
                launch_offset_m = arith.MulIOp(launch_ivx_s, c_tile_m_herd_m)
                launch_offset_n = arith.MulIOp(launch_ivy_s, c_tile_n_herd_n)

                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[
                        launch_offset_m,
                        launch_offset_n,
                        l3_a_s,
                        l3_out_s,
                    ],
                )
                def herd_body(tx, ty, _sx, _sy, _loff_m, _loff_n, _l3_a, _l3_out):
                    l1_tile_in = AllocOp(l1TileTy_f32, [], [])
                    l1_tile_out = AllocOp(l1TileTy_f32, [], [])
                    l1_tile_bf16 = AllocOp(l1TileTy_bf16, [], [])

                    # m_offset = launch_offset_m + tx * tile_m
                    # n_offset = launch_offset_n + ty * tile_n
                    m_offset_map = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(tile_m),
                                ),
                            )
                        ],
                    )
                    n_offset_map = AffineMap.get(
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
                    m_offset = affine_apply(m_offset_map, [_loff_m, tx])
                    n_offset = affine_apply(n_offset_map, [_loff_n, ty])

                    # DMA matrix tile in
                    dma_memcpy_nd(
                        l1_tile_in,
                        _l3_a,
                        src_offsets=[m_offset, n_offset],
                        src_sizes=[tile_m, tile_n],
                        src_strides=[n, 1],
                    )

                    # Compute: max(x, 0) via bf16 comparison + select.
                    # AIE2P has no f32 vector cmp/sel; truncf to bf16,
                    # do cmp/sel in bf16, extf result back to f32.
                    c0 = ConstantOp(index_type, 0)
                    c1 = ConstantOp(index_type, 1)
                    c_vec_size = ConstantOp(index_type, vector_size)
                    c_tile_m_cst = ConstantOp(index_type, tile_m)
                    c_tile_n_cst = ConstantOp(index_type, tile_n)
                    cst0_f32 = arith.ConstantOp(xrt_dtype_f32, 0.0)
                    cst0_bf16 = arith.ConstantOp(xrt_dtype_bf16, 0.0)
                    v_zero_bf16 = BroadcastOp(vecTy_bf16, cst0_bf16)
                    identity_map_1d = AffineMapAttr.get(AffineMap.get_identity(1))

                    for i in range_(c0, c_tile_m_cst, c1):
                        for j in range_(c0, c_tile_n_cst, c_vec_size):
                            sub_in_f32 = subview(
                                l1_tile_in.result,
                                [i, j],
                                [1, vector_size],
                                [1, 1],
                                result_type=l1SubviewTy_f32,
                            )
                            sub_in_bf16 = subview(
                                l1_tile_bf16.result,
                                [i, j],
                                [1, vector_size],
                                [1, 1],
                                result_type=l1SubviewTy_bf16,
                            )
                            sub_out_f32 = subview(
                                l1_tile_out.result,
                                [i, j],
                                [1, vector_size],
                                [1, 1],
                                result_type=l1SubviewTy_f32,
                            )

                            # Read f32, truncate to bf16
                            v_f32 = transfer_read(
                                vecTy_f32,
                                sub_in_f32,
                                [c0],
                                identity_map_1d,
                                cst0_f32,
                                [True],
                            )
                            v_bf16 = arith.TruncFOp(vecTy_bf16, v_f32)
                            # Write bf16 to L1 temp
                            transfer_write(
                                None,
                                v_bf16,
                                sub_in_bf16,
                                [c0],
                                identity_map_1d,
                                [True],
                            )

                            # ReLU in bf16: x > 0 ? x : 0
                            v_bf16_read = transfer_read(
                                vecTy_bf16,
                                sub_in_bf16,
                                [c0],
                                identity_map_1d,
                                cst0_bf16,
                                [True],
                            )
                            cmp = arith.CmpFOp(
                                arith.CmpFPredicate.OGT,
                                v_bf16_read,
                                v_zero_bf16,
                            )
                            v_relu_bf16 = arith.SelectOp(cmp, v_bf16_read, v_zero_bf16)

                            # Extend back to f32 and write output
                            v_relu_f32 = arith.ExtFOp(vecTy_f32, v_relu_bf16)
                            transfer_write(
                                None,
                                v_relu_f32,
                                sub_out_f32,
                                [c0],
                                identity_map_1d,
                                [True],
                            )
                            yield_([])
                        yield_([])

                    # DMA output tile back to L3
                    dma_memcpy_nd(
                        _l3_out,
                        l1_tile_out,
                        dst_offsets=[m_offset, n_offset],
                        dst_sizes=[tile_m, tile_n],
                        dst_strides=[n, 1],
                    )

                    DeallocOp(l1_tile_in)
                    DeallocOp(l1_tile_out)
                    DeallocOp(l1_tile_bf16)


if __name__ == "__main__":
    M_ACTUAL = 500
    N_ACTUAL = 500
    TILE_M = 64
    TILE_N = 32
    HERD_M = 1
    HERD_N = 4
    VECTOR_SIZE = 16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="2D ReLU: C[i,j] = max(A[i,j], 0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_ACTUAL, help="Number of rows")
    parser.add_argument("--N", type=int, default=N_ACTUAL, help="Number of columns")
    parser.add_argument("--tile-m", type=int, default=TILE_M)
    parser.add_argument("--tile-n", type=int, default=TILE_N)
    parser.add_argument("--herd-m", type=int, default=HERD_M)
    parser.add_argument("--herd-n", type=int, default=HERD_N)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )

    args = parser.parse_args()

    M_actual = args.M
    N_actual = args.N
    TILE_M = args.tile_m
    TILE_N = args.tile_n
    HERD_M = args.herd_m
    HERD_N = args.herd_n
    VECTOR_SIZE = args.vector_size

    # Pad to tile-aligned dimensions
    M_padded = math.ceil(M_actual / (TILE_M * HERD_M)) * (TILE_M * HERD_M)
    N_padded = math.ceil(N_actual / (TILE_N * HERD_N)) * (TILE_N * HERD_N)

    if args.verbose:
        print(f"M_actual={M_actual}, N_actual={N_actual}")
        print(f"M_padded={M_padded}, N_padded={N_padded}")
        print(f"TILE_M={TILE_M}, TILE_N={TILE_N}, HERD_M={HERD_M}, HERD_N={HERD_N}")

    mlir_module = build_module(
        M_padded, N_padded, TILE_M, TILE_N, HERD_M, HERD_N, VECTOR_SIZE
    )

    # Add actual_sizes attribute for device-side padding
    needs_padding = (M_actual != M_padded) or (N_actual != N_padded)
    if needs_padding:
        with mlir_module.context:
            for op in mlir_module.body.operations:
                for inner_op in op.body.blocks[0].operations:
                    if inner_op.name == "air.launch":
                        inner_op.attributes["air.actual_sizes"] = DenseI64ArrayAttr.get(
                            [M_actual, N_actual, 1]
                        )
                        break

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Host data: mix of positive and negative values
    input_a = np.zeros((M_padded, N_padded), dtype=np.float32)
    input_a[:M_actual, :N_actual] = (np.random.randn(M_actual, N_actual) * 4).astype(
        np.float32
    )

    if args.compile_mode == "compile-and-run":
        # Golden reference: max(x, 0)
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, M_actual, num_samples),
                np.random.randint(0, N_actual, num_samples),
            ]
        )

        # Add boundary samples
        boundary_m = list(
            set(
                [
                    min(M_actual - 1, m)
                    for m in [M_actual - 1, M_actual - TILE_M + 1, 0]
                    if m >= 0
                ]
            )
        )
        boundary_n = list(
            set(
                [
                    min(N_actual - 1, n)
                    for n in [N_actual - 1, N_actual - TILE_N + 1, 0]
                    if n >= 0
                ]
            )
        )
        boundary_indices = np.array([[m, n] for m in boundary_m for n in boundary_n]).T
        sampled_indices = np.hstack([sampled_indices, boundary_indices])

        # Golden: truncate f32 to bf16 (matching hardware), relu in bf16,
        # then extend back to f32
        input_a_bf16 = input_a.astype(bfloat16)
        sampled_values = np.array(
            [max(float(input_a_bf16[i, j]), 0.0) for i, j in zip(*sampled_indices)],
            dtype=np.float32,
        )

        sampled_data = {
            "shape": (M_padded, N_padded),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf" if needs_padding else "xclbin",
            instance_name="relu",
            runtime_loop_tiling_sizes=[4, 4],
        )
        # bf16 truncation introduces rounding; use bf16-appropriate tolerance
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf" if needs_padding else "xclbin",
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
