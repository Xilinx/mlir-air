# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Broadcast bias add: C[row,col] = A[row,col] + bias[col]
# Adds a row vector (ne0-aligned) to every row of a matrix.
# All data is f32.
#
# GGML layout: [ne0, ne1] where ne0 is contiguous.
# GGML op: x+y: [ne0, ne1] + [ne0, 1] -> [ne0, ne1]
# In numpy row-major: matrix is (ne1, ne0), bias is (ne0,).
# The bias is along the contiguous (column) dimension, broadcast along rows.
#
# MNIST context: ops #2 and #5 in the GGML MNIST-FC pipeline.
# Op #2: [500,500] + [500,1] -> [500,500], bias has ne0=500 elements.

import argparse
import math
import numpy as np

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview, load
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu
from air.extras import types as extrasT

np.random.seed(42)

range_ = for_


@module_builder
def build_module(m, n, tile_m, tile_n, herd_m, herd_n, vector_size=16):
    """Build broadcast bias add module.

    m = ne1 (rows, padded), n = ne0 (cols, contiguous, padded).
    Bias has length n (ne0). C[row,col] = A[row,col] + bias[col].
    """
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert tile_n % vector_size == 0

    xrt_dtype = type_mapper(np.float32)
    index_type = IndexType.get()
    l1_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)

    # L3 MemRefTypes
    memrefTyA = MemRefType.get([m, n], xrt_dtype)
    # Bias has length n (ne0, contiguous dimension)
    memrefTyBias = MemRefType.get([n], xrt_dtype)
    memrefTyOut = MemRefType.get([m, n], xrt_dtype)

    # L1 MemRefTypes
    l1TileTy = MemRefType.get(
        shape=[tile_m, tile_n], element_type=xrt_dtype, memory_space=l1_mem_space
    )
    # Bias tile: tile_n elements (one per column in the tile)
    l1BiasTy = MemRefType.get(
        shape=[tile_n], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    vecTy = VectorType.get([vector_size], xrt_dtype)
    # Rank-reduced 1D subview type
    l1SubviewTy = MemRefType.get(
        [vector_size],
        xrt_dtype,
        layout=StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [1]),
        memory_space=l1_mem_space,
    )

    @FuncOp.from_py_func(memrefTyA, memrefTyBias, memrefTyOut)
    def broadcast_bias_add(arg_a, arg_bias, arg_out):
        launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

        @launch(operands=[arg_a, arg_bias, arg_out], sizes=launch_size)
        def launch_body(
            launch_ivx, launch_ivy, launch_sizex, launch_sizey, l3_a, l3_bias, l3_out
        ):

            @segment(
                name="bias_add_seg",
                operands=[launch_ivx, launch_ivy, l3_a, l3_bias, l3_out],
            )
            def segment_body(launch_ivx_s, launch_ivy_s, l3_a_s, l3_bias_s, l3_out_s):
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
                        l3_bias_s,
                        l3_out_s,
                    ],
                )
                def herd_body(
                    tx, ty, _sx, _sy, _loff_m, _loff_n, _l3_a, _l3_bias, _l3_out
                ):
                    l1_tile_in = AllocOp(l1TileTy, [], [])
                    l1_tile_out = AllocOp(l1TileTy, [], [])
                    l1_bias = AllocOp(l1BiasTy, [], [])

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

                    # DMA bias slice: bias[n_offset : n_offset + tile_n]
                    # Bias is along ne0 (columns, contiguous dimension).
                    dma_memcpy_nd(
                        l1_bias,
                        _l3_bias,
                        src_offsets=[n_offset],
                        src_sizes=[tile_n],
                        src_strides=[1],
                    )

                    # DMA matrix tile
                    dma_memcpy_nd(
                        l1_tile_in,
                        _l3_a,
                        src_offsets=[m_offset, n_offset],
                        src_sizes=[tile_m, tile_n],
                        src_strides=[n, 1],
                    )

                    # Compute: C[row,col] = A[row,col] + bias[col]
                    # Bias is along columns; load as vector, add to each row.
                    c0 = ConstantOp(index_type, 0)
                    c1 = ConstantOp(index_type, 1)
                    c_vec_size = ConstantOp(index_type, vector_size)
                    c_tile_m_cst = ConstantOp(index_type, tile_m)
                    c_tile_n_cst = ConstantOp(index_type, tile_n)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    identity_map_1d = AffineMapAttr.get(AffineMap.get_identity(1))

                    for i in range_(c0, c_tile_m_cst, c1):
                        for j in range_(c0, c_tile_n_cst, c_vec_size):
                            # Load bias vector (same for every row)
                            sub_bias = subview(
                                l1_bias.result,
                                [j],
                                [vector_size],
                                [1],
                            )
                            v_bias = transfer_read(
                                vecTy,
                                sub_bias,
                                [c0],
                                identity_map_1d,
                                cst0,
                                [True],
                            )

                            sub_in = subview(
                                l1_tile_in.result,
                                [i, j],
                                [1, vector_size],
                                [1, 1],
                                result_type=l1SubviewTy,
                            )
                            sub_out = subview(
                                l1_tile_out.result,
                                [i, j],
                                [1, vector_size],
                                [1, 1],
                                result_type=l1SubviewTy,
                            )
                            v_in = transfer_read(
                                vecTy, sub_in, [c0], identity_map_1d, cst0, [True]
                            )
                            v_out = arith.AddFOp(v_in, v_bias)
                            transfer_write(
                                None, v_out, sub_out, [c0], identity_map_1d, [True]
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
                    DeallocOp(l1_bias)


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
        description="Broadcast bias add: C[row,col] = A[row,col] + bias[col]",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_ACTUAL, help="Number of rows (ne1)")
    parser.add_argument(
        "--N", type=int, default=N_ACTUAL, help="Number of columns (ne0)"
    )
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
        print(f"M_actual={M_actual} (ne1), N_actual={N_actual} (ne0)")
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

    # Host data: matrix (ne1 x ne0), bias (ne0,)
    input_a = np.zeros((M_padded, N_padded), dtype=np.float32)
    input_a[:M_actual, :N_actual] = (np.random.randn(M_actual, N_actual) * 4).astype(
        np.float32
    )
    # Bias along ne0 (columns, contiguous dimension)
    input_bias = np.zeros(N_padded, dtype=np.float32)
    input_bias[:N_actual] = (np.random.randn(N_actual) * 2).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        # Golden: C[row,col] = A[row,col] + bias[col]
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

        # bias indexed by column (ne0)
        sampled_values = np.array(
            [input_a[i, j] + input_bias[j] for i, j in zip(*sampled_indices)],
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
            instance_name="broadcast_bias_add",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_bias],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-6,
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
