# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Non-tile-aligned f32 matmul with bf16 computation on NPU1.
# Host data is f32. A is stored in K×M layout (same as test 54).
# L3→L2 DMA transposes A from K×M to M×K using f32 strides (4-byte aligned).
# A dedicated truncf herd converts f32→bf16 in L1 before the compute herd.
# This 4-herd pattern (prologue, truncf, compute, epilogue) avoids the
# problematic combined truncf+matmul pattern that fails on NPU1.
# Output is f32.
#
# Target: NPU1/Phoenix, aie2 architecture with native 4x8x4 bf16 matmul.

import argparse
import math
import os
import sys
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.linalg import fill
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store, subview
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt import compile_air, get_air_runtime
from air.backend.xrt_runner import type_mapper
from air.compiler.util import run_transform
import aie.utils
from air.extras import types as extrasT
from air.dialects.linalg.opdsl.lang import *
import air.dialects.linalg.opdsl.lang as linalg_lang

import numpy as np

np.random.seed(42)

range_ = for_


# Element-wise truncation: f32 → bf16
@linalg_structured_op()
def truncf_op(
    A=TensorDef(linalg_lang.TV.T1, S.a, S.b, S.c, S.d, S.e, S.f),
    B=TensorDef(linalg_lang.TV.T2, S.a, S.b, S.c, S.d, S.e, S.f, output=True),
):
    domain(D.a, D.b, D.c, D.d, D.e, D.f)
    B[D.a, D.b, D.c, D.d, D.e, D.f] = TypeFn.cast_signed(
        linalg_lang.TV.T2, A[D.a, D.b, D.c, D.d, D.e, D.f]
    )


@linalg_structured_op()
def block_matmul(
    A=TensorDef(linalg_lang.TV.T1, S.a, S.c, S.f, S.d, S.g, S.i),
    B=TensorDef(linalg_lang.TV.T2, S.b, S.c, S.e, S.f, S.i, S.h),
    C=TensorDef(linalg_lang.TV.U, S.b, S.a, S.e, S.d, S.g, S.h, output=True),
):
    domain(D.a, D.b, D.c, D.d, D.e, D.f, D.g, D.h, D.i)
    C[D.b, D.a, D.e, D.d, D.g, D.h] += (
        TypeFn.cast_signed(linalg_lang.TV.U, A[D.a, D.c, D.f, D.d, D.g, D.i])
    ) * (TypeFn.cast_signed(linalg_lang.TV.U, B[D.b, D.c, D.e, D.f, D.i, D.h]))


@module_builder
def build_module(
    m,
    k,
    n,
    m_alloc,
    n_alloc,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
):
    """Build matmul module with 4-herd pattern: prologue, truncf, compute, epilogue.

    L3 inputs are f32 in K×M / K×N layout. L3→L2 DMA transposes A to M×K.
    A dedicated truncf herd converts f32→bf16 in L1.
    The compute herd reads bf16 from L1 and runs block_matmul.
    This avoids the problematic combined truncf+matmul herd pattern on NPU1."""
    assert m % tile_m == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert n % tile_n == 0
    assert (
        tile_k_l2 == tile_k_l1
    ), "truncf herd approach requires tile_k_l2 == tile_k_l1"

    mmul_mkn = [4, 8, 4]  # aie2 native bf16 matmul

    xrt_dtype_f32 = type_mapper(np.float32)
    xrt_dtype_bf16 = type_mapper(bfloat16)

    # L3 MemRefTypes: A is K×M_alloc (f32), B is K×N_alloc (f32), C is M×N (f32)
    memrefTyA = MemRefType.get([k, m_alloc], xrt_dtype_f32)
    memrefTyB = MemRefType.get([k, n_alloc], xrt_dtype_f32)
    memrefTyOut = MemRefType.get([m, n], xrt_dtype_f32)

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)
    a_l1_size = [
        1,
        1,
        tile_k_l1 // mmul_mkn[1],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
        mmul_mkn[1],
    ]
    b_l1_size = [
        1,
        1,
        tile_n // mmul_mkn[2],
        tile_k_l1 // mmul_mkn[1],
        mmul_mkn[1],
        mmul_mkn[2],
    ]
    c_herd_l1_size = [
        herd_m,
        herd_n,
        tile_n // mmul_mkn[2],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
        mmul_mkn[2],
    ]

    # L1 buffers: f32 for DMA input, bf16 for matmul, f32 for output
    l1MemrefTyA_f32 = MemRefType.get(
        shape=a_l1_size, element_type=xrt_dtype_f32, memory_space=l1_mem_space
    )
    l1MemrefTyB_f32 = MemRefType.get(
        shape=b_l1_size, element_type=xrt_dtype_f32, memory_space=l1_mem_space
    )
    l1MemrefTyA_bf16 = MemRefType.get(
        shape=a_l1_size, element_type=xrt_dtype_bf16, memory_space=l1_mem_space
    )
    l1MemrefTyB_bf16 = MemRefType.get(
        shape=b_l1_size, element_type=xrt_dtype_bf16, memory_space=l1_mem_space
    )
    l1MemrefTyCHerd = MemRefType.get(
        shape=c_herd_l1_size, element_type=xrt_dtype_f32, memory_space=l1_mem_space
    )

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyOut)
    def matmul_f32(arg0, arg1, arg2):
        launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx, launch_ivy, launch_sizex, launch_sizey, l3_a, l3_b, l3_c
        ):
            @segment(
                name="matmul_seg",
                operands=[launch_ivx, launch_ivy, l3_a, l3_b, l3_c],
            )
            def segment_body(launch_ivx_s, launch_ivy_s, l3_a_s, l3_b_s, l3_c_s):
                l2_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L2)

                # L2 buffers: f32 (DMA from L3 is f32 for 4-byte stride alignment)
                l2MemrefTyA = MemRefType.get(
                    shape=[herd_m, 1, tile_m, tile_k_l2],
                    element_type=xrt_dtype_f32,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyB = MemRefType.get(
                    shape=[1, herd_n, tile_k_l2, tile_n],
                    element_type=xrt_dtype_f32,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyC = MemRefType.get(
                    shape=[herd_m, herd_n, tile_m, tile_n],
                    element_type=xrt_dtype_f32,
                    memory_space=l2_mem_space,
                )

                l2_a = AllocOp(l2MemrefTyA, [], [])
                l2_b = AllocOp(l2MemrefTyB, [], [])
                l2_c = AllocOp(l2MemrefTyC, [], [])
                l1_a_f32 = AllocOp(l1MemrefTyA_f32, [], [])
                l1_b_f32 = AllocOp(l1MemrefTyB_f32, [], [])
                l1_a_bf16 = AllocOp(l1MemrefTyA_bf16, [], [])
                l1_b_bf16 = AllocOp(l1MemrefTyB_bf16, [], [])
                l1_c = AllocOp(l1MemrefTyCHerd, [], [])

                # Launch offsets
                c_tile_m_herd_m = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_m * herd_m), None
                )
                c_tile_n_herd_n = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_n * herd_n), None
                )
                launch_offset_x = arith.MulIOp(launch_ivx_s, c_tile_m_herd_m)
                launch_offset_y = arith.MulIOp(launch_ivy_s, c_tile_n_herd_n)

                # Herd 1 (prologue): zero-fill C accumulator
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[
                        l1_a_f32,
                        l1_b_f32,
                        l1_a_bf16,
                        l1_b_bf16,
                        l1_c,
                        l2_a,
                        l2_b,
                    ],
                )
                def prologue_herd(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _af,
                    _bf,
                    _ab,
                    _bb,
                    _c,
                    _l2a,
                    _l2b,
                ):
                    l1_c_sv = subview(
                        _c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[
                            1,
                            1,
                            tile_n // mmul_mkn[2],
                            tile_m // mmul_mkn[0],
                            mmul_mkn[0],
                            mmul_mkn[2],
                        ],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype_f32, 0.0), None)
                    fill(zero_const, outs=[l1_c_sv])

                # K-reduction loop
                for i in range_(0, k // tile_k_l2):
                    reduction_l2_iv_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(tile_k_l2),
                            )
                        ],
                    )
                    reduction_offset = affine_apply(reduction_l2_iv_map, [i])

                    # L3→L2 DMA for A: TRANSPOSE K×M → M×K (f32).
                    # f32 stride=1 = 4 bytes satisfies shim DMA 4-byte alignment.
                    dma_memcpy_nd(
                        l2_a,
                        l3_a_s,
                        src_offsets=[0, 0, launch_offset_x, reduction_offset],
                        src_sizes=[herd_m, 1, tile_m, tile_k_l2],
                        src_strides=[tile_m, tile_k_l2 * m_alloc, 1, m_alloc],
                    )
                    # L3→L2 DMA for B: K×N layout (f32).
                    dma_memcpy_nd(
                        l2_b,
                        l3_b_s,
                        src_offsets=[0, 0, reduction_offset, launch_offset_y],
                        src_sizes=[1, herd_n, tile_k_l2, tile_n],
                        src_strides=[n_alloc * tile_k_l2, tile_n, n_alloc, 1],
                    )

                    # Herd 2 (truncf): DMA f32 L2→L1, convert f32→bf16 in L1
                    @herd(
                        name="herd_0",
                        sizes=[herd_m, herd_n],
                        operands=[
                            l1_a_f32,
                            l1_b_f32,
                            l1_a_bf16,
                            l1_b_bf16,
                            l1_c,
                            l2_a,
                            l2_b,
                        ],
                    )
                    def truncf_herd(
                        _tx,
                        _ty,
                        _sx,
                        _sy,
                        _l1_a_f32,
                        _l1_b_f32,
                        _l1_a_bf16,
                        _l1_b_bf16,
                        _l1_c,
                        _l2_a,
                        _l2_b,
                    ):
                        # DMA f32 L2→L1 for A
                        dma_memcpy_nd(
                            _l1_a_f32,
                            _l2_a,
                            src_offsets=[_tx, 0, 0, 0, 0, 0],
                            src_sizes=[
                                1,
                                1,
                                tile_k_l1 // mmul_mkn[1],
                                tile_m // mmul_mkn[0],
                                mmul_mkn[0],
                                mmul_mkn[1],
                            ],
                            src_strides=[
                                tile_m * tile_k_l2,
                                tile_m * tile_k_l2,
                                mmul_mkn[1],
                                tile_k_l2 * mmul_mkn[0],
                                tile_k_l2,
                                1,
                            ],
                        )
                        # DMA f32 L2→L1 for B
                        dma_memcpy_nd(
                            _l1_b_f32,
                            _l2_b,
                            src_offsets=[0, _ty, 0, 0, 0, 0],
                            src_sizes=[
                                1,
                                1,
                                tile_n // mmul_mkn[2],
                                tile_k_l1 // mmul_mkn[1],
                                mmul_mkn[1],
                                mmul_mkn[2],
                            ],
                            src_strides=[
                                herd_n * tile_n * tile_k_l2,
                                tile_n * tile_k_l2,
                                mmul_mkn[2],
                                tile_n * mmul_mkn[1],
                                tile_n,
                                1,
                            ],
                        )
                        # Convert f32→bf16 in L1
                        truncf_op(_l1_a_f32, outs=[_l1_a_bf16])
                        truncf_op(_l1_b_f32, outs=[_l1_b_bf16])

                    # Herd 3 (compute): read bf16 from L1, block_matmul
                    @herd(
                        name="herd_0",
                        sizes=[herd_m, herd_n],
                        operands=[
                            l1_a_f32,
                            l1_b_f32,
                            l1_a_bf16,
                            l1_b_bf16,
                            l1_c,
                            l2_a,
                            l2_b,
                        ],
                    )
                    def compute_herd(
                        _tx,
                        _ty,
                        _sx,
                        _sy,
                        _af,
                        _bf,
                        _l1_a,
                        _l1_b,
                        _l1_c,
                        _l2a,
                        _l2b,
                    ):
                        l1_c_sv = subview(
                            _l1_c,
                            offsets=[_tx, _ty, 0, 0, 0, 0],
                            sizes=[
                                1,
                                1,
                                tile_n // mmul_mkn[2],
                                tile_m // mmul_mkn[0],
                                mmul_mkn[0],
                                mmul_mkn[2],
                            ],
                            strides=[1, 1, 1, 1, 1, 1],
                        )
                        block_matmul(_l1_a, _l1_b, outs=[l1_c_sv])

                    yield_([])

                # Herd 4 (epilogue): write C from L1→L2
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[
                        l1_a_f32,
                        l1_b_f32,
                        l1_a_bf16,
                        l1_b_bf16,
                        l1_c,
                        l2_a,
                        l2_b,
                        l2_c,
                    ],
                )
                def epilogue_herd(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _af,
                    _bf,
                    _ab,
                    _bb,
                    _l1_c,
                    _l2a,
                    _l2b,
                    _l2_c,
                ):
                    dma_memcpy_nd(
                        _l2_c,
                        _l1_c,
                        dst_offsets=[_tx, _ty, 0, 0],
                        dst_sizes=[1, 1, tile_m, tile_n],
                        dst_strides=[
                            herd_n * tile_m * tile_n,
                            tile_m * tile_n,
                            tile_n,
                            1,
                        ],
                        src_offsets=[_tx, _ty, 0, 0, 0, 0],
                        src_sizes=[
                            1,
                            1,
                            tile_m // mmul_mkn[0],
                            mmul_mkn[0],
                            tile_n // mmul_mkn[2],
                            mmul_mkn[2],
                        ],
                        src_strides=[
                            herd_n * tile_m * tile_n,
                            tile_m * tile_n,
                            mmul_mkn[2] * mmul_mkn[0],
                            mmul_mkn[2],
                            tile_m * mmul_mkn[2],
                            1,
                        ],
                    )

                # L2→L3 DMA: write f32 output
                dma_memcpy_nd(
                    l3_c_s,
                    l2_c,
                    dst_offsets=[launch_offset_x, launch_offset_y],
                    dst_sizes=[herd_m * tile_m, herd_n * tile_n],
                    dst_strides=[n, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_m, tile_m, herd_n, tile_n],
                    src_strides=[tile_m * herd_n * tile_n, tile_n, tile_m * tile_n, 1],
                )

                DeallocOp(l2_a)
                DeallocOp(l2_b)
                DeallocOp(l2_c)
                DeallocOp(l1_a_f32)
                DeallocOp(l1_b_f32)
                DeallocOp(l1_a_bf16)
                DeallocOp(l1_b_bf16)
                DeallocOp(l1_c)


if __name__ == "__main__":
    # Default values
    M_actual = 500
    K = 784
    N_actual = 500
    TILE_M = 64
    TILE_K_L2 = 16
    TILE_K_L1 = 16
    TILE_N = 32
    HERD_M = 4
    HERD_N = 4

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Non-tile-aligned f32 matmul with bf16 computation on NPU1",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_actual)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--N", type=int, default=N_actual)
    parser.add_argument(
        "--k-l2-tile",
        type=int,
        default=TILE_K_L2,
        dest="k_l2_tile",
        help="L2 K-dimension tile size (K must be a multiple of this)",
    )
    parser.add_argument("--herd-m", type=int, default=HERD_M, dest="herd_m")
    parser.add_argument("--herd-n", type=int, default=HERD_N, dest="herd_n")
    args = parser.parse_args()

    M_actual = args.M
    K = args.K
    N_actual = args.N
    TILE_K_L2 = args.k_l2_tile
    TILE_K_L1 = args.k_l2_tile
    HERD_M = args.herd_m
    HERD_N = args.herd_n

    # Pad M, N to tile-aligned for the module
    M_padded = math.ceil(M_actual / (TILE_M * HERD_M)) * (TILE_M * HERD_M)
    N_padded = math.ceil(N_actual / (TILE_N * HERD_N)) * (TILE_N * HERD_N)

    if args.verbose:
        print(f"M_actual={M_actual}, N_actual={N_actual}, K={K}")
        print(f"M_padded={M_padded}, N_padded={N_padded}")

    # Host-side padding: alloc at tile-aligned size.
    M_alloc = M_padded
    N_alloc = N_padded

    mlir_module = build_module(
        M_padded,
        K,
        N_padded,
        M_alloc,
        N_alloc,
        TILE_M,
        TILE_K_L2,
        TILE_K_L1,
        TILE_N,
        HERD_M,
        HERD_N,
    )

    # Vectorization transform: tile truncf and block_matmul for vectorization.
    # 4 herds → split_handle produces 4 handles.
    # Truncf herd (herd2) has 2 truncf_op generics.
    # Compute herd (herd3) has 1 block_matmul generic.
    transform_ir_string = """
        module attributes {transform.with_named_sequence} {
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

            %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %func0 {
                transform.apply_patterns.linalg.tiling_canonicalization
                transform.apply_patterns.scf.for_loop_canonicalization
                transform.apply_patterns.canonicalization
            } : !transform.any_op
            %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

            // Match 2 truncf_ops + 1 block_matmul (3 linalg.generics total)
            %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %truncf_a_g, %truncf_b_g, %matmul = transform.split_handle %all_generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            // Tile truncf_ops to [1,1,0,0] for vectorization
            %tiled_truncf_a, %truncf_a_loops:2 =
              transform.structured.tile_using_for %truncf_a_g tile_sizes [1, 1, 0, 0]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
            %tiled_truncf_b, %truncf_b_loops:2 =
              transform.structured.tile_using_for %truncf_b_g tile_sizes [1, 1, 0, 0]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            // Tile block_matmul for vectorization [2,2,1,0,0,0] then unroll
            %inner_most_matmul, %vec_loops:3 =
              transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
            %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
              transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
            transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !transform.any_op
            transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !transform.any_op

            %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %inner_most_fills, %vec_fill_loops:2 =
              transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            // Vectorize all herds (4 herds: prologue, truncf, compute, epilogue)
            %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op
            %herd1, %herd2, %herd3, %herd4 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

            %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %func1 {
                transform.apply_patterns.linalg.tiling_canonicalization
                transform.apply_patterns.scf.for_loop_canonicalization
                transform.apply_patterns.canonicalization
                transform.apply_patterns.memref.fold_memref_alias_ops
            } : !transform.any_op
            %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

            %func1_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op

            // Re-vectorize after cleanup
            %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
            %h1, %h2, %h3, %h4 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

            // No vector_type_cast needed — accumulator is already f32.
            // The arith.extf on bf16 inputs before vector.contract will be
            // fused into aievec.matmul by convert-vector-to-aievec in aircc.

            %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %func2 {
                transform.apply_patterns.linalg.tiling_canonicalization
                transform.apply_patterns.scf.for_loop_canonicalization
                transform.apply_patterns.canonicalization
                transform.apply_patterns.memref.fold_memref_alias_ops
            } : !transform.any_op
            %func_fold_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %func_folded_3 = transform.air.fold_unit_extent_dims %func_fold_3 : (!transform.any_op) -> !transform.any_op
          transform.yield
        }
        }
    """
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    transform_ir = Module.parse(transform_ir_string, context=mlir_module.context)
    run_transform(transform_ir, mlir_module)

    if args.verbose:
        module_str = str(mlir_module)
        print(
            f"After vectorization: {module_str.count('arith.extf')} extf, "
            f"{module_str.count('arith.truncf')} truncf, "
            f"{module_str.count('vector.contract')} contracts"
        )

    # Host data: f32 in K×M / K×N layout. Zero-padded beyond M_actual/N_actual.
    input_a = np.zeros((K, M_alloc), dtype=np.float32)
    input_a[:, :M_actual] = (np.random.rand(K, M_actual) * 4).astype(np.float32)
    input_b = np.zeros((K, N_alloc), dtype=np.float32)
    input_b[:, :N_actual] = (np.random.rand(K, N_actual) * 4).astype(np.float32)

    num_samples = 100
    sampled_indices = np.vstack(
        [
            np.random.randint(0, M_actual, num_samples),
            np.random.randint(0, N_actual, num_samples),
        ]
    )

    # Add boundary-tile samples
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

    # Golden: truncate f32→bf16 (matching hardware truncf), compute dot product
    # with f32 accumulation. A is K×M so golden uses A[:, i] for row i of result.
    golden_a_bf16 = input_a.astype(bfloat16)
    golden_b_bf16 = input_b.astype(bfloat16)
    sampled_values = np.array(
        [
            np.sum(
                golden_a_bf16[:, i].astype(np.float32)
                * golden_b_bf16[:, j].astype(np.float32),
                dtype=np.float32,
            )
            for i, j in zip(*sampled_indices)
        ],
        dtype=np.float32,
    )

    sampled_data = {
        "shape": (M_padded, N_padded),
        "indices": sampled_indices,
        "values": sampled_values,
    }

    npu_kernel = compile_air(
        mlir_module,
        verbose=args.verbose,
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[4, 4],
        instance_name="matmul_f32",
    )
    runtime = get_air_runtime()
    io_args = [
        aie.utils.tensor(input_a),
        aie.utils.tensor(input_b),
        aie.utils.tensor(np.zeros((M_padded, N_padded), np.float32)),
    ]
    exit(
        runtime.run_test(
            npu_kernel,
            io_args,
            refs={},
            stochastic_refs=[sampled_data],
            rtol=0.1,
            max_mismatch_percentage=10,
        )
    )
