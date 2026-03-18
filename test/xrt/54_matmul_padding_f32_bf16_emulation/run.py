# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# F32 matmul with bf16/bfp16 emulation using @module_builder approach.
# All host data is f32. A is stored in K×M layout (transposed).
# DMA carries f32 data L3→L2→L1 (f32 strides satisfy 4-byte DMA alignment).
# L3→L2 DMA transposes A from K×M to M×K layout.
# Inside the herd, truncf_op converts f32→bf16 before block_matmul.
# Output is f32.
#
# Target: NPU2/Strix, aie2p architecture.

import argparse
import os
import sys

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects.linalg import fill
from air.dialects.affine import apply as affine_apply
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.compiler.util import run_transform
from air.extras import types as extrasT
from air.dialects.linalg.opdsl.lang import *
import air.dialects.linalg.opdsl.lang as linalg_lang
from ml_dtypes import bfloat16

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


# Packed block matmul: bf16 inputs, f32 accumulation
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
    """Build matmul module. m/n are padded (tile-aligned) dimensions for the
    launch grid. m_alloc/n_alloc are the actual host buffer sizes (block-aligned)
    used for DMA strides. The air.actual_sizes attribute is added after building."""
    assert m % tile_m == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert n % tile_n == 0

    xrt_dtype_f32 = type_mapper(np.float32)
    xrt_dtype_bf16 = type_mapper(bfloat16)
    mmul_mkn = [8, 8, 8]  # aie2p

    # L3 uses actual alloc sizes (not padded). A is K×M_alloc, B is K×N_alloc.
    # C uses padded sizes (output shim DMAs write full tiles).
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
    c_l1_size = [
        1,
        1,
        tile_n // mmul_mkn[2],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
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

    # L1: f32 for DMA input, bf16 for matmul, f32 for output
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

    layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(),
        [
            tile_m * tile_n * herd_n,
            tile_m * tile_n,
            tile_m * mmul_mkn[2],
            mmul_mkn[0] * mmul_mkn[2],
            mmul_mkn[2],
            1,
        ],
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
                name="matmul_seg", operands=[launch_ivx, launch_ivy, l3_a, l3_b, l3_c]
            )
            def segment_body(launch_ivx_s, launch_ivy_s, l3_a_s, l3_b_s, l3_c_s):
                l2_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L2)

                # L2: f32 for inputs (f32 strides satisfy 4-byte DMA alignment for transpose).
                # A stored in M×K layout (transposed from L3 K×M by shim DMA).
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

                # Compute launch offsets using arith.muli (required by
                # air-split-launch-for-padding which matches this pattern).
                c_tile_m_herd_m = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_m * herd_m), None
                )
                c_tile_n_herd_n = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_n * herd_n), None
                )
                launch_offset_x = arith.MulIOp(launch_ivx_s, c_tile_m_herd_m)
                launch_offset_y = arith.MulIOp(launch_ivy_s, c_tile_n_herd_n)

                # Prologue herd: zero-fill C accumulator
                @herd(name="herd_0", sizes=[herd_m, herd_n], operands=[l1_c])
                def prologue_herd(_tx, _ty, _sx, _sy, _l1_c):
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

                    # L3→L2 DMA for A: TRANSPOSE K×M → M×K.
                    # L3 A[k, m] at offset k*M + m (K×M, f32).
                    # Read as [herd_m, 1, tile_m, tile_k_l2] where M is inner (stride=1),
                    # K is outer (stride=m). This transposes K×M to M×K in L2.
                    # f32 stride=1 = 4 bytes: satisfies >=4-byte alignment.
                    dma_memcpy_nd(
                        l2_a,
                        l3_a_s,
                        src_offsets=[0, 0, launch_offset_x, reduction_offset],
                        src_sizes=[herd_m, 1, tile_m, tile_k_l2],
                        src_strides=[tile_m, tile_k_l2 * m_alloc, 1, m_alloc],
                    )
                    # L3→L2 DMA for B: K×N (same as original).
                    dma_memcpy_nd(
                        l2_b,
                        l3_b_s,
                        src_offsets=[0, 0, reduction_offset, launch_offset_y],
                        src_sizes=[1, herd_n, tile_k_l2, tile_n],
                        src_strides=[n_alloc * tile_k_l2, tile_n, n_alloc, 1],
                    )

                    # Compute herd: DMA f32 L2→L1, truncf to bf16, block_matmul
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
                        _l1_a_f32,
                        _l1_b_f32,
                        _l1_a_bf16,
                        _l1_b_bf16,
                        _l1_c,
                        _l2_a,
                        _l2_b,
                    ):
                        for j in range_(0, tile_k_l2 // tile_k_l1):
                            reduction_l1_iv_map = AffineMap.get(
                                0,
                                1,
                                [
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(tile_k_l1),
                                    )
                                ],
                            )
                            reduction_l1_offset = affine_apply(reduction_l1_iv_map, [j])

                            # L2→L1 DMA for A: f32, M×K layout at L2 (already transposed).
                            # Same strides as original bf16 example.
                            dma_memcpy_nd(
                                _l1_a_f32,
                                _l2_a,
                                src_offsets=[_tx, 0, 0, 0, 0, reduction_l1_offset],
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
                            # L2→L1 DMA for B: f32 (same as original)
                            dma_memcpy_nd(
                                _l1_b_f32,
                                _l2_b,
                                src_offsets=[0, _ty, 0, 0, reduction_l1_offset, 0],
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

                            # Truncf f32→bf16 in core
                            truncf_op(_l1_a_f32, outs=[_l1_a_bf16])
                            truncf_op(_l1_b_f32, outs=[_l1_b_bf16])

                            # Block matmul: bf16 inputs, f32 accumulation
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
                            block_matmul(_l1_a_bf16, _l1_b_bf16, outs=[l1_c_sv])
                            yield_([])

                    yield_([])

                # Epilogue herd: write C from L1→L2
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a_f32, l1_b_f32, l1_c, l2_c],
                )
                def epilogue_herd(_tx, _ty, _sx, _sy, _l1_a, _l1_b, _l1_c, _l2_c):
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
    import math

    # Actual (non-tile-aligned) dimensions, same as test 53
    M_actual = 500
    K = 784
    N_actual = 500
    TILE_M = 64
    TILE_K_L2 = 16  # 784 % 16 == 0
    TILE_K_L1 = 16
    TILE_N = 32
    HERD_M = 4
    HERD_N = 4

    parser = argparse.ArgumentParser(
        prog="run.py", description="F32 matmul with bf16 emulation, A in K×M layout"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_actual)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--N", type=int, default=N_actual)
    parser.add_argument(
        "--transform-script",
        type=str,
        dest="transform_script",
        default=None,
        help="Path to transform script MLIR file (overrides inline transform)",
    )
    parser.add_argument(
        "--k-l2-tile",
        type=int,
        default=TILE_K_L2,
        dest="k_l2_tile",
        help="L2 K-dimension tile size (K must be a multiple of this)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    args = parser.parse_args()

    M_actual = args.M
    K = args.K
    N_actual = args.N
    TILE_K_L2 = args.k_l2_tile
    TILE_K_L1 = args.k_l2_tile

    # Pad M, N to tile-aligned for the module
    M_padded = math.ceil(M_actual / (TILE_M * HERD_M)) * (TILE_M * HERD_M)
    N_padded = math.ceil(N_actual / (TILE_N * HERD_N)) * (TILE_N * HERD_N)

    if args.verbose:
        print(f"M_actual={M_actual}, N_actual={N_actual}, K={K}")
        print(f"M_padded={M_padded}, N_padded={N_padded}")

    # Block-aligned allocation sizes (only pad to innerBlockSize=8, not full tile).
    INNER_BLOCK = 8
    M_alloc = math.ceil(M_actual / INNER_BLOCK) * INNER_BLOCK
    N_alloc = math.ceil(N_actual / INNER_BLOCK) * INNER_BLOCK

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

    # Add actual_sizes attribute to air.launch for device-side padding.
    # air-split-launch-for-padding reads this to split boundary blocks.
    import air.passmanager

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={M_padded // TILE_M // HERD_M},{N_padded // TILE_N // HERD_N},1 actual-sizes={M_actual},{N_actual},1}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
            ]
        )
        + ")"
    )
    # Note: can't use air-wrap-func-with-parallel because air.launch already
    # exists. Instead, manually set the attribute on the existing launch.
    with mlir_module.context:
        for op in mlir_module.body.operations:
            for inner_op in op.body.blocks[0].operations:
                if inner_op.name == "air.launch":
                    inner_op.attributes["air.actual_sizes"] = DenseI64ArrayAttr.get(
                        [M_actual, N_actual, 1]
                    )
                    break

    # Vectorization transform: tile block_matmul for vectorization, vectorize
    # herds, cast vector types, hoist transfers and extf/truncf pairs.
    # Adapted from programming_examples/matrix_multiplication/bf16/run.py
    # direct_codegen transform. The compute herd (herd2) has truncf_op +
    # block_matmul; we match block_matmul by annotation attribute.
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

            // Match 2 truncf_ops + 1 block_matmul
            %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %truncf_a_g, %truncf_b_g, %matmul = transform.split_handle %all_generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            // Tile truncf_ops to [1,1,0,0] → vector<8x8> for aievec
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

            // Vectorize all herds
            %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op

            %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

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

            // Re-vectorize after cleanup, then hoist transfers
            %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
            %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
            %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

            // Cast vector.contract accumulator types (bf16→f32 for matmul)
            %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

            %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
            %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
            %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

            // Hoist extf/truncf pairs from the innermost loop.
            // The compute herd has truncf_op (f32→bf16) + block_matmul (bf16→f32 cast).
            // After vectorization, there are 4 extf and 4 truncf from the matmul contracts,
            // plus additional truncf from truncf_op. We hoist the 4 matmul pairs.
            %fors_to_hoist = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
            %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)
            %all_extf = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
            %all_truncf = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op

            // Skip extf/truncf hoisting for now — the truncf_op adds extra
            // cast ops that change the count. The matmul will still vectorize
            // correctly; hoisting is a performance optimization.

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

    # Use external transform script if provided, otherwise use inline transform.
    if args.transform_script:
        with open(args.transform_script, "r") as f:
            transform_ir_string = f.read()

    transform_ir = Module.parse(transform_ir_string, context=mlir_module.context)
    run_transform(transform_ir, mlir_module)

    if args.verbose:
        # Count extf/truncf in the vectorized module
        module_str = str(mlir_module)
        print(
            f"After vectorization: {module_str.count('arith.extf')} extf, {module_str.count('arith.truncf')} truncf, {module_str.count('vector.contract')} contracts"
        )

    # Host data: f32. A is K×M_alloc (transposed, block-aligned actual size).
    # B is K×N_alloc. Zero-padded beyond M_actual/N_actual.
    # Device-side padding (air-split-launch-for-padding) handles boundary tiles.
    #
    # Use random inputs scaled by 4 (matching IRON methodology from PR #1440).
    input_a = np.zeros((K, M_alloc), dtype=np.float32)
    input_a[:, :M_actual] = (np.random.rand(K, M_actual) * 4).astype(np.float32)
    input_b = np.zeros((K, N_alloc), dtype=np.float32)
    input_b[:, :N_actual] = (np.random.rand(K, N_actual) * 4).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        num_samples = 200
        sampled_indices = np.vstack(
            [
                np.random.randint(0, M_actual, num_samples),
                np.random.randint(0, N_actual, num_samples),
            ]
        )

        # Add deterministic boundary-tile samples to catch padding errors.
        # These sample the last few rows/cols of each boundary herd.
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

        # Golden: truncate f32 inputs to bf16 (matching hardware truncf_op),
        # then compute dot product with f32 accumulation. This eliminates the
        # f32-vs-bf16 rounding bias that inflated the tolerance (PR #1440).
        input_a_bf16 = input_a.astype(bfloat16)
        input_b_bf16 = input_b.astype(bfloat16)
        sampled_values = np.array(
            [
                np.sum(
                    input_a_bf16[:, i].astype(np.float32)
                    * input_b_bf16[:, j].astype(np.float32),
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

        needs_padding = (M_actual % TILE_M != 0) or (N_actual % TILE_N != 0)
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf" if needs_padding else "xclbin",
            instance_name="matmul_f32",
            bf16_emulation=True,
        )
        # With bf16-truncated golden, remaining error is from BFP16 block
        # floating point quantization and floor rounding in direct-codegen.
        # Tolerance matches PR #1440 (rtol=0.05, atol=4); requires
        # mlir-aie#2987 (conv_even rounding) for full precision.
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=0.05,
                atol=4,
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            bf16_emulation=True,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
