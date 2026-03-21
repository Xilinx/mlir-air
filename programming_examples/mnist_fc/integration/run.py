# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MNIST-FC inference integration test.
# Chains: matmul1(500x500x784) -> bias_add+relu -> bias_add2 -> argmax
# in a single multi-launch module on NPU2.
#
# Data layout: test 54's matmul outputs (M, N) with N contiguous.
# Bias is per-row (axis=0). Argmax reduces along axis=0.
#
# Pipeline (4 launches):
#   Launch 1: matmul1      W1[K,M1] x X[K,N1] -> C1[M1,N1]     (784,500 x 784,500 -> 500,500)
#   Launch 2: bias_add2    C3[i,j] = matmul2_out[i,j] + bias2[i] (10x500)
#   Launch 3: argmax       out[j]  = argmax_i(C3[i,j])            (10x500 -> 500)
#   Launch 4: bias+relu    C2[i,j] = max(C1[i,j]+bias1[i], 0)    (500x500)
#
# Strategy: build matmul module from test54, apply its transform,
# then parse and extend with element-wise launches.

import argparse
import math
import os
import sys
import numpy as np

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview, load, store
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.compiler.util import run_transform
from air.extras import types as extrasT

# Import test 54's matmul builder and structured ops
test54_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "test",
    "xrt",
    "54_matmul_padding_f32_bf16_emulation",
)
sys.path.insert(0, test54_dir)
from run import (
    build_module as build_matmul_module,
    truncf_op,
    block_matmul,
)  # noqa: E402

sys.path.pop(0)

np.random.seed(42)

range_ = for_

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_offset_map(tile_size):
    """Return affine_map: (s0, s1) -> s0 + s1 * tile_size."""
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1),
                    AffineConstantExpr.get(tile_size),
                ),
            )
        ],
    )


def build_bias_add_launch(
    l3_in,
    l3_bias,
    l3_out,
    m,
    n,
    tile_m,
    tile_n,
    herd_m,
    herd_n,
    vector_size,
    xrt_dtype,
    l1_mem_space,
    name_suffix="",
):
    """Emit a bias_add launch: out[i,j] = in[i,j] + bias[i].
    Bias per-row (axis=0), broadcast along cols."""
    index_type = IndexType.get()
    vecTy = VectorType.get([vector_size], xrt_dtype)
    l1TileTy = MemRefType.get(
        shape=[tile_m, tile_n], element_type=xrt_dtype, memory_space=l1_mem_space
    )
    l1BiasTy = MemRefType.get(
        shape=[tile_m], element_type=xrt_dtype, memory_space=l1_mem_space
    )
    l1SubviewTy = MemRefType.get(
        [vector_size],
        xrt_dtype,
        layout=StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [1]),
        memory_space=l1_mem_space,
    )
    launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

    @launch(operands=[l3_in, l3_bias, l3_out], sizes=launch_size)
    def launch_body(livx, livy, lsx, lsy, _in, _bias, _out):
        @segment(
            name=f"bias_add_seg{name_suffix}", operands=[livx, livy, _in, _bias, _out]
        )
        def seg(livx_s, livy_s, in_s, bias_s, out_s):
            c_tmhm = ConstantOp(IntegerAttr.get(IndexType.get(), tile_m * herd_m), None)
            c_tnhn = ConstantOp(IntegerAttr.get(IndexType.get(), tile_n * herd_n), None)
            loff_m = arith.MulIOp(livx_s, c_tmhm)
            loff_n = arith.MulIOp(livy_s, c_tnhn)

            @herd(
                name="herd_0",
                sizes=[herd_m, herd_n],
                operands=[loff_m, loff_n, in_s, bias_s, out_s],
            )
            def herd_body(tx, ty, _sx, _sy, _lm, _ln, _l3i, _l3b, _l3o):
                l1i = AllocOp(l1TileTy, [], [])
                l1o = AllocOp(l1TileTy, [], [])
                l1b = AllocOp(l1BiasTy, [], [])
                m_off = affine_apply(make_offset_map(tile_m), [_lm, tx])
                n_off = affine_apply(make_offset_map(tile_n), [_ln, ty])
                dma_memcpy_nd(
                    l1b, _l3b, src_offsets=[m_off], src_sizes=[tile_m], src_strides=[1]
                )
                dma_memcpy_nd(
                    l1i,
                    _l3i,
                    src_offsets=[m_off, n_off],
                    src_sizes=[tile_m, tile_n],
                    src_strides=[n, 1],
                )
                c0 = ConstantOp(index_type, 0)
                c1 = ConstantOp(index_type, 1)
                cvs = ConstantOp(index_type, vector_size)
                ctm = ConstantOp(index_type, tile_m)
                ctn = ConstantOp(index_type, tile_n)
                cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                id1 = AffineMapAttr.get(AffineMap.get_identity(1))
                for i in range_(c0, ctm, c1):
                    bias_val = load(l1b, [i])
                    bias_vec = BroadcastOp(vecTy, bias_val)
                    for j in range_(c0, ctn, cvs):
                        si = subview(
                            l1i.result,
                            [i, j],
                            [1, vector_size],
                            [1, 1],
                            result_type=l1SubviewTy,
                        )
                        so = subview(
                            l1o.result,
                            [i, j],
                            [1, vector_size],
                            [1, 1],
                            result_type=l1SubviewTy,
                        )
                        v = transfer_read(vecTy, si, [c0], id1, cst0, [True])
                        vo = arith.AddFOp(v, bias_vec)
                        transfer_write(None, vo, so, [c0], id1, [True])
                        yield_([])
                    yield_([])
                dma_memcpy_nd(
                    _l3o,
                    l1o,
                    dst_offsets=[m_off, n_off],
                    dst_sizes=[tile_m, tile_n],
                    dst_strides=[n, 1],
                )
                DeallocOp(l1i)
                DeallocOp(l1o)
                DeallocOp(l1b)


def build_bias_relu_fused_launch(
    l3_in,
    l3_bias,
    l3_out,
    m,
    n,
    tile_m,
    tile_n,
    herd_m,
    herd_n,
    vector_size,
    xrt_dtype_f32,
    xrt_dtype_bf16,
    l1_mem_space,
):
    """Fused bias_add + relu: out[i,j] = max(in[i,j] + bias[i], 0)."""
    index_type = IndexType.get()
    vecTy_f32 = VectorType.get([vector_size], xrt_dtype_f32)
    vecTy_bf16 = VectorType.get([vector_size], xrt_dtype_bf16)
    l1Ty_f32 = MemRefType.get(
        shape=[tile_m, tile_n], element_type=xrt_dtype_f32, memory_space=l1_mem_space
    )
    l1Ty_bf16 = MemRefType.get(
        shape=[tile_m, tile_n], element_type=xrt_dtype_bf16, memory_space=l1_mem_space
    )
    l1BiasTy = MemRefType.get(
        shape=[tile_m], element_type=xrt_dtype_f32, memory_space=l1_mem_space
    )
    svTy_f32 = MemRefType.get(
        [vector_size],
        xrt_dtype_f32,
        layout=StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [1]),
        memory_space=l1_mem_space,
    )
    svTy_bf16 = MemRefType.get(
        [vector_size],
        xrt_dtype_bf16,
        layout=StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [1]),
        memory_space=l1_mem_space,
    )
    launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

    @launch(operands=[l3_in, l3_bias, l3_out], sizes=launch_size)
    def launch_body(livx, livy, lsx, lsy, _in, _bias, _out):
        @segment(name="bias_relu_seg", operands=[livx, livy, _in, _bias, _out])
        def seg(livx_s, livy_s, in_s, bias_s, out_s):
            c_tmhm = ConstantOp(IntegerAttr.get(IndexType.get(), tile_m * herd_m), None)
            c_tnhn = ConstantOp(IntegerAttr.get(IndexType.get(), tile_n * herd_n), None)
            loff_m = arith.MulIOp(livx_s, c_tmhm)
            loff_n = arith.MulIOp(livy_s, c_tnhn)

            @herd(
                name="herd_0",
                sizes=[herd_m, herd_n],
                operands=[loff_m, loff_n, in_s, bias_s, out_s],
            )
            def herd_body(tx, ty, _sx, _sy, _lm, _ln, _l3i, _l3b, _l3o):
                l1i = AllocOp(l1Ty_f32, [], [])
                l1o = AllocOp(l1Ty_f32, [], [])
                l1bf = AllocOp(l1Ty_bf16, [], [])
                l1b = AllocOp(l1BiasTy, [], [])
                m_off = affine_apply(make_offset_map(tile_m), [_lm, tx])
                n_off = affine_apply(make_offset_map(tile_n), [_ln, ty])
                dma_memcpy_nd(
                    l1b, _l3b, src_offsets=[m_off], src_sizes=[tile_m], src_strides=[1]
                )
                dma_memcpy_nd(
                    l1i,
                    _l3i,
                    src_offsets=[m_off, n_off],
                    src_sizes=[tile_m, tile_n],
                    src_strides=[n, 1],
                )
                c0 = ConstantOp(index_type, 0)
                c1 = ConstantOp(index_type, 1)
                cvs = ConstantOp(index_type, vector_size)
                ctm = ConstantOp(index_type, tile_m)
                ctn = ConstantOp(index_type, tile_n)
                cst0_f32 = arith.ConstantOp(xrt_dtype_f32, 0.0)
                cst0_bf16 = arith.ConstantOp(xrt_dtype_bf16, 0.0)
                vz_bf16 = BroadcastOp(vecTy_bf16, cst0_bf16)
                id1 = AffineMapAttr.get(AffineMap.get_identity(1))
                for i in range_(c0, ctm, c1):
                    bias_val = load(l1b, [i])
                    bias_vec = BroadcastOp(vecTy_f32, bias_val)
                    for j in range_(c0, ctn, cvs):
                        si = subview(
                            l1i.result,
                            [i, j],
                            [1, vector_size],
                            [1, 1],
                            result_type=svTy_f32,
                        )
                        sb = subview(
                            l1bf.result,
                            [i, j],
                            [1, vector_size],
                            [1, 1],
                            result_type=svTy_bf16,
                        )
                        so = subview(
                            l1o.result,
                            [i, j],
                            [1, vector_size],
                            [1, 1],
                            result_type=svTy_f32,
                        )
                        vf = transfer_read(vecTy_f32, si, [c0], id1, cst0_f32, [True])
                        vbiased = arith.AddFOp(vf, bias_vec)
                        vb = arith.TruncFOp(vecTy_bf16, vbiased)
                        transfer_write(None, vb, sb, [c0], id1, [True])
                        vb2 = transfer_read(
                            vecTy_bf16, sb, [c0], id1, cst0_bf16, [True]
                        )
                        cmp = arith.CmpFOp(arith.CmpFPredicate.OGT, vb2, vz_bf16)
                        vr = arith.SelectOp(cmp, vb2, vz_bf16)
                        vrf = arith.ExtFOp(vecTy_f32, vr)
                        transfer_write(None, vrf, so, [c0], id1, [True])
                        yield_([])
                    yield_([])
                dma_memcpy_nd(
                    _l3o,
                    l1o,
                    dst_offsets=[m_off, n_off],
                    dst_sizes=[tile_m, tile_n],
                    dst_strides=[n, 1],
                )
                DeallocOp(l1i)
                DeallocOp(l1o)
                DeallocOp(l1bf)
                DeallocOp(l1b)


def build_argmax_launch(
    l3_in,
    l3_out,
    num_rows,
    num_cols,
    m_actual,
    tile_n,
    herd_n,
    xrt_dtype_f32,
    l1_mem_space,
):
    """Emit argmax launch: out[j] = argmax_i(in[i,j]) for i in [0, m_actual)."""
    index_type = IndexType.get()
    xrt_dtype_i32 = IntegerType.get_signless(32)
    l1TileTy = MemRefType.get(
        shape=[num_rows, tile_n], element_type=xrt_dtype_f32, memory_space=l1_mem_space
    )
    l1OutTy = MemRefType.get(
        shape=[tile_n], element_type=xrt_dtype_i32, memory_space=l1_mem_space
    )
    launch_size = [1, num_cols // tile_n // herd_n]

    @launch(operands=[l3_in, l3_out], sizes=launch_size)
    def launch_body(livx, livy, lsx, lsy, _in, _out):
        @segment(name="argmax_seg", operands=[livy, _in, _out])
        def seg(livy_s, in_s, out_s):
            c_tnhn = ConstantOp(IntegerAttr.get(IndexType.get(), tile_n * herd_n), None)
            loff_n = arith.MulIOp(livy_s, c_tnhn)

            @herd(name="herd_0", sizes=[1, herd_n], operands=[loff_n, in_s, out_s])
            def herd_body(tx, ty, _sx, _sy, _ln, _l3i, _l3o):
                l1t = AllocOp(l1TileTy, [], [])
                l1o = AllocOp(l1OutTy, [], [])
                n_off = affine_apply(make_offset_map(tile_n), [_ln, ty])
                dma_memcpy_nd(
                    l1t,
                    _l3i,
                    src_offsets=[0, n_off],
                    src_sizes=[num_rows, tile_n],
                    src_strides=[num_cols, 1],
                )
                c0 = ConstantOp(index_type, 0)
                c1 = ConstantOp(index_type, 1)
                ctn = ConstantOp(index_type, tile_n)
                cma = ConstantOp(index_type, m_actual)
                neg_inf = arith.ConstantOp(
                    xrt_dtype_f32, FloatAttr.get(xrt_dtype_f32, float("-inf"))
                )
                c0_i32 = arith.ConstantOp(xrt_dtype_i32, 0)
                for j in range_(c0, ctn, c1):
                    for i, (cur_val, cur_idx), results in range_(
                        c0, cma, c1, iter_args=[neg_inf, c0_i32]
                    ):
                        val = load(l1t, [i, j])
                        cmp = arith.CmpFOp(arith.CmpFPredicate.OGT, val, cur_val)
                        nv = arith.SelectOp(cmp, val, cur_val)
                        ii = arith.IndexCastOp(xrt_dtype_i32, i)
                        ni = arith.SelectOp(cmp, ii, cur_idx)
                        yield_([nv, ni])
                    store(results[1], l1o, [j])
                    yield_([])
                dma_memcpy_nd(
                    _l3o, l1o, dst_offsets=[n_off], dst_sizes=[tile_n], dst_strides=[1]
                )
                DeallocOp(l1t)
                DeallocOp(l1o)


# ─── Two-phase module build ──────────────────────────────────────────────────
# Phase 1: build matmul module (test 54) + apply transform to vectorize.
# Phase 2: extend the function with element-wise launches.
# This avoids transform IR conflicts (element-wise herds have no linalg ops
# but the transform's split_handle expects exactly 3 herds from the matmul).


def build_integration_module(
    # Matmul params
    m1_pad,
    k1,
    n1_pad,
    m1_alloc,
    n1_alloc,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m_mm,
    herd_n_mm,
    mmul_mkn,
    # Element-wise params
    m2_pad,
    n2_pad,
    m2_actual,
    herd_m_ew,
    herd_n_ew,
    tile_n_argmax,
    herd_n_argmax,
    vector_size,
):
    """Build integration module in two phases."""

    # ── Phase 1: matmul ──
    matmul_module = build_matmul_module(
        m1_pad,
        k1,
        n1_pad,
        m1_alloc,
        n1_alloc,
        tile_m,
        tile_k_l2,
        tile_k_l1,
        tile_n,
        herd_m_mm,
        herd_n_mm,
        mmul_mkn,
    )

    # Apply test 54's transform for vectorization
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

            %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %truncf_a_g, %truncf_b_g, %matmul = transform.split_handle %all_generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            %tiled_truncf_a, %truncf_a_loops:2 =
              transform.structured.tile_using_for %truncf_a_g tile_sizes [1, 1, 0, 0]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
            %tiled_truncf_b, %truncf_b_loops:2 =
              transform.structured.tile_using_for %truncf_b_g tile_sizes [1, 1, 0, 0]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

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

            %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
            %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

            %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
            %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

            %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

            %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
            %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
            %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

            %fors_to_hoist = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
            %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)
            %all_extf = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
            %all_truncf = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op

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
    transform_ir = Module.parse(transform_ir_string, context=matmul_module.context)
    run_transform(transform_ir, matmul_module)

    # ── Phase 2: serialize vectorized matmul, extend with element-wise ──
    # Get the matmul function's IR, then rebuild a module that includes
    # both the matmul launch and the element-wise launches.
    matmul_ir = str(matmul_module)

    # Parse the vectorized matmul module, then add element-wise launches
    # by modifying the function to accept additional arguments.
    # Strategy: re-parse the matmul IR, add extra func arguments and
    # append element-wise launches into the function body.

    # We need to extend the function signature. The matmul func has:
    #   func @matmul_f32(%A, %B, %C) -> ()
    # We need to change it to:
    #   func @mnist_fc(%A, %B, %C, %bias1, %relu_out, %mat2_out, %bias2, %bias2_out, %argmax_out) -> ()
    # And add the element-wise launches after the matmul launch.

    # Simplest approach: rebuild from scratch with the matmul IR as a string
    # embedded in the new module. But this is fragile.

    # Better approach: use the matmul module directly, modify its function
    # to add more arguments and more launches.

    ctx = matmul_module.context
    with ctx, Location.unknown():
        return _extend_with_elementwise(
            matmul_module,
            m1_pad,
            n1_pad,
            m2_pad,
            n2_pad,
            m2_actual,
            tile_m,
            tile_n,
            herd_m_ew,
            herd_n_ew,
            tile_n_argmax,
            herd_n_argmax,
            vector_size,
        )


def _extend_with_elementwise(
    matmul_module,
    m1_pad,
    n1_pad,
    m2_pad,
    n2_pad,
    m2_actual,
    tile_m,
    tile_n,
    herd_m_ew,
    herd_n_ew,
    tile_n_argmax,
    herd_n_argmax,
    vector_size,
):
    xrt_f32 = type_mapper(np.float32)
    xrt_bf16 = type_mapper(bfloat16)
    xrt_i32 = IntegerType.get_signless(32)
    l1ms = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)

    # Additional L3 types for element-wise ops
    tyBias1 = MemRefType.get([m1_pad], xrt_f32)
    tyRelu = MemRefType.get([m1_pad, n1_pad], xrt_f32)
    tyMat2 = MemRefType.get([m2_pad, n2_pad], xrt_f32)
    tyBias2 = MemRefType.get([m2_pad], xrt_f32)
    tyBias2Out = MemRefType.get([m2_pad, n2_pad], xrt_f32)
    tyArgmax = MemRefType.get([n2_pad], xrt_i32)

    extra_arg_types = [tyBias1, tyRelu, tyMat2, tyBias2, tyBias2Out, tyArgmax]

    # Get the existing function from the matmul module
    with matmul_module.context:
        # After transform, the module may be in generic form.
        # The func op appears as '"matmul_f32"' (its sym_name).
        # Just grab the first op with a body region.
        func_op = list(matmul_module.body.operations)[0]

        # Add extra arguments to the function
        entry_block = func_op.body.blocks[0]
        loc = Location.unknown()
        for ty in extra_arg_types:
            entry_block.add_argument(ty, loc)

        # Update function type
        old_func_type_attr = func_op.attributes["function_type"]
        old_func_type = FunctionType(TypeAttr(old_func_type_attr).value)
        new_input_types = list(old_func_type.inputs) + extra_arg_types
        new_func_type = FunctionType.get(new_input_types, [])
        func_op.attributes["function_type"] = TypeAttr.get(new_func_type)

        # Rename function
        func_op.attributes["sym_name"] = StringAttr.get("mnist_fc")

        # Get the new arguments (after the original 3 matmul args)
        bias1_arg = entry_block.arguments[3]
        relu_arg = entry_block.arguments[4]
        mat2_arg = entry_block.arguments[5]
        bias2_arg = entry_block.arguments[6]
        bias2_out_arg = entry_block.arguments[7]
        argmax_arg = entry_block.arguments[8]

        # Insert element-wise launches before the function return
        return_op = list(entry_block.operations)[-1]
        assert return_op.name == "func.return"

        with InsertionPoint(return_op):
            # Launch 2: bias_add2
            build_bias_add_launch(
                mat2_arg,
                bias2_arg,
                bias2_out_arg,
                m2_pad,
                n2_pad,
                tile_m,
                tile_n,
                1,
                herd_n_ew,
                vector_size,
                xrt_f32,
                l1ms,
                name_suffix="_2",
            )
            # Launch 3: argmax
            build_argmax_launch(
                bias2_out_arg,
                argmax_arg,
                m2_pad,
                n2_pad,
                m2_actual,
                tile_n_argmax,
                herd_n_argmax,
                xrt_f32,
                l1ms,
            )
            # Launch 4: bias+relu fused
            build_bias_relu_fused_launch(
                entry_block.arguments[2],
                bias1_arg,
                relu_arg,  # C1, bias1, relu_out
                m1_pad,
                n1_pad,
                tile_m,
                tile_n,
                herd_m_ew,
                herd_n_ew,
                vector_size,
                xrt_f32,
                xrt_bf16,
                l1ms,
            )

    return matmul_module


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    M1_ACTUAL = 500
    K1 = 784
    N1_ACTUAL = 500
    M2_ACTUAL = 10
    N2_ACTUAL = 500

    TILE_M = 64
    TILE_N = 32
    TILE_K_L2 = 16
    TILE_K_L1 = 16
    HERD_M_MM = 4
    HERD_N_MM = 4
    HERD_M_EW = 1
    HERD_N_EW = 4
    TILE_N_ARGMAX = 32
    HERD_N_ARGMAX = 4
    VECTOR_SIZE = 16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="MNIST-FC integration: matmul -> bias+relu -> bias_add -> argmax",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    args = parser.parse_args()

    # aie2p mmul dimensions
    mmul_mkn = [8, 8, 8]
    INNER_BLOCK = 8

    # Pad dimensions
    M1_pad = math.ceil(M1_ACTUAL / (TILE_M * HERD_M_MM)) * (TILE_M * HERD_M_MM)  # 512
    N1_pad = math.ceil(N1_ACTUAL / (TILE_N * HERD_N_MM)) * (TILE_N * HERD_N_MM)  # 512
    M1_alloc = math.ceil(M1_ACTUAL / INNER_BLOCK) * INNER_BLOCK  # 504
    N1_alloc = math.ceil(N1_ACTUAL / INNER_BLOCK) * INNER_BLOCK  # 504
    M2_pad = math.ceil(M2_ACTUAL / TILE_M) * TILE_M  # 64
    N2_pad = math.ceil(N2_ACTUAL / (TILE_N * HERD_N_EW)) * (TILE_N * HERD_N_EW)  # 512

    if args.verbose:
        print(f"Matmul1: M={M1_ACTUAL}→{M1_pad}, K={K1}, N={N1_ACTUAL}→{N1_pad}")
        print(f"M1_alloc={M1_alloc}, N1_alloc={N1_alloc}")
        print(f"M2={M2_ACTUAL}→{M2_pad}, N2={N2_ACTUAL}→{N2_pad}")

    mlir_module = build_integration_module(
        M1_pad,
        K1,
        N1_pad,
        M1_alloc,
        N1_alloc,
        TILE_M,
        TILE_K_L2,
        TILE_K_L1,
        TILE_N,
        HERD_M_MM,
        HERD_N_MM,
        mmul_mkn,
        M2_pad,
        N2_pad,
        M2_ACTUAL,
        HERD_M_EW,
        HERD_N_EW,
        TILE_N_ARGMAX,
        HERD_N_ARGMAX,
        VECTOR_SIZE,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # ─── Host data ────────────────────────────────────────────────────────────
    # Matmul1 inputs: W1 (K x M_alloc), X (K x N_alloc)
    W1 = np.zeros((K1, M1_alloc), dtype=np.float32)
    W1[:, :M1_ACTUAL] = (np.random.randn(K1, M1_ACTUAL) * 0.01).astype(np.float32)

    X = np.zeros((K1, N1_alloc), dtype=np.float32)
    X[:, :N1_ACTUAL] = (np.random.randn(K1, N1_ACTUAL) * 0.01).astype(np.float32)

    # Matmul1 output buffer
    C1 = np.zeros((M1_pad, N1_pad), dtype=np.float32)

    # bias1 (per row of C1, length M1_pad)
    bias1 = np.zeros(M1_pad, dtype=np.float32)
    bias1[:M1_ACTUAL] = (np.random.randn(M1_ACTUAL) * 0.1).astype(np.float32)

    # relu_out buffer
    relu_out = np.zeros((M1_pad, N1_pad), dtype=np.float32)

    # Simulate matmul2 output (provided as input for now)
    matmul2_out = np.zeros((M2_pad, N2_pad), dtype=np.float32)
    matmul2_out[:M2_ACTUAL, :N2_ACTUAL] = (
        np.random.randn(M2_ACTUAL, N2_ACTUAL) * 4
    ).astype(np.float32)

    bias2 = np.zeros(M2_pad, dtype=np.float32)
    bias2[:M2_ACTUAL] = (np.random.randn(M2_ACTUAL) * 0.5).astype(np.float32)

    bias2_out = np.zeros((M2_pad, N2_pad), dtype=np.float32)
    argmax_out = np.zeros(N2_pad, dtype=np.int32)

    if args.compile_mode == "compile-and-run":
        # ─── Golden reference ─────────────────────────────────────────────
        # Step 1: matmul1 (bf16 compute, f32 output)
        W1_bf16 = W1.astype(bfloat16)
        X_bf16 = X.astype(bfloat16)
        golden_C1 = np.zeros((M1_pad, N1_pad), dtype=np.float32)
        for i in range(M1_ACTUAL):
            for j in range(N1_ACTUAL):
                golden_C1[i, j] = np.sum(
                    W1_bf16[:, i].astype(np.float32) * X_bf16[:, j].astype(np.float32)
                )

        # Step 2: bias+relu (bf16 truncation for relu)
        golden_biased = golden_C1.copy()
        golden_biased[:M1_ACTUAL, :N1_ACTUAL] += bias1[:M1_ACTUAL, np.newaxis]
        golden_biased_bf16 = golden_biased.astype(bfloat16)
        golden_relu = np.maximum(golden_biased_bf16, 0).astype(np.float32)

        # Step 3: bias_add2
        golden_biased2 = matmul2_out.copy()
        golden_biased2[:M2_ACTUAL, :N2_ACTUAL] += bias2[:M2_ACTUAL, np.newaxis]

        # Step 4: argmax
        golden_argmax = np.argmax(
            golden_biased2[:M2_ACTUAL, :N2_ACTUAL], axis=0
        ).astype(np.int32)
        golden_argmax_pad = np.zeros(N2_pad, dtype=np.int32)
        golden_argmax_pad[:N2_ACTUAL] = golden_argmax

        # Sample for argmax verification
        num_samples = min(100, N2_ACTUAL)
        sampled_cols = np.random.choice(N2_ACTUAL, num_samples, replace=False)
        sampled_cols = np.unique(np.concatenate([sampled_cols, [0, N2_ACTUAL - 1]]))
        sampled_data = {
            "shape": (N2_pad,),
            "indices": np.vstack([sampled_cols]),
            "values": golden_argmax_pad[sampled_cols],
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="mnist_fc",
            runtime_loop_tiling_sizes=[1, 1],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[
                    W1,
                    X,
                    C1,
                    bias1,
                    relu_out,
                    matmul2_out,
                    bias2,
                    bias2_out,
                ],
                stochastic_expected_outputs=[sampled_data],
                rtol=0,
                atol=0,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            runtime_loop_tiling_sizes=[1, 1],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
