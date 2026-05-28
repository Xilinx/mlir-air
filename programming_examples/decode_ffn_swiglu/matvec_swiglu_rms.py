# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Single-token GEMV with weighted-RMSNorm input and fused SwiGLU output:
#
#   normed = rms_norm(input_vec, norm_weight)        # row 0 + row 1 of B
#   raw[M] = A_interleaved[M, K] @ normed             # interleaved gate/up
#   swiglu[M/2] = silu(raw[2i]) * raw[2i+1]           # per pair
#
# A is interleaved at compile time: A[2i, :] = gate[i], A[2i+1, :] = up[i].
# B is a packed [2, K] buffer carrying the RMSNorm input row and the
# per-element norm weight; the kernel does the RMSNorm inline and feeds
# the normalized vector into the cascade-reduced matvec. The cascade tail
# pairs adjacent output rows and emits silu(gate)*up per pair, so the
# output is M/2 elements. tile_m must be even.
#
# BF16 in/out, accfloat accumulation. SiLU is computed in f32 via the
# tanh form `silu(x) = x * 0.5 * (1 + tanh(x/2))`.

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects import affine
from air.dialects.air import *
from air.dialects.air import channel as channel_decl
from air.dialects import arith, scf
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    subview,
    load as memref_load,
    store as memref_store,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
    fma,
)
from air.dialects import math as math_dialect
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


def compute_partial_dot(
    row,
    _l1_a,
    _l1_b,
    l1_acc_tmp,
    c0,
    k_chunk,
    f32_vec_size,
    vecTy_bf16,
    vecTy_f32,
    identity_map,
    read_map_2d,
    cst0_bf16,
    cst0_f32,
    f32_type,
):
    """Single-row bf16 dot product accumulated into f32. Returns the
    horizontal sum as an f32 scalar."""
    zero_vec_f32 = BroadcastOp(vecTy_f32, cst0_f32)
    transfer_write(None, zero_vec_f32, l1_acc_tmp, [c0], identity_map, [True])

    for j_k in range_(0, k_chunk, f32_vec_size):
        sub_a = subview(_l1_a, [row, j_k], [1, f32_vec_size], [1, 1])
        sub_b = subview(_l1_b, [j_k], [f32_vec_size], [1])
        v_a_bf16 = transfer_read(
            vecTy_bf16, sub_a, [c0, c0], read_map_2d, cst0_bf16, [True]
        )
        v_b_bf16 = transfer_read(
            vecTy_bf16, sub_b, [c0], identity_map, cst0_bf16, [True]
        )
        v_a_f32 = arith.extf(vecTy_f32, v_a_bf16)
        v_b_f32 = arith.extf(vecTy_f32, v_b_bf16)
        v_acc = transfer_read(
            vecTy_f32, l1_acc_tmp, [c0], identity_map, cst0_f32, [True]
        )
        v_result = fma(v_a_f32, v_b_f32, v_acc)
        transfer_write(None, v_result, l1_acc_tmp, [c0], identity_map, [True])
        yield_([])

    v_final = transfer_read(vecTy_f32, l1_acc_tmp, [c0], identity_map, cst0_f32, [True])
    return vector_reduction(f32_type, "add", v_final)


@module_builder
def build_module(
    m, k, tile_m, m_input, herd_cols, n_cascade, np_dtype_in, np_dtype_out
):
    assert (
        n_cascade >= 2
    ), f"n_cascade ({n_cascade}) must be >= 2 for a cascade pipeline"
    k_chunk = k // n_cascade
    assert (
        m % (tile_m * herd_cols) == 0
    ), f"M ({m}) must be divisible by tile_m * herd_cols ({tile_m * herd_cols})"
    assert (
        tile_m % m_input == 0
    ), f"tile_m ({tile_m}) must be divisible by m_input ({m_input})"
    assert tile_m % 2 == 0, f"tile_m ({tile_m}) must be even (gate/up pairs)"
    assert (
        m_input % 2 == 0
    ), f"m_input ({m_input}) must be even (rows iterated in (gate,up) pairs)"
    assert k % n_cascade == 0, f"K ({k}) must be divisible by n_cascade ({n_cascade})"
    assert (
        k_chunk % 64 == 0
    ), f"k_chunk ({k_chunk}) must be divisible by 64 (vector width)"
    # Vectorized silu in the tail uses vector<16 x bf16> tanh / mul. Peano
    # AIE2P only legalizes 16- and 32-lane bf16 vectors, so tile_m/2 (swiglu
    # outputs per (col, ty=0) tile) must be a positive multiple of 16.
    assert (tile_m // 2) >= 16 and (
        tile_m // 2
    ) % 16 == 0, f"tile_m/2 ({tile_m // 2}) must be a positive multiple of 16"

    bytes_per_elem_in = np.dtype(np_dtype_in).itemsize
    bytes_per_elem_out = np.dtype(np_dtype_out).itemsize
    # L2 budget: per col, 1 bulk A buffer (tile_m*k bf16). Output is
    # halved swiglu (tile_m/2 per col).
    a_bulk_bytes = tile_m * k * bytes_per_elem_in
    l2_per_col = a_bulk_bytes
    d_l2_bytes = herd_cols * (tile_m // 2) * bytes_per_elem_out
    # Per-memtile capacity (NPU2 = 512 KB). Per-col allocs are distinct
    # memrefs; the placer distributes them across memtiles, so the binding
    # constraint is per-col size (worst case 1 col per memtile).
    L2_CAPACITY = 512 * 1024
    assert (
        l2_per_col <= L2_CAPACITY
    ), f"L2 per-col exceeds memtile: per-col={l2_per_col}B > {L2_CAPACITY}B."

    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)
    f32_type = F32Type.get()

    # L3 MemRefTypes. To stay within the AIE2P 2-S2MM/tile budget we pack
    # res1 and ffn_norm_w into ONE L3 buffer of shape [2, k]: row 0 = res1,
    # row 1 = ffn_norm_w. Single broadcast DMA delivers both to each tile.
    memrefTyA = MemRefType.get([m, k], xrt_dtype_in)  # interleaved (gate,up)
    memrefTyRmsIn = MemRefType.get([2, k], xrt_dtype_in)
    memrefTyD = MemRefType.get([m // 2], xrt_dtype_out)  # swiglu output

    # L2 staging: per-col bulk A buffer + bulk swiglu output (halved).
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyAbulk = MemRefType.get(
        shape=[tile_m, k],
        element_type=xrt_dtype_in,
        memory_space=l2_mem_space,
    )
    l2MemrefTyD = MemRefType.get(
        shape=[herd_cols, tile_m // 2],
        element_type=xrt_dtype_out,
        memory_space=l2_mem_space,
    )

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get(
        shape=[tile_m, k_chunk],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    # L1 B holds the FULL k of post-RMSNorm normed vector — each tile reads
    # its k_chunk slice for GEMV but needs full k to compute the RMS scale.
    l1MemrefTyB = MemRefType.get(
        shape=[k],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    # Packed [2, k] bf16 scratch for (res1, ffn_norm_w) per compute tile.
    # Single L3->L1 channel, demuxed at-use (subview on row 0 vs row 1).
    l1MemrefTyRmsIn = MemRefType.get(
        shape=[2, k],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyD = MemRefType.get(
        shape=[tile_m // 2],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    # tile_m/2 * sizeof(bf16) >= 4 bytes for AIE DMA alignment on writeback.
    assert (tile_m // 2) * np.dtype(
        np_dtype_in
    ).itemsize >= 4, (
        f"tile_m/2 ({tile_m // 2}) * sizeof({np_dtype_in}) must be >= 4 bytes"
    )
    CASCADE_WIDTH = 16
    cascade_buf_len = max(tile_m, CASCADE_WIDTH)
    cascade_buf_len = (
        (cascade_buf_len + CASCADE_WIDTH - 1) // CASCADE_WIDTH
    ) * CASCADE_WIDTH
    l1MemrefTyScratch = MemRefType.get(
        shape=[cascade_buf_len],
        element_type=f32_type,
        memory_space=l1_mem_space,
    )

    # ar_L3toL2: per-col channel from L3 (carries A bulk only — no R).
    # ar_L2toL1: per-(col, cascade_row) memtile MM2S → compute tile.
    channel_decl("ar_L3toL2", size=[herd_cols])
    channel_decl("ar_L2toL1", size=[herd_cols, n_cascade])
    channel_decl(
        "chan_cascade",
        size=[herd_cols, n_cascade - 1],
        channel_type="npu_cascade",
    )

    # Signature: (A, rms_in[2, k], D) where rms_in[0] = res1, rms_in[1] =
    # ffn_norm_w. Packing both into one buffer keeps the compute tile under
    # the AIE2P 2-S2MM-per-tile budget.
    @FuncOp.from_py_func(memrefTyA, memrefTyRmsIn, memrefTyD)
    def matvec_swiglu_rms(arg0, arg1, arg2):

        launch_size = [m // tile_m // herd_cols, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_rms_in_data,
            l3_d_data,
        ):
            # Row offset for this launch iter
            launch_ivx_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(tile_m * herd_cols),
                    )
                ],
            )
            launch_offset_m_l = affine_apply(launch_ivx_map, [launch_ivx])

            # L3-side puts on ar_L3toL2[col]: 1 A bulk per launch iter.
            for col in range(herd_cols):
                c_col_idx_l = arith.ConstantOp.create_index(col)
                col_off_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(col * tile_m),
                        )
                    ],
                )
                col_off = affine_apply(col_off_map, [launch_offset_m_l])
                # A bulk: tile_m × k for this col
                ChannelPut(
                    "ar_L3toL2",
                    l3_a_data,
                    indices=[c_col_idx_l],
                    offsets=[col_off, 0],
                    sizes=[tile_m, k],
                    strides=[k, 1],
                )

            @segment(
                name="matvec_cascade_swiglu_rms_seg",
                operands=[launch_ivx, l3_rms_in_data, l3_d_data],
            )
            def segment_body(
                launch_ivx_s,
                l3_rms_in_data_s,
                l3_d_data_s,
            ):
                # L2: bulk A buffer per col + bulk swiglu output (halved).
                a_l2_bufs = [AllocOp(l2MemrefTyAbulk, [], []) for _ in range(herd_cols)]
                l2_d_data = AllocOp(l2MemrefTyD, [], [])

                # Memtile streaming per col: 1 A bulk get from L3, then
                # per-(col, ty) MM2S puts of A k_chunk slices.
                for col in range(herd_cols):
                    c_col_idx = arith.ConstantOp.create_index(col)
                    a_l2 = a_l2_bufs[col].result
                    # A bulk: GET tile_m × k from L3 → a_l2
                    ChannelGet(
                        "ar_L3toL2",
                        a_l2,
                        indices=[c_col_idx],
                        offsets=[0, 0],
                        sizes=[tile_m, k],
                        strides=[k, 1],
                    )
                    # A slices: PUT per ty (each MM2S reads its k_chunk slice)
                    for ty_v in range(n_cascade):
                        c_ty_idx = arith.ConstantOp.create_index(ty_v)
                        ChannelPut(
                            "ar_L2toL1",
                            a_l2,
                            indices=[c_col_idx, c_ty_idx],
                            offsets=[0, ty_v * k_chunk],
                            sizes=[tile_m, k_chunk],
                            strides=[k, 1],
                        )

                # L1 buffers (passed into herd as operands).
                l1_a_data = AllocOp(l1MemrefTyA, [], [])
                l1_b_data = AllocOp(l1MemrefTyB, [], [])  # full K, post-RMSNorm
                l1_rms_in_data = AllocOp(l1MemrefTyRmsIn, [], [])
                l1_d_data = AllocOp(l1MemrefTyD, [], [])
                l1_scratch = AllocOp(l1MemrefTyScratch, [], [])
                l1_recv = AllocOp(l1MemrefTyScratch, [], [])

                @herd(
                    name="herd_0",
                    sizes=[herd_cols, n_cascade],
                    operands=[
                        l1_a_data,
                        l1_b_data,
                        l1_rms_in_data,
                        l1_d_data,
                        l1_scratch,
                        l1_recv,
                        l3_rms_in_data_s,
                        l2_d_data,
                    ],
                )
                def herd_body(
                    tx,
                    ty,
                    sx,
                    sy,
                    _l1_a,
                    _l1_b,
                    _l1_rms_in,
                    _l1_d,
                    _l1_scratch,
                    _l1_recv,
                    _l3_rms_in,
                    _l2_d,
                ):
                    c0 = arith.ConstantOp.create_index(0)
                    c1_idx = arith.ConstantOp.create_index(1)
                    last_ty = arith.ConstantOp.create_index(n_cascade - 1)

                    # k_offset = ty * k_chunk
                    ty_k_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(k_chunk),
                            )
                        ],
                    )
                    k_offset = affine_apply(ty_k_map, [ty])

                    # RMSNorm absorbed (L-C3): pull packed [res1; ffn_norm_w]
                    # from L3 in ONE broadcast DMA (stays under the 2-S2MM/tile
                    # budget), then compute normed = (res1 * rsqrt(mean(res1^2)
                    # + eps)) * ffn_norm_w into _l1_b (full K). Each tile reads
                    # its own k_chunk slice for GEMV.
                    dma_memcpy_nd(
                        _l1_rms_in,
                        _l3_rms_in,
                        src_offsets=[0, 0],
                        src_sizes=[2, k],
                        src_strides=[k, 1],
                    )
                    # Sum-of-squares: mul in bf16 (Peano AIE2P has no vector
                    # f32 mul), extf to f32 between mul and add, accumulate in
                    # f32 (avoids bf16 accumulator precision loss summing K
                    # squared values — K=2048 lost ~9 % in pure bf16).
                    # Use store/read on the bf16 product to break the aievec
                    # mul→add chain (which the convert-vector-to-aievec pass
                    # rejects).
                    rms_vec_size = 16
                    rms_vecTy_bf16 = VectorType.get([rms_vec_size], xrt_dtype_in)
                    rms_vecTy_f32 = VectorType.get([rms_vec_size], f32_type)
                    rms_identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    # Read map for 2D `[1, vec_size]` subviews of `_l1_rms_in`:
                    # selects dim 1 (vec_size) and ignores dim 0 (the row index).
                    read_map_2d_rms = AffineMapAttr.get(
                        AffineMap.get(2, 0, [AffineExpr.get_dim(1)])
                    )
                    rms_cst0_bf16 = arith.ConstantOp(xrt_dtype_in, 0.0)
                    rms_cst0_f32 = arith.ConstantOp(f32_type, 0.0)
                    l1MemrefTyRmsAccF32 = MemRefType.get(
                        shape=[rms_vec_size],
                        element_type=f32_type,
                        memory_space=l1_mem_space,
                    )
                    l1MemrefTyRmsTmpBf16 = MemRefType.get(
                        shape=[rms_vec_size],
                        element_type=xrt_dtype_in,
                        memory_space=l1_mem_space,
                    )
                    rms_acc = AllocOp(l1MemrefTyRmsAccF32, [], [])
                    rms_tmp = AllocOp(l1MemrefTyRmsTmpBf16, [], [])
                    zero_vec_f32 = BroadcastOp(rms_vecTy_f32, rms_cst0_f32)
                    transfer_write(
                        None,
                        zero_vec_f32,
                        rms_acc,
                        [c0],
                        rms_identity_map,
                        [True],
                    )
                    c_k = arith.ConstantOp.create_index(k)
                    c_rms_vec = arith.ConstantOp.create_index(rms_vec_size)
                    for j in range_(0, c_k, c_rms_vec):
                        sub_r = subview(_l1_rms_in, [0, j], [1, rms_vec_size], [1, 1])
                        v_x = transfer_read(
                            rms_vecTy_bf16,
                            sub_r,
                            [c0, c0],
                            read_map_2d_rms,
                            rms_cst0_bf16,
                            [True],
                        )
                        # mul (bf16) → store → read to break aievec mul→add.
                        v_sq_bf16 = arith.mulf(v_x, v_x)
                        transfer_write(
                            None,
                            v_sq_bf16,
                            rms_tmp,
                            [c0],
                            rms_identity_map,
                            [True],
                        )
                        v_sq_rd_bf16 = transfer_read(
                            rms_vecTy_bf16,
                            rms_tmp,
                            [c0],
                            rms_identity_map,
                            rms_cst0_bf16,
                            [True],
                        )
                        v_sq_f32 = arith.extf(rms_vecTy_f32, v_sq_rd_bf16)
                        v_acc = transfer_read(
                            rms_vecTy_f32,
                            rms_acc,
                            [c0],
                            rms_identity_map,
                            rms_cst0_f32,
                            [True],
                        )
                        v_sum = arith.addf(v_acc, v_sq_f32)
                        transfer_write(
                            None,
                            v_sum,
                            rms_acc,
                            [c0],
                            rms_identity_map,
                            [True],
                        )
                        yield_([])

                    # Horizontal reduce → scalar f32 sum, mean, rstd.
                    v_final_f32 = transfer_read(
                        rms_vecTy_f32,
                        rms_acc,
                        [c0],
                        rms_identity_map,
                        rms_cst0_f32,
                        [True],
                    )
                    total_sum_f32 = vector_reduction(f32_type, "add", v_final_f32)
                    k_f32_const = arith.ConstantOp(f32_type, float(k))
                    eps_f32_const = arith.ConstantOp(f32_type, 1.0e-5)
                    mean_f32 = arith.divf(total_sum_f32, k_f32_const)
                    mean_eps_f32 = arith.addf(mean_f32, eps_f32_const)
                    rstd_f32 = math_dialect.rsqrt(mean_eps_f32)
                    rstd_bf16 = arith.truncf(xrt_dtype_in, rstd_f32)
                    v_rstd = BroadcastOp(rms_vecTy_bf16, rstd_bf16)

                    # normed = res1 * rstd * ffn_norm_w → _l1_b (full K, bf16).
                    for j in range_(0, c_k, c_rms_vec):
                        sub_r = subview(_l1_rms_in, [0, j], [1, rms_vec_size], [1, 1])
                        sub_w = subview(_l1_rms_in, [1, j], [1, rms_vec_size], [1, 1])
                        sub_b = subview(_l1_b, [j], [rms_vec_size], [1])
                        v_r = transfer_read(
                            rms_vecTy_bf16,
                            sub_r,
                            [c0, c0],
                            read_map_2d_rms,
                            rms_cst0_bf16,
                            [True],
                        )
                        v_w = transfer_read(
                            rms_vecTy_bf16,
                            sub_w,
                            [c0, c0],
                            read_map_2d_rms,
                            rms_cst0_bf16,
                            [True],
                        )
                        v_n = arith.mulf(v_r, v_rstd.result)
                        v_y = arith.mulf(v_n, v_w)
                        transfer_write(
                            None,
                            v_y,
                            sub_b,
                            [c0],
                            rms_identity_map,
                            [True],
                        )
                        yield_([])

                    DeallocOp(rms_acc)
                    DeallocOp(rms_tmp)

                    # head_set fires when cascade_row == n_cascade-1.
                    head_set = IntegerSet.get(
                        0,
                        1,
                        [
                            AffineSymbolExpr.get(0)
                            - AffineConstantExpr.get(n_cascade - 1)
                        ],
                        [True],
                    )

                    # Cascade pipeline setup (vector dot product utilities).
                    f32_vec_size = 16
                    vecTy_bf16 = VectorType.get([f32_vec_size], xrt_dtype_in)
                    vecTy_f32 = VectorType.get([f32_vec_size], f32_type)
                    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    read_map_2d = AffineMapAttr.get(
                        AffineMap.get(2, 0, [AffineExpr.get_dim(1)])
                    )
                    cst0_bf16 = arith.ConstantOp(xrt_dtype_in, 0.0)
                    cst0_f32 = arith.ConstantOp(f32_type, 0.0)
                    row_out_map = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineSymbolExpr.get(1),
                            )
                        ],
                    )

                    l1MemrefTyAccTmp = MemRefType.get(
                        shape=[f32_vec_size],
                        element_type=f32_type,
                        memory_space=l1_mem_space,
                    )
                    l1_acc_tmp = AllocOp(l1MemrefTyAccTmp, [], [])

                    # Tail buffers for vectorized silu: gate and up partials
                    # land in SEPARATE bf16 scratches of size tile_m/2. Simple
                    # identity index pattern keeps air-shrink-memref-sizes
                    # analysis happy. Read as contiguous vectors of
                    # SILU_VEC_SIZE for vector<bf16> math.tanh.
                    SILU_VEC_SIZE = 16
                    l1MemrefTyHalf = MemRefType.get(
                        shape=[tile_m // 2],
                        element_type=xrt_dtype_out,
                        memory_space=l1_mem_space,
                    )
                    l1_bf16_gate = AllocOp(l1MemrefTyHalf, [], [])
                    l1_bf16_up = AllocOp(l1MemrefTyHalf, [], [])
                    vecTyOut = VectorType.get([SILU_VEC_SIZE], xrt_dtype_out)
                    cst_half_bf16 = arith.ConstantOp(xrt_dtype_out, 0.5)
                    cst_one_bf16 = arith.ConstantOp(xrt_dtype_out, 1.0)
                    v_half_bf16 = BroadcastOp(vecTyOut, cst_half_bf16)
                    v_one_bf16 = BroadcastOp(vecTyOut, cst_one_bf16)

                    # _l1_b is full K (post-RMSNorm normed); each tile's
                    # GEMV reads its k_chunk slice at offset ty*k_chunk.
                    l1_b_slice = subview(_l1_b, [k_offset], [k_chunk], [1])
                    dot_args = (
                        _l1_a,
                        l1_b_slice,
                        l1_acc_tmp,
                        c0,
                        k_chunk,
                        f32_vec_size,
                        vecTy_bf16,
                        vecTy_f32,
                        identity_map,
                        read_map_2d,
                        cst0_bf16,
                        cst0_f32,
                        f32_type,
                    )

                    # Single bulk A slice receive per launch iter.
                    ChannelGet(
                        "ar_L2toL1",
                        _l1_a,
                        indices=[tx, ty],
                    )

                    cst_half_f32 = arith.ConstantOp(f32_type, 0.5)
                    cst_one_f32 = arith.ConstantOp(f32_type, 1.0)

                    # Hot loop: per j_m iter, compute partial dot from
                    # rows [j_m*m_input : (j_m+1)*m_input] of _l1_a (which
                    # holds the full tile_m rows for this (col, ty)).
                    for j_m in range_(0, tile_m // m_input):
                        j_m_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        j_m_offset = affine_apply(j_m_map, [j_m])
                        # Map (j_m_offset, row) → row index in _l1_a (= j_m_offset + row)
                        abs_row_map = AffineMap.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineSymbolExpr.get(1),
                                )
                            ],
                        )

                        # === Cascade compute ===
                        # HEAD (ty == n_cascade-1): partial = A·B → scratch → cascade.
                        # MIDDLE: get cascade; partial; sum; put cascade.
                        # TAIL (ty == 0): get cascade; partial; sum;
                        #   pair adjacent (gate, up) rows → swiglu out.
                        cmp_first = arith.CmpIOp(arith.CmpIPredicate.eq, ty, last_ty)
                        if_first = scf.IfOp(cmp_first, has_else=True)
                        with InsertionPoint(if_first.then_block):
                            # HEAD: own partial → scratch → cascade.
                            for row in range_(0, m_input):
                                abs_row = affine_apply(abs_row_map, [j_m_offset, row])
                                partial_sum = compute_partial_dot(abs_row, *dot_args)
                                sub_scratch = subview(_l1_scratch, [row], [1], [1])
                                memref_store(partial_sum, sub_scratch, [c0])
                                yield_([])

                            prev_ty = arith.SubIOp(ty, c1_idx)
                            ChannelPut(
                                "chan_cascade",
                                _l1_scratch,
                                indices=[tx, prev_ty],
                            )
                            yield_([])

                        with InsertionPoint(if_first.else_block):
                            # TAIL or MIDDLE
                            cmp_last = arith.CmpIOp(arith.CmpIPredicate.eq, ty, c0)
                            if_last = scf.IfOp(cmp_last, has_else=True)
                            with InsertionPoint(if_last.then_block):
                                # TAIL: get cascade, add own partial, truncate
                                # to bf16, store gate partials and up partials
                                # into SEPARATE bf16 scratches indexed by pair
                                # position. Vectorized silu+mul runs after the
                                # j_m loop (scalar tanh isn't legalizable on
                                # AIE2P).
                                ChannelGet(
                                    "chan_cascade",
                                    _l1_recv,
                                    indices=[tx, ty],
                                )

                                # j_m_pair_offset = j_m * (m_input / 2)
                                j_m_pair_map = AffineMap.get(
                                    0,
                                    1,
                                    [
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(0),
                                            AffineConstantExpr.get(m_input // 2),
                                        )
                                    ],
                                )
                                j_m_pair_offset = affine_apply(j_m_pair_map, [j_m])
                                pair_idx_map = AffineMap.get(
                                    0,
                                    2,
                                    [
                                        AffineExpr.get_add(
                                            AffineSymbolExpr.get(0),
                                            AffineSymbolExpr.get(1),
                                        )
                                    ],
                                )
                                row_g_map = AffineMap.get(
                                    0,
                                    1,
                                    [
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(0),
                                            AffineConstantExpr.get(2),
                                        )
                                    ],
                                )
                                row_u_map = AffineMap.get(
                                    0,
                                    1,
                                    [
                                        AffineExpr.get_add(
                                            AffineExpr.get_mul(
                                                AffineSymbolExpr.get(0),
                                                AffineConstantExpr.get(2),
                                            ),
                                            AffineConstantExpr.get(1),
                                        )
                                    ],
                                )

                                for pair in range_(0, m_input // 2):
                                    row_g_local = affine_apply(row_g_map, [pair])
                                    row_u_local = affine_apply(row_u_map, [pair])
                                    abs_row_g = affine_apply(
                                        abs_row_map,
                                        [j_m_offset, row_g_local],
                                    )
                                    abs_row_u = affine_apply(
                                        abs_row_map,
                                        [j_m_offset, row_u_local],
                                    )
                                    g_partial = compute_partial_dot(
                                        abs_row_g, *dot_args
                                    )
                                    u_partial = compute_partial_dot(
                                        abs_row_u, *dot_args
                                    )
                                    sub_recv_g = subview(
                                        _l1_recv, [row_g_local], [1], [1]
                                    )
                                    sub_recv_u = subview(
                                        _l1_recv, [row_u_local], [1], [1]
                                    )
                                    g_recv = memref_load(sub_recv_g, [c0])
                                    u_recv = memref_load(sub_recv_u, [c0])
                                    g_total = arith.addf(g_recv, g_partial)
                                    u_total = arith.addf(u_recv, u_partial)
                                    g_bf16 = arith.truncf(xrt_dtype_out, g_total)
                                    u_bf16 = arith.truncf(xrt_dtype_out, u_total)
                                    pair_pos = affine_apply(
                                        pair_idx_map,
                                        [j_m_pair_offset, pair],
                                    )
                                    sub_g_out = subview(
                                        l1_bf16_gate.result,
                                        [pair_pos],
                                        [1],
                                        [1],
                                    )
                                    sub_u_out = subview(
                                        l1_bf16_up.result,
                                        [pair_pos],
                                        [1],
                                        [1],
                                    )
                                    memref_store(g_bf16, sub_g_out, [c0])
                                    memref_store(u_bf16, sub_u_out, [c0])
                                    yield_([])

                                yield_([])

                            with InsertionPoint(if_last.else_block):
                                # Middle tiles: cascade get → compute → cascade put
                                ChannelGet(
                                    "chan_cascade",
                                    _l1_recv,
                                    indices=[tx, ty],
                                )

                                for row in range_(0, m_input):
                                    abs_row = affine_apply(
                                        abs_row_map, [j_m_offset, row]
                                    )
                                    partial_sum = compute_partial_dot(
                                        abs_row, *dot_args
                                    )
                                    sub_recv = subview(_l1_recv, [row], [1], [1])
                                    recv_val = memref_load(sub_recv, [c0])
                                    total = arith.addf(recv_val, partial_sum)
                                    sub_scratch = subview(_l1_scratch, [row], [1], [1])
                                    memref_store(total, sub_scratch, [c0])
                                    yield_([])

                                prev_ty_mid = arith.SubIOp(ty, c1_idx)
                                ChannelPut(
                                    "chan_cascade",
                                    _l1_scratch,
                                    indices=[tx, prev_ty_mid],
                                )
                                yield_([])

                            yield_([])

                        yield_([])

                    # ty=0 tiles vectorize silu+mul on the gate/up bf16
                    # scratches, then DMA the result to L2.
                    cmp_writer = arith.CmpIOp(arith.CmpIPredicate.eq, ty, c0)
                    if_writer = scf.IfOp(cmp_writer)
                    with InsertionPoint(if_writer.then_block):
                        # Vectorized silu(gate) * up — see swiglu.py reference.
                        c_vec_size = arith.ConstantOp.create_index(SILU_VEC_SIZE)
                        c_tile_m_half = arith.ConstantOp.create_index(tile_m // 2)
                        for kk in range_(0, c_tile_m_half, c_vec_size):
                            sub_g = subview(
                                l1_bf16_gate.result,
                                [kk],
                                [SILU_VEC_SIZE],
                                [1],
                            )
                            sub_u = subview(
                                l1_bf16_up.result,
                                [kk],
                                [SILU_VEC_SIZE],
                                [1],
                            )
                            sub_out = subview(_l1_d, [kk], [SILU_VEC_SIZE], [1])
                            v_g = transfer_read(
                                vecTyOut,
                                sub_g,
                                [c0],
                                identity_map,
                                cst0_bf16,
                                [True],
                            )
                            v_u = transfer_read(
                                vecTyOut,
                                sub_u,
                                [c0],
                                identity_map,
                                cst0_bf16,
                                [True],
                            )
                            v_half_g = arith.mulf(v_g, v_half_bf16.result)
                            v_tanh = math_dialect.tanh(v_half_g)
                            v_tanh_p1 = arith.addf(v_tanh, v_one_bf16.result)
                            v_sig = arith.mulf(v_tanh_p1, v_half_bf16.result)
                            v_silu = arith.mulf(v_g, v_sig)
                            v_out = arith.mulf(v_silu, v_u)
                            transfer_write(
                                None,
                                v_out,
                                sub_out,
                                [c0],
                                identity_map,
                                [True],
                            )
                            yield_([])

                        dma_memcpy_nd(
                            _l2_d,
                            _l1_d,
                            dst_offsets=[tx, 0],
                            dst_sizes=[1, tile_m // 2],
                            dst_strides=[tile_m // 2, 1],
                            src_offsets=[],
                            src_sizes=[tile_m // 2],
                            src_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_acc_tmp)
                    DeallocOp(l1_bf16_gate)
                    DeallocOp(l1_bf16_up)

                # L2 -> L3: swiglu writeback for this launch slice (halved).
                launch_ivx_map_s = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get((tile_m // 2) * herd_cols),
                        )
                    ],
                )
                launch_offset_m_d = affine_apply(launch_ivx_map_s, [launch_ivx_s])
                dma_memcpy_nd(
                    l3_d_data_s,
                    l2_d_data,
                    dst_offsets=[launch_offset_m_d],
                    dst_sizes=[herd_cols * (tile_m // 2)],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_cols, tile_m // 2],
                    src_strides=[tile_m // 2, 1],
                )

                for a_l2 in a_l2_bufs:
                    DeallocOp(a_l2)
                DeallocOp(l2_d_data)
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_d_data)
                DeallocOp(l1_scratch)
                DeallocOp(l1_recv)


if __name__ == "__main__":
    # Defaults sized for an interleaved-gate/up FFN at K=2048, hidden=8192:
    # M = 2 * hidden = 16384. tile_m / m_input / n_cascade tuned for an 8-col herd.
    M = 16384
    K = 2048
    TILE_M = 32
    M_INPUT = 4
    HERD_COLS = 8
    N_CASCADE = 4
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="matvec_swiglu_rms.py",
        description="BF16 GEMV with fused RMSNorm input and SwiGLU output: "
        "swiglu = silu(A_interleaved · rms_norm(B[0], B[1])); output is M/2.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--tile-m", type=int, default=TILE_M, dest="tile_m")
    parser.add_argument("--m-input", type=int, default=M_INPUT, dest="m_input")
    parser.add_argument("--herd-cols", type=int, default=HERD_COLS, dest="herd_cols")
    parser.add_argument("--n-cascade", type=int, default=N_CASCADE, dest="n_cascade")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-and-xclbin"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument("--debug-ir", action="store_true", dest="debug_ir")

    args = parser.parse_args()

    mlir_module = build_module(
        args.m,
        args.k,
        args.tile_m,
        args.m_input,
        args.herd_cols,
        args.n_cascade,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(42)
        # Interleaved input: rows 2i = gate[i], rows 2i+1 = up[i].
        n_out = args.m // 2
        gate = (np.random.randn(n_out, args.k) * 0.02).astype(INPUT_DATATYPE)
        up = (np.random.randn(n_out, args.k) * 0.02).astype(INPUT_DATATYPE)
        input_a = np.empty((args.m, args.k), dtype=INPUT_DATATYPE)
        input_a[0::2] = gate
        input_a[1::2] = up
        # Packed [2, K] input: row 0 = vector to be normalized,
        # row 1 = per-element RMSNorm scale. One broadcast DMA stays
        # under the 2-S2MM-per-tile budget on AIE2P.
        input_vec = (np.random.randn(args.k)).astype(INPUT_DATATYPE)
        norm_weight = (np.random.randn(args.k) * 0.1 + 1.0).astype(INPUT_DATATYPE)
        input_rms = np.stack([input_vec, norm_weight], axis=0).astype(INPUT_DATATYPE)
        # CPU reference: RMSNorm inline (matches hardware), then GEMV + SwiGLU.
        eps = 1.0e-5
        x_f32 = input_vec.astype(np.float32)
        w_f32 = norm_weight.astype(np.float32)
        mean_sq = float((x_f32 * x_f32).sum()) / args.k
        rstd = 1.0 / np.sqrt(mean_sq + eps)
        normed = (x_f32 * rstd) * w_f32
        normed_bf16 = normed.astype(INPUT_DATATYPE).astype(np.float32)
        g_scalars = gate.astype(np.float32) @ normed_bf16
        u_scalars = up.astype(np.float32) @ normed_bf16
        silu = g_scalars * 0.5 * (np.tanh(g_scalars / 2.0) + 1.0)
        output_d = (silu * u_scalars).astype(OUTPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="matvec_swiglu_rms",
            debug_ir=args.debug_ir,
            use_lock_race_condition_fix=True,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_rms],
                expected_outputs=[output_d],
                rtol=0.08,
                atol=1e-2,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            use_lock_race_condition_fix=True,
        )
        backend.compile(mlir_module)
        backend.unload()
