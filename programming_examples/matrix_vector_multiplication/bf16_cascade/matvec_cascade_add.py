# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cascade-based GEMV with fused residual add: D[M] = A[M,K] @ B[K] + R[M]
# BF16 input/output, accfloat accumulation.
#
# R is added at the cascade HEAD as the initial accumulator; the cascade
# carries R + Σpartials south to the TAIL which writes D.
#
# A and R share the ar_L3toL2 / ar_L2toL1 channels (per-(col, cascade_row)
# bundle slot), each fill targeting its own L2 buffer to avoid shared-
# buffer write/read races between independent lock pairs.

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
    """Single-row bf16 dot product into f32 accumulator (verbatim from
    matvec_cascade.py). Returns the f32 horizontal sum."""
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
    assert k % n_cascade == 0, f"K ({k}) must be divisible by n_cascade ({n_cascade})"
    assert (
        k_chunk % 64 == 0
    ), f"k_chunk ({k_chunk}) must be divisible by 64 (vector width)"

    bytes_per_elem_in = np.dtype(np_dtype_in).itemsize
    bytes_per_elem_out = np.dtype(np_dtype_out).itemsize
    # Per-(col, cascade_row) L2 chunk buffer (m_input * k_chunk bf16),
    # plus one R buffer per col.
    ar_l2_chunk_bytes = m_input * k_chunk * bytes_per_elem_in
    d_l2_bytes = herd_cols * tile_m * bytes_per_elem_out
    # Per col: 1 R chunk buf + n_cascade A chunk bufs (each m_input*k_chunk).
    l2_per_col = (1 + n_cascade) * ar_l2_chunk_bytes
    L2_CAPACITY = 512 * 1024
    assert herd_cols * l2_per_col + d_l2_bytes <= L2_CAPACITY, (
        f"L2 capacity exceeded: per-col={l2_per_col}B × {herd_cols} cols "
        f"+ D={d_l2_bytes}B = {herd_cols * l2_per_col + d_l2_bytes}B "
        f"> {L2_CAPACITY}B."
    )

    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)
    f32_type = F32Type.get()
    index_type = IndexType.get()

    # L3 MemRefTypes
    memrefTyA = MemRefType.get([m, k], xrt_dtype_in)
    memrefTyB = MemRefType.get([k], xrt_dtype_in)
    memrefTyR = MemRefType.get([m], xrt_dtype_in)
    memrefTyD = MemRefType.get([m], xrt_dtype_out)

    # L2 chunk buffers: shape [m_input, k_chunk] (sized for A; R fills
    # only first m_input cells). Per-fill separate buffers — see segment
    # body below for the rationale.
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyARchunk = MemRefType.get(
        shape=[m_input, k_chunk],
        element_type=xrt_dtype_in,
        memory_space=l2_mem_space,
    )
    l2MemrefTyD = MemRefType.get(
        shape=[herd_cols, tile_m],
        element_type=xrt_dtype_out,
        memory_space=l2_mem_space,
    )

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get(
        shape=[m_input, k_chunk],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyB = MemRefType.get(
        shape=[k_chunk],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyD = MemRefType.get(
        shape=[tile_m],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    # Per j_m iter, HEAD's L1 R buffer holds m_input elements (one per
    # output row computed this iter). m_input * sizeof(bf16) must be
    # ≥ 4 bytes for AIE DMA alignment → m_input ≥ 2 required.
    assert (
        m_input * np.dtype(np_dtype_in).itemsize >= 4
    ), f"m_input ({m_input}) * sizeof({np_dtype_in}) must be ≥ 4 bytes (AIE DMA alignment)"
    l1MemrefTyR = MemRefType.get(
        shape=[m_input],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
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

    # ar_L3toL2: per-col channel from L3.
    # ar_L2toL1: per-(col, cascade_row) memtile MM2S → compute tile.
    # R goes to slot (col, n_cascade-1) = HEAD; A_r goes to (col, r).
    channel_decl("ar_L3toL2", size=[herd_cols])
    channel_decl("ar_L2toL1", size=[herd_cols, n_cascade])
    channel_decl(
        "chan_cascade",
        size=[herd_cols, n_cascade - 1],
        channel_type="npu_cascade",
    )

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyR, memrefTyD)
    def matvec_cascade_add_bf16(arg0, arg1, arg2, arg3):

        launch_size = [m // tile_m // herd_cols, 1]

        @launch(operands=[arg0, arg1, arg2, arg3], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_b_data,
            l3_r_data,
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

            # L3-side puts on ar_L3toL2[col]: per j_m iter, R then per-ty A
            # Order MUST match memtile-side gets in segment body.
            for col in range(herd_cols):
                c_col_idx_l = arith.ConstantOp.create_index(col)
                for j_m_v in range(tile_m // m_input):
                    add_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(col * tile_m + j_m_v * m_input),
                            )
                        ],
                    )
                    r_off = affine_apply(add_map, [launch_offset_m_l])
                    ChannelPut(
                        "ar_L3toL2",
                        l3_r_data,
                        indices=[c_col_idx_l],
                        offsets=[r_off],
                        sizes=[m_input],
                        strides=[1],
                    )
                    # Reverse order so HEAD's K-slice is sent right after R
                    # (segment-side memtile loop reads in same reverse order).
                    for ty_v in reversed(range(n_cascade)):
                        a_off_m = affine_apply(add_map, [launch_offset_m_l])
                        ChannelPut(
                            "ar_L3toL2",
                            l3_a_data,
                            indices=[c_col_idx_l],
                            offsets=[a_off_m, ty_v * k_chunk],
                            sizes=[m_input, k_chunk],
                            strides=[k, 1],
                        )

            @segment(
                name="matvec_cascade_add_seg",
                operands=[launch_ivx, l3_b_data, l3_d_data],
            )
            def segment_body(
                launch_ivx_s,
                l3_b_data_s,
                l3_d_data_s,
            ):
                # Per-fill L2 buffers: one R buffer per col + one A buffer
                # per (col, cascade_row). Required so each S2MM/MM2S BD
                # pair owns its own L2 buffer; sharing one L2 buffer
                # across BD pairs with independent lock pairs lets later
                # fills overwrite the buffer before earlier reads drain.
                r_l2_bufs = [
                    AllocOp(l2MemrefTyARchunk, [], []) for _ in range(herd_cols)
                ]
                a_l2_bufs = [
                    [AllocOp(l2MemrefTyARchunk, [], []) for _ in range(n_cascade)]
                    for _ in range(herd_cols)
                ]
                l2_d_data = AllocOp(l2MemrefTyD, [], [])

                # Memtile streaming: R to (col, HEAD), A_r to (col, r).
                c_head_idx = arith.ConstantOp.create_index(n_cascade - 1)
                for col in range(herd_cols):
                    c_col_idx = arith.ConstantOp.create_index(col)
                    r_l2 = r_l2_bufs[col].result
                    for j_m_v in range(tile_m // m_input):
                        # R: GET → r_l2 (separate buffer), PUT → (col, HEAD)
                        ChannelGet(
                            "ar_L3toL2",
                            r_l2,
                            indices=[c_col_idx],
                            offsets=[0, 0],
                            sizes=[1, m_input],
                            strides=[k_chunk, 1],
                        )
                        ChannelPut(
                            "ar_L2toL1",
                            r_l2,
                            indices=[c_col_idx, c_head_idx],
                            offsets=[0, 0],
                            sizes=[1, m_input],
                            strides=[k_chunk, 1],
                        )
                        # A per cascade row: GET → its own a_l2 buffer.
                        # Reverse order keeps HEAD's A adjacent to R in the
                        # L3-side put stream (compiler pairs adjacent puts).
                        for ty_v in reversed(range(n_cascade)):
                            c_ty_idx = arith.ConstantOp.create_index(ty_v)
                            a_l2 = a_l2_bufs[col][ty_v].result
                            ChannelGet(
                                "ar_L3toL2",
                                a_l2,
                                indices=[c_col_idx],
                                offsets=[0, 0],
                                sizes=[m_input, k_chunk],
                                strides=[k_chunk, 1],
                            )
                            ChannelPut(
                                "ar_L2toL1",
                                a_l2,
                                indices=[c_col_idx, c_ty_idx],
                                offsets=[0, 0],
                                sizes=[m_input, k_chunk],
                                strides=[k_chunk, 1],
                            )

                # L1 buffers (passed into herd as operands)
                l1_a_data = AllocOp(l1MemrefTyA, [], [])
                l1_b_data = AllocOp(l1MemrefTyB, [], [])
                l1_d_data = AllocOp(l1MemrefTyD, [], [])
                l1_r_data = AllocOp(l1MemrefTyR, [], [])
                l1_scratch = AllocOp(l1MemrefTyScratch, [], [])
                l1_recv = AllocOp(l1MemrefTyScratch, [], [])

                @herd(
                    name="herd_0",
                    sizes=[herd_cols, n_cascade],
                    operands=[
                        l1_a_data,
                        l1_b_data,
                        l1_d_data,
                        l1_r_data,
                        l1_scratch,
                        l1_recv,
                        l3_b_data_s,
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
                    _l1_d,
                    _l1_r,
                    _l1_scratch,
                    _l1_recv,
                    _l3_b,
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

                    # B (L3->L1) — per-tile k_chunk slice (broadcast within row
                    # by air-broadcast-detection).
                    dma_memcpy_nd(
                        _l1_b,
                        _l3_b,
                        src_offsets=[k_offset],
                        src_sizes=[k_chunk],
                        src_strides=[1],
                    )

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

                    dot_args = (
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
                    )

                    # Hot loop: per j_m iter, HEAD receives R + A; other
                    # cascade tiles just receive A. Cascade pipeline runs
                    # with R added at HEAD's init accumulator.
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

                        # R receive: HEAD only, point-to-point
                        if_head_r = affine.AffineIfOp(head_set, cond_operands=[ty])
                        with InsertionPoint(if_head_r.then_block):
                            ChannelGet(
                                "ar_L2toL1",
                                _l1_r,
                                indices=[tx, ty],
                            )
                            affine.AffineYieldOp([])

                        # A receive: each tile gets its own
                        ChannelGet(
                            "ar_L2toL1",
                            _l1_a,
                            indices=[tx, ty],
                        )

                        # === Cascade compute ===
                        # HEAD (ty == n_cascade-1): partial = A·B; init acc
                        # with R; put cascade.
                        # MIDDLE: get cascade; partial; sum; put cascade.
                        # TAIL (ty == 0): get cascade; partial; sum; write D.
                        cmp_first = arith.CmpIOp(arith.CmpIPredicate.eq, ty, last_ty)
                        if_first = scf.IfOp(cmp_first, has_else=True)
                        with InsertionPoint(if_first.then_block):
                            # HEAD: partial + R → scratch → cascade
                            for row in range_(0, m_input):
                                partial_sum = compute_partial_dot(row, *dot_args)
                                sub_r = subview(_l1_r, [row], [1], [1])
                                r_val_bf16 = memref_load(sub_r, [c0])
                                r_val_f32 = arith.extf(f32_type, r_val_bf16)
                                init_acc = arith.addf(partial_sum, r_val_f32)
                                sub_scratch = subview(_l1_scratch, [row], [1], [1])
                                memref_store(init_acc, sub_scratch, [c0])
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
                                # TAIL: get cascade, add own partial, write D
                                # (no R add — R was added at HEAD)
                                ChannelGet(
                                    "chan_cascade",
                                    _l1_recv,
                                    indices=[tx, ty],
                                )

                                for row in range_(0, m_input):
                                    partial_sum = compute_partial_dot(row, *dot_args)
                                    sub_recv = subview(_l1_recv, [row], [1], [1])
                                    recv_val = memref_load(sub_recv, [c0])
                                    total_f32 = arith.addf(recv_val, partial_sum)
                                    total_bf16 = arith.truncf(xrt_dtype_out, total_f32)
                                    out_idx = affine_apply(
                                        row_out_map, [j_m_offset, row]
                                    )
                                    sub_d_out = subview(_l1_d, [out_idx], [1], [1])
                                    memref_store(total_bf16, sub_d_out, [c0])
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
                                    partial_sum = compute_partial_dot(row, *dot_args)
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

                    # ty=0 tiles write D to L2 (same pattern as matvec_cascade.py)
                    cmp_writer = arith.CmpIOp(arith.CmpIPredicate.eq, ty, c0)
                    if_writer = scf.IfOp(cmp_writer)
                    with InsertionPoint(if_writer.then_block):
                        dma_memcpy_nd(
                            _l2_d,
                            _l1_d,
                            dst_offsets=[tx, 0],
                            dst_sizes=[1, tile_m],
                            dst_strides=[tile_m, 1],
                            src_offsets=[],
                            src_sizes=[tile_m],
                            src_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_acc_tmp)

                # L2 -> L3: D writeback for this launch slice.
                launch_ivx_map_s = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_cols),
                        )
                    ],
                )
                launch_offset_m_d = affine_apply(launch_ivx_map_s, [launch_ivx_s])
                dma_memcpy_nd(
                    l3_d_data_s,
                    l2_d_data,
                    dst_offsets=[launch_offset_m_d],
                    dst_sizes=[herd_cols * tile_m],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_cols, tile_m],
                    src_strides=[tile_m, 1],
                )

                for r_l2 in r_l2_bufs:
                    DeallocOp(r_l2)
                for col_bufs in a_l2_bufs:
                    for a_l2 in col_bufs:
                        DeallocOp(a_l2)
                DeallocOp(l2_d_data)
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_d_data)
                DeallocOp(l1_r_data)
                DeallocOp(l1_scratch)
                DeallocOp(l1_recv)


if __name__ == "__main__":
    M = 2048
    K = 8192
    TILE_M = 2
    M_INPUT = 1
    HERD_COLS = 8
    N_CASCADE = 4
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="matvec_cascade_add.py",
        description="Cascade BF16 GEMV with fused residual add: D = A @ B + R",
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
        default="xclbin",
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
        input_a = (np.random.randn(args.m, args.k) * 4).astype(INPUT_DATATYPE)
        input_b = (np.random.randn(args.k) * 4).astype(INPUT_DATATYPE)
        input_r = (np.random.randn(args.m) * 4).astype(INPUT_DATATYPE)
        output_d = (
            np.dot(input_a.astype(np.float32), input_b.astype(np.float32))
            + input_r.astype(np.float32)
        ).astype(OUTPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            # Per-col ar_L3toL2 carries 1 R + n_cascade A puts per launch
            # iter. Shim DMA limit = 4 BDs queued per channel; tiling=[1,1]
            # keeps it at (1+n_cascade) ≤ 4 for n_cascade up to 3, and at
            # n_cascade=4 the BDs fit just within the limit.
            runtime_loop_tiling_sizes=[1, 1],
            output_format=args.output_format,
            instance_name="matvec_cascade_add_bf16",
            debug_ir=args.debug_ir,
            use_lock_race_condition_fix=True,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b, input_r],
                expected_outputs=[output_d],
                rtol=0.04,
                atol=1e-3,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[1, 1],
            output_format=args.output_format,
            use_lock_race_condition_fix=True,
        )
        backend.compile(mlir_module)
        backend.unload()
