# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# bf16 A x bfp16ebs8 B -> bf16 C mixed-precision GEMM on NPU2.
# B is uint8 at the AIR boundary (no MLIR bfp16ebs8 element type) and the
# kernel reinterprets via aie::block_vector<bfp16ebs8>.

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    BF16Type,
    F32Type,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    StridedLayoutAttr,
    StringAttr,
    UnitAttr,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects import arith
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.scf import ForOp, for_, yield_
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner


def for_disable_pp(start, stop=None, step=None):
    """`for_` variant that tags the underlying scf.for with
    `air.disable_ping_pong`. The K-l1 fill loop's L1 budget can't absorb
    the ping-pong unroll's 2x buffer copies on AIE2P (74 KiB > 64 KiB)."""
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            params[i] = arith.ConstantOp.create_index(p)
    start, stop, step = params
    for_op = ForOp(start, stop, step, [])
    for_op.operation.attributes["air.disable_ping_pong"] = UnitAttr.get()
    with InsertionPoint(for_op.body):
        yield for_op.induction_variable


KERNEL_OBJ_NAME = "mm_bf16_x_bfp16.o"
BFP16_BLOCK = 8
BFP16_BYTES_PER_BLOCK = 9


def bfp_tile_bytes(tile_n, tile_k_l1):
    nelem = tile_n * tile_k_l1
    assert nelem % BFP16_BLOCK == 0
    return (nelem // BFP16_BLOCK) * BFP16_BYTES_PER_BLOCK


def _bf16_block_to_bfp16ebs8(block_f32):
    """Reference per-block scalar packer; use pack_b_bfp16ebs8 in production."""
    assert block_f32.shape == (BFP16_BLOCK,)
    bits = np.frombuffer(block_f32.astype(np.float32).tobytes(), dtype=np.uint32).copy()
    sign = (bits & 0x80000000) != 0
    exp = (bits >> 23) & 0xFF
    mant_explicit = (bits & 0x007FFFFF) | np.where(exp != 0, 0x00800000, 0).astype(
        np.uint32
    )
    max_exp = int(exp.max())
    signed_mant = np.where(
        sign, (~mant_explicit + 1) & 0xFFFFFFFF, mant_explicit
    ).astype(np.int64)
    base = (signed_mant.astype(np.int64) >> (23 - 7 + 1)).astype(np.int64)
    shift = max_exp - exp.astype(np.int64)
    aligned = np.where(shift >= 32, np.where(sign, -1, 0), base >> shift).astype(
        np.int8
    )
    out = np.empty(BFP16_BYTES_PER_BLOCK, dtype=np.uint8)
    out[0] = np.uint8(max_exp)
    out[1:] = aligned.view(np.uint8)
    return out


def pack_b_bfp16ebs8(B_bf16, tile_n, tile_k_l1):
    """[K, N] bf16 -> [N/tile_n, K/tile_k_l1, tile_bytes] uint8 (3D BO).

    Vectorized over all blocks at once. Each 9-byte record packs 8
    K-contiguous elements for one N row (B is consumed transposed).
    """
    K, N = B_bf16.shape
    r = s = t = 8
    assert tile_n % t == 0 and tile_k_l1 % s == 0
    assert K % tile_k_l1 == 0 and N % tile_n == 0
    Nb = N // tile_n
    Kb = K // tile_k_l1
    NB_in = tile_n // t  # n MMUL sub-tiles per tile_n
    KB_in = tile_k_l1 // s  # k MMUL sub-tiles per tile_k_l1
    n_blocks_total = Nb * Kb * NB_in * KB_in  # one MMUL sub-block per row of output

    # Permute B into per-block 8-element views: [Nb, Kb, NB_in, KB_in, t, s].
    # sub[..., n_i, k_i] = B[kb * tile_k_l1 + kbi * s + k_i, nb * tile_n + nbi * t + n_i]
    Bf = B_bf16.astype(np.float32)
    Bv = Bf.reshape(Kb, KB_in, s, Nb, NB_in, t)  # [Kb, KB_in, s, Nb, NB_in, t]
    Bv = np.transpose(Bv, (3, 0, 4, 1, 5, 2))  # [Nb, Kb, NB_in, KB_in, t, s]
    # Each (block_idx, n_i) line of 8 elements gets one bfp16ebs8 9-byte record.
    blocks = np.ascontiguousarray(Bv).reshape(n_blocks_total * t, s)  # [n_records, 8]

    # Vectorized bit math (same per-element logic as the scalar reference).
    bits = blocks.view(np.uint32)  # reinterpret f32 bits
    sign = (bits & 0x80000000) != 0  # [n_records, 8] bool
    exp = ((bits >> 23) & 0xFF).astype(np.int32)  # [n_records, 8]
    mant_explicit = (bits & 0x007FFFFF) | np.where(
        exp != 0, np.uint32(0x00800000), np.uint32(0)
    )
    max_exp = exp.max(axis=1, keepdims=True)  # [n_records, 1] shared exp per block
    signed_mant = np.where(
        sign, (~mant_explicit + np.uint32(1)) & np.uint32(0xFFFFFFFF), mant_explicit
    ).astype(np.int64)
    base = signed_mant >> (23 - 7 + 1)  # 24-bit -> 8-bit-with-sign
    shift = (max_exp - exp).astype(np.int64)  # [n_records, 8] >= 0
    big_shift = shift >= 32
    aligned = np.where(
        big_shift, np.where(sign, -1, 0), base >> np.minimum(shift, 31)
    ).astype(np.int8)

    # Interleave [shared_exp, m0..m7] per record into 9-byte stride.
    records = np.empty((n_blocks_total * t, BFP16_BYTES_PER_BLOCK), dtype=np.uint8)
    records[:, 0] = max_exp.flatten().astype(np.uint8)
    records[:, 1:] = aligned.view(np.uint8)

    # Reassemble: per tile we have NB_in * KB_in * t records, each 9 bytes.
    tb = bfp_tile_bytes(tile_n, tile_k_l1)
    out = records.reshape(Nb, Kb, NB_in * KB_in * t * BFP16_BYTES_PER_BLOCK)
    assert out.shape[2] == tb
    return out


def cpu_reference_from_bfp_packed(B_packed, A_bf16, m, k, n, tile_n, tile_k_l1):
    r = s = t = 8
    NB_in = tile_n // t
    KB_in = tile_k_l1 // s
    Bf = np.zeros((k, n), dtype=np.float32)
    for nb in range(n // tile_n):
        for kb in range(k // tile_k_l1):
            cursor = 0
            for nbi in range(NB_in):
                for kbi in range(KB_in):
                    n0 = nb * tile_n + nbi * t
                    k0 = kb * tile_k_l1 + kbi * s
                    sub_T = np.zeros((t, s), dtype=np.float32)
                    for n_i in range(t):
                        block = B_packed[
                            nb, kb, cursor : cursor + BFP16_BYTES_PER_BLOCK
                        ]
                        cursor += BFP16_BYTES_PER_BLOCK
                        shared_exp = int(block[0])
                        mults = (
                            (1.0 * (1 << (shared_exp - 127)))
                            if shared_exp >= 127
                            else (1.0 / (1 << (127 - shared_exp)))
                        ) / 64.0
                        mants = block[1:].view(np.int8).astype(np.int32)
                        sub_T[n_i, :] = mants.astype(np.float32) * mults
                    Bf[k0 : k0 + s, n0 : n0 + t] = sub_T.T
    C = A_bf16.astype(np.float32) @ Bf
    return C.astype(bfloat16)


@module_builder
def build_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
):
    """bf16 A x bfp16ebs8 B mixed-precision GEMM with f32 L1 accumulator."""
    r, s, t = 8, 8, 8
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert tile_m % (2 * r) == 0
    assert tile_n % (2 * t) == 0
    assert tile_k_l1 % s == 0

    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    N_div = n // tile_n
    K_div = k // tile_k_l1

    bf16_ty = BF16Type.get()
    f32_ty = F32Type.get()
    u8_ty = IntegerType.get_signless(8)

    # L3 (caller-facing)
    A_l3_ty = MemRefType.get([m, k], bf16_ty)
    B_l3_ty = MemRefType.get([N_div, K_div, tile_bytes], u8_ty)
    C_l3_ty = MemRefType.get([m, n], bf16_ty)

    # L1 MMUL block layouts. C is N-outer M-inner so the L1->L2 drain DMA
    # fits in the AIE2P 3-dim BD step limit.
    l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
    a_l1_size = [1, 1, tile_m // r, tile_k_l1 // s, r, s]
    b_l1_size = [tile_bytes]
    c_l1_size = [1, 1, tile_n // t, tile_m // r, r, t]
    c_herd_l1_size = [herd_m, herd_n, tile_n // t, tile_m // r, r, t]

    l1MemrefTyA = MemRefType.get(a_l1_size, bf16_ty, memory_space=l1_ms)
    l1MemrefTyB = MemRefType.get(b_l1_size, u8_ty, memory_space=l1_ms)
    c_subview_layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(),
        [
            tile_m * tile_n * herd_n,
            tile_m * tile_n,
            tile_m * t,  # n_b stride: skip full M-block column = tile_m * t
            r * t,  # m_b stride: skip one M-block = r * t
            t,  # m_i stride: skip one row within block
            1,  # n_i stride
        ],
    )
    l1MemrefTyC = MemRefType.get(
        c_l1_size, f32_ty, memory_space=l1_ms, layout=c_subview_layout
    )
    l1MemrefTyCHerd = MemRefType.get(c_herd_l1_size, f32_ty, memory_space=l1_ms)
    c_drain_l1_size = [herd_m, herd_n, tile_n // t, tile_m // r, r, t]
    l1MemrefTyCDrainHerd = MemRefType.get(c_drain_l1_size, bf16_ty, memory_space=l1_ms)

    # External funcs: matmul (block-layout I/O) + f32->bf16 row-major drain
    # operating on the flattened tile (length tile_m*tile_n).
    matmul_func = FuncOp(
        "matmul_bf16_x_bfp16_packed_f32",
        ([l1MemrefTyA, l1MemrefTyB, l1MemrefTyC], []),
        visibility="private",
    )
    matmul_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
    matmul_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # CallOp(zero_func) over linalg.fill: kernel writes contiguously, avoids
    # affine-codegen edge cases on the strided 6D subview.
    zero_func = FuncOp(
        "zero_vectorized_f32_mn",
        ([l1MemrefTyC], []),
        visibility="private",
    )
    zero_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
    zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # The drain kernel reads/writes flat tile_m*tile_n elements; pass per-PE
    # subviews from the segment-level buffers.
    drain_acc_ty = MemRefType.get(
        c_l1_size, f32_ty, memory_space=l1_ms, layout=c_subview_layout
    )
    drain_dst_layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(),
        [
            tile_m * tile_n * herd_n,
            tile_m * tile_n,
            tile_n * r,
            r * t,
            t,
            1,
        ],
    )
    drain_dst_ty = MemRefType.get(
        c_l1_size, bf16_ty, memory_space=l1_ms, layout=drain_dst_layout
    )
    f32_to_bf16_func = FuncOp(
        "f32_to_bf16_mn",
        ([drain_acc_ty, drain_dst_ty], []),
        visibility="private",
    )
    f32_to_bf16_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
    f32_to_bf16_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(A_l3_ty, B_l3_ty, C_l3_ty)
    def matmul_bf16_x_bfp16(arg0, arg1, arg2):
        launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_b_data,
            l3_c_data,
        ):
            @segment(
                name="matmul_seg",
                operands=[launch_ivx, launch_ivy, l3_a_data, l3_b_data, l3_c_data],
            )
            def segment_body(
                launch_ivx_s,
                launch_ivy_s,
                l3_a_data_s,
                l3_b_data_s,
                l3_c_data_s,
            ):
                k_per_l2 = tile_k_l2 // tile_k_l1
                a_size_l2 = [herd_m, 1, tile_m, tile_k_l2]
                b_size_l2 = [1, herd_n, k_per_l2, tile_bytes]
                c_size_l2 = [herd_m, herd_n, tile_m, tile_n]
                l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)
                l2MemrefTyA = MemRefType.get(a_size_l2, bf16_ty, memory_space=l2_ms)
                l2MemrefTyB = MemRefType.get(b_size_l2, u8_ty, memory_space=l2_ms)
                l2MemrefTyC = MemRefType.get(c_size_l2, bf16_ty, memory_space=l2_ms)

                l2_a_data = AllocOp(l2MemrefTyA, [], [])
                l2_b_data = AllocOp(l2MemrefTyB, [], [])
                l2_c_data = AllocOp(l2MemrefTyC, [], [])
                # Segment-shared L1_C f32 accumulator persists across the
                # three herd invocations; drain holds the bf16-narrowed result.
                l1_c_data = AllocOp(l1MemrefTyCHerd, [], [])
                l1_c_drain = AllocOp(l1MemrefTyCDrainHerd, [], [])

                launch_ix_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                launch_iy_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_n * herd_n),
                        )
                    ],
                )
                launch_offset_x = affine_apply(launch_ix_map, [launch_ivx_s])
                launch_offset_y = affine_apply(launch_iy_map, [launch_ivy_s])

                n_outer_off_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(herd_n),
                        )
                    ],
                )
                n_outer_off = affine_apply(n_outer_off_map, [launch_ivy_s])

                # ---- Herd #1: zero-init L1 C f32 (segment-shared).
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_c_data],
                )
                def herd_init(_tx, _ty, _sx, _sy, _l1_c):
                    l1_c_subview = subview(
                        _l1_c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, tile_n // t, tile_m // r, r, t],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    CallOp(zero_func, [l1_c_subview])

                # ---- Segment-level K-l2 loop (matches bf16).
                for i in for_(0, k // tile_k_l2):
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
                    # K-l2 chunk offset in B (in units of K-l1 tile slots).
                    k_l2_b_off_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(k_per_l2),
                            )
                        ],
                    )
                    k_l2_b_off = affine_apply(k_l2_b_off_map, [i])
                    dma_memcpy_nd(
                        l2_a_data,
                        l3_a_data_s,
                        src_offsets=[0, 0, launch_offset_x, reduction_offset],
                        src_sizes=[herd_m, 1, tile_m, tile_k_l2],
                        src_strides=[k * tile_m, tile_k_l2, k, 1],
                    )
                    dma_memcpy_nd(
                        l2_b_data,
                        l3_b_data_s,
                        src_offsets=[0, n_outer_off, k_l2_b_off, 0],
                        src_sizes=[1, herd_n, k_per_l2, tile_bytes],
                        src_strides=[
                            K_div * tile_bytes,
                            K_div * tile_bytes,
                            tile_bytes,
                            1,
                        ],
                    )

                    # ---- Herd #2: compute (K-l1 loop inside).
                    @herd(
                        name="herd_0",
                        sizes=[herd_m, herd_n],
                        operands=[l1_c_data, l2_a_data, l2_b_data],
                    )
                    def herd_compute(
                        _tx,
                        _ty,
                        _sx,
                        _sy,
                        _l1_c,
                        _l2_a,
                        _l2_b,
                    ):
                        _l1_a = AllocOp(l1MemrefTyA, [], [])
                        _l1_b = AllocOp(l1MemrefTyB, [], [])
                        for j in for_disable_pp(0, k_per_l2):
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
                            dma_memcpy_nd(
                                _l1_a,
                                _l2_a,
                                src_offsets=[_tx, 0, 0, 0, 0, reduction_l1_offset],
                                src_sizes=[
                                    1,
                                    1,
                                    tile_m // r,
                                    tile_k_l1 // s,
                                    r,
                                    s,
                                ],
                                src_strides=[
                                    tile_m * tile_k_l2,
                                    tile_m * tile_k_l2,
                                    tile_k_l2 * r,
                                    s,
                                    tile_k_l2,
                                    1,
                                ],
                            )
                            # L2->L1 B: per-PE per-K-l1-chunk byte tile.
                            dma_memcpy_nd(
                                _l1_b,
                                _l2_b,
                                src_offsets=[0, _ty, j, 0],
                                src_sizes=[1, 1, 1, tile_bytes],
                                src_strides=[
                                    herd_n * k_per_l2 * tile_bytes,
                                    k_per_l2 * tile_bytes,
                                    tile_bytes,
                                    1,
                                ],
                            )
                            l1_c_subview = subview(
                                _l1_c,
                                offsets=[_tx, _ty, 0, 0, 0, 0],
                                sizes=[
                                    1,
                                    1,
                                    tile_n // t,
                                    tile_m // r,
                                    r,
                                    t,
                                ],
                                strides=[1, 1, 1, 1, 1, 1],
                            )
                            CallOp(matmul_func, [_l1_a, _l1_b, l1_c_subview])
                            yield_([])

                        DeallocOp(_l1_a)
                        DeallocOp(_l1_b)

                    yield_([])

                # ---- Herd #3: drain f32 L1 C -> bf16 L1 C drain -> L2.
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_c_data, l1_c_drain, l2_c_data],
                )
                def herd_drain(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_c,
                    _l1_c_drain,
                    _l2_c,
                ):
                    l1_c_acc_sv = subview(
                        _l1_c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, tile_n // t, tile_m // r, r, t],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    l1_c_drain_sv = subview(
                        _l1_c_drain,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[1, 1, tile_n // t, tile_m // r, r, t],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    CallOp(f32_to_bf16_func, [l1_c_acc_sv, l1_c_drain_sv])
                    # L1 (block) -> L2 (row-major), 6D src permute matches bf16.
                    dma_memcpy_nd(
                        _l2_c,
                        _l1_c_drain,
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
                            tile_m // r,
                            r,
                            tile_n // t,
                            t,
                        ],
                        # 6D src strides chosen so the BD optimizer collapses
                        # (m_b, m_i) into a single dim, fitting AIE2P 3-dim BDs.
                        src_strides=[
                            herd_n * tile_m * tile_n,
                            tile_m * tile_n,
                            r * t,  # m_b: skip one M-block
                            t,  # m_i: next row within block
                            tile_m * t,  # n_b: skip one M-block column
                            1,  # n_i
                        ],
                    )

                dma_memcpy_nd(
                    l3_c_data_s,
                    l2_c_data,
                    dst_offsets=[launch_offset_x, launch_offset_y],
                    dst_sizes=[herd_m * tile_m, herd_n * tile_n],
                    dst_strides=[n, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_m, tile_m, herd_n, tile_n],
                    src_strides=[
                        tile_m * herd_n * tile_n,
                        tile_n,
                        tile_m * tile_n,
                        1,
                    ],
                )

                DeallocOp(l2_a_data)
                DeallocOp(l2_b_data)
                DeallocOp(l2_c_data)
                DeallocOp(l1_c_data)
                DeallocOp(l1_c_drain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=32, dest="tile_m")
    parser.add_argument("--tile-k-l2", type=int, default=128, dest="tile_k_l2")
    parser.add_argument("--tile-k-l1", type=int, default=128, dest="tile_k_l1")
    parser.add_argument("--tile-n", type=int, default=32, dest="tile_n")
    parser.add_argument("--herd-m", type=int, default=2, dest="herd_m")
    parser.add_argument("--herd-n", type=int, default=4, dest="herd_n")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-and-run", "compile-only", "compile-and-xclbin"],
        default="compile-and-run",
        dest="compile_mode",
    )
    args = parser.parse_args()

    module = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    A = (np.random.randn(args.m, args.k) * (1.0 / np.sqrt(args.k))).astype(bfloat16)
    B = (np.random.randn(args.k, args.n) * (1.0 / np.sqrt(args.k))).astype(bfloat16)

    B_packed = pack_b_bfp16ebs8(B, args.tile_n, args.tile_k_l1)
    C_ref = cpu_reference_from_bfp_packed(
        B_packed, A, args.m, args.k, args.n, args.tile_n, args.tile_k_l1
    )

    # runtime_loop_tiling_sizes=[2,2] keeps the runtime DMA loop within the
    # ~4-task shim BD pool at large launch axes.
    common_kwargs = dict(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="xclbin",
        stack_size=2048,
        runtime_loop_tiling_sizes=[2, 2],
        instance_name="matmul_bf16_x_bfp16",
    )

    if args.compile_mode == "compile-only":
        backend = XRTBackend(**common_kwargs)
        backend.compile(module)
        backend.unload()
        sys.exit(0)

    if args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(**common_kwargs)
        backend.compile(module)
        backend.unload()
        sys.exit(0)

    runner = XRTRunner(**common_kwargs)
    sys.exit(
        runner.run_test(
            module,
            inputs=[A, B_packed],
            expected_outputs=[C_ref],
            rtol=0.1,
            atol=0.05,
            max_mismatch_percentage=0.05,
            min_correlation=0.999,
        )
    )
