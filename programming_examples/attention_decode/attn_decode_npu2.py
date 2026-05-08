# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from math import cos, sin, sqrt, exp

import numpy as np
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store, subview, collapse_shape
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    fma as vector_fma,
    reduction as vector_reduction,
)
from air.dialects import arith as arith_dialect, math as math_dialect
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.dialects import linalg
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from ml_dtypes import bfloat16

range_ = for_

# RoPE freq table for dk=64. out[i] = pos / rope_base^(2i/dk), rope_base=500000
# (LLaMA-3 / LLaMA-3.2 rope_theta). Mirrors freq_table_dk64 in attn_decode_npu2.cc.
_FREQ_TABLE_DK64_BF16 = np.array(
    [
        1.000000,
        0.663601,
        0.440367,
        0.292228,
        0.193923,
        0.128687,
        0.085397,
        0.056670,
        0.037606,
        0.024955,
        0.016560,
        0.010990,
        0.007293,
        0.004839,
        0.003211,
        0.002131,
        0.001414,
        0.000938,
        0.000623,
        0.000413,
        0.000274,
        0.000182,
        0.000121,
        0.000080,
        0.000053,
        0.000035,
        0.000023,
        0.000016,
        0.000010,
        0.000007,
        0.000005,
        0.000003,
    ],
    dtype=bfloat16,
)


@module_builder
def build_module(
    k,
    n,
    tile_k,
    tile_n,
    seq_len,
    np_dtype_in,
    np_dtype_vm_acc,
    np_dtype_out,
    pos_host,
    group_size=4,
    nkv=2,
):

    GROUP_SIZE = group_size
    GEMV_COUNT = GROUP_SIZE + 2  # group_size Q heads + 1 K + 1 V
    KV_COUNT = 2  # K and V (cache writeback)
    NKV = nkv

    assert k % tile_k == 0
    assert n % tile_n == 0
    # xrms is padded to match the weight packet shape [tile_k, tile_n] so
    # bL3ToL2 carries one uniform packet shape across xrms and weights —
    # the channel becomes a single self-loop BD chain (no repeat_count, no
    # lightweight-reset PDI between tokens). Real data (x_raw, w_rms) lives
    # in the first 2*k flat elements of the [tile_k, tile_n] packet; the
    # remainder is padding read by the kernel demux but ignored.
    assert tile_k * tile_n >= 2 * k, (
        f"xrms padded packet ([tile_k, tile_n]={tile_k * tile_n} elems) must "
        f"hold 2*k={2 * k} bf16 (x_raw + w_rms)"
    )
    xrms_size = [k]
    xrms_pack_size = [tile_k, tile_n]
    b_size = [NKV, GEMV_COUNT, k, n]
    kv_cache_size = [NKV, seq_len, n]
    xb_size_l3 = [NKV, GROUP_SIZE, n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_vm_acc = type_mapper(np_dtype_vm_acc)
    xrt_dtype_out = type_mapper(np_dtype_out)

    # L3 MemRefTypes
    memrefTyXRmsPack = MemRefType.get(xrms_pack_size, xrt_dtype_in)
    memrefTyB = MemRefType.get(b_size, xrt_dtype_in)
    memrefTyKCache = MemRefType.get(kv_cache_size, xrt_dtype_out)
    memrefTyVCache = MemRefType.get(kv_cache_size, xrt_dtype_out)

    # Channels: every memtile/L1 channel is single-task self-loop. xrms is
    # fused onto the weight stream as one [tile_k, tile_n]-shaped padded
    # packet so bL3ToL2's BD config is uniform across all 49 fires per
    # token (1 padded xrms + 48 weight chunks). Each col reads its own
    # xrms copy from L3 (no broadcast); rms is computed on the gqa tile.
    #   - aL3ToL2 / aL2ToL1: per-col K + V cache rows ([tile_n] packets).
    #   - bL3ToL2 / bL2ToL1: per-col weights + padded xrms head packet
    #     ([tile_k, tile_n] packets, all identical config).
    #   - cL1ToL2 / cL2ToL3: KV writeback. dL1ToL2 / dL2ToL3: xb output.
    channel("aL3ToL2", size=[NKV])
    channel("bL3ToL2", size=[NKV])
    channel("aL2ToL1", size=[NKV])
    channel("bL2ToL1", size=[NKV])
    channel("cL1ToL2", size=[NKV])
    channel("cL2ToL3", size=[NKV])
    channel("dL1ToL2", size=[NKV])
    channel("dL2ToL3", size=[NKV])

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # x_norm L1 buffer (full k=n_in, cascade-received as one shot from rms);
    # weight L1 buffer (per inner-k chunk, sized [tile_k, tile_n]). For
    # backward compatibility tile_k may equal k; in that case the inner-k
    # loop has a single iteration and x_offset is always 0.
    a_l1_size = [k]
    a_chunk_l1_size = [tile_k]
    b_l1_size = [tile_k, tile_n]
    l1MemrefTyA = MemRefType.get(
        shape=a_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyAChunk = MemRefType.get(
        shape=a_chunk_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyB = MemRefType.get(
        shape=b_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    c_l1_size = [tile_n]
    l1MemrefTyC = MemRefType.get(
        shape=c_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyACC = MemRefType.get(
        shape=c_l1_size,
        element_type=xrt_dtype_vm_acc,
        memory_space=l1_mem_space,
    )
    # GEMV output: [GROUP_SIZE Q heads + 1 K + 1 V, dk]
    l1MemrefTyThreeByFortyEightVec = MemRefType.get(
        shape=[GEMV_COUNT, n],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyHSByTwo = MemRefType.get(
        shape=[32],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyVec = MemRefType.get(
        shape=[8],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    cosf_poly_func = FuncOp(
        "cosf_bf16_32_16",
        ([l1MemrefTyHSByTwo, l1MemrefTyHSByTwo], []),
        visibility="private",
    )
    sinf_poly_func = FuncOp(
        "sinf_bf16_32_16",
        ([l1MemrefTyHSByTwo, l1MemrefTyHSByTwo], []),
        visibility="private",
    )
    shuffle_apply_rope_poly_func = FuncOp(
        "shuffle_apply_rope_bf16_64",
        (
            [
                T.i32(),
                l1MemrefTyHSByTwo,
                l1MemrefTyHSByTwo,
                l1MemrefTyThreeByFortyEightVec,
            ],
            [],
        ),
        visibility="private",
    )
    l1MemrefTyQKV = MemRefType.get(
        shape=[GEMV_COUNT, tile_n],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTySharedL1BDBuf = MemRefType.get(
        shape=[64],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    # GQA: q has GROUP_SIZE rows, attn has GROUP_SIZE rows of seq_len, xb
    # has GROUP_SIZE rows. K/V cache rows are shared across the group.
    q_l1_size = [GROUP_SIZE, n]
    xb_l1_size = [GROUP_SIZE, n]
    attn_l1_size = [GROUP_SIZE, seq_len]
    # L3 xb is per-col: [NKV, GROUP_SIZE, n]. Each col's xb output writes to
    # arg4[col, :, :].
    memrefTyXb = MemRefType.get(xb_size_l3, xrt_dtype_out)
    l1MemrefTyQ = MemRefType.get(
        shape=q_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyXb = MemRefType.get(
        shape=xb_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyAttn = MemRefType.get(
        shape=attn_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    attn_func = FuncOp(
        "attn_1_group",
        ([l1MemrefTyQ, l1MemrefTySharedL1BDBuf, T.i32(), l1MemrefTyAttn], []),
        visibility="private",
    )
    attn2_func = FuncOp(
        "attn_2_group",
        ([l1MemrefTyAttn, l1MemrefTySharedL1BDBuf, T.i32(), l1MemrefTyXb], []),
        visibility="private",
    )
    # rms now runs on the gqa tile: receives a padded [tile_k, tile_n] xrms
    # packet on bL2ToL1 (same channel as weights), demuxes into x_raw + w_rms
    # L1 buffers, then runs simple_rms_bf16 in-place (x_raw -> x_norm).
    l1MemrefTyXRms = MemRefType.get(
        shape=xrms_size, element_type=xrt_dtype_in, memory_space=l1_mem_space
    )

    for func in [
        cosf_poly_func,
        sinf_poly_func,
        shuffle_apply_rope_poly_func,
        attn_func,
        attn2_func,
    ]:
        func.attributes["link_with"] = StringAttr.get("attn_decode_npu2.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(
        memrefTyXRmsPack,
        memrefTyB,
        memrefTyKCache,
        memrefTyVCache,
        memrefTyXb,
    )
    def mha_bf16(arg_xrms, arg_b, arg_kc, arg_vc, arg_xb):

        launch_size = [1, 1]

        @launch(
            operands=[arg_xrms, arg_b, arg_kc, arg_vc, arg_xb],
            sizes=launch_size,
        )
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_xrms_data,
            l3_b_data,
            l3_k_cache_data,
            l3_v_cache_data,
            l3_xb_data,
        ):

            # Affine map for launch iv
            launch_ivy_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(tile_n),
                    )
                ],
            )
            launch_offset_y = affine_apply(launch_ivy_map, [launch_ivy])

            # Per-col data movement: Python-unrolled over NKV. Each col gets
            # its own indices=[c_tx_i] slot on the size=[NKV] channels.
            for tx_i in range(NKV):
                c_tx_i = arith.ConstantOp.create_index(tx_i)

                # bL3ToL2 packet 0: padded xrms ([tile_k, tile_n] = same
                # shape as a weight chunk). Each col reads its own copy from
                # the shared L3 xrms buffer (no broadcast). Real x_raw +
                # w_rms occupy the first 2*k flat elements; the rest is
                # padding the demux kernel ignores.
                ChannelPut(
                    "bL3ToL2",
                    l3_xrms_data,
                    offsets=[0, 0],
                    sizes=[tile_k, tile_n],
                    strides=[tile_n, 1],
                    indices=[c_tx_i],
                )

                # bL3ToL2 packets 1..GEMV_COUNT*chunks: weight chunks. Same
                # [tile_k, tile_n] packet shape as the xrms head packet —
                # the memtile S2MM and L1 S2MM are single self-loop BDs.
                for mm_iter in range(GEMV_COUNT):
                    ChannelPut(
                        "bL3ToL2",
                        l3_b_data,
                        offsets=[tx_i, mm_iter, 0, launch_offset_y],
                        sizes=[1, 1, k, tile_n],
                        strides=[GEMV_COUNT * n * k, n * k, n, 1],
                        indices=[c_tx_i],
                    )

                # KV cache writeback at slot pos_host for this col.
                ChannelGet(
                    "cL2ToL3",
                    l3_k_cache_data,
                    offsets=[tx_i, pos_host, launch_offset_y],
                    sizes=[1, 1, tile_n],
                    strides=[seq_len * n, n, 1],
                    indices=[c_tx_i],
                )
                ChannelGet(
                    "cL2ToL3",
                    l3_v_cache_data,
                    offsets=[tx_i, pos_host, launch_offset_y],
                    sizes=[1, 1, tile_n],
                    strides=[seq_len * n, n, 1],
                    indices=[c_tx_i],
                )

                # KV cache reads for attention: pos+1 K rows, then pos+1 V rows.
                for i in range_(0, pos_host + 1):
                    ChannelPut(
                        "aL3ToL2",
                        l3_k_cache_data,
                        offsets=[tx_i, i, 0],
                        sizes=[1, 1, tile_n],
                        strides=[seq_len * n, n, 1],
                        indices=[c_tx_i],
                    )
                    yield_([])
                for i in range_(0, pos_host + 1):
                    ChannelPut(
                        "aL3ToL2",
                        l3_v_cache_data,
                        offsets=[tx_i, i, 0],
                        sizes=[1, 1, tile_n],
                        strides=[seq_len * n, n, 1],
                        indices=[c_tx_i],
                    )
                    yield_([])

                # xb output back to L3 for this col.
                ChannelGet(
                    "dL2ToL3",
                    l3_xb_data,
                    offsets=[tx_i, 0, 0],
                    sizes=[1, GROUP_SIZE, n],
                    strides=[GROUP_SIZE * n, n, 1],
                    indices=[c_tx_i],
                )

            @segment(name="vecmat_i8_0")
            def segment_body():
                # L2 MemRefTypes. Inner-k chunked: weight L2 buf shrinks from
                # [k, tile_n] to [tile_k, tile_n] (8 KB/col at tile_k=64,
                # vs 256 KB/col at k=2048 without chunking).
                a_size_l2 = [tile_n]  # K/V cache row staging (single row at a time)
                b_size_l2 = [tile_k, tile_n]
                c_size_l2 = [KV_COUNT, tile_n]
                l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
                l2MemrefTyA = MemRefType.get(
                    shape=a_size_l2,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyB = MemRefType.get(
                    shape=b_size_l2,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyC = MemRefType.get(
                    shape=c_size_l2,
                    element_type=xrt_dtype_out,
                    memory_space=l2_mem_space,
                )
                # rms now runs on the gqa tile — no L2 xrms staging, no
                # rms_herd, no xRms L2-to-L1 broadcast, no x_norm cascade.
                # The padded xrms packet arrives on bL3ToL2 like a weight
                # chunk and is consumed by the gqa tile as the first
                # bL2ToL1 acquire.
                k_i32_const = arith.ConstantOp(IntegerAttr.get(T.i32(), k), None)

                # Per-col L2 staging: one buffer per col so memtile DMA channels
                # don't collide. Same allocation pattern as v2.
                l2_a_bufs = [AllocOp(l2MemrefTyA, [], []) for _ in range(NKV)]
                l2_b_bufs = [AllocOp(l2MemrefTyB, [], []) for _ in range(NKV)]
                l2_c_bufs = [AllocOp(l2MemrefTyC, [], []) for _ in range(NKV)]

                l1_c_data = AllocOp(l1MemrefTyQKV, [], [])

                # Per-col data forwarding (Step 3-style):
                #   - aL3ToL2 → aL2ToL1 streams K+V cache rows (single get/put
                #     pair, BD chain handles streaming)
                #   - bL3ToL2 → bL2ToL1 streams weights (separate channel)
                for tx_i in range(NKV):
                    c_tx_i = arith.ConstantOp.create_index(tx_i)
                    ChannelGet(
                        "aL3ToL2",
                        l2_a_bufs[tx_i].result,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[c_tx_i],
                    )
                    ChannelPut(
                        "aL2ToL1",
                        l2_a_bufs[tx_i].result,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[c_tx_i],
                    )

                    # Inner-k weight relay: nested scf.for over GEMV_COUNT *
                    # (k/tile_k) chunks. Folds to one repeat_count BD on each
                    # of bL3ToL2 RX and bL2ToL1 TX (memtile BD limit is 24/ch
                    # — Python-unrolling explodes here at large N_IN). The
                    # repeat_count BD does NOT reset across host invocations
                    # in xclbin format, so multi-iter profiling requires ELF.
                    for mm_iter in range_(0, GEMV_COUNT * (k // tile_k)):
                        ChannelGet(
                            "bL3ToL2",
                            l2_b_bufs[tx_i].result,
                            offsets=[],
                            sizes=[],
                            strides=[],
                            indices=[c_tx_i],
                        )
                        ChannelPut(
                            "bL2ToL1",
                            l2_b_bufs[tx_i].result,
                            offsets=[],
                            sizes=[],
                            strides=[],
                            indices=[c_tx_i],
                        )
                        yield_([])

                    # KV cache writeback: pull from herd then forward to L3.
                    ChannelGet(
                        "cL1ToL2",
                        l2_c_bufs[tx_i].result,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[c_tx_i],
                    )

                l1_out_data = AllocOp(l1MemrefTyC, [], [])

                pos_c = arith.ConstantOp.create_index(pos_host)

                @herd(
                    name="herd_0",
                    sizes=[NKV, 1],
                    operands=[l1_c_data, l1_out_data, pos_c, k_i32_const],
                )
                def herd_body_0(_tx, _ty, _sx, _sy, c_data, out_data, pos, k_i32):

                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype_vm_acc, 0), None)
                    l1_a_data = AllocOp(l1MemrefTyA, [], [])
                    l1_b_data = AllocOp(l1MemrefTyB, [], [])
                    l1_shared_bd_buf_data = AllocOp(l1MemrefTySharedL1BDBuf, [], [])

                    # First bL2ToL1 acquire: padded xrms packet (same shape
                    # as a weight chunk). Demux into x_raw_l1 + w_rms_l1
                    # scratch buffers, then run rms into l1_a_data. Cannot
                    # alias x and y (rms_kernel_bf16 has __restrict on both).
                    # Subsequent GEMV_COUNT*chunks acquires reuse l1_b_data
                    # for weights.
                    x_raw_l1 = AllocOp(l1MemrefTyXRms, [], [])
                    w_rms_l1 = AllocOp(l1MemrefTyXRms, [], [])
                    ChannelGet(
                        "bL2ToL1",
                        l1_b_data,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[_tx],
                    )
                    # Inline-MLIR demux of padded [tile_k, tile_n] packet into
                    # x_raw[k] (first k elems) and w_rms[k] (next k elems).
                    b_flat_type = MemRefType.get(
                        (tile_k * tile_n,),
                        xrt_dtype_in,
                        memory_space=l1_mem_space,
                    )
                    b_flat = collapse_shape(b_flat_type, l1_b_data.result, [[0, 1]])
                    vec_d_lanes = 32
                    vec_d_type = VectorType.get([vec_d_lanes], xrt_dtype_in)
                    c0_idx_d = arith.ConstantOp.create_index(0)
                    k_total_idx = arith.ConstantOp.create_index(k)
                    vec_d_step = arith.ConstantOp.create_index(vec_d_lanes)
                    cst0_bf_d = arith.ConstantOp(xrt_dtype_in, 0.0)
                    identity_map_1d_d = AffineMapAttr.get(AffineMap.get_identity(1))
                    # Pass 1: x_raw_l1 ← b_flat[0..k]
                    for j in range_(c0_idx_d, k_total_idx, vec_d_step):
                        sub_b = subview(b_flat, [j], [vec_d_lanes], [1])
                        sub_x = subview(x_raw_l1.result, [j], [vec_d_lanes], [1])
                        v = transfer_read(
                            vec_d_type,
                            sub_b,
                            [c0_idx_d],
                            identity_map_1d_d,
                            cst0_bf_d,
                            [True],
                        )
                        transfer_write(
                            None, v, sub_x, [c0_idx_d], identity_map_1d_d, [True]
                        )
                        yield_([])
                    # Pass 2: w_rms_l1 ← b_flat[k..2k]
                    for j in range_(c0_idx_d, k_total_idx, vec_d_step):
                        # b offset = k + j
                        k_const = arith.ConstantOp.create_index(k)
                        j_plus_k = arith.addi(j, k_const)
                        sub_b_w = subview(b_flat, [j_plus_k], [vec_d_lanes], [1])
                        sub_w = subview(w_rms_l1.result, [j], [vec_d_lanes], [1])
                        v = transfer_read(
                            vec_d_type,
                            sub_b_w,
                            [c0_idx_d],
                            identity_map_1d_d,
                            cst0_bf_d,
                            [True],
                        )
                        transfer_write(
                            None, v, sub_w, [c0_idx_d], identity_map_1d_d, [True]
                        )
                        yield_([])
                    # Inline-MLIR simple RMS norm: y = x * w * rsqrt(mean(x^2) + eps)
                    # x = x_raw_l1 [k], w = w_rms_l1 [k], y = l1_a_data [k]
                    rms_vl = 16
                    rms_vec_type = VectorType.get([rms_vl], xrt_dtype_in)
                    f32_type = F32Type.get()
                    rms_acc_l1_type = MemRefType.get(
                        (rms_vl,), xrt_dtype_in, memory_space=l1_mem_space
                    )
                    rms_acc_buf = AllocOp(rms_acc_l1_type, [], [])
                    rms_zero_const = arith.ConstantOp(xrt_dtype_in, 0.0)
                    rms_eps_const = arith.ConstantOp(xrt_dtype_in, 1e-5)
                    rms_inv_n_const_f32 = arith.ConstantOp(f32_type, 1.0 / float(k))
                    linalg.fill(rms_zero_const, outs=[rms_acc_buf])
                    rms_c0_idx = arith.ConstantOp.create_index(0)
                    rms_k_idx = arith.ConstantOp.create_index(k)
                    rms_vl_idx = arith.ConstantOp.create_index(rms_vl)
                    rms_id_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    # Phase 1: ssq = sum(x*x) via vector.fma chain
                    for j in range_(rms_c0_idx, rms_k_idx, rms_vl_idx):
                        sub_x_rms = subview(x_raw_l1.result, [j], [rms_vl], [1])
                        v_x_rms = transfer_read(
                            rms_vec_type,
                            sub_x_rms,
                            [rms_c0_idx],
                            rms_id_map,
                            rms_zero_const,
                            [True],
                        )
                        v_acc_rms = transfer_read(
                            rms_vec_type,
                            rms_acc_buf,
                            [rms_c0_idx],
                            rms_id_map,
                            rms_zero_const,
                            [True],
                        )
                        v_acc_new = vector_fma(v_x_rms, v_x_rms, v_acc_rms)
                        transfer_write(
                            None,
                            v_acc_new,
                            rms_acc_buf,
                            [rms_c0_idx],
                            rms_id_map,
                            [True],
                        )
                        yield_([])
                    # Reduce to scalar, compute rstd
                    v_final_rms = transfer_read(
                        rms_vec_type,
                        rms_acc_buf,
                        [rms_c0_idx],
                        rms_id_map,
                        rms_zero_const,
                        [True],
                    )
                    ssq_bf = vector_reduction(xrt_dtype_in, "add", v_final_rms)
                    ssq_f32 = arith.extf(f32_type, ssq_bf)
                    mean_f32 = arith.mulf(ssq_f32, rms_inv_n_const_f32)
                    eps_f32 = arith.extf(f32_type, rms_eps_const)
                    mean_eps_f32 = arith.addf(mean_f32, eps_f32)
                    rstd_f32 = math_dialect.rsqrt(mean_eps_f32)
                    rstd_bf = arith.truncf(xrt_dtype_in, rstd_f32)
                    v_rstd = BroadcastOp(rms_vec_type, rstd_bf)
                    # Phase 2: y = x * rstd * w
                    for j in range_(rms_c0_idx, rms_k_idx, rms_vl_idx):
                        sub_x2 = subview(x_raw_l1.result, [j], [rms_vl], [1])
                        sub_w2 = subview(w_rms_l1.result, [j], [rms_vl], [1])
                        sub_y2 = subview(l1_a_data.result, [j], [rms_vl], [1])
                        v_x2 = transfer_read(
                            rms_vec_type,
                            sub_x2,
                            [rms_c0_idx],
                            rms_id_map,
                            rms_zero_const,
                            [True],
                        )
                        v_w2 = transfer_read(
                            rms_vec_type,
                            sub_w2,
                            [rms_c0_idx],
                            rms_id_map,
                            rms_zero_const,
                            [True],
                        )
                        v_xr = arith.mulf(v_x2, v_rstd.result)
                        v_y2 = arith.mulf(v_xr, v_w2)
                        transfer_write(
                            None, v_y2, sub_y2, [rms_c0_idx], rms_id_map, [True]
                        )
                        yield_([])
                    DeallocOp(rms_acc_buf)
                    DeallocOp(x_raw_l1)
                    DeallocOp(w_rms_l1)

                    # GEMV: GEMV_COUNT iterations (4 Q + 1 K + 1 V), each
                    # producing one [tile_n] row of c_data at offset g*tile_n.
                    # Pre-zero entire c_data once via inline linalg.fill;
                    # each row is then overwritten by its own vecmat acc chain
                    # (saves GEMV_COUNT-1=5 per-row extern fill_bf16 calls).
                    zero_const_c = arith.ConstantOp(xrt_dtype_out, 0.0)
                    linalg.fill(zero_const_c, outs=[c_data])
                    gemv_offset_consts = [
                        ConstantOp(IntegerAttr.get(T.i32(), g * tile_n), None)
                        for g in range(GEMV_COUNT)
                    ]
                    # Inline vecmat (per matvec_cascade.py pattern):
                    # f32 L1 accumulator, bf16->f32 extf, vector.fma in f32.
                    # linalg.vecmat lowering was 14x slower; this hand-written
                    # equivalent of the extern vecmat_bf16_bf16 should match.
                    vm_vl = 16  # 512-bit f32 = full SIMD width
                    vm_vec_bf16 = VectorType.get([vm_vl], xrt_dtype_in)
                    vm_vec_f32 = VectorType.get([vm_vl], F32Type.get())
                    vm_acc_buf_type = MemRefType.get(
                        (vm_vl,),
                        F32Type.get(),
                        memory_space=l1_mem_space,
                    )
                    vm_acc_buf = AllocOp(vm_acc_buf_type, [], [])
                    vm_c0 = arith.ConstantOp.create_index(0)
                    vm_cst0_bf = arith.ConstantOp(xrt_dtype_in, 0.0)
                    vm_cst0_f32 = arith.ConstantOp(F32Type.get(), 0.0)
                    vm_id_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    vm_2d_to_1d_map = AffineMapAttr.get(
                        AffineMap.get(2, 0, [AffineDimExpr.get(1)])
                    )
                    g_idx_consts_v = [
                        arith.ConstantOp.create_index(g) for g in range(GEMV_COUNT)
                    ]
                    n_idx_v = arith.ConstantOp.create_index(n)
                    vm_vl_idx_v = arith.ConstantOp.create_index(vm_vl)
                    tile_k_idx_v = arith.ConstantOp.create_index(tile_k)
                    one_idx_v = arith.ConstantOp.create_index(1)
                    for g in range(GEMV_COUNT):
                        for i in range_(0, k, tile_k):
                            ChannelGet(
                                "bL2ToL1",
                                l1_b_data,
                                offsets=[],
                                sizes=[],
                                strides=[],
                                indices=[_tx],
                            )
                            # scf.for col chunks (n/vm_vl=4) so body emits once
                            for jc in range_(vm_c0, n_idx_v, vm_vl_idx_v):
                                # Load c[g, jc:jc+vl], extend to f32, init acc
                                sub_c_init = subview(
                                    c_data,
                                    [g_idx_consts_v[g], jc],
                                    [1, vm_vl],
                                    [1, 1],
                                )
                                v_c_init_bf = transfer_read(
                                    vm_vec_bf16,
                                    sub_c_init,
                                    [vm_c0, vm_c0],
                                    vm_2d_to_1d_map,
                                    vm_cst0_bf,
                                    [True],
                                )
                                v_c_init_f32 = arith.extf(vm_vec_f32, v_c_init_bf)
                                transfer_write(
                                    None,
                                    v_c_init_f32,
                                    vm_acc_buf,
                                    [vm_c0],
                                    vm_id_map,
                                    [True],
                                )
                                # Loop over tile_k rows, accumulate into vm_acc_buf
                                for row in range_(vm_c0, tile_k_idx_v, one_idx_v):
                                    a_idx = arith.addi(i, row)
                                    a_scalar = load(l1_a_data.result, [a_idx])
                                    a_v_bf = BroadcastOp(vm_vec_bf16, a_scalar)
                                    a_v_f32 = arith.extf(vm_vec_f32, a_v_bf.result)
                                    sub_b = subview(
                                        l1_b_data.result,
                                        [row, jc],
                                        [1, vm_vl],
                                        [1, 1],
                                    )
                                    v_b_bf = transfer_read(
                                        vm_vec_bf16,
                                        sub_b,
                                        [vm_c0, vm_c0],
                                        vm_2d_to_1d_map,
                                        vm_cst0_bf,
                                        [True],
                                    )
                                    v_b_f32 = arith.extf(vm_vec_f32, v_b_bf)
                                    v_acc = transfer_read(
                                        vm_vec_f32,
                                        vm_acc_buf,
                                        [vm_c0],
                                        vm_id_map,
                                        vm_cst0_f32,
                                        [True],
                                    )
                                    v_new = vector_fma(a_v_f32, v_b_f32, v_acc)
                                    transfer_write(
                                        None,
                                        v_new,
                                        vm_acc_buf,
                                        [vm_c0],
                                        vm_id_map,
                                        [True],
                                    )
                                    yield_([])
                                # Truncate acc back to bf16, store to c[g, jc:jc+vl]
                                v_acc_final = transfer_read(
                                    vm_vec_f32,
                                    vm_acc_buf,
                                    [vm_c0],
                                    vm_id_map,
                                    vm_cst0_f32,
                                    [True],
                                )
                                v_c_final = arith.truncf(vm_vec_bf16, v_acc_final)
                                transfer_write(
                                    None,
                                    v_c_final,
                                    sub_c_init,
                                    [vm_c0, vm_c0],
                                    vm_2d_to_1d_map,
                                    [True],
                                )
                                yield_([])  # col chunk scf.for terminator
                            yield_([])  # inner-k scf.for terminator
                    DeallocOp(vm_acc_buf)

                    DeallocOp(l1_a_data)
                    DeallocOp(l1_b_data)

                    l1_freq_pos_data = AllocOp(l1MemrefTyHSByTwo, [], [])
                    zero_constindex = ConstantOp.create_index(0)

                    # Inline freq_pos_bf16_32_16: l1_freq_pos_data[i] =
                    # bf16(pos) * freq_table[i] for i in 0..32. Two 16-lane
                    # arith.mulf chunks with a precomputed dense<vector<16xbf16>>
                    # constant for each half of freq_table_dk64.
                    fp_vec_bf16 = VectorType.get([16], xrt_dtype_out)
                    fp_pos_i32 = arith.index_cast(T.i32(), pos)
                    fp_pos_f32 = arith_dialect.sitofp(F32Type.get(), fp_pos_i32)
                    fp_pos_bf = arith_dialect.truncf(xrt_dtype_out, fp_pos_f32)
                    fp_pos_v = BroadcastOp(fp_vec_bf16, fp_pos_bf)
                    fp_freq_lo_attr = DenseElementsAttr.get(
                        _FREQ_TABLE_DK64_BF16[:16].copy(), type=fp_vec_bf16
                    )
                    fp_freq_hi_attr = DenseElementsAttr.get(
                        _FREQ_TABLE_DK64_BF16[16:].copy(), type=fp_vec_bf16
                    )
                    fp_freq_lo = arith_dialect.ConstantOp(fp_vec_bf16, fp_freq_lo_attr)
                    fp_freq_hi = arith_dialect.ConstantOp(fp_vec_bf16, fp_freq_hi_attr)
                    fp_out_lo = arith_dialect.mulf(fp_pos_v.result, fp_freq_lo.result)
                    fp_out_hi = arith_dialect.mulf(fp_pos_v.result, fp_freq_hi.result)
                    fp_c16 = arith.ConstantOp.create_index(16)
                    fp_id_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    transfer_write(
                        None,
                        fp_out_lo,
                        l1_freq_pos_data.result,
                        [zero_constindex],
                        fp_id_map,
                        [True],
                    )
                    transfer_write(
                        None,
                        fp_out_hi,
                        l1_freq_pos_data.result,
                        [fp_c16],
                        fp_id_map,
                        [True],
                    )

                    l1_sinf_vec = AllocOp(l1MemrefTyHSByTwo, [], [])
                    l1_cosf_vec = AllocOp(l1MemrefTyHSByTwo, [], [])
                    sinf_poly_call = CallOp(
                        sinf_poly_func, [l1_freq_pos_data, l1_sinf_vec]
                    )
                    cosf_poly_call = CallOp(
                        cosf_poly_func, [l1_freq_pos_data, l1_cosf_vec]
                    )

                    # RoPE: rotate Q[0..GROUP_SIZE-1] (4 calls) and K (1 call,
                    # at offset GROUP_SIZE*tile_n). V (offset (GROUP_SIZE+1)*
                    # tile_n) is NOT rotated.
                    # Inline-MLIR attempt revealed a Peano codegen issue: any
                    # combination of {arith.subf following arith.mulf} +
                    # {vector.fma following arith.mulf} in the same body
                    # produces correct-looking aievec IR (sub_elem + mac_elem)
                    # but emits an ELF that writes all-zeros at runtime.
                    # Simpler patterns (mul-only, mul+subf only, mul+fma only)
                    # work; mul+addf is rejected at peano llc time. Reverted to
                    # extern shuffle_apply_rope_bf16_64 — only 5 calls/token.
                    for g in range(GROUP_SIZE):
                        rope_off = ConstantOp(
                            IntegerAttr.get(T.i32(), g * tile_n), None
                        )
                        CallOp(
                            shuffle_apply_rope_poly_func,
                            [rope_off, l1_cosf_vec, l1_sinf_vec, c_data],
                        )
                    rope_off_k = ConstantOp(
                        IntegerAttr.get(T.i32(), GROUP_SIZE * tile_n), None
                    )
                    CallOp(
                        shuffle_apply_rope_poly_func,
                        [rope_off_k, l1_cosf_vec, l1_sinf_vec, c_data],
                    )

                    DeallocOp(l1_sinf_vec)
                    DeallocOp(l1_cosf_vec)
                    DeallocOp(l1_freq_pos_data)

                    # KV cache writeback: send K (row GROUP_SIZE) and V (row
                    # GROUP_SIZE+1) of c_data on this col's slot.
                    ChannelPut(
                        "cL1ToL2",
                        c_data,
                        offsets=[GROUP_SIZE, 0],
                        sizes=[KV_COUNT, tile_n],
                        strides=[tile_n, 1],
                        indices=[_tx],
                    )

                    # launch 2 herd

                    pos_p1 = arith.addi(pos, arith.ConstantOp.create_index(1))

                    q_l1_data = AllocOp(l1MemrefTyQ, [], [])
                    attn_l1_data = AllocOp(l1MemrefTyAttn, [], [])
                    # softmax in-place: reuse attn_l1_data buffer (saves 16 KB
                    # at lk=2048, important for L1 budget at large lk).
                    softmax_l1_data = attn_l1_data

                    # Inline-MLIR -99 fill of attn buffer (GROUP_SIZE * seq_len)
                    # via linalg.fill — costs ~30us extra over the hand-tuned
                    # 16-lane vectorized extern kernel, accepted to remove
                    # external dependency.
                    neg99_const = arith.ConstantOp(xrt_dtype_out, -99.0)
                    linalg.fill(neg99_const, outs=[attn_l1_data])

                    # Inline-MLIR vectorized copy + 1/sqrt(DIM_N) pre-scale of
                    # the first GROUP_SIZE rows of c_data (Q heads) into
                    # q_l1_data. Pre-scaling Q here lets attn_1_group skip the
                    # per-call (bf16)(scalar * 0.125f) tail.
                    inv_sqrt_dk_const = arith.ConstantOp(xrt_dtype_out, 0.125)
                    vec_q_lanes = 32
                    vec_q_type = VectorType.get([vec_q_lanes], xrt_dtype_out)
                    v_inv_sqrt_dk = BroadcastOp(vec_q_type, inv_sqrt_dk_const)
                    c_flat_type = MemRefType.get(
                        (GEMV_COUNT * n,),
                        xrt_dtype_out,
                        memory_space=l1_mem_space,
                    )
                    q_flat_type = MemRefType.get(
                        (GROUP_SIZE * n,),
                        xrt_dtype_out,
                        memory_space=l1_mem_space,
                    )
                    c_flat_q = collapse_shape(c_flat_type, c_data, [[0, 1]])
                    q_flat = collapse_shape(q_flat_type, q_l1_data.result, [[0, 1]])
                    c0_idx_q = arith.ConstantOp.create_index(0)
                    n_total_q = arith.ConstantOp.create_index(GROUP_SIZE * n)
                    vec_q_step = arith.ConstantOp.create_index(vec_q_lanes)
                    cst0_bf = arith.ConstantOp(xrt_dtype_out, 0.0)
                    identity_map_1d_q = AffineMapAttr.get(AffineMap.get_identity(1))
                    for j in range_(c0_idx_q, n_total_q, vec_q_step):
                        sub_c_q = subview(c_flat_q, [j], [vec_q_lanes], [1])
                        sub_q = subview(q_flat, [j], [vec_q_lanes], [1])
                        v = transfer_read(
                            vec_q_type,
                            sub_c_q,
                            [c0_idx_q],
                            identity_map_1d_q,
                            cst0_bf,
                            [True],
                        )
                        v_scaled = arith.mulf(v, v_inv_sqrt_dk.result)
                        transfer_write(
                            None,
                            v_scaled,
                            sub_q,
                            [c0_idx_q],
                            identity_map_1d_q,
                            [True],
                        )
                        yield_([])

                    # Inline attn_1_group: dot(Q[g], K) per g via 16-lane bf16
                    # multiply + f32 fma accumulator (matches the C kernel's
                    # accum<accfloat, 32> precision; pure bf16 acc loses enough
                    # precision that dot(Q, K_new) ≈ 0, softmax becomes uniform
                    # and attn@V degenerates to V[pos]/seq_len). vector<32xf32>
                    # fma is not legal on AIE2P (16-lane f32 max), so we use
                    # four 16-lane halves chained into a single f32 accumulator
                    # per Q head, then reduce + truncf + store.
                    a1_vl = 16
                    a1_n_chunks = n // a1_vl  # 64/16 = 4
                    a1_vec_bf_t = VectorType.get([a1_vl], xrt_dtype_out)
                    a1_f32 = T.f32()
                    a1_vec_f32_t = VectorType.get([a1_vl], a1_f32)
                    a1_id_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    a1_2d_to_1d = AffineMapAttr.get(
                        AffineMap.get(2, 0, [AffineDimExpr.get(1)])
                    )
                    a1_c0 = arith.ConstantOp.create_index(0)
                    a1_offs = [
                        arith.ConstantOp.create_index(c * a1_vl)
                        for c in range(a1_n_chunks)
                    ]
                    a1_cst0 = arith.ConstantOp(xrt_dtype_out, 0.0)
                    a1_zero_f32_v = arith.ConstantOp(
                        a1_vec_f32_t,
                        DenseElementsAttr.get(
                            np.zeros(a1_vl, dtype=np.float32),
                            type=a1_vec_f32_t,
                        ),
                    )
                    for i in range_(0, pos_p1):
                        ChannelGet(
                            "aL2ToL1",
                            l1_shared_bd_buf_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                            indices=[_tx],
                        )
                        k_chunks_f = []
                        for c in range(a1_n_chunks):
                            k_bf = transfer_read(
                                a1_vec_bf_t,
                                l1_shared_bd_buf_data.result,
                                [a1_offs[c]],
                                a1_id_map,
                                a1_cst0,
                                [True],
                            )
                            k_chunks_f.append(arith.extf(a1_vec_f32_t, k_bf))
                        accs = [a1_zero_f32_v.result for _ in range(GROUP_SIZE)]
                        for g in range(GROUP_SIZE):
                            sub_q = subview(
                                q_l1_data.result, [g, 0], [1, n], [1, 1]
                            )
                            for c in range(a1_n_chunks):
                                q_bf = transfer_read(
                                    a1_vec_bf_t,
                                    sub_q,
                                    [a1_c0, a1_offs[c]],
                                    a1_2d_to_1d,
                                    a1_cst0,
                                    [True],
                                )
                                q_f = arith.extf(a1_vec_f32_t, q_bf)
                                accs[g] = vector_fma(
                                    q_f, k_chunks_f[c], accs[g]
                                )
                        for g in range(GROUP_SIZE):
                            sc_f32 = vector_reduction(a1_f32, "add", accs[g])
                            sc_bf = arith.truncf(xrt_dtype_out, sc_f32)
                            sub_o = subview(
                                attn_l1_data.result,
                                [g, 0],
                                [1, seq_len],
                                [1, 1],
                            )
                            store(sc_bf, sub_o, [a1_c0, i])
                        yield_([])

                    # Inline-MLIR softmax over [GROUP_SIZE, SEQ_LEN]: per-row
                    # 3-pass softmax (max-reduce, exp+sum-reduce, divide).
                    # Uses L1 accumulator buffers (not scf.for iter_args — see
                    # Xilinx/mlir-air#1591). Matches the extern softmax_group
                    # semantics including reading -99 fill at positions past
                    # pos (exp(-99-max) ~ 0 contributes nothing to sum).
                    sm_vl = 16
                    sm_vec_type = VectorType.get([sm_vl], xrt_dtype_out)
                    sm_acc_buf_type = MemRefType.get(
                        (sm_vl,), xrt_dtype_out, memory_space=l1_mem_space
                    )
                    sm_neg_big = arith.ConstantOp(xrt_dtype_out, -1e30)
                    sm_zero = arith.ConstantOp(xrt_dtype_out, 0.0)
                    sm_one_const = arith.ConstantOp(xrt_dtype_out, 1.0)
                    sm_one_vec = BroadcastOp(sm_vec_type, sm_one_const)
                    sm_c0 = arith.ConstantOp.create_index(0)
                    sm_seq_idx = arith.ConstantOp.create_index(seq_len)
                    sm_vl_idx = arith.ConstantOp.create_index(sm_vl)
                    sm_id_map = AffineMapAttr.get(AffineMap.get_identity(1))
                    sm_2d_to_1d_map = AffineMapAttr.get(
                        AffineMap.get(2, 0, [AffineDimExpr.get(1)])
                    )
                    f32_type_sm = F32Type.get()
                    sm_group_size_idx = arith.ConstantOp.create_index(GROUP_SIZE)
                    sm_one_step_idx = arith.ConstantOp.create_index(1)
                    # Allocate the two per-row accumulator buffers ONCE,
                    # outside the g loop; reset each g iter via linalg.fill.
                    sm_max_buf = AllocOp(sm_acc_buf_type, [], [])
                    sm_sum_buf = AllocOp(sm_acc_buf_type, [], [])
                    for g in range_(sm_c0, sm_group_size_idx, sm_one_step_idx):
                        # Pass 1: max reduce. acc_buf[VL] = -inf, max-update each chunk
                        linalg.fill(sm_neg_big, outs=[sm_max_buf])
                        for j in range_(sm_c0, sm_seq_idx, sm_vl_idx):
                            sub_in_max = subview(
                                softmax_l1_data.result,
                                [g, j],
                                [1, sm_vl],
                                [1, 1],
                            )
                            v_in_max = transfer_read(
                                sm_vec_type,
                                sub_in_max,
                                [sm_c0, sm_c0],
                                sm_2d_to_1d_map,
                                sm_zero,
                                [True],
                            )
                            v_max_cur = transfer_read(
                                sm_vec_type,
                                sm_max_buf,
                                [sm_c0],
                                sm_id_map,
                                sm_zero,
                                [True],
                            )
                            v_max_new = arith.maximumf(v_max_cur, v_in_max)
                            transfer_write(
                                None,
                                v_max_new,
                                sm_max_buf,
                                [sm_c0],
                                sm_id_map,
                                [True],
                            )
                            yield_([])
                        v_max_final = transfer_read(
                            sm_vec_type,
                            sm_max_buf,
                            [sm_c0],
                            sm_id_map,
                            sm_zero,
                            [True],
                        )
                        max_scalar = vector_reduction(
                            xrt_dtype_out, "maximumf", v_max_final
                        )
                        v_max_bcast = BroadcastOp(sm_vec_type, max_scalar)

                        # Pass 2: exp(x-max), write back, accumulate sum
                        linalg.fill(sm_zero, outs=[sm_sum_buf])
                        for j in range_(sm_c0, sm_seq_idx, sm_vl_idx):
                            sub_inout = subview(
                                softmax_l1_data.result,
                                [g, j],
                                [1, sm_vl],
                                [1, 1],
                            )
                            v_in = transfer_read(
                                sm_vec_type,
                                sub_inout,
                                [sm_c0, sm_c0],
                                sm_2d_to_1d_map,
                                sm_zero,
                                [True],
                            )
                            v_diff = arith.subf(v_in, v_max_bcast.result)
                            v_exp = math_dialect.exp(v_diff)
                            transfer_write(
                                None,
                                v_exp,
                                sub_inout,
                                [sm_c0, sm_c0],
                                sm_2d_to_1d_map,
                                [True],
                            )
                            v_sum_cur = transfer_read(
                                sm_vec_type,
                                sm_sum_buf,
                                [sm_c0],
                                sm_id_map,
                                sm_zero,
                                [True],
                            )
                            # vector.fma(1.0, v_exp, v_sum_cur) avoids the
                            # mulf->addf rejection (CLAUDE.md known pitfall).
                            v_sum_new = vector_fma(sm_one_vec.result, v_exp, v_sum_cur)
                            transfer_write(
                                None,
                                v_sum_new,
                                sm_sum_buf,
                                [sm_c0],
                                sm_id_map,
                                [True],
                            )
                            yield_([])
                        v_sum_final = transfer_read(
                            sm_vec_type,
                            sm_sum_buf,
                            [sm_c0],
                            sm_id_map,
                            sm_zero,
                            [True],
                        )
                        sum_scalar_bf = vector_reduction(
                            xrt_dtype_out, "add", v_sum_final
                        )
                        # Compute 1/sum in f32 (bf16 div not legalized)
                        sum_scalar_f32 = arith.extf(f32_type_sm, sum_scalar_bf)
                        one_f32 = arith.ConstantOp(f32_type_sm, 1.0)
                        inv_sum_f32 = arith.divf(one_f32, sum_scalar_f32)
                        inv_sum_bf = arith.truncf(xrt_dtype_out, inv_sum_f32)
                        v_inv_sum = BroadcastOp(sm_vec_type, inv_sum_bf)

                        # Pass 3: out *= inv_sum
                        for j in range_(sm_c0, sm_seq_idx, sm_vl_idx):
                            sub_out_div = subview(
                                softmax_l1_data.result,
                                [g, j],
                                [1, sm_vl],
                                [1, 1],
                            )
                            v_out = transfer_read(
                                sm_vec_type,
                                sub_out_div,
                                [sm_c0, sm_c0],
                                sm_2d_to_1d_map,
                                sm_zero,
                                [True],
                            )
                            v_scaled = arith.mulf(v_out, v_inv_sum.result)
                            transfer_write(
                                None,
                                v_scaled,
                                sub_out_div,
                                [sm_c0, sm_c0],
                                sm_2d_to_1d_map,
                                [True],
                            )
                            yield_([])
                        yield_([])  # outer g scf.for terminator
                    DeallocOp(sm_max_buf)
                    DeallocOp(sm_sum_buf)

                    xb_l1_data = AllocOp(l1MemrefTyXb, [], [])
                    zero_const_xb = arith.ConstantOp(xrt_dtype_out, 0.0)
                    linalg.fill(zero_const_xb, outs=[xb_l1_data])
                    # attn_2_group: kept extern. Inline attempts:
                    # (1) collapse_shape + load: read returned -99 fill (memory-
                    #     effect tracking miss through softmax buffer alias).
                    # (2) 2D direct load: same issue.
                    # (3) scf.for iter_args (carry 8 accs): IR looks correct
                    #     (memref.subview strides fixed [1,1]), but at runtime
                    #     produces single-iter result — accumulation across
                    #     iterations doesn't materialize. Replacing fma with
                    #     arith.addf in iter_args body crashes aircc.
                    # Conclusion: vector iter_args + L1 channel.get + AIE2P
                    # lowering interact badly here. Hot path, keep extern.
                    for i in range_(0, pos_p1):
                        ChannelGet(
                            "aL2ToL1",
                            l1_shared_bd_buf_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                            indices=[_tx],
                        )
                        i32_iv = arith.index_cast(T.i32(), i)
                        CallOp(
                            attn2_func,
                            [
                                softmax_l1_data,
                                l1_shared_bd_buf_data,
                                i32_iv,
                                xb_l1_data,
                            ],
                        )
                        yield_([])

                    ChannelPut(
                        "dL1ToL2",
                        xb_l1_data,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[_tx],
                    )

                    DeallocOp(q_l1_data)
                    DeallocOp(attn_l1_data)
                    # softmax_l1_data aliases attn_l1_data (in-place softmax) —
                    # only one dealloc.
                    DeallocOp(xb_l1_data)
                    DeallocOp(l1_shared_bd_buf_data)

                herd_body_0.attributes["link_with"] = StringAttr.get(
                    "attn_decode_npu2.o"
                )

                # Per-col KV cache writeback forwarding to L3.
                for tx_i in range(NKV):
                    c_tx_i = arith.ConstantOp.create_index(tx_i)
                    ChannelPut(
                        "cL2ToL3",
                        l2_c_bufs[tx_i].result,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[c_tx_i],
                    )

                # Per-col xb output staging L1->L2->L3.
                xb_l2_size = xb_l1_size
                l2MemrefTyXb = MemRefType.get(
                    shape=xb_l2_size,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                xb_l2_bufs = [AllocOp(l2MemrefTyXb, [], []) for _ in range(NKV)]
                for tx_i in range(NKV):
                    c_tx_i = arith.ConstantOp.create_index(tx_i)
                    ChannelGet(
                        "dL1ToL2",
                        xb_l2_bufs[tx_i].result,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[c_tx_i],
                    )
                    ChannelPut(
                        "dL2ToL3",
                        xb_l2_bufs[tx_i].result,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        indices=[c_tx_i],
                    )
                for tx_i in range(NKV):
                    DeallocOp(xb_l2_bufs[tx_i])
                    DeallocOp(l2_a_bufs[tx_i])
                    DeallocOp(l2_b_bufs[tx_i])
                    DeallocOp(l2_c_bufs[tx_i])
                DeallocOp(l1_c_data)


if __name__ == "__main__":
    # Step 2a defaults: dk=dv=64 (LLaMA head_size), keep n_in (K) modest
    # so the [k, tile_n] L2 weight buffer stays tiny. tile_k=k for now
    # (single inner-k iter); larger n_in handled later when L2 streaming
    # kicks in.
    M = 1
    K = 64
    N = 64
    pos = 16
    seq_len = 128
    TILE_K = 64
    TILE_N = 64
    QKV_COUNT = 3  # Running the design 3 times for Q, K and V, respectively.
    KV_COUNT = 2
    INPUT_DATATYPE = bfloat16
    VM_ACC_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--k", type=int, default=K, help="K dimension size in a (1xK) * (KxN) matmul"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="N dimension size in a (1xK) * (KxN) matmul",
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=pos,
        help="Position runtime variable",
    )
    parser.add_argument(
        "--tile-k", type=int, default=TILE_K, help="K dimension size of each L1 tile"
    )
    parser.add_argument(
        "--tile-n", type=int, default=TILE_N, help="N dimension size of each L1 tile"
    )
    parser.add_argument("--seq-len", type=int, default=seq_len, help="Sequence length")
    parser.add_argument(
        "--xclbin-kernel-name",
        dest="kernel_name",
        default="",
        help="Kernel name in xclbin file",
    )
    parser.add_argument(
        "--xclbin-instance-name",
        dest="instance_name",
        default="",
        help="Instance name in xclbin metadata",
    )
    parser.add_argument(
        "--xclbin-kernel-id",
        dest="kernel_id",
        default="",
        help="Kernel id in xclbin file",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--nkv",
        type=int,
        default=2,
        help="Number of KV heads / cols (Step 3 default 2; LLaMA target 8)",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.k,
        args.n,
        args.tile_k,
        args.tile_n,
        args.seq_len,
        INPUT_DATATYPE,
        VM_ACC_DATATYPE,
        OUTPUT_DATATYPE,
        args.pos,
        nkv=args.nkv,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    GROUP_SIZE = 4
    GEMV_COUNT = GROUP_SIZE + 2  # 4 Q + 1 K + 1 V
    NKV = args.nkv

    # Realistic uniform random inputs (matches prefill attn_npu2.py pattern).
    # val_range=4.0 keeps values in a magnitude bf16 can represent without
    # cascading overflow through the GEMV+softmax+attn_2 chain.
    rng = np.random.default_rng(42)
    val_range = 4.0
    # xrms is padded to [tile_k, tile_n] so it shares one BD shape with the
    # weight stream on bL3ToL2 (single self-loop per memtile/L1 channel,
    # no repeat_count). Real x_raw + w_rms occupy the first 2*k flat
    # elements; the rest is padding the kernel demux ignores.
    input_xrms = np.zeros((args.tile_k, args.tile_n), dtype=INPUT_DATATYPE)
    input_xraw = rng.uniform(0, val_range, (args.k,)).astype(INPUT_DATATYPE)
    input_wrms = rng.uniform(0, val_range, (args.k,)).astype(INPUT_DATATYPE)
    input_xrms_flat = input_xrms.reshape(-1)
    input_xrms_flat[: args.k] = input_xraw
    input_xrms_flat[args.k : 2 * args.k] = input_wrms
    input_b = rng.uniform(0, val_range, (NKV, GEMV_COUNT, args.k, args.n)).astype(
        INPUT_DATATYPE
    )

    output_kc = np.zeros(shape=(NKV, args.seq_len, args.n), dtype=OUTPUT_DATATYPE)
    output_vc = np.zeros(shape=(NKV, args.seq_len, args.n), dtype=OUTPUT_DATATYPE)
    output_xb = np.zeros(shape=(NKV, GROUP_SIZE, args.n), dtype=OUTPUT_DATATYPE)

    # x_norm = RMSNorm(x_raw, w_rms) — float32 reference, bf16 round-trip
    # (the kernel computes in mixed precision; final stored as bf16).
    xraw_f = input_xraw.astype(np.float32)
    wrms_f = input_wrms.astype(np.float32)
    rstd = 1.0 / np.sqrt(np.mean(xraw_f * xraw_f) + 1e-5)
    x_norm = (xraw_f * rstd * wrms_f).astype(OUTPUT_DATATYPE).astype(np.float32)

    inv_sqrt_n = 1.0 / sqrt(args.n)
    for kv in range(NKV):
        # GEMV: x_norm (shared) @ per-col W. float32 accumulation, bf16 round-trip.
        Bf = input_b[kv].astype(np.float32)
        output_vm = np.zeros(shape=(GEMV_COUNT, args.n), dtype=np.float32)
        for qkv_iter in range(GEMV_COUNT):
            output_vm[qkv_iter] = (x_norm @ Bf[qkv_iter]).astype(np.float32)
        # Round-trip through bf16 to mirror in-kernel storage precision.
        output_vm = output_vm.astype(OUTPUT_DATATYPE).astype(np.float32)

        # RoPE on Q rows (0..GROUP_SIZE-1) and K row (GROUP_SIZE); V not rotated.
        for row in list(range(GROUP_SIZE)) + [GROUP_SIZE]:
            for s in range(0, args.n, 2):
                freq = 1.0 / pow(10000.0, float(s) / float(args.n))
                val = args.pos * freq
                fcr = cos(val)
                fci = sin(val)
                v0 = output_vm[row][s]
                v1 = output_vm[row][s + 1]
                output_vm[row][s] = v0 * fcr - v1 * fci
                output_vm[row][s + 1] = v0 * fci + v1 * fcr
        output_vm = output_vm.astype(OUTPUT_DATATYPE).astype(np.float32)

        # KV cache slot pos populated by GEMV outputs.
        Q_kv_f = output_vm[:GROUP_SIZE]  # [GROUP_SIZE, n]
        K_new_f = output_vm[GROUP_SIZE]
        V_new_f = output_vm[GROUP_SIZE + 1]
        # KV cache must be bf16 since the host hands it to the kernel as bf16.
        for i in range(args.n):
            output_kc[kv][args.pos][i] = K_new_f[i]
            output_vc[kv][args.pos][i] = V_new_f[i]

        # Per-Q-head attention: float32 throughout, bf16 at the end.
        Kc_f = output_kc[kv].astype(np.float32)
        Vc_f = output_vc[kv].astype(np.float32)
        for g in range(GROUP_SIZE):
            scores = np.zeros(args.pos + 1, dtype=np.float32)
            for t in range(args.pos + 1):
                scores[t] = float(np.dot(Q_kv_f[g], Kc_f[t])) * inv_sqrt_n
            scores -= scores.max()
            P = np.exp(scores)
            P /= P.sum()
            xb_f = np.zeros(args.n, dtype=np.float32)
            for t in range(args.pos + 1):
                xb_f += P[t] * Vc_f[t]
            output_xb[kv][g] = xb_f.astype(OUTPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        ###### Compile and test (prefill-style tolerance + correlation check)
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            output_format=args.output_format,
            instance_name="mha_bf16",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_xrms, input_b, output_kc, output_vc],
                expected_outputs=[output_xb],
                atol=0.15,
                rtol=0.04,
                max_mismatch_percentage=0.5,
                min_correlation=0.99,
            )
        )

    elif args.compile_mode == "compile-only":
        ####### Compile only — for profiling via test_xclbin_decode.exe.
        # omit_while_true_loop=False keeps the runtime self-loop intact —
        # required because the all-self-loop DMA design (no repeat_count)
        # has no producer-side terminator; only the runtime while loop
        # gates the per-iteration shim BD launch sequence.
        # once and exits, allowing multi-iteration profiling. (run_test path
        # above uses False — XRTRunner only does one invocation per call.)
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            kernel_name=args.kernel_name,
            instance_name=args.instance_name,
            kernel_id=args.kernel_id,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
            target_device="npu2",
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
