# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from math import cos, sin, sqrt, exp

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from ml_dtypes import bfloat16

range_ = for_


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
    linalg_fill_func = FuncOp(
        "linalg_fill_bf16",
        ([T.i32(), l1MemrefTyThreeByFortyEightVec], []),
        visibility="private",
    )
    # vecmat_bf16_bf16(x_offset, c_offset, a, b, c): x_offset selects the
    # tile_k-sized slice of the full-k x_norm buffer for this inner-k chunk.
    vecmat_func = FuncOp(
        "vecmat_bf16_bf16",
        (
            [
                T.i32(),
                T.i32(),
                l1MemrefTyA,
                l1MemrefTyB,
                l1MemrefTyThreeByFortyEightVec,
            ],
            [],
        ),
        visibility="private",
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
    freq_pos_func = FuncOp(
        "freq_pos_bf16_32_16",
        ([T.i32(), l1MemrefTyHSByTwo], []),
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
    softmax_func = FuncOp(
        "softmax_group",
        ([l1MemrefTyAttn, T.i32(), l1MemrefTyAttn], []),
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
    simple_rms_func = FuncOp(
        "simple_rms_bf16",
        ([l1MemrefTyXRms, l1MemrefTyXRms, l1MemrefTyXRms, T.i32()], []),
        visibility="private",
    )
    xrms_demux_func = FuncOp(
        "xrms_demux_bf16",
        ([l1MemrefTyB, l1MemrefTyXRms, l1MemrefTyXRms, T.i32()], []),
        visibility="private",
    )
    # Vectorized fill / copy helpers replace per-element scf.for stores
    # in the herd body — the scalar variants unroll into thousands of
    # stores in the AIE2P core ELF, exceeding the 16 KB program memory.
    fill_neg99_func = FuncOp(
        "fill_neg99_bf16",
        ([l1MemrefTyAttn, T.i32()], []),
        visibility="private",
    )
    fill_zero_func = FuncOp(
        "fill_zero_bf16",
        ([l1MemrefTyXb, T.i32()], []),
        visibility="private",
    )
    vec_copy_n_func = FuncOp(
        "vector_copy_n_bf16",
        ([l1MemrefTyThreeByFortyEightVec, l1MemrefTyQ, T.i32()], []),
        visibility="private",
    )

    for func in [
        linalg_fill_func,
        vecmat_func,
        cosf_poly_func,
        sinf_poly_func,
        freq_pos_func,
        shuffle_apply_rope_poly_func,
        attn_func,
        softmax_func,
        attn2_func,
        simple_rms_func,
        xrms_demux_func,
        fill_neg99_func,
        fill_zero_func,
        vec_copy_n_func,
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
                    CallOp(xrms_demux_func, [l1_b_data, x_raw_l1, w_rms_l1, k_i32])
                    CallOp(simple_rms_func, [x_raw_l1, w_rms_l1, l1_a_data, k_i32])
                    DeallocOp(x_raw_l1)
                    DeallocOp(w_rms_l1)

                    # GEMV: GEMV_COUNT iterations (4 Q + 1 K + 1 V), each
                    # producing one [tile_n] row of c_data at offset g*tile_n.
                    gemv_offset_consts = [
                        ConstantOp(IntegerAttr.get(T.i32(), g * tile_n), None)
                        for g in range(GEMV_COUNT)
                    ]
                    for g in range(GEMV_COUNT):
                        zero_fill = CallOp(
                            linalg_fill_func, [gemv_offset_consts[g], c_data]
                        )
                        for i in range_(0, k, tile_k):
                            # Weight chunk via bL2ToL1 (one [tile_k, tile_n]
                            # tile per inner-k iter).
                            ChannelGet(
                                "bL2ToL1",
                                l1_b_data,
                                offsets=[],
                                sizes=[],
                                strides=[],
                                indices=[_tx],
                            )
                            # x_offset = i (in elements) selects the
                            # tile_k-sized slice of l1_a_data for this chunk.
                            i32_x_offset = arith.index_cast(T.i32(), i)
                            vecmat = CallOp(
                                vecmat_func,
                                [
                                    i32_x_offset,
                                    gemv_offset_consts[g],
                                    l1_a_data,
                                    l1_b_data,
                                    c_data,
                                ],
                            )
                            yield_([])

                    DeallocOp(l1_a_data)
                    DeallocOp(l1_b_data)

                    l1_freq_pos_data = AllocOp(l1MemrefTyHSByTwo, [], [])
                    zero_constindex = ConstantOp.create_index(0)

                    freq_pos_call = CallOp(
                        freq_pos_func,
                        [arith.index_cast(T.i32(), pos), l1_freq_pos_data],
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

                    # Vectorized -99 fill of attn buffer (GROUP_SIZE * seq_len);
                    # the previous scalar scf.for variant unrolled into thousands
                    # of stores in the AIE2P core ELF.
                    attn_total = arith.ConstantOp(
                        IntegerAttr.get(T.i32(), GROUP_SIZE * seq_len), None
                    )
                    CallOp(fill_neg99_func, [attn_l1_data, attn_total])

                    # Vectorized copy of the first GROUP_SIZE rows of c_data
                    # (Q heads) into q_l1_data.
                    q_total = arith.ConstantOp(
                        IntegerAttr.get(T.i32(), GROUP_SIZE * n), None
                    )
                    CallOp(vec_copy_n_func, [c_data, q_l1_data, q_total])

                    # attn_1_group: K rows shared across all GROUP_SIZE Q heads.
                    # aL2ToL1 only carries K then V rows now (no x, no weights),
                    # so the Step 3 scf.for + cycling-BD pattern works.
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
                            attn_func,
                            [q_l1_data, l1_shared_bd_buf_data, i32_iv, attn_l1_data],
                        )
                        yield_([])

                    # softmax_group: per-row softmax across the group.
                    CallOp(
                        softmax_func,
                        [
                            attn_l1_data,
                            arith.index_cast(T.i32(), pos_p1),
                            softmax_l1_data,
                        ],
                    )

                    xb_l1_data = AllocOp(l1MemrefTyXb, [], [])
                    xb_total = arith.ConstantOp(
                        IntegerAttr.get(T.i32(), GROUP_SIZE * n), None
                    )
                    CallOp(fill_zero_func, [xb_l1_data, xb_total])
                    # attn_2_group: V rows shared across the group.
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
                    DeallocOp(softmax_l1_data)
                    DeallocOp(xb_l1_data)
                    DeallocOp(l1_shared_bd_buf_data)

                herd_body_0.attributes["link_with"] = StringAttr.get("attn_decode_npu2.o")

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
