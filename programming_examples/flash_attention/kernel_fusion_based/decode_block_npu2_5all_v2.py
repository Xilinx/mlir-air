# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode block on NPU2: full 4-herd RMS->GEMV->RoPE->Attention chain.

Step 5-all V2 of DECODE_FUSION_PLAN.md: extends V1a by adding a
real `rms_herd` upstream of gemv_herd. rms_herd consumes MMIO
x_raw + MMIO RMS weights w_rms, computes x_norm via RMSNorm, and
broadcasts x_norm to all NKV gemv_herd tiles via an L1 broadcast
channel. gemv_herd consumes broadcast x_norm (no longer X_mmio)
plus MMIO W_qkv, and the rest of the chain (rope -> attention)
is unchanged from V1a.

Layout:
  rms_herd     [1, 1]      : MMIO x_raw + w_rms -> rms -> broadcast
                              ↓ (L1 broadcast to all NKV gemv tiles)
  gemv_herd    [NKV, 1]    : x_norm + MMIO W_qkv -> matvec -> 3 cascades
                              ↓
  rope_herd    [NKV, 1]    : 3 cascades + MMIO LUT -> rope -> 3 cascades
                              ↓
  decode_herd  [NKV, 1]    : 3 cascades + MMIO pos
                            + L3->memtile->L1 K/V cache reads
                            -> flash attention + K_new/V_new splice

Toy input dim N=64 keeps W_qkv in L1 — no L2 staging needed.
Per-col shim DMA: 2 MM2S (KIn, VIn). MMIO + cascade + L1 broadcast
do not consume shim.
"""

import argparse
from math import sqrt

import numpy as np
from ml_dtypes import bfloat16

import air
from air.ir import *
from air.dialects.air import *
from air.dialects.air import channel, Channel
from air.dialects.arith import ConstantOp
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    GlobalOp,
    get_global,
    load,
    store,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_ as scf_range, yield_, IfOp as ScfIfOp
from air.ir import InsertionPoint
from air.dialects import scf, affine, arith
from air.dialects.affine import apply as affine_apply


def make_rope_lut(dk, pos, theta=10000.0):
    """LUT layout matches mlir-aie/aie_kernels/aie2p/rope.cc:
    LUT[2k]   = cos(pos * 1/theta^(2k/dk))
    LUT[2k+1] = sin(pos * 1/theta^(2k/dk))
    """
    half = dk // 2
    inv_freq = 1.0 / (theta ** (np.arange(half) * 2.0 / dk))
    angles = pos * inv_freq
    lut = np.empty(dk, dtype=np.float32)
    lut[0::2] = np.cos(angles)
    lut[1::2] = np.sin(angles)
    return lut.astype(bfloat16)


def apply_rope_ref(q, lut):
    """NumPy reference matching the rope.cc kernel semantics:
    even/odd split; out_even = even*cos - odd*sin; out_odd = even*sin + odd*cos
    (per (q_head, dk) row; same lut shared across all heads)
    """
    q_f32 = q.astype(np.float32)
    cos = lut[0::2].astype(np.float32)
    sin = lut[1::2].astype(np.float32)
    even = q_f32[..., 0::2]
    odd = q_f32[..., 1::2]
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos
    out = np.empty_like(q_f32)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out.astype(bfloat16)


@module_builder
def build_module(
    w_qkv_data,
    x_raw_data,
    w_rms_data,
    pos_value,
    n_in=64,
    lk=2048,
    lkp=64,
    dk=64,
    dv=64,
    group_size=4,
    num_kv_heads=8,
):
    NS = 1
    NKV = num_kv_heads
    num_heads = group_size * num_kv_heads
    m_per_col = group_size * dk + dk + dv  # Q + K_new + V_new per col
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    chunks = lk // lkp
    assert w_qkv_data.shape == (
        NKV,
        m_per_col,
        n_in,
    ), f"w_qkv shape {w_qkv_data.shape} != [{NKV}, {m_per_col}, {n_in}]"
    assert x_raw_data.shape == (n_in,), f"x_raw shape != [{n_in}]"
    assert w_rms_data.shape == (n_in,), f"w_rms shape != [{n_in}]"

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    q_per_col_shape = [group_size, dk]
    knew_per_col_shape = [dk]
    vnew_per_col_shape = [dv]

    q_l1_t = MemRefType.get(q_per_col_shape, bf16, memory_space=l1_space)
    knew_l1_t = MemRefType.get(knew_per_col_shape, bf16, memory_space=l1_space)
    vnew_l1_t = MemRefType.get(vnew_per_col_shape, bf16, memory_space=l1_space)
    # gemv L1: per-col W_qkv weight matrix (rows for Q+K+V) and the
    # input vector x. Both loaded via MMIO initial_value at device init.
    w_q_l1_t = MemRefType.get([group_size * dk, n_in], bf16, memory_space=l1_space)
    w_k_l1_t = MemRefType.get([dk, n_in], bf16, memory_space=l1_space)
    w_v_l1_t = MemRefType.get([dv, n_in], bf16, memory_space=l1_space)
    q_flat_l1_t = MemRefType.get([group_size * dk], bf16, memory_space=l1_space)
    x_l1_t = MemRefType.get([n_in], bf16, memory_space=l1_space)
    lut_l1_t = MemRefType.get([dk], bf16, memory_space=l1_space)
    k_l1_t = MemRefType.get([lkp, dk], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv], bf16, memory_space=l1_space)
    scores_l1_t = MemRefType.get([group_size, lkp], bf16, memory_space=l1_space)
    out_l1_t = MemRefType.get([group_size, dv], bf16, memory_space=l1_space)
    ms_l1_t = MemRefType.get([max(group_size, 32)], bf16, memory_space=l1_space)
    pos_l1_t = MemRefType.get([1], i32, memory_space=l1_space)

    out_l2_t = MemRefType.get([NKV, group_size, dv], bf16, memory_space=l2_space)
    k_l2_t = MemRefType.get([lkp, dk], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, dv], bf16, memory_space=l2_space)

    k_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    v_l3_t = MemRefType.get([num_kv_heads, lk, dv], bf16)
    out_l3_t = MemRefType.get([num_heads, dv], bf16)

    # MMIO W_qkv globals: per-col [m_per_col, n_in] weight matrix.
    # Split into per-output-stream (Q, K, V) globals so the destination
    # MMIO L1 buffer types (W_q_l1, W_k_l1, W_v_l1) match cleanly.
    w_q_global_t = MemRefType.get([group_size * dk, n_in], bf16)
    w_k_global_t = MemRefType.get([dk, n_in], bf16)
    w_v_global_t = MemRefType.get([dv, n_in], bf16)
    w_q_global_names = []
    w_k_global_names = []
    w_v_global_names = []
    for c in range(NKV):
        # Q rows: [c, 0:group_size*dk, :]
        sym_q = f"w_q_const_col{c}"
        w_q_global_names.append(sym_q)
        slice_q = w_qkv_data[c, : group_size * dk, :]
        tensor_t_q = RankedTensorType.get([group_size * dk, n_in], bf16)
        attr_q = DenseElementsAttr.get(
            np.ascontiguousarray(slice_q).view(np.uint16), type=tensor_t_q
        )
        GlobalOp(
            sym_name=sym_q,
            type_=TypeAttr.get(w_q_global_t),
            sym_visibility="private",
            initial_value=attr_q,
        )

        # K_new rows: [c, group_size*dk : group_size*dk + dk, :]
        sym_k = f"w_k_const_col{c}"
        w_k_global_names.append(sym_k)
        slice_k = w_qkv_data[c, group_size * dk : group_size * dk + dk, :]
        tensor_t_k = RankedTensorType.get([dk, n_in], bf16)
        attr_k = DenseElementsAttr.get(
            np.ascontiguousarray(slice_k).view(np.uint16), type=tensor_t_k
        )
        GlobalOp(
            sym_name=sym_k,
            type_=TypeAttr.get(w_k_global_t),
            sym_visibility="private",
            initial_value=attr_k,
        )

        # V_new rows: [c, group_size*dk + dk : group_size*dk + dk + dv, :]
        sym_v = f"w_v_const_col{c}"
        w_v_global_names.append(sym_v)
        slice_v = w_qkv_data[c, group_size * dk + dk : group_size * dk + dk + dv, :]
        tensor_t_v = RankedTensorType.get([dv, n_in], bf16)
        attr_v = DenseElementsAttr.get(
            np.ascontiguousarray(slice_v).view(np.uint16), type=tensor_t_v
        )
        GlobalOp(
            sym_name=sym_v,
            type_=TypeAttr.get(w_v_global_t),
            sym_visibility="private",
            initial_value=attr_v,
        )

    # MMIO x_raw + w_rms globals: single tile (rms_herd consumes them
    # via per-tile MMIO). One global each — rms_herd is [1, 1].
    xraw_global_t = MemRefType.get([n_in], bf16)
    xraw_tensor_t = RankedTensorType.get([n_in], bf16)
    GlobalOp(
        sym_name="x_raw_const",
        type_=TypeAttr.get(xraw_global_t),
        sym_visibility="private",
        initial_value=DenseElementsAttr.get(
            np.ascontiguousarray(x_raw_data).view(np.uint16),
            type=xraw_tensor_t,
        ),
    )
    wrms_global_t = MemRefType.get([n_in], bf16)
    GlobalOp(
        sym_name="w_rms_const",
        type_=TypeAttr.get(wrms_global_t),
        sym_visibility="private",
        initial_value=DenseElementsAttr.get(
            np.ascontiguousarray(w_rms_data).view(np.uint16),
            type=xraw_tensor_t,
        ),
    )

    # MMIO LUT globals: per-pos rope cos/sin LUT, shared across all
    # heads of all cols. Delivered per-col for symmetry with mmio
    # bundle dispatch.
    rope_lut = make_rope_lut(dk, pos_value)
    lut_global_t = MemRefType.get([dk], bf16)
    lut_global_names = []
    for c in range(NKV):
        sym = f"lut_const_col{c}"
        lut_global_names.append(sym)
        tensor_t = RankedTensorType.get([dk], bf16)
        const_attr = DenseElementsAttr.get(
            np.ascontiguousarray(rope_lut).view(np.uint16),
            type=tensor_t,
        )
        GlobalOp(
            sym_name=sym,
            type_=TypeAttr.get(lut_global_t),
            sym_visibility="private",
            initial_value=const_attr,
        )

    pos_global_t = MemRefType.get([1], i32)
    pos_tensor_t = RankedTensorType.get([1], i32)
    pos_global_names = []
    for c in range(NKV):
        sym = f"pos_const_col{c}"
        pos_global_names.append(sym)
        pos_attr = DenseElementsAttr.get(
            np.array([pos_value], dtype=np.int32),
            type=pos_tensor_t,
        )
        GlobalOp(
            sym_name=sym,
            type_=TypeAttr.get(pos_global_t),
            sym_visibility="private",
            initial_value=pos_attr,
        )

    def external_func(name, inputs, link_with=None):
        func_type = FunctionType.get(inputs, [])
        f = FuncOp(name=name, type=func_type, visibility="private")
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            f.attributes["link_with"] = StringAttr.get(link_with)
        return f

    kobj = "attn_decode_npu2.o"
    external_func("decode_zero_output", [out_l1_t], link_with=kobj)
    external_func("decode_zero_sum", [ms_l1_t], link_with=kobj)
    external_func("decode_neg_inf_max", [ms_l1_t], link_with=kobj)
    external_func(
        "compute_qk_scores_bf16", [k_l1_t, q_l1_t, scores_l1_t], link_with=kobj
    )
    external_func("apply_decode_mask", [scores_l1_t, i32, i32], link_with=kobj)
    external_func("decode_softmax_max", [scores_l1_t, ms_l1_t], link_with=kobj)
    external_func(
        "decode_softmax_exp",
        [scores_l1_t, ms_l1_t, ms_l1_t, ms_l1_t],
        link_with=kobj,
    )
    external_func("decode_softmax_sum", [scores_l1_t, ms_l1_t, ms_l1_t], link_with=kobj)
    external_func("decode_rescale_output", [ms_l1_t, out_l1_t], link_with=kobj)
    external_func(
        "compute_pv_output_bf16", [scores_l1_t, v_l1_t, out_l1_t], link_with=kobj
    )
    external_func("decode_div_output", [ms_l1_t, out_l1_t], link_with=kobj)

    # rope_multi: applies rope to nrows consecutive bf16 rows of size
    # dim, sharing one LUT. Wraps the upstream rope kernel — avoids
    # per-row subview type acrobatics in the herd body.
    rope_kobj = "rope_multi.o"
    external_func(
        "rope_multi",
        [q_l1_t, lut_l1_t, q_l1_t, i32, i32],
        link_with=rope_kobj,
    )
    # Same wrapper, called for K_new (single row).
    external_func(
        "rope_multi_knew",
        [knew_l1_t, lut_l1_t, knew_l1_t, i32, i32],
        link_with=rope_kobj,
    )

    # Note: row-patching of K_new/V_new into the last K/V chunk is
    # done with scalar memref.load/store loops in the herd body
    # rather than a kernel call — keeps decode_herd link_with
    # pinned to attn_decode_npu2.o (one .o per herd) and avoids
    # the per-row strided-memref subview type juggling that
    # rope_multi originally sidestepped.

    # simple_matvec_bf16(W, x, y, M, K) : y[m] = sum_k W[m, k] * x[k].
    # Lives in simple_matvec.o. Used by gemv_herd at toy dim N=64.
    matvec_kobj = "simple_matvec.o"
    # The kernel C ABI takes a flat bf16* — MLIR memref shape is just
    # used by the c_interface wrapper to extract the data pointer.
    # Declare the Q output as q_l1_t [group_size, dk] for clean
    # downstream cascade put (flat-shape would need an expand_shape).
    external_func(
        "simple_matvec_bf16",
        [w_q_l1_t, x_l1_t, q_l1_t, i32, i32],
        link_with=matvec_kobj,
    )
    external_func(
        "simple_matvec_bf16_k",
        [w_k_l1_t, x_l1_t, knew_l1_t, i32, i32],
        link_with=matvec_kobj,
    )
    external_func(
        "simple_matvec_bf16_v",
        [w_v_l1_t, x_l1_t, vnew_l1_t, i32, i32],
        link_with=matvec_kobj,
    )

    # simple_rms_bf16(x, w, y, N) : y = x * rsqrt(mean(x^2)+eps) * w.
    # Lives in simple_rms.o. Used by rms_herd at toy dim N=64.
    rms_kobj = "simple_rms.o"
    external_func(
        "simple_rms_bf16",
        [x_l1_t, x_l1_t, x_l1_t, i32],
        link_with=rms_kobj,
    )

    # MMIO weight channels for gemv_herd (W_qkv split per output stream).
    channel("W_q_mmio", size=[NKV], channel_type="mmio")
    channel("W_k_mmio", size=[NKV], channel_type="mmio")
    channel("W_v_mmio", size=[NKV], channel_type="mmio")
    # MMIO inputs for rms_herd (single tile, indices=[0]).
    channel("XRaw_mmio", size=[1], channel_type="mmio")
    channel("WRms_mmio", size=[1], channel_type="mmio")
    # L1 broadcast: rms_herd's single tile -> all NKV gemv tiles.
    # Uppercase Channel() supports broadcast_shape (lowercase channel
    # forwards channel_type but not broadcast_shape — see CLAUDE.md).
    Channel("x_norm_bcast", size=[1, 1], broadcast_shape=[NKV, 1])

    # gemv -> rope cascades (3 hops on the gemv->rope segment of the
    # column cascade wire). gemv_herd_stub gets MMIO into L1 and
    # then immediately puts onto these cascades.
    channel("G2R_Q", size=[NKV, 1], channel_type="cascade")
    channel("G2R_KNew", size=[NKV, 1], channel_type="cascade")
    channel("G2R_VNew", size=[NKV, 1], channel_type="cascade")

    # rope -> decode cascades (3 hops on the rope->decode segment).
    channel("LUT_mmio", size=[NKV], channel_type="mmio")
    channel("Q_cascade", size=[NKV, 1], channel_type="cascade")
    channel("KNew_cascade", size=[NKV, 1], channel_type="cascade")
    channel("VNew_cascade", size=[NKV, 1], channel_type="cascade")
    channel("pos_mmio", size=[NKV], channel_type="mmio")
    channel("KIn", size=[NKV])
    channel("K2L1", size=[NKV])
    channel("VIn", size=[NKV])
    channel("V2L1", size=[NKV])
    channel("out_dec", size=[NKV])

    @FuncOp.from_py_func(k_l3_t, v_l3_t, out_l3_t)
    def decode_attention_bf16(k_in, v_in, out):
        c1_idx = ConstantOp(index_type, 1)

        @launch(operands=[k_in, v_in, out], sizes=[c1_idx])
        def launch_body(_lx, _lsx, k_l, v_l, out_l):
            # Single put for rms_herd's MMIO inputs (size=[1]).
            c0_idx = ConstantOp(index_type, 0)
            xraw_src = get_global(xraw_global_t, "x_raw_const")
            ChannelPut("XRaw_mmio", xraw_src, indices=[c0_idx])
            wrms_src = get_global(wrms_global_t, "w_rms_const")
            ChannelPut("WRms_mmio", wrms_src, indices=[c0_idx])

            for c in range(NKV):
                c_idx = ConstantOp(index_type, c)
                w_q_src = get_global(w_q_global_t, w_q_global_names[c])
                ChannelPut("W_q_mmio", w_q_src, indices=[c_idx])
                w_k_src = get_global(w_k_global_t, w_k_global_names[c])
                ChannelPut("W_k_mmio", w_k_src, indices=[c_idx])
                w_v_src = get_global(w_v_global_t, w_v_global_names[c])
                ChannelPut("W_v_mmio", w_v_src, indices=[c_idx])
                lut_src = get_global(lut_global_t, lut_global_names[c])
                ChannelPut("LUT_mmio", lut_src, indices=[c_idx])
                pos_src = get_global(pos_global_t, pos_global_names[c])
                ChannelPut("pos_mmio", pos_src, indices=[c_idx])

            for tx_i in range(NKV):
                c_tx_i = ConstantOp(index_type, tx_i)
                ChannelPut(
                    "KIn",
                    k_l,
                    offsets=[c_tx_i, 0, 0, 0],
                    sizes=[1, chunks, lkp, dk],
                    strides=[lk * dk, lkp * dk, dk, 1],
                    indices=[c_tx_i],
                )
                ChannelPut(
                    "VIn",
                    v_l,
                    offsets=[c_tx_i, 0, 0, 0],
                    sizes=[1, chunks, lkp, dv],
                    strides=[lk * dv, lkp * dv, dv, 1],
                    indices=[c_tx_i],
                )

            @segment(name="decode_seg", operands=[out_l])
            def segment_body(out_s):
                out_l2 = AllocOp(out_l2_t, [], [])
                k_l2_bufs = [AllocOp(k_l2_t, [], []) for _ in range(NKV)]
                v_l2_bufs = [AllocOp(v_l2_t, [], []) for _ in range(NKV)]

                # rms_herd L1: x_raw (MMIO), w_rms (MMIO), x_norm (out
                # via L1 broadcast to gemv).
                xraw_rms_l1 = AllocOp(x_l1_t, [], [])
                wrms_rms_l1 = AllocOp(x_l1_t, [], [])
                xnorm_rms_l1 = AllocOp(x_l1_t, [], [])

                # gemv_herd L1 buffers (V2: x_norm via broadcast).
                #   W_q/W_k/W_v: per-output-stream weights (MMIO load)
                #   x_gemv: input vector (broadcast load from rms_herd)
                #   q_gemv/knew_gemv/vnew_gemv: matvec outputs
                #     cascaded south to rope_herd
                w_q_gemv_l1 = AllocOp(w_q_l1_t, [], [])
                w_k_gemv_l1 = AllocOp(w_k_l1_t, [], [])
                w_v_gemv_l1 = AllocOp(w_v_l1_t, [], [])
                x_gemv_l1 = AllocOp(x_l1_t, [], [])
                q_gemv_l1 = AllocOp(q_l1_t, [], [])
                knew_gemv_l1 = AllocOp(knew_l1_t, [], [])
                vnew_gemv_l1 = AllocOp(vnew_l1_t, [], [])

                # rope_herd L1: now receives Q/K_new/V_new from G2R
                # cascades (not MMIO) and post-rope into _out buffers.
                q_prod_in_l1 = AllocOp(q_l1_t, [], [])
                lut_l1 = AllocOp(lut_l1_t, [], [])
                q_prod_out_l1 = AllocOp(q_l1_t, [], [])
                knew_in_l1_prod = AllocOp(knew_l1_t, [], [])
                knew_out_l1_prod = AllocOp(knew_l1_t, [], [])
                vnew_l1_prod = AllocOp(vnew_l1_t, [], [])

                # Consumer L1: drain buffers for K_new/V_new cascades.
                # V0 doesn't fold them into attention, just drains.
                knew_drain_l1 = AllocOp(knew_l1_t, [], [])
                vnew_drain_l1 = AllocOp(vnew_l1_t, [], [])

                q_l1 = AllocOp(q_l1_t, [], [])
                k_l1 = AllocOp(k_l1_t, [], [])
                v_l1 = AllocOp(v_l1_t, [], [])
                scores_l1 = AllocOp(scores_l1_t, [], [])
                out_l1 = AllocOp(out_l1_t, [], [])
                max_l1 = AllocOp(ms_l1_t, [], [])
                sum_l1 = AllocOp(ms_l1_t, [], [])
                score_max_tmp = AllocOp(ms_l1_t, [], [])
                rescale_tmp = AllocOp(ms_l1_t, [], [])
                pos_l1 = AllocOp(pos_l1_t, [], [])

                c_nkv = ConstantOp(index_type, NKV)
                c_ns = ConstantOp(index_type, 1)
                c_chunks_seg = ConstantOp(index_type, chunks)

                for tx_i in range(NKV):
                    c_tx_i = ConstantOp(index_type, tx_i)
                    for _ in scf_range(0, c_chunks_seg, 1):
                        ChannelGet(
                            "KIn",
                            k_l2_bufs[tx_i].result,
                            indices=[c_tx_i],
                        )
                        ChannelPut(
                            "K2L1",
                            k_l2_bufs[tx_i].result,
                            indices=[c_tx_i],
                        )
                        yield_([])
                    for _ in scf_range(0, c_chunks_seg, 1):
                        ChannelGet(
                            "VIn",
                            v_l2_bufs[tx_i].result,
                            indices=[c_tx_i],
                        )
                        ChannelPut(
                            "V2L1",
                            v_l2_bufs[tx_i].result,
                            indices=[c_tx_i],
                        )
                        yield_([])

                # rms_herd: single tile, MMIO x_raw + w_rms, computes
                # x_norm and broadcasts to all NKV gemv_herd tiles.
                # Declared BEFORE gemv so the placer puts it north.
                c1_seg = ConstantOp(index_type, 1)

                @herd(
                    name="rms_herd",
                    sizes=[c1_seg, c1_seg],
                    operands=[xraw_rms_l1, wrms_rms_l1, xnorm_rms_l1],
                    link_with=rms_kobj,
                )
                def rms_body(_tx, _ty, _hsx, _hsy, _xraw, _wrms, _xnorm):
                    ChannelGet("XRaw_mmio", _xraw, indices=[_tx])
                    ChannelGet("WRms_mmio", _wrms, indices=[_tx])
                    n_i32_rms = ConstantOp(i32, n_in)
                    CallOp([], "simple_rms_bf16", [_xraw, _wrms, _xnorm, n_i32_rms])
                    ChannelPut("x_norm_bcast", _xnorm, indices=[_tx, _ty])

                # gemv_herd: real matvec. Receives MMIO W_qkv (split
                # into 3 per-output-stream globals) + broadcast x_norm
                # from rms_herd, computes Q/K_new/V_new via
                # simple_matvec_bf16, cascades south to rope_herd.
                # Declared BEFORE rope_herd so the cascade-aware
                # placer puts it at the highest row (north of rope,
                # which is north of decode).
                @herd(
                    name="gemv_herd",
                    sizes=[c_nkv, c_ns],
                    operands=[
                        w_q_gemv_l1,
                        w_k_gemv_l1,
                        w_v_gemv_l1,
                        x_gemv_l1,
                        q_gemv_l1,
                        knew_gemv_l1,
                        vnew_gemv_l1,
                    ],
                    link_with=matvec_kobj,
                )
                def gemv_body(
                    _tx,
                    _ty,
                    _hsx,
                    _hsy,
                    _w_q,
                    _w_k,
                    _w_v,
                    _x,
                    _q,
                    _knew,
                    _vnew,
                ):
                    ChannelGet("W_q_mmio", _w_q, indices=[_tx])
                    ChannelGet("W_k_mmio", _w_k, indices=[_tx])
                    ChannelGet("W_v_mmio", _w_v, indices=[_tx])
                    # x_norm arrives via L1 broadcast from rms_herd
                    # (size=[1, 1] producer, broadcast_shape=[NKV, 1]).
                    ChannelGet("x_norm_bcast", _x, indices=[_tx, _ty])

                    n_i32 = ConstantOp(i32, n_in)
                    m_q_i32 = ConstantOp(i32, group_size * dk)
                    m_k_i32 = ConstantOp(i32, dk)
                    m_v_i32 = ConstantOp(i32, dv)
                    CallOp([], "simple_matvec_bf16", [_w_q, _x, _q, m_q_i32, n_i32])
                    CallOp(
                        [], "simple_matvec_bf16_k", [_w_k, _x, _knew, m_k_i32, n_i32]
                    )
                    CallOp(
                        [], "simple_matvec_bf16_v", [_w_v, _x, _vnew, m_v_i32, n_i32]
                    )

                    # Cascade south in the same order rope_herd will
                    # consume (Q, K_new, V_new) — single physical
                    # cascade wire serializes them.
                    ChannelPut("G2R_Q", _q, indices=[_tx, _ty])
                    ChannelPut("G2R_KNew", _knew, indices=[_tx, _ty])
                    ChannelPut("G2R_VNew", _vnew, indices=[_tx, _ty])

                # rope_herd: now receives Q/K_new/V_new from G2R
                # cascades (not MMIO) and LUT from MMIO. Applies rope
                # to Q and K_new (V passthrough), cascades 3 outputs
                # south to decode_herd.
                @herd(
                    name="rope_herd",
                    sizes=[c_nkv, c_ns],
                    operands=[
                        q_prod_in_l1,
                        lut_l1,
                        q_prod_out_l1,
                        knew_in_l1_prod,
                        knew_out_l1_prod,
                        vnew_l1_prod,
                    ],
                    link_with=rope_kobj,
                )
                def rope_body(
                    _tx,
                    _ty,
                    _hsx,
                    _hsy,
                    _q_in,
                    _lut,
                    _q_out,
                    _knew_in,
                    _knew_out,
                    _vnew,
                ):
                    # Q/K_new/V_new now arrive from gemv_herd_stub
                    # via the G2R_* cascades (north). Order matches
                    # gemv_stub_body's puts.
                    ChannelGet("G2R_Q", _q_in, indices=[_tx, _ty])
                    ChannelGet("G2R_KNew", _knew_in, indices=[_tx, _ty])
                    ChannelGet("G2R_VNew", _vnew, indices=[_tx, _ty])
                    ChannelGet("LUT_mmio", _lut, indices=[_tx])

                    q_nrows_i32 = ConstantOp(i32, group_size)
                    knew_nrows_i32 = ConstantOp(i32, 1)
                    dim_i32 = ConstantOp(i32, dk)
                    CallOp(
                        [], "rope_multi", [_q_in, _lut, _q_out, q_nrows_i32, dim_i32]
                    )
                    CallOp(
                        [],
                        "rope_multi_knew",
                        [_knew_in, _lut, _knew_out, knew_nrows_i32, dim_i32],
                    )

                    # Cascade order: Q, K_new, V_new (consumer must
                    # get in the same order on the same physical
                    # cascade link).
                    ChannelPut("Q_cascade", _q_out, indices=[_tx, _ty])
                    ChannelPut("KNew_cascade", _knew_out, indices=[_tx, _ty])
                    ChannelPut("VNew_cascade", _vnew, indices=[_tx, _ty])

                @herd(
                    name="decode_herd",
                    sizes=[c_nkv, c_ns],
                    operands=[
                        q_l1,
                        k_l1,
                        v_l1,
                        scores_l1,
                        out_l1,
                        max_l1,
                        sum_l1,
                        score_max_tmp,
                        rescale_tmp,
                        pos_l1,
                        knew_drain_l1,
                        vnew_drain_l1,
                    ],
                    link_with=kobj,
                )
                def herd_body(
                    tx,
                    ty,
                    _hsx,
                    _hsy,
                    _q_l1,
                    _k_l1,
                    _v_l1,
                    _scores_l1,
                    _out_l1,
                    _max_l1,
                    _sum_l1,
                    _score_max_tmp,
                    _rescale_tmp,
                    _pos_l1,
                    _knew_drain,
                    _vnew_drain,
                ):
                    # Get all 3 cascades in producer-put order
                    # (Q, K_new, V_new) — same physical cascade link.
                    ChannelGet("Q_cascade", _q_l1, indices=[tx, ty])
                    ChannelGet("KNew_cascade", _knew_drain, indices=[tx, ty])
                    ChannelGet("VNew_cascade", _vnew_drain, indices=[tx, ty])
                    ChannelGet("pos_mmio", _pos_l1, indices=[tx])
                    c0_idx = ConstantOp(index_type, 0)
                    pos_val = load(_pos_l1, [c0_idx])

                    CallOp([], "decode_zero_output", [_out_l1])
                    CallOp([], "decode_neg_inf_max", [_max_l1])
                    CallOp([], "decode_zero_sum", [_sum_l1])

                    c_chunks = ConstantOp(index_type, chunks)
                    chunk_pos_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(lkp),
                            )
                        ],
                    )

                    c_chunks_minus_1 = ConstantOp(index_type, chunks - 1)
                    c_lkp_minus_1 = ConstantOp(index_type, lkp - 1)

                    for chunk_idx in scf_range(0, c_chunks, 1):
                        chunk_pos = affine_apply(chunk_pos_map, [chunk_idx])

                        ChannelGet("K2L1", _k_l1, indices=[tx])
                        ChannelGet("V2L1", _v_l1, indices=[tx])

                        # On the last chunk, splice K_new/V_new
                        # (cascade-delivered) into row lkp-1 of the
                        # K/V tile. The host zero-stuffed slot lk-1
                        # of the cache so cascade values are the
                        # source of truth for that position.
                        is_last = arith.cmpi(
                            arith.CmpIPredicate.eq, chunk_idx, c_chunks_minus_1
                        )
                        if_patch = ScfIfOp(is_last)
                        with InsertionPoint(if_patch.then_block):
                            for i in range(dk):
                                ci = ConstantOp(index_type, i)
                                val = load(_knew_drain, [ci])
                                store(val, _k_l1, [c_lkp_minus_1, ci])
                            for i in range(dv):
                                ci = ConstantOp(index_type, i)
                                val = load(_vnew_drain, [ci])
                                store(val, _v_l1, [c_lkp_minus_1, ci])
                            yield_([])

                        CallOp(
                            [],
                            "compute_qk_scores_bf16",
                            [_k_l1, _q_l1, _scores_l1],
                        )
                        chunk_pos_i32 = arith.IndexCastOp(i32, chunk_pos).result
                        CallOp(
                            [],
                            "apply_decode_mask",
                            [_scores_l1, chunk_pos_i32, pos_val],
                        )
                        CallOp([], "decode_softmax_max", [_scores_l1, _score_max_tmp])
                        CallOp(
                            [],
                            "decode_softmax_exp",
                            [_scores_l1, _max_l1, _score_max_tmp, _rescale_tmp],
                        )
                        CallOp([], "decode_rescale_output", [_rescale_tmp, _out_l1])
                        CallOp(
                            [],
                            "compute_pv_output_bf16",
                            [_scores_l1, _v_l1, _out_l1],
                        )
                        CallOp(
                            [],
                            "decode_softmax_sum",
                            [_scores_l1, _rescale_tmp, _sum_l1],
                        )
                        yield_([])

                    CallOp([], "decode_div_output", [_sum_l1, _out_l1])
                    ChannelPut("out_dec", _out_l1, indices=[tx])

                for tx_i in range(NKV):
                    c_tx_i_out = ConstantOp(index_type, tx_i)
                    ChannelGet(
                        "out_dec",
                        out_l2.result,
                        offsets=[tx_i, 0, 0],
                        sizes=[1, group_size, dv],
                        strides=[group_size * dv, dv, 1],
                        indices=[c_tx_i_out],
                    )

                dma_memcpy_nd(
                    out_s,
                    out_l2.result,
                    dst_offsets=[0, 0],
                    dst_sizes=[NKV * group_size, dv],
                    dst_strides=[dv, 1],
                    src_offsets=[0, 0, 0],
                    src_sizes=[NKV, group_size, dv],
                    src_strides=[group_size * dv, dv, 1],
                )

                DeallocOp(out_l2)
                DeallocOp(xraw_rms_l1)
                DeallocOp(wrms_rms_l1)
                DeallocOp(xnorm_rms_l1)
                DeallocOp(w_q_gemv_l1)
                DeallocOp(w_k_gemv_l1)
                DeallocOp(w_v_gemv_l1)
                DeallocOp(x_gemv_l1)
                DeallocOp(q_gemv_l1)
                DeallocOp(knew_gemv_l1)
                DeallocOp(vnew_gemv_l1)
                DeallocOp(q_prod_in_l1)
                DeallocOp(lut_l1)
                DeallocOp(q_prod_out_l1)
                DeallocOp(knew_in_l1_prod)
                DeallocOp(knew_out_l1_prod)
                DeallocOp(vnew_l1_prod)
                DeallocOp(knew_drain_l1)
                DeallocOp(vnew_drain_l1)
                DeallocOp(q_l1)
                DeallocOp(k_l1)
                DeallocOp(v_l1)
                DeallocOp(scores_l1)
                DeallocOp(out_l1)
                DeallocOp(max_l1)
                DeallocOp(sum_l1)
                DeallocOp(score_max_tmp)
                DeallocOp(rescale_tmp)
                DeallocOp(pos_l1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="decode_block_npu2_5all_v1a.py",
        description="Decode block with real GEMV in 3-herd cascade chain (toy N)",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--n-in",
        type=int,
        default=64,
        help="GEMV input dimension (must match kernel's K=64)",
    )
    parser.add_argument("--lk", type=int, default=2048)
    parser.add_argument("--lkp", type=int, default=64)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--current-pos", type=int, default=None)
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output-format", choices=["elf"], default="elf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_heads = args.group_size * args.num_kv_heads
    current_pos = args.current_pos if args.current_pos is not None else args.lk - 1
    m_per_col = args.group_size * args.dk + args.dk + args.dv

    rng = np.random.default_rng(args.seed)
    # Smaller weight scale so [N=64, sum] doesn't blow up bf16 range
    # in the matvec accumulation. With val_range=0.5 and N=64 the
    # max |dot| ~ 64 * 0.5^2 ~ 16, still well within bf16 dynamic.
    w_range = 0.5
    x_range = 1.0
    val_range = 2.0

    # RMS inputs: x_raw vector + w_rms scaling weights.
    x_raw = rng.uniform(-x_range, x_range, (args.n_in,)).astype(bfloat16)
    w_rms = rng.uniform(0.5, 1.5, (args.n_in,)).astype(bfloat16)

    # GEMV inputs: per-kv-head W_qkv slice (input vec is rms output, computed below).
    w_qkv = rng.uniform(
        -w_range, w_range, (args.num_kv_heads, m_per_col, args.n_in)
    ).astype(bfloat16)

    # Reference RMS: x_norm = x_raw * rsqrt(mean(x_raw^2) + eps) * w_rms
    eps = 1e-5
    x_raw_f32 = x_raw.astype(np.float32)
    w_rms_f32 = w_rms.astype(np.float32)
    rstd = 1.0 / np.sqrt(np.mean(x_raw_f32**2) + eps)
    x_norm_f32 = x_raw_f32 * rstd * w_rms_f32
    x_norm_bf16 = x_norm_f32.astype(bfloat16)

    # Reference matvec uses x_norm cast back to fp32 (matches kernel
    # which loads x_norm bf16 then promotes for accumulation).
    w_f32 = w_qkv.astype(np.float32)
    x_f32 = x_norm_bf16.astype(np.float32)
    qkv_out = (w_f32 @ x_f32).astype(bfloat16)  # [NKV, m_per_col]

    # Slice into Q/K/V per kv-head, then unflatten Q into heads.
    q_pre_rope = qkv_out[:, : args.group_size * args.dk].reshape(
        args.num_kv_heads, args.group_size, args.dk
    )  # [NKV, group_size, dk]
    k_new_pre_rope = qkv_out[
        :, args.group_size * args.dk : args.group_size * args.dk + args.dk
    ]  # [NKV, dk]
    v_new = qkv_out[
        :,
        args.group_size * args.dk
        + args.dk : args.group_size * args.dk
        + args.dk
        + args.dv,
    ]  # [NKV, dv]

    # K_cache, V_cache from L3 — zeroed at slot lk-1 (cascade K_new/V_new
    # are the source of truth for that position).
    input_k = rng.uniform(
        -val_range, val_range, (args.num_kv_heads, args.lk, args.dk)
    ).astype(bfloat16)
    input_v = rng.uniform(
        -val_range, val_range, (args.num_kv_heads, args.lk, args.dv)
    ).astype(bfloat16)
    input_k[:, args.lk - 1, :] = 0
    input_v[:, args.lk - 1, :] = 0

    mlir_module = build_module(
        w_qkv_data=w_qkv,
        x_raw_data=x_raw,
        w_rms_data=w_rms,
        pos_value=current_pos,
        n_in=args.n_in,
        lk=args.lk,
        lkp=args.lkp,
        dk=args.dk,
        dv=args.dv,
        group_size=args.group_size,
        num_kv_heads=args.num_kv_heads,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    from air.backend.xrt_runner import XRTRunner

    rope_lut = make_rope_lut(args.dk, current_pos)
    # Reference Q after rope: per-head [num_heads, dk]
    rotated_q = apply_rope_ref(q_pre_rope.reshape(num_heads, args.dk), rope_lut)
    rotated_k_new = apply_rope_ref(k_new_pre_rope, rope_lut)
    input_k_new = rotated_k_new  # alias for the splice reference below
    input_v_new = v_new

    inv_sqrt_dk = 1.0 / sqrt(args.dk)
    expected_out = np.zeros((num_heads, args.dv), dtype=bfloat16)
    for h in range(num_heads):
        kv_h = h // args.group_size
        q_h = rotated_q[h].astype(np.float32)
        K_full = input_k[kv_h].astype(np.float32).copy()
        V_full = input_v[kv_h].astype(np.float32).copy()
        K_full[args.lk - 1] = rotated_k_new[kv_h].astype(np.float32)
        V_full[args.lk - 1] = input_v_new[kv_h].astype(np.float32)
        K_valid = K_full[: current_pos + 1, :]
        V_valid = V_full[: current_pos + 1, :]
        scores = K_valid @ q_h * inv_sqrt_dk
        mx = scores.max()
        P = np.exp(scores - mx)
        P = P / P.sum()
        expected_out[h] = (P @ V_valid).astype(bfloat16)

    runner = XRTRunner(
        omit_while_true_loop=False,
        omit_pingpong="all",
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="decode_attention_bf16",
        target_device="npu2",
    )

    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_k, input_v],
            expected_outputs=[expected_out],
            atol=0.15,
            rtol=0.04,
            max_mismatch_percentage=2.0,
            min_correlation=0.99,
        )
    )
