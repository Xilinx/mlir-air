# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode flash attention on NPU2 with MMIO-fed Q (cascade-Q mock).

Stepping stone toward the cascade-Q fusion plan (DECODE_FUSION_PLAN.md):
this variant runs attention on a single 8-tile row with NS=1 and
delivers Q via `channel_type="mmio"` instead of a shim DMA channel.
That mocks what RoPE will eventually produce on-chip via cascade.

Differences from attn_decode_npu2.py:
  * Herd shape is [NKV, 1] (one row, no cascade merge).
  * Each tile sequentially processes lk/lkp chunks of K/V.
  * Q is a compile-time constant baked in via NKV per-col
    memref.globals, delivered through a `channel_type="mmio"`
    channel of size [NKV]. No shim channel allocated for Q.
  * pos is also a compile-time MMIO constant (per-col globals).
    Baking it in keeps shim input pressure at 2 channels/col
    (K + V) and avoids the dma_packet auto-upgrade.
  * K and V each take an L3->L2->L1 memtile relay path:
    one multi-dim shim BD per col covers ALL chunks (KIn/VIn),
    then the memtile DMA chain doles them out chunk-by-chunk to L1
    (K2L1/V2L1). Per-col shim BD count is 2 (K + V) — constant in
    `chunks` — so this scales to arbitrary lk.
    K lands on tile S2MM 0, V on S2MM 1: split channels through
    memtile sidestep the AIE2P S2MM-1 shim BD-chain ordering bug
    that prevents the same split L3->L1-direct.
  * No cascade channels, no _cascade_merge.

DMA channel budget per col (2 S2MM + 2 MM2S):
  Shim S2MM 0: KIn (1 BD)
  Shim S2MM 1: VIn (1 BD)
  Memtile DMAs: 2 in + 2 out (K + V relay)
  Tile S2MM 0: K2L1, S2MM 1: V2L1
  Tile MM2S 0: output L1->L2
  Q + pos: MMIO (no shim channel allocated)

Q and pos values must be supplied to build_module() at compile time.
To test different values at runtime requires re-compilation (V1 MMIO
limitation).
"""

import argparse
from math import sqrt

import numpy as np
from ml_dtypes import bfloat16

import air
from air.ir import *
from air.dialects.air import *
from air.dialects.air import channel
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
from air.dialects.scf import for_ as scf_range, yield_
from air.dialects import scf, affine, arith
from air.dialects.affine import apply as affine_apply


@module_builder
def build_module(
    q_data,
    pos_value,
    lk=2048,
    lkp=64,
    dk=64,
    dv=64,
    group_size=4,
    num_kv_heads=8,
):
    """Build the MMIO-Q decode attention module.

    Args:
        q_data: numpy bfloat16 array of shape [num_heads, dk] baked in
            at compile time as the Q payload (per-col MMIO globals).
        pos_value: int32, current decode position. Baked in via MMIO
            (broadcast to all cols).
        lk, lkp, dk, dv, group_size, num_kv_heads: standard decode params.
            num_kv_parallel is fixed at num_kv_heads (single launch
            iteration), and NS is fixed at 1.
    """
    NS = 1
    NKV = num_kv_heads
    num_heads = group_size * num_kv_heads
    assert lk % lkp == 0, f"lk ({lk}) must be divisible by lkp ({lkp})"
    chunks = lk // lkp
    assert q_data.shape == (
        num_heads,
        dk,
    ), f"q_data shape {q_data.shape} != [{num_heads}, {dk}]"
    assert (
        q_data.dtype == bfloat16
    ), f"q_data dtype must be bfloat16, got {q_data.dtype}"

    bf16 = Type.parse("bf16")
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # Per-tile L1 type for Q is [group_size, dk]. The MMIO global
    # backing each col is [group_size, dk] of bf16.
    q_per_col_shape = [group_size, dk]

    q_l1_t = MemRefType.get(q_per_col_shape, bf16, memory_space=l1_space)
    k_l1_t = MemRefType.get([lkp, dk], bf16, memory_space=l1_space)
    v_l1_t = MemRefType.get([lkp, dv], bf16, memory_space=l1_space)
    scores_l1_t = MemRefType.get([group_size, lkp], bf16, memory_space=l1_space)
    out_l1_t = MemRefType.get([group_size, dv], bf16, memory_space=l1_space)
    # max/sum buffers must be at least 32 elements (cascade alignment),
    # though no cascade is used here we keep the kernel ABI consistent.
    ms_l1_t = MemRefType.get([max(group_size, 32)], bf16, memory_space=l1_space)
    pos_l1_t = MemRefType.get([1], i32, memory_space=l1_space)

    out_l2_t = MemRefType.get([NKV, group_size, dv], bf16, memory_space=l2_space)
    # L2 staging buffers for K and V — one chunk's worth each, per col.
    # The L3->L2 transfer is multi-dim covering all chunks (one shim BD
    # per col), the memtile reuses the buffer chunk-by-chunk on the
    # L2->L1 path.
    k_l2_t = MemRefType.get([lkp, dk], bf16, memory_space=l2_space)
    v_l2_t = MemRefType.get([lkp, dv], bf16, memory_space=l2_space)

    # L3 memref types for runtime args (Q is NOT a runtime arg).
    k_l3_t = MemRefType.get([num_kv_heads, lk, dk], bf16)
    v_l3_t = MemRefType.get([num_kv_heads, lk, dv], bf16)
    pos_l3_t = MemRefType.get([1], i32)
    out_l3_t = MemRefType.get([num_heads, dv], bf16)

    # Per-col Q global: bf16, matching the destination L1 buffer's
    # type. The MMIO lowering loads this into the L1 buffer at
    # device-init time via aie.buffer initial_value.
    q_global_t = MemRefType.get(q_per_col_shape, bf16)

    q_global_names = []
    for c in range(NKV):
        sym = f"q_const_col{c}"
        q_global_names.append(sym)
        q_slice = q_data[c * group_size : (c + 1) * group_size, :]
        # MLIR Python bindings can't take bf16 numpy directly — pass
        # bytes as uint16 with a bf16-typed tensor type.
        tensor_t = RankedTensorType.get(q_per_col_shape, bf16)
        const_attr = DenseElementsAttr.get(
            np.ascontiguousarray(q_slice).view(np.uint16),
            type=tensor_t,
        )
        GlobalOp(
            sym_name=sym,
            type_=TypeAttr.get(q_global_t),
            sym_visibility="private",
            initial_value=const_attr,
        )

    # pos as a compile-time MMIO constant. Baking pos in (rather than
    # streaming via shim) keeps shim input pressure at 2 channels/col
    # (K + V) and avoids the dma_packet auto-upgrade. NKV separate
    # globals keep the structure parallel to Q (one put per col into a
    # size=[NKV] mmio channel).
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

    # Kernel externs (same .o as the original decode example).
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

    # Channels. Q and pos via MMIO (no shim).
    # K and V each take an L3->L2->L1 memtile relay path:
    #   * KIn (L3->L2): one multi-dim shim BD per col covering all
    #     chunks. Per-col shim BD count is constant in `chunks` -> 1.
    #   * K2L1 (L2->L1): per-chunk transfers from memtile to tile;
    #     the memtile DMA chain handles the chunk iteration.
    # Same shape for V. With memtile in the middle, K and V can sit
    # on separate tile S2MM channels without hitting the shim-side
    # AIE2P S2MM-1 BD-chain ordering bug (that bug is shim-only;
    # memtile DMAs aren't affected, as confirmed by attn_npu2.py
    # prefill which uses the same split-channel + memtile pattern).
    channel("Q_mmio", size=[NKV], channel_type="mmio")
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
            # MMIO Q + pos puts: NKV separate puts each pulling from
            # its own compile-time constant memref.global.
            for c in range(NKV):
                c_idx = ConstantOp(index_type, c)
                q_src = get_global(q_global_t, q_global_names[c])
                ChannelPut("Q_mmio", q_src, indices=[c_idx])
                pos_src = get_global(pos_global_t, pos_global_names[c])
                ChannelPut("pos_mmio", pos_src, indices=[c_idx])

            # Launch-side L3 puts: one multi-dim shim BD per col covers
            # ALL chunks for K (and similarly for V). Per-col shim BD
            # count is 2 (K + V), constant in `chunks` -> scales to any
            # lk. 4D iteration: outer dim selects the KV head (stride
            # lk*dk for K, lk*dv for V — the natural row-major stride
            # of K/V's first dim); middle dim iterates the chunks
            # within that head; inner two dims are (lkp rows, dk/dv
            # elements). The memtile breaks each multi-dim transfer
            # into chunks for the L2->L1 path.
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
                # Per-col L2 staging buffers for K and V (one chunk's
                # worth each). The memtile DMA chain reuses these buffers
                # across chunks: fill from L3, drain to L1, repeat.
                k_l2_bufs = [AllocOp(k_l2_t, [], []) for _ in range(NKV)]
                v_l2_bufs = [AllocOp(v_l2_t, [], []) for _ in range(NKV)]
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

                # Memtile L2->L1 relay: per col, drive `chunks`
                # iterations of (get from L3->L2, put L2->L1). The
                # memtile DMA chain handles the buffer reuse and pacing.
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
                    ],
                    link_with="attn_decode_npu2.o",
                )
                def herd_body(
                    tx,
                    _ty,
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
                ):
                    # Q and pos via MMIO: the aie.buffer initial_value
                    # path loads them at device-init time (before any
                    # core starts), so there is no host↔core race —
                    # safe to read at the top of the loop.
                    ChannelGet("Q_mmio", _q_l1, indices=[tx])
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

                    for chunk_idx in scf_range(0, c_chunks, 1):
                        chunk_pos = affine_apply(chunk_pos_map, [chunk_idx])

                        # K on S2MM 0, V on S2MM 1 — both fed by the
                        # memtile L2->L1 relay (KIn/K2L1 and VIn/V2L1).
                        ChannelGet("K2L1", _k_l1, indices=[tx])
                        ChannelGet("V2L1", _v_l1, indices=[tx])

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

                    # NS=1: no cascade merge, divide and emit directly.
                    CallOp([], "decode_div_output", [_sum_l1, _out_l1])
                    ChannelPut("out_dec", _out_l1, indices=[tx])

                # Drain herd outputs into L2 staging.
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

                # L2 -> L3.
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
        prog="attn_decode_npu2_mmioq.py",
        description="Decode flash attention on NPU2 with MMIO-fed Q (NS=1, [NKV,1] herd)",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
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
    parser.add_argument(
        "--zero-q",
        action="store_true",
        help="Inject Q=0 via MMIO and verify against the closed-form "
        "Q=0 reference (uniform softmax => mean(V) per head). Bisect "
        "tool for isolating the K/V/pos data path from MMIO Q correctness.",
    )
    args = parser.parse_args()

    num_heads = args.group_size * args.num_kv_heads
    current_pos = args.current_pos if args.current_pos is not None else args.lk - 1

    rng = np.random.default_rng(args.seed)
    val_range = 2.0
    input_q = rng.uniform(-val_range, val_range, (num_heads, args.dk)).astype(bfloat16)
    if args.zero_q:
        input_q = np.zeros_like(input_q)
    input_k = rng.uniform(
        -val_range, val_range, (args.num_kv_heads, args.lk, args.dk)
    ).astype(bfloat16)
    input_v = rng.uniform(
        -val_range, val_range, (args.num_kv_heads, args.lk, args.dv)
    ).astype(bfloat16)

    mlir_module = build_module(
        q_data=input_q,
        pos_value=current_pos,
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

    inv_sqrt_dk = 1.0 / sqrt(args.dk)
    expected_out = np.zeros((num_heads, args.dv), dtype=bfloat16)
    for h in range(num_heads):
        kv_h = h // args.group_size
        q_h = input_q[h].astype(np.float32)
        K_valid = input_k[kv_h, : current_pos + 1, :].astype(np.float32)
        V_valid = input_v[kv_h, : current_pos + 1, :].astype(np.float32)
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
