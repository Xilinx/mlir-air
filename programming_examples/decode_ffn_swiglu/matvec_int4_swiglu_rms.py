# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4-AWQ FFN GEMV with weighted-RMSNorm input and fused SwiGLU output:
#
#   normed = rms_norm(input_vec, norm_weight)        # row 0 + row 1 of RMS
#   raw[M]  = dequant(A_packed[M, K]) @ normed        # gate/up interleaved
#   swiglu[M/2] = silu(raw[2i]) * raw[2i+1]
#
# A is the AWQ packed Q+S+Z BO (matvec_int4_packed.pack_inputs layout).
# Single matvec herd (no cascade) — simpler than the bf16 4-cascade design
# but stays under the AIE2P 2-S2MM/tile cap: PACKED via aL2ToL1 (1 S2MM)
# + RMS via inRMS broadcast (1 S2MM, replayed once per launch). Gate/up
# partials accumulate into separate L1 scratches; a final vectorized
# SwiGLU pass emits M_per_core/2 outputs per tile.
#
# Reuses mv_int4_bf16.o (same kernel symbol/dim macros as stages 1, 3).

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "matrix_vector_multiplication",
        "int4_awq",
    ),
)

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineSymbolExpr,
    BF16Type,
    BoolAttr,
    F32Type,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    UnitAttr,
    VectorType,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    Channel,
    ChannelGet,
    ChannelPut,
    MemorySpace,
    T,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.air import channel as channel_decl
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    subview,
    load as memref_load,
    store as memref_store,
    cast as memref_cast,
)
from air.dialects import arith
from air.dialects import math as math_dialect
from air.dialects.scf import for_, yield_
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

from matvec_int4_packed import pack_inputs

KERNEL_OBJ_NAME = "mv_int4_bf16.o"

range_ = for_


def build_module(M, K, GS=128, M_TILE=8, K_CHUNK=2048, N_CORES=8):
    """int4-AWQ fused FFN: RMSNorm → int4 GEMV → SwiGLU pair.

    Output is M/2 elements (gate/up paired). M is the interleaved gate+up
    row count (= 2 * hidden_dim). Single-launch, single-herd design.
    """
    assert M % 2 == 0
    assert M % N_CORES == 0
    M_per_core = M // N_CORES
    assert M_per_core % M_TILE == 0
    M_div = M_per_core // M_TILE
    assert K == K_CHUNK, "int4 FFN: K must equal K_CHUNK (single chunk per outer)"
    assert K % GS == 0
    assert M_TILE % 2 == 0
    SILU_VEC = 16
    half_M_per_core = M_per_core // 2
    assert (
        half_M_per_core % SILU_VEC == 0
    ), f"M_per_core/2 ({half_M_per_core}) must be a multiple of {SILU_VEC}"

    n_gpc = K_CHUNK // GS
    q_bytes = M_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * M_TILE * 2
    z_bytes = n_gpc * M_TILE
    tile_bytes = q_bytes + s_bytes + z_bytes
    assert q_bytes % 32 == 0
    assert (q_bytes + s_bytes) % 32 == 0

    total_tiles = N_CORES * M_div

    @module_builder
    def build():
        bf16_ty = BF16Type.get()
        i8_ty = IntegerType.get_signless(8)
        f32_ty = F32Type.get()

        packed_l3 = MemRefType.get([total_tiles, tile_bytes], i8_ty)
        RMS_l3 = MemRefType.get([2, K], bf16_ty)
        D_l3 = MemRefType.get([M // 2], bf16_ty)

        l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

        # Allocate the partial buffer as a 32-lane (CASCADE_WIDTH-style) buffer
        # and pass a subview-cast to the M_TILE-shaped kernel — matches the
        # matvec_int4_packed_add layout so air-shrink-memref-sizes-by-access
        # rewrites the kernel signature consistently across all stitched calls.
        CASCADE_WIDTH = 32
        packed_l2 = MemRefType.get([tile_bytes], i8_ty, memory_space=l2_ms)
        packed_l1 = MemRefType.get([tile_bytes], i8_ty, memory_space=l1_ms)
        rms_l1_ty = MemRefType.get([2, K], bf16_ty, memory_space=l1_ms)
        normed_l1_ty = MemRefType.get([K], bf16_ty, memory_space=l1_ms)
        partial_full_ty = MemRefType.get([CASCADE_WIDTH], bf16_ty, memory_space=l1_ms)
        partial_slice_ty = MemRefType.get([M_TILE], bf16_ty, memory_space=l1_ms)
        gate_l1_ty = MemRefType.get([half_M_per_core], bf16_ty, memory_space=l1_ms)
        up_l1_ty = MemRefType.get([half_M_per_core], bf16_ty, memory_space=l1_ms)
        D_l1_ty = MemRefType.get([half_M_per_core], bf16_ty, memory_space=l1_ms)

        channel_decl("inL3", size=[N_CORES])
        channel_decl("aL2ToL1", size=[N_CORES])
        Channel("inRMS", size=[1, 1], broadcast_shape=[N_CORES, 1])
        channel_decl("outD", size=[N_CORES])

        zero_func = FuncOp(
            "zero_vectorized_bf16",
            ([partial_slice_ty], []),
            visibility="private",
        )
        zero_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        matvec_func = FuncOp(
            "matvec_int4_bf16_packed",
            ([packed_l1, normed_l1_ty, partial_slice_ty], []),
            visibility="private",
        )
        matvec_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matvec_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(packed_l3, RMS_l3, D_l3)
        def matvec_int4_swiglu_rms(PACKED, RMS, D):
            @launch(sizes=[1, 1], operands=[PACKED, RMS, D])
            def launch_body(li, lj, lsx, lsy, packed, rms, d):
                # L3-side puts: per-core PACKED slab + per-core D get.
                for c in range(N_CORES):
                    c_idx = arith.ConstantOp.create_index(c)
                    c_tile_const = arith.ConstantOp.create_index(c * M_div)
                    ChannelPut(
                        "inL3",
                        packed,
                        indices=[c_idx],
                        offsets=[c_tile_const, 0],
                        sizes=[M_div, tile_bytes],
                        strides=[tile_bytes, 1],
                    )
                    c_d_off = arith.ConstantOp.create_index(c * half_M_per_core)
                    ChannelGet(
                        "outD",
                        d,
                        indices=[c_idx],
                        offsets=[c_d_off],
                        sizes=[half_M_per_core],
                        strides=[1],
                    )
                # RMS input broadcast: a single S2MM per compute tile, replayed
                # once per launch (RMS input does not change across outer iters).
                ChannelPut(
                    "inRMS",
                    rms,
                    offsets=[0, 0],
                    sizes=[2, K],
                    strides=[K, 1],
                )

                @segment(name="seg")
                def segment_body():
                    # Memtile staging: M_div iters of 1 packed tile each.
                    for c in range(N_CORES):
                        c_idx_s = arith.ConstantOp.create_index(c)
                        for _ in for_(M_div):
                            l2_op = AllocOp(packed_l2, [], [])
                            ChannelGet("inL3", l2_op, indices=[c_idx_s])
                            ChannelPut("aL2ToL1", l2_op, indices=[c_idx_s])
                            DeallocOp(l2_op)
                            yield_([])

                    @herd(name="matvec_h", sizes=[N_CORES, 1])
                    def herd_body(tx, ty, _sx, _sy):
                        c0 = arith.ConstantOp.create_index(0)

                        l1_rms_op = AllocOp(rms_l1_ty, [], [])
                        l1_normed_op = AllocOp(normed_l1_ty, [], [])
                        l1_gate_op = AllocOp(gate_l1_ty, [], [])
                        l1_up_op = AllocOp(up_l1_ty, [], [])
                        l1_d_op = AllocOp(D_l1_ty, [], [])
                        l1_rms = l1_rms_op.result
                        l1_normed = l1_normed_op.result
                        l1_gate = l1_gate_op.result
                        l1_up = l1_up_op.result
                        l1_d = l1_d_op.result

                        # Single-shot S2MM: RMS input via broadcast.
                        ChannelGet("inRMS", l1_rms, indices=[tx, ty])

                        # ---- Inline RMSNorm (faithful port of bf16 stage 2) ----
                        rms_vec_size = 16
                        rms_vecTy_bf16 = VectorType.get([rms_vec_size], bf16_ty)
                        rms_vecTy_f32 = VectorType.get([rms_vec_size], f32_ty)
                        rms_identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
                        read_map_2d_rms = AffineMapAttr.get(
                            AffineMap.get(2, 0, [AffineExpr.get_dim(1)])
                        )
                        rms_cst0_bf16 = arith.ConstantOp(bf16_ty, 0.0)
                        rms_cst0_f32 = arith.ConstantOp(f32_ty, 0.0)
                        rms_acc_ty = MemRefType.get(
                            shape=[rms_vec_size],
                            element_type=f32_ty,
                            memory_space=l1_ms,
                        )
                        rms_tmp_ty = MemRefType.get(
                            shape=[rms_vec_size],
                            element_type=bf16_ty,
                            memory_space=l1_ms,
                        )
                        rms_acc = AllocOp(rms_acc_ty, [], [])
                        rms_tmp = AllocOp(rms_tmp_ty, [], [])
                        zero_vec_f32 = BroadcastOp(rms_vecTy_f32, rms_cst0_f32)
                        transfer_write(
                            None,
                            zero_vec_f32,
                            rms_acc,
                            [c0],
                            rms_identity_map,
                            [True],
                        )
                        c_k = arith.ConstantOp.create_index(K)
                        c_rms_vec = arith.ConstantOp.create_index(rms_vec_size)
                        for j in range_(0, c_k, c_rms_vec):
                            sub_r = subview(l1_rms, [0, j], [1, rms_vec_size], [1, 1])
                            v_x = transfer_read(
                                rms_vecTy_bf16,
                                sub_r,
                                [c0, c0],
                                read_map_2d_rms,
                                rms_cst0_bf16,
                                [True],
                            )
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

                        v_final_f32 = transfer_read(
                            rms_vecTy_f32,
                            rms_acc,
                            [c0],
                            rms_identity_map,
                            rms_cst0_f32,
                            [True],
                        )
                        total_sum_f32 = vector_reduction(f32_ty, "add", v_final_f32)
                        k_f32_const = arith.ConstantOp(f32_ty, float(K))
                        eps_f32_const = arith.ConstantOp(f32_ty, 1.0e-5)
                        mean_f32 = arith.divf(total_sum_f32, k_f32_const)
                        mean_eps_f32 = arith.addf(mean_f32, eps_f32_const)
                        rstd_f32 = math_dialect.rsqrt(mean_eps_f32)
                        rstd_bf16 = arith.truncf(bf16_ty, rstd_f32)
                        v_rstd = BroadcastOp(rms_vecTy_bf16, rstd_bf16)

                        for j in range_(0, c_k, c_rms_vec):
                            sub_r = subview(l1_rms, [0, j], [1, rms_vec_size], [1, 1])
                            sub_w = subview(l1_rms, [1, j], [1, rms_vec_size], [1, 1])
                            sub_b = subview(l1_normed, [j], [rms_vec_size], [1])
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

                        # ---- Hot int4 GEMV loop: M_div outers ----
                        # Each iter produces M_TILE partials = (M_TILE/2) (gate, up)
                        # pairs. Deinterleave into separate gate/up scratches indexed
                        # at outer * (M_TILE/2).
                        pair_off_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(M_TILE // 2),
                                )
                            ],
                        )
                        for outer in for_(M_div):
                            l1_partial_op = AllocOp(partial_full_ty, [], [])
                            l1_partial_op.attributes["air.shrinkage"] = BoolAttr.get(
                                False
                            )
                            l1_partial_slice_strided = subview(
                                l1_partial_op.result, [0], [M_TILE], [1]
                            )
                            l1_partial_slice = memref_cast(
                                partial_slice_ty, l1_partial_slice_strided
                            )
                            l1_packed_op = AllocOp(packed_l1, [], [])
                            ChannelGet("aL2ToL1", l1_packed_op, indices=[tx])
                            CallOp(zero_func, [l1_partial_slice])
                            CallOp(
                                matvec_func,
                                [l1_packed_op, l1_normed, l1_partial_slice],
                            )
                            pair_off = affine_apply(pair_off_map, [outer])
                            for i in range(M_TILE // 2):
                                ci_g = arith.ConstantOp.create_index(2 * i)
                                ci_u = arith.ConstantOp.create_index(2 * i + 1)
                                v_g = memref_load(l1_partial_slice, [ci_g])
                                v_u = memref_load(l1_partial_slice, [ci_u])
                                pair_pos_map = AffineMap.get(
                                    0,
                                    1,
                                    [
                                        AffineExpr.get_add(
                                            AffineSymbolExpr.get(0),
                                            AffineConstantExpr.get(i),
                                        )
                                    ],
                                )
                                pair_pos = affine_apply(pair_pos_map, [pair_off])
                                memref_store(v_g, l1_gate, [pair_pos])
                                memref_store(v_u, l1_up, [pair_pos])
                            DeallocOp(l1_packed_op)
                            DeallocOp(l1_partial_op)
                            yield_([])

                        # ---- Vectorized SwiGLU(gate, up) → D ----
                        vecTyOut = VectorType.get([SILU_VEC], bf16_ty)
                        cst_half_bf16 = arith.ConstantOp(bf16_ty, 0.5)
                        cst_one_bf16 = arith.ConstantOp(bf16_ty, 1.0)
                        v_half_bf16 = BroadcastOp(vecTyOut, cst_half_bf16)
                        v_one_bf16 = BroadcastOp(vecTyOut, cst_one_bf16)
                        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
                        cst0_bf16_v = arith.ConstantOp(bf16_ty, 0.0)
                        c_half = arith.ConstantOp.create_index(half_M_per_core)
                        c_silu = arith.ConstantOp.create_index(SILU_VEC)
                        for kk in for_(0, c_half, c_silu):
                            sub_g = subview(l1_gate, [kk], [SILU_VEC], [1])
                            sub_u = subview(l1_up, [kk], [SILU_VEC], [1])
                            sub_out = subview(l1_d, [kk], [SILU_VEC], [1])
                            v_g = transfer_read(
                                vecTyOut,
                                sub_g,
                                [c0],
                                identity_map,
                                cst0_bf16_v,
                                [True],
                            )
                            v_u = transfer_read(
                                vecTyOut,
                                sub_u,
                                [c0],
                                identity_map,
                                cst0_bf16_v,
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

                        ChannelPut("outD", l1_d, indices=[tx])

                        DeallocOp(l1_rms_op)
                        DeallocOp(l1_normed_op)
                        DeallocOp(l1_gate_op)
                        DeallocOp(l1_up_op)
                        DeallocOp(l1_d_op)

                    herd_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
                    herd_body.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    herd_body.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

    return build()


def cpu_reference(A_q, A_s, A_z, RMS_in, eps=1e-5):
    """CPU bf16 reference: RMSNorm + int4 GEMV + SwiGLU pair. Output M/2."""
    M_ = A_q.shape[0]
    K_ = RMS_in.shape[1]
    n_groups = A_s.shape[0]
    gs = K_ // n_groups
    x = RMS_in[0].astype(np.float32)
    w = RMS_in[1].astype(np.float32)
    mean_sq = float((x * x).sum()) / K_
    rstd = 1.0 / np.sqrt(mean_sq + eps)
    normed = ((x * rstd) * w).astype(bfloat16).astype(np.float32)
    # Vectorized dequant: unpack nibbles, then broadcast scales/zeros per group.
    A_q_i = A_q.astype(np.int32)
    low = A_q_i & 0x0F
    high = (A_q_i >> 4) & 0x0F
    nibs = np.empty((M_, K_), dtype=np.int32)
    nibs[:, 0::2] = low
    nibs[:, 1::2] = high
    s_per_kk = np.repeat(A_s.astype(np.float32), gs, axis=0)  # (K_, M_)
    z_per_kk = np.repeat(A_z.astype(np.int32), gs, axis=0)  # (K_, M_)
    dequant = (nibs - z_per_kk.T) * s_per_kk.T  # (M_, K_)
    raw = dequant @ normed
    raw_bf16 = raw.astype(bfloat16).astype(np.float32)
    gate = raw_bf16[0::2]
    up = raw_bf16[1::2]
    silu = gate * 0.5 * (np.tanh(gate / 2.0) + 1.0)
    return (silu * up).astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="matvec_int4_swiglu_rms.py",
        description="int4-AWQ FFN GEMV with fused RMSNorm input and SwiGLU output.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--m-tile", type=int, default=8, dest="m_tile")
    parser.add_argument("--k-chunk", type=int, default=2048, dest="k_chunk")
    parser.add_argument("--n-cores", type=int, default=8, dest="n_cores")
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"], default="elf"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-only"],
        default="compile-and-run",
    )
    args = parser.parse_args()

    module = build_module(
        args.m,
        args.k,
        GS=args.gs,
        M_TILE=args.m_tile,
        K_CHUNK=args.k_chunk,
        N_CORES=args.n_cores,
    )
    if args.print_module_only:
        print(module)
        exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="matvec_int4_swiglu_rms",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        backend.compile(module)
        backend.unload()
        exit(0)

    np.random.seed(42)
    A_q_unp = np.random.randint(0, 16, size=(args.m, args.k), dtype=np.uint8)
    A_q = (A_q_unp[:, 0::2] | (A_q_unp[:, 1::2] << 4)).astype(np.uint8)
    n_groups = args.k // args.gs
    A_s = np.random.uniform(0.005, 0.02, size=(n_groups, args.m)).astype(bfloat16)
    A_z = np.random.randint(7, 9, size=(n_groups, args.m), dtype=np.uint8)
    input_vec = np.random.randn(args.k).astype(bfloat16)
    norm_weight = (np.random.randn(args.k) * 0.1 + 1.0).astype(bfloat16)
    RMS_in = np.stack([input_vec, norm_weight], axis=0).astype(bfloat16)

    D_ref = cpu_reference(A_q, A_s, A_z, RMS_in)
    PACKED = pack_inputs(
        A_q,
        A_s,
        A_z,
        args.m,
        args.k,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        args.m,
    )

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="matvec_int4_swiglu_rms",
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )
    exit(
        runner.run_test(
            module,
            inputs=[PACKED, RMS_in],
            expected_outputs=[D_ref],
            rtol=0.15,
            atol=0.5,
            min_correlation=0.99,
        )
    )
