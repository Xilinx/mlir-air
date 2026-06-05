# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""o_gemv_ffn_int4_fused — full-int4 1-launch ELF2 for the LLAMA decode block.

Single-launch alternative to o_gemv_ffn_int4_multi.py. Same post-attention
math (O proj + residual #1, RMSNorm, FFN gate/up + SwiGLU, FFN down +
residual #2), packed into ONE air.launch with three herds wired together
by two W->E cascade-shift chains, so res1 (LA->LGU/LD) and swiglu
(LGU->LD) flow in-array rather than round-tripping through L3.

  LA  (row 4 cols 0..7): matvec_int4 + partial_plus_r -> per-core
      M_LA/N_LA=256 bf16 res1 slab. W->E cascade-shift assembles full
      M_LA=2048 res1 (= W_O @ attn_out + x_residual). Eastmost LA
      broadcasts full res1 to BOTH the 8 LGU and 8 LD cores via
      res1ToCons packet broadcast (16 destinations).

  LGU (row 2 cols 0..7): receives res1 + gamma. RMS(res1, gamma) ->
      int4 matvec -> SwiGLU produces a 1024 bf16 swiglu slab per core.
      W->E cascade-shift assembles the full M_OUT=8192 bf16 swiglu.
      Eastmost LGU broadcasts the swiglu as K_LD_div=4 K_CHUNK chunks
      in FIFO sequence on one swigluToLd packet channel to all 8 LD
      cores.

  LD  (row 3 cols 0..7): receives 4 swiglu chunks + res1 (residual
      addend, from LA via res1ToCons). matvec wd @ swiglu (K_LD=8192
      reduction in K_CHUNK=2048 slices, 4 inner iters per output tile)
      + partial_plus_r adds res1 at the per-core slab offset. Per-col
      writes via ldOutD to L3.

ABI (8 args; arg0/arg3/arg5 are packed int4 BOs):

    arg0:  memref<n_la_tiles x tile_bytes xi8>  wo_packed         STATIC
    arg1:  memref<emb xbf16>                    attn_out          INPUT
    arg2:  memref<emb xbf16>                    x_residual        INPUT
    arg3:  memref<n_lgu_tiles x tile_bytes xi8> gate/up_packed    STATIC
    arg4:  memref<emb xbf16>                    ffn_norm_w        STATIC
    arg5:  memref<n_ld_tiles x ld_tile_bytes xi8> wdown_packed    STATIC
    arg6:  memref<emb xbf16>                    output            OUTPUT
    arg7:  memref<emb xbf16>                    res1 debug copy   OUTPUT

K_CHUNK is fixed at 2048 (=K=emb) → all three stages link the same
mv_int4_bf16.o (DIM_K=2048, DIM_M=8). LD's K=hidden=8192 splits into
K_LD_div=4 inner iters; LA and LGU do a single K chunk each.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineSymbolExpr,
    BF16Type,
    BoolAttr,
    F32Type,
    InsertionPoint,
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
from air.dialects import linalg
from air.dialects import math as math_dialect
from air.dialects import scf
from air.dialects.scf import for_, yield_
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "matrix_vector_multiplication",
        "int4_awq",
    ),
)
from matvec_int4_packed import pack_inputs
from matvec_int4_packed_add import cpu_reference as la_cpu_reference

KERNEL_OBJ_NAME = "mv_int4_bf16.o"

range_ = for_


def lgu_cpu_reference(A_q, A_s, A_z, res1, gamma, eps=1e-5):
    K_ = res1.shape[0]
    n_groups = A_s.shape[0]
    gs = K_ // n_groups
    x = res1.astype(np.float32)
    w = gamma.astype(np.float32)
    mean_sq = float((x * x).sum()) / K_
    rstd = 1.0 / np.sqrt(mean_sq + eps)
    normed = ((x * rstd) * w).astype(bfloat16).astype(np.float32)
    M_ = A_q.shape[0]
    A_q_i = A_q.astype(np.int32)
    low = A_q_i & 0x0F
    high = (A_q_i >> 4) & 0x0F
    nibs = np.empty((M_, K_), dtype=np.int32)
    nibs[:, 0::2] = low
    nibs[:, 1::2] = high
    s_per_kk = np.repeat(A_s.astype(np.float32), gs, axis=0)
    z_per_kk = np.repeat(A_z.astype(np.int32), gs, axis=0)
    dequant = (nibs - z_per_kk.T) * s_per_kk.T
    raw = dequant @ normed
    raw_bf16 = raw.astype(bfloat16).astype(np.float32)
    gate = raw_bf16[0::2]
    up = raw_bf16[1::2]
    silu = gate * 0.5 * (np.tanh(gate / 2.0) + 1.0)
    return (silu * up).astype(bfloat16)


def build_o_gemv_ffn_int4_fused_module(
    emb_dim=2048, hidden_dim=8192, gs=128, m_tile=8, k_chunk=2048, n_cores=8
):
    """Public API matching o_gemv_ffn_int4_multi.build_o_gemv_ffn_int4_module."""
    return build_module(
        K=emb_dim,
        M_LA=emb_dim,
        M_LGU=2 * hidden_dim,
        K_LD=hidden_dim,
        GS=gs,
        M_TILE=m_tile,
        K_CHUNK=k_chunk,
        N_LA=n_cores,
        N_LGU=n_cores,
        N_LD=n_cores,
    )


def build_module(
    K=2048,
    M_LA=2048,
    M_LGU=16384,
    K_LD=8192,
    GS=128,
    M_TILE=8,
    K_CHUNK=2048,
    N_LA=8,
    N_LGU=8,
    N_LD=8,
    skip_inline=False,
):
    assert K == K_CHUNK
    assert K_CHUNK % GS == 0
    assert M_LGU % 2 == 0
    assert M_LGU % N_LGU == 0
    assert K_LD % K_CHUNK == 0
    assert M_LA % N_LA == 0
    assert (M_LA // N_LA) % M_TILE == 0
    assert M_LA % N_LD == 0
    assert (M_LA // N_LD) % M_TILE == 0
    # Pair two outer iters per add so the bf16 vector add is 16-wide
    # (8-wide bf16 doesn't legalize on AIE2P).
    assert (M_LA // N_LA) % (2 * M_TILE) == 0
    assert (M_LA // N_LD) % (2 * M_TILE) == 0
    # Sanity: swiglu width must match LD's K reduction.
    assert M_LGU // 2 == K_LD
    # LA and LGU use W->E cascade chains (size=N-1) and the eastmost
    # core broadcasts to consumers; a 1-core herd has no cascade hop
    # and no eastmost-vs-rest split, so the broadcast paths would
    # never fire. LD has no cascade so N_LD >= 1 is fine.
    assert N_LA >= 2, "LA cascade requires N_LA >= 2"
    assert N_LGU >= 2, "LGU cascade requires N_LGU >= 2"

    M_la_per_core = M_LA // N_LA  # 256
    M_la_div = M_la_per_core // M_TILE  # 32 outer iters per LA core
    la_tiles_per_col = M_la_div  # 32

    M_lgu_per_core = M_LGU // N_LGU  # 2048
    M_lgu_div = M_lgu_per_core // M_TILE  # 256
    SILU_VEC = 32
    half_M_per_core = M_lgu_per_core // 2  # 1024
    M_OUT = M_LGU // 2  # 8192 (assembled swiglu)
    assert half_M_per_core % SILU_VEC == 0

    K_LD_div = K_LD // K_CHUNK  # 4
    M_ld_per_core = M_LA // N_LD  # 256
    M_ld_div = M_ld_per_core // M_TILE  # 32
    ld_tiles_per_col = M_ld_div * K_LD_div  # 128

    n_gpc = K_CHUNK // GS
    q_bytes = M_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * M_TILE * 2
    z_bytes = n_gpc * M_TILE
    tile_bytes = q_bytes + s_bytes + z_bytes
    total_lgu_tiles = N_LGU * M_lgu_div

    @module_builder
    def build():
        bf16_ty = BF16Type.get()
        i8_ty = IntegerType.get_signless(8)
        i32_ty = T.i32()
        f32_ty = F32Type.get()
        l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

        # ---- L3 ----
        # LA inputs
        packed_la_l3 = MemRefType.get([N_LA * la_tiles_per_col, tile_bytes], i8_ty)
        B_la_l3 = MemRefType.get([K], bf16_ty)
        R_la_l3 = MemRefType.get([M_LA], bf16_ty)
        # LGU inputs
        packed_lgu_l3 = MemRefType.get([total_lgu_tiles, tile_bytes], i8_ty)
        gamma_l3 = MemRefType.get([K], bf16_ty)
        # LD inputs / outputs
        packed_ld_l3 = MemRefType.get([N_LD * ld_tiles_per_col, tile_bytes], i8_ty)
        D_ld_l3 = MemRefType.get([M_LA], bf16_ty)
        # DEBUG: capture LA's assembled res1 to L3 so we can verify it
        # matches the CPU reference for LA. Same shape as R/res1.
        D_dbg_l3 = MemRefType.get([M_LA], bf16_ty)

        # ---- LGU L1 / L2 ----
        packed_lgu_l2 = MemRefType.get([tile_bytes], i8_ty, memory_space=l2_ms)
        packed_lgu_l1 = MemRefType.get([tile_bytes], i8_ty, memory_space=l1_ms)
        res1_l1 = MemRefType.get([K], bf16_ty, memory_space=l1_ms)
        gamma_l1 = MemRefType.get([K], bf16_ty, memory_space=l1_ms)
        normed_l1 = MemRefType.get([K], bf16_ty, memory_space=l1_ms)
        gate_l1_ty = MemRefType.get([half_M_per_core], bf16_ty, memory_space=l1_ms)
        up_l1_ty = MemRefType.get([half_M_per_core], bf16_ty, memory_space=l1_ms)
        # Per-core SwiGLU output (2 KB). Copied into l1_recv at the
        # cascade hop -- avoids a second 16 KB cascade buffer.
        swiglu_out_l1_ty = MemRefType.get(
            [half_M_per_core], bf16_ty, memory_space=l1_ms
        )
        # Full assembled swiglu L1 scratch: 8192 bf16 = 16 KB.
        full_l1 = MemRefType.get([M_OUT], bf16_ty, memory_space=l1_ms)

        CASCADE_WIDTH = 32
        partial_full_ty = MemRefType.get([CASCADE_WIDTH], bf16_ty, memory_space=l1_ms)
        partial_slice_ty = MemRefType.get([M_TILE], bf16_ty, memory_space=l1_ms)
        D_la_l1 = MemRefType.get([M_TILE], bf16_ty, memory_space=l1_ms)

        # ---- LA L1 ----
        packed_la_l1 = MemRefType.get([tile_bytes], i8_ty, memory_space=l1_ms)
        B_la_l1 = MemRefType.get([K], bf16_ty, memory_space=l1_ms)
        # R is full M_LA; inline partial+r subviews it at the per-core
        # slab offset (shared shape with LD).
        R_la_l1 = MemRefType.get([M_LA], bf16_ty, memory_space=l1_ms)
        full_la_l1 = MemRefType.get([M_LA], bf16_ty, memory_space=l1_ms)

        # ---- LD L1 / L2 ----
        packed_ld_l2 = MemRefType.get([tile_bytes], i8_ty, memory_space=l2_ms)
        packed_ld_l1 = MemRefType.get([tile_bytes], i8_ty, memory_space=l1_ms)
        # Full assembled swiglu in L1: K_LD bf16 = 16 KB. Inner k loop
        # uses matvec_*_b_offset to skip into the right chunk.
        swiglu_full_l1 = MemRefType.get([K_LD], bf16_ty, memory_space=l1_ms)
        # Residual addend on LD side: full M_LA, indexed via offset.
        R_ld_l1 = MemRefType.get([M_LA], bf16_ty, memory_space=l1_ms)
        slab_ld_l1 = MemRefType.get([M_ld_per_core], bf16_ty, memory_space=l1_ms)

        # ---- Channels ----
        # LA per-col coalesced inputs: B + R-slab + multi-dim PACKED
        # all on ONE packet channel per col.
        for c in range(N_LA):
            channel_decl(f"laAll_{c}", channel_type="npu_dma_packet")
        # W->E cascade chain across LA row (row 4). N_LA-1 edges.
        channel_decl("chan_cascade_la", size=[N_LA - 1], channel_type="npu_cascade")

        # LGU: packet broadcast for gamma (shares LGU S2MM:0). Per-col
        # packed via memtile, default circuit for non-shared flows.
        channel_decl("lguPACKED", size=[N_LGU])
        channel_decl("lguL2ToL1", size=[N_LGU])
        # Eastmost LA L1 -> all LGU + LD cores: 16-dest packet broadcast.
        N_RES1_CONS = N_LGU + N_LD
        res1_ch = Channel("res1ToCons", size=[1, 1], broadcast_shape=[N_RES1_CONS, 1])
        res1_ch.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
        # DEBUG: also emit res1 to L3 so we can inspect what LA produces.
        channel_decl("laResDebug", size=[1])
        gamma_ch = Channel("lguGAMMA", size=[1, 1], broadcast_shape=[N_LGU, 1])
        gamma_ch.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
        channel_decl("chan_cascade_lgu", size=[N_LGU - 1], channel_type="npu_cascade")

        # Eastmost LGU L1 -> all LD cores: ONE packet broadcast carrying
        # K_LD_div K_CHUNK-bf16 chunks in FIFO sequence. Collapsing the
        # 4 chunks onto a single channel keeps the LD stream switch
        # arbiter under the 4-msel multicast limit (4 separate broadcast
        # channels + ldR1 would exceed it).
        sw_ch = Channel("swigluToLd", size=[1, 1], broadcast_shape=[N_LD, 1])
        sw_ch.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")

        # LD: per-col packed via memtile (packet for shim sharing).
        channel_decl("ldPACKED", size=[N_LD], channel_type="npu_dma_packet")
        channel_decl("ldL2ToL1", size=[N_LD])
        # LD output per-col to L3.
        channel_decl("ldOutD", size=[N_LD])

        # ---- Private kernel decls ----
        matvec_func = FuncOp(
            "matvec_int4_bf16_packed",
            ([packed_lgu_l1, normed_l1, partial_slice_ty], []),
            visibility="private",
        )
        matvec_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matvec_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        matvec_store_func = FuncOp(
            "matvec_int4_bf16_packed_store",
            ([packed_lgu_l1, normed_l1, partial_slice_ty], []),
            visibility="private",
        )
        matvec_store_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matvec_store_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # LD path: matvec that takes a b-offset (elements) so we can keep
        # one big swiglu buffer and iterate k via scf.for + ping-pong.
        matvec_offset_func = FuncOp(
            "matvec_int4_bf16_packed_b_offset",
            ([packed_ld_l1, swiglu_full_l1, i32_ty, partial_slice_ty], []),
            visibility="private",
        )
        matvec_offset_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matvec_offset_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(
            packed_la_l3,
            B_la_l3,
            R_la_l3,
            packed_lgu_l3,
            gamma_l3,
            packed_ld_l3,
            D_ld_l3,
            D_dbg_l3,
        )  # DEBUG: capture LA's res1 output
        def o_gemv_ffn_int4_fused(P_la, B_la, R_la, P_lgu, G_l3, P_ld, D_ld, D_dbg):
            @launch(operands=[P_la, B_la, R_la, P_lgu, G_l3, P_ld, D_ld, D_dbg])
            def launch_body(p_la, b_la, r_la, p_lgu, g_l3, p_ld, d_ld, d_dbg):
                c0 = arith.ConstantOp.create_index(0)
                c1 = arith.ConstantOp.create_index(1)
                cK = arith.ConstantOp.create_index(K)
                cMLA = arith.ConstantOp.create_index(M_LA)
                c_tb = arith.ConstantOp.create_index(tile_bytes)

                # LA per-col puts (B + full R + multi-dim PACKED) on the
                # same per-col packet channel. Order matches herd-side gets.
                for c in range(N_LA):
                    ChannelPut(
                        f"laAll_{c}", b_la, offsets=[c0], sizes=[cK], strides=[c1]
                    )
                    # Push FULL R so each LA core subviews its own per-core
                    # slab offset (shared shape with LD).
                    ChannelPut(
                        f"laAll_{c}", r_la, offsets=[c0], sizes=[cMLA], strides=[c1]
                    )
                    c_packed_off = arith.ConstantOp.create_index(c * la_tiles_per_col)
                    c_tpc = arith.ConstantOp.create_index(la_tiles_per_col)
                    ChannelPut(
                        f"laAll_{c}",
                        p_la,
                        offsets=[c_packed_off, c0],
                        sizes=[c_tpc, c_tb],
                        strides=[c_tb, c1],
                    )

                # LGU per-col packed + gamma broadcast. res1 is supplied
                # by LA via res1ToCons; no L3 res1 put here.
                for c in range(N_LGU):
                    c_idx = arith.ConstantOp.create_index(c)
                    c_tile_off = arith.ConstantOp.create_index(c * M_lgu_div)
                    ChannelPut(
                        "lguPACKED",
                        p_lgu,
                        indices=[c_idx],
                        offsets=[c_tile_off, 0],
                        sizes=[M_lgu_div, tile_bytes],
                        strides=[tile_bytes, 1],
                    )
                ChannelPut("lguGAMMA", g_l3, offsets=[0], sizes=[K], strides=[1])

                # DEBUG: catch LA's broadcast output to L3.
                ChannelGet("laResDebug", d_dbg)

                # LD per-col packed + per-col output. Residual addend is
                # res1 (from LA via res1ToCons).
                for c in range(N_LD):
                    c_idx = arith.ConstantOp.create_index(c)
                    c_ld_tile_off = arith.ConstantOp.create_index(c * ld_tiles_per_col)
                    ChannelPut(
                        "ldPACKED",
                        p_ld,
                        indices=[c_idx],
                        offsets=[c_ld_tile_off, 0],
                        sizes=[ld_tiles_per_col, tile_bytes],
                        strides=[tile_bytes, 1],
                    )
                    c_ld_d_off = arith.ConstantOp.create_index(c * M_ld_per_core)
                    c_mldpc = arith.ConstantOp.create_index(M_ld_per_core)
                    c_one = arith.ConstantOp.create_index(1)
                    ChannelGet(
                        "ldOutD",
                        d_ld,
                        indices=[c_idx],
                        offsets=[c_ld_d_off],
                        sizes=[c_mldpc],
                        strides=[c_one],
                    )

                @segment(name="seg")
                def segment_body():
                    # LGU L2 staging.
                    for c in range(N_LGU):
                        c_idx_s = arith.ConstantOp.create_index(c)
                        for _ in for_(M_lgu_div):
                            l2_op = AllocOp(packed_lgu_l2, [], [])
                            ChannelGet("lguPACKED", l2_op, indices=[c_idx_s])
                            ChannelPut("lguL2ToL1", l2_op, indices=[c_idx_s])
                            DeallocOp(l2_op)
                            yield_([])

                    # LD L2 staging.
                    for c in range(N_LD):
                        c_idx_s = arith.ConstantOp.create_index(c)
                        for _ in for_(ld_tiles_per_col):
                            l2_op = AllocOp(packed_ld_l2, [], [])
                            ChannelGet("ldPACKED", l2_op, indices=[c_idx_s])
                            ChannelPut("ldL2ToL1", l2_op, indices=[c_idx_s])
                            DeallocOp(l2_op)
                            yield_([])

                    # ---- LA herds (N_LA cores at row 4, per-col) ----
                    # Each LA core c does matvec_int4 + partial_plus_r
                    # producing its M_la_per_core slab into l1_local at
                    # offset col*M_la_per_core, then participates in the
                    # W->E chan_cascade_la chain. Eastmost LA (col N_LA-1)
                    # broadcasts the assembled res1 to LGU+LD cores.
                    for c in range(N_LA):

                        def make_la_herd(col):
                            is_first = col == 0
                            is_last = col == N_LA - 1
                            col_base_const = col * M_la_per_core

                            @herd(name=f"la_{col}", sizes=[1, 1])
                            def la_body(tx, ty, _sx, _sy):
                                c0 = arith.ConstantOp.create_index(0)
                                vec_w_la = CASCADE_WIDTH
                                vecTy_la = VectorType.get([vec_w_la], bf16_ty)
                                vecTy_mt_la = VectorType.get([M_TILE], bf16_ty)
                                cst0_bf16_la = arith.ConstantOp(bf16_ty, 0.0)
                                id_map_la = AffineMapAttr.get(AffineMap.get_identity(1))
                                c_mla_idx = arith.ConstantOp.create_index(M_LA)
                                c_vec_idx_la = arith.ConstantOp.create_index(vec_w_la)
                                zero_vec_la = BroadcastOp(vecTy_la, cst0_bf16_la)

                                l1_b = AllocOp(B_la_l1, [], [])
                                l1_r = AllocOp(R_la_l1, [], [])
                                l1_local_la = AllocOp(full_la_l1, [], [])
                                if not skip_inline:
                                    # Vectorized zero-fill of full-M_LA scratch.
                                    for j in for_(c0, c_mla_idx, c_vec_idx_la):
                                        sub_l = subview(
                                            l1_local_la.result, [j], [vec_w_la], [1]
                                        )
                                        transfer_write(
                                            None,
                                            zero_vec_la.result,
                                            sub_l,
                                            [c0],
                                            id_map_la,
                                            [True],
                                        )
                                        yield_([])
                                # Get B, R, then PACKED tiles in matvec loop
                                ChannelGet(f"laAll_{col}", l1_b)
                                ChannelGet(f"laAll_{col}", l1_r)

                                c_mt_la = arith.ConstantOp.create_index(M_TILE)
                                col_base_la = arith.ConstantOp.create_index(
                                    col_base_const
                                )
                                for outer in for_(M_la_div):
                                    l1_p = AllocOp(packed_la_l1, [], [])
                                    ChannelGet(f"laAll_{col}", l1_p)
                                    l1_partial_full_la = AllocOp(
                                        partial_full_ty, [], []
                                    )
                                    l1_partial_full_la.attributes["air.shrinkage"] = (
                                        BoolAttr.get(False)
                                    )
                                    l1_partial_strided_la = subview(
                                        l1_partial_full_la.result, [0], [M_TILE], [1]
                                    )
                                    l1_partial_la = memref_cast(
                                        D_la_l1, l1_partial_strided_la
                                    )
                                    l1_d_full_la = AllocOp(partial_full_ty, [], [])
                                    l1_d_full_la.attributes["air.shrinkage"] = (
                                        BoolAttr.get(False)
                                    )
                                    l1_d_strided_la = subview(
                                        l1_d_full_la.result, [0], [M_TILE], [1]
                                    )
                                    l1_d_la = memref_cast(D_la_l1, l1_d_strided_la)

                                    local_off = arith.muli(outer, c_mt_la)
                                    global_off = arith.addi(col_base_la, local_off)
                                    CallOp(
                                        matvec_store_func, [l1_p, l1_b, l1_partial_la]
                                    )
                                    # Inline partial+r: one 8-wide bf16
                                    # vector add (patched aievec pads to
                                    # 16-wide internally).
                                    sub_r_la = subview(
                                        l1_r.result, [global_off], [M_TILE], [1]
                                    )
                                    v_p_la = transfer_read(
                                        vecTy_mt_la,
                                        l1_partial_la,
                                        [c0],
                                        id_map_la,
                                        cst0_bf16_la,
                                        [True],
                                    )
                                    v_r_la = transfer_read(
                                        vecTy_mt_la,
                                        sub_r_la,
                                        [c0],
                                        id_map_la,
                                        cst0_bf16_la,
                                        [True],
                                    )
                                    v_sum_la = arith.addf(v_p_la, v_r_la)
                                    transfer_write(
                                        None,
                                        v_sum_la,
                                        l1_d_la,
                                        [c0],
                                        id_map_la,
                                        [True],
                                    )
                                    if not skip_inline:
                                        # Scatter M_TILE outputs into l1_local
                                        # at offset global_off.
                                        for i in range(M_TILE):
                                            ci = arith.ConstantOp.create_index(i)
                                            abs_idx = arith.addi(global_off, ci)
                                            v = memref_load(l1_d_la, [ci])
                                            memref_store(v, l1_local_la, [abs_idx])
                                    DeallocOp(l1_p)
                                    DeallocOp(l1_partial_full_la)
                                    DeallocOp(l1_d_full_la)
                                    yield_([])

                                # ---- W->E cascade pad-and-reduce ----
                                if is_first:
                                    edge_idx = arith.ConstantOp.create_index(col)
                                    ChannelPut(
                                        "chan_cascade_la",
                                        l1_local_la,
                                        indices=[edge_idx],
                                    )
                                else:
                                    l1_recv_la = AllocOp(full_la_l1, [], [])
                                    prev_edge_idx = arith.ConstantOp.create_index(
                                        col - 1
                                    )
                                    ChannelGet(
                                        "chan_cascade_la",
                                        l1_recv_la,
                                        indices=[prev_edge_idx],
                                    )
                                    if not skip_inline:
                                        # Vectorized add at CASCADE_WIDTH lanes.
                                        for j in for_(c0, c_mla_idx, c_vec_idx_la):
                                            sub_r = subview(
                                                l1_recv_la.result, [j], [vec_w_la], [1]
                                            )
                                            sub_l = subview(
                                                l1_local_la.result, [j], [vec_w_la], [1]
                                            )
                                            v_r = transfer_read(
                                                vecTy_la,
                                                sub_r,
                                                [c0],
                                                id_map_la,
                                                cst0_bf16_la,
                                                [True],
                                            )
                                            v_l = transfer_read(
                                                vecTy_la,
                                                sub_l,
                                                [c0],
                                                id_map_la,
                                                cst0_bf16_la,
                                                [True],
                                            )
                                            v_sum = arith.addf(v_r, v_l)
                                            transfer_write(
                                                None,
                                                v_sum,
                                                sub_l,
                                                [c0],
                                                id_map_la,
                                                [True],
                                            )
                                            yield_([])
                                    DeallocOp(l1_recv_la)
                                    if is_last:
                                        ChannelPut("res1ToCons", l1_local_la)
                                        # DEBUG: also send to L3.
                                        ChannelPut("laResDebug", l1_local_la)
                                    else:
                                        edge_idx = arith.ConstantOp.create_index(col)
                                        ChannelPut(
                                            "chan_cascade_la",
                                            l1_local_la,
                                            indices=[edge_idx],
                                        )

                                DeallocOp(l1_b)
                                DeallocOp(l1_r)
                                DeallocOp(l1_local_la)

                            la_body.attributes["link_with"] = StringAttr.get(
                                KERNEL_OBJ_NAME
                            )
                            la_body.attributes["x_loc"] = IntegerAttr.get(T.i64(), col)
                            la_body.attributes["y_loc"] = IntegerAttr.get(T.i64(), 4)

                        make_la_herd(c)

                    @herd(name="lgu_h", sizes=[N_LGU, 1])
                    def lgu_body(tx, ty, _sx, _sy):
                        c0 = arith.ConstantOp.create_index(0)
                        vec_w = CASCADE_WIDTH
                        vecTy = VectorType.get([vec_w], bf16_ty)
                        cst0_bf16 = arith.ConstantOp(bf16_ty, 0.0)
                        id_map = AffineMapAttr.get(AffineMap.get_identity(1))
                        c_m_out_idx = arith.ConstantOp.create_index(M_OUT)
                        c_vec_idx = arith.ConstantOp.create_index(vec_w)
                        zero_vec = BroadcastOp(vecTy, cst0_bf16)

                        l1_gamma = AllocOp(gamma_l1, [], [])
                        l1_res1 = AllocOp(res1_l1, [], [])
                        # RMS writes back in place into res1 (no separate
                        # normed buffer) -- saves 4 KB L1.
                        l1_normed = l1_res1
                        l1_gate = AllocOp(gate_l1_ty, [], [])
                        l1_up = AllocOp(up_l1_ty, [], [])
                        # Per-core SwiGLU output (2 KB). SwiGLU writes here;
                        # the cascade step copies it into l1_recv at the
                        # right offset. Eliminates the 16 KB l1_local
                        # scratch we used to have (-14 KB net, restores
                        # L1 budget for ping-pong on packed).
                        l1_swiglu_out = AllocOp(swiglu_out_l1_ty, [], [])

                        # Faster-arriving broadcast (gamma) first.
                        ChannelGet("lguGAMMA", l1_gamma, indices=[tx, ty])
                        # LGU dests are first N_LGU indices of res1ToCons.
                        ChannelGet("res1ToCons", l1_res1, indices=[tx, ty])

                        # ---- Inline RMSNorm (mirrors la_lgu_cascade_fused) ----
                        rms_vec_size = 16
                        rms_vecTy_bf16 = VectorType.get([rms_vec_size], bf16_ty)
                        rms_vecTy_f32 = VectorType.get([rms_vec_size], f32_ty)
                        rms_cst0_f32 = arith.ConstantOp(f32_ty, 0.0)
                        rms_acc_ty = MemRefType.get(
                            shape=[rms_vec_size],
                            element_type=f32_ty,
                            memory_space=l1_ms,
                        )
                        rms_acc = AllocOp(rms_acc_ty, [], [])
                        if not skip_inline:
                            zero_vec_f32 = BroadcastOp(rms_vecTy_f32, rms_cst0_f32)
                            transfer_write(
                                None, zero_vec_f32, rms_acc, [c0], id_map, [True]
                            )
                            c_k = arith.ConstantOp.create_index(K)
                            c_rms_vec = arith.ConstantOp.create_index(rms_vec_size)
                            for j in range_(0, c_k, c_rms_vec):
                                sub_r = subview(
                                    l1_res1.result, [j], [rms_vec_size], [1]
                                )
                                v_x = transfer_read(
                                    rms_vecTy_bf16,
                                    sub_r,
                                    [c0],
                                    id_map,
                                    cst0_bf16,
                                    [True],
                                )
                                v_sq_bf16 = arith.mulf(v_x, v_x)
                                v_sq_f32 = arith.extf(rms_vecTy_f32, v_sq_bf16)
                                v_acc = transfer_read(
                                    rms_vecTy_f32,
                                    rms_acc,
                                    [c0],
                                    id_map,
                                    rms_cst0_f32,
                                    [True],
                                )
                                v_sum = arith.addf(v_acc, v_sq_f32)
                                transfer_write(
                                    None, v_sum, rms_acc, [c0], id_map, [True]
                                )
                                yield_([])

                            v_final_f32 = transfer_read(
                                rms_vecTy_f32,
                                rms_acc,
                                [c0],
                                id_map,
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
                                sub_r = subview(
                                    l1_res1.result, [j], [rms_vec_size], [1]
                                )
                                sub_w = subview(
                                    l1_gamma.result, [j], [rms_vec_size], [1]
                                )
                                v_r = transfer_read(
                                    rms_vecTy_bf16,
                                    sub_r,
                                    [c0],
                                    id_map,
                                    cst0_bf16,
                                    [True],
                                )
                                v_w = transfer_read(
                                    rms_vecTy_bf16,
                                    sub_w,
                                    [c0],
                                    id_map,
                                    cst0_bf16,
                                    [True],
                                )
                                v_n = arith.mulf(v_r, v_rstd.result)
                                v_y = arith.mulf(v_n, v_w)
                                # Write back into res1 in place (l1_normed === l1_res1).
                                transfer_write(None, v_y, sub_r, [c0], id_map, [True])
                                yield_([])

                        DeallocOp(rms_acc)
                        # Gamma not needed after RMS -- free its 4 KB.
                        DeallocOp(l1_gamma)

                        # ---- Hot int4 GEMV loop (ORIGINAL structure with
                        #      ping-pong on packed_l1 via per-iter alloc) ----
                        l1_partial_op = AllocOp(partial_full_ty, [], [])
                        l1_partial_op.attributes["air.shrinkage"] = BoolAttr.get(False)
                        l1_partial_strided = subview(
                            l1_partial_op.result, [0], [M_TILE], [1]
                        )
                        l1_partial_slice = memref_cast(
                            partial_slice_ty, l1_partial_strided
                        )
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
                        for outer in for_(M_lgu_div):
                            l1_packed_op = AllocOp(packed_lgu_l1, [], [])
                            ChannelGet("lguL2ToL1", l1_packed_op, indices=[tx])
                            CallOp(
                                matvec_store_func,
                                [l1_packed_op, l1_normed, l1_partial_slice],
                            )
                            if not skip_inline:
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
                            yield_([])
                        DeallocOp(l1_partial_op)

                        # ---- Vectorized SwiGLU(gate, up) -> l1_swiglu_out
                        #      (per-core 2 KB output; copied into l1_recv
                        #      during the cascade step). ----
                        vecTyOut = VectorType.get([SILU_VEC], bf16_ty)
                        cst_half_bf16 = arith.ConstantOp(bf16_ty, 0.5)
                        cst_one_bf16 = arith.ConstantOp(bf16_ty, 1.0)
                        v_half_bf16 = BroadcastOp(vecTyOut, cst_half_bf16)
                        v_one_bf16 = BroadcastOp(vecTyOut, cst_one_bf16)
                        c_half_idx = arith.ConstantOp.create_index(half_M_per_core)
                        c_silu_idx = arith.ConstantOp.create_index(SILU_VEC)
                        if not skip_inline:
                            for kk in for_(c0, c_half_idx, c_silu_idx):
                                sub_g = subview(l1_gate.result, [kk], [SILU_VEC], [1])
                                sub_u = subview(l1_up.result, [kk], [SILU_VEC], [1])
                                sub_out = subview(
                                    l1_swiglu_out.result, [kk], [SILU_VEC], [1]
                                )
                                v_g = transfer_read(
                                    vecTyOut, sub_g, [c0], id_map, cst0_bf16, [True]
                                )
                                v_u = transfer_read(
                                    vecTyOut, sub_u, [c0], id_map, cst0_bf16, [True]
                                )
                                v_half_g = arith.mulf(v_g, v_half_bf16.result)
                                v_tanh = math_dialect.tanh(v_half_g)
                                v_tanh_p1 = arith.addf(v_tanh, v_one_bf16.result)
                                v_sig = arith.mulf(v_tanh_p1, v_half_bf16.result)
                                v_silu = arith.mulf(v_g, v_sig)
                                v_out = arith.mulf(v_silu, v_u)
                                transfer_write(
                                    None, v_out, sub_out, [c0], id_map, [True]
                                )
                                yield_([])

                        # ---- W->E cascade pad-and-copy ----
                        # The single 16 KB cascade scratch is l1_recv. For
                        # col 0 it starts zero-filled (we just write our
                        # slab in); for other cols it gets the assembled
                        # value from the previous cascade hop and we copy
                        # our slab in at col*half_M_per_core (overwriting
                        # the zero that's there because earlier cores
                        # don't write to our slot).
                        l1_recv = AllocOp(full_l1, [], [])
                        c1_idx = arith.ConstantOp.create_index(1)
                        c0_tx = arith.ConstantOp.create_index(0)
                        last_tx = arith.ConstantOp.create_index(N_LGU - 1)
                        # Per-core base offset into the assembled vector
                        # (also the SwiGLU's "global" position).
                        c_hmpc_idx = arith.ConstantOp.create_index(half_M_per_core)
                        col_base = arith.muli(tx, c_hmpc_idx)

                        cmp_first = arith.CmpIOp(arith.CmpIPredicate.eq, tx, c0_tx)
                        if_first = scf.IfOp(cmp_first, has_else=True)
                        with InsertionPoint(if_first.then_block):
                            # Col 0: zero-fill, then write own slab in.
                            for j in for_(c0, c_m_out_idx, c_vec_idx):
                                sub_r = subview(l1_recv.result, [j], [vec_w], [1])
                                transfer_write(
                                    None, zero_vec.result, sub_r, [c0], id_map, [True]
                                )
                                yield_([])
                            yield_([])
                        with InsertionPoint(if_first.else_block):
                            # Receive assembled-so-far from prev cascade.
                            prev_tx = arith.SubIOp(tx, c1_idx)
                            ChannelGet("chan_cascade_lgu", l1_recv, indices=[prev_tx])
                            yield_([])
                        if not skip_inline:
                            # Both branches: copy our 2 KB swiglu slab into
                            # l1_recv at offset col_base. Vectorized at SILU_VEC.
                            for kk in for_(c0, c_hmpc_idx, c_silu_idx):
                                sub_s = subview(
                                    l1_swiglu_out.result, [kk], [SILU_VEC], [1]
                                )
                                global_idx = arith.addi(col_base, kk)
                                sub_r = subview(
                                    l1_recv.result, [global_idx], [SILU_VEC], [1]
                                )
                                v_s = transfer_read(
                                    vecTyOut, sub_s, [c0], id_map, cst0_bf16, [True]
                                )
                                transfer_write(None, v_s, sub_r, [c0], id_map, [True])
                                yield_([])

                        # Forward or broadcast.
                        cmp_last = arith.CmpIOp(arith.CmpIPredicate.eq, tx, last_tx)
                        if_last = scf.IfOp(cmp_last, has_else=True)
                        with InsertionPoint(if_last.then_block):
                            # Eastmost LGU: broadcast K_LD_div K_CHUNK
                            # chunks on one packet channel (FIFO).
                            for k_chunk in range(K_LD_div):
                                ck_off = arith.ConstantOp.create_index(
                                    k_chunk * K_CHUNK
                                )
                                ck_n = arith.ConstantOp.create_index(K_CHUNK)
                                ck_one = arith.ConstantOp.create_index(1)
                                ChannelPut(
                                    "swigluToLd",
                                    l1_recv,
                                    offsets=[ck_off],
                                    sizes=[ck_n],
                                    strides=[ck_one],
                                )
                            yield_([])
                        with InsertionPoint(if_last.else_block):
                            ChannelPut("chan_cascade_lgu", l1_recv, indices=[tx])
                            yield_([])

                        DeallocOp(l1_res1)
                        DeallocOp(l1_gate)
                        DeallocOp(l1_up)
                        DeallocOp(l1_swiglu_out)
                        DeallocOp(l1_recv)

                    lgu_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
                    lgu_body.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    lgu_body.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    # ---- LD herd (N_LD cores at row 3) ----
                    @herd(name="ld_herd", sizes=[N_LD, 1])
                    def ld_herd_body(tx, ty, _sx, _sy):
                        c0 = arith.ConstantOp.create_index(0)
                        c_mt = arith.ConstantOp.create_index(M_TILE)
                        c_mldpc = arith.ConstantOp.create_index(M_ld_per_core)
                        c_kchunk = arith.ConstantOp.create_index(K_CHUNK)
                        c_one = arith.ConstantOp.create_index(1)
                        c_kchunk_i32 = arith.ConstantOp(i32_ty, K_CHUNK)
                        tx_base = arith.muli(tx, c_mldpc)

                        # Single full swiglu buffer; LGU sends K_LD_div
                        # K_CHUNK chunks in FIFO order with offsets.
                        l1_swiglu = AllocOp(swiglu_full_l1, [], [])
                        l1_r = AllocOp(R_ld_l1, [], [])
                        l1_slab = AllocOp(slab_ld_l1, [], [])

                        # Residual addend first (was correct order for
                        # standalone LGU+LD). LD dests are second half of
                        # res1ToCons: tx + N_LGU.
                        n_lgu_const = arith.ConstantOp.create_index(N_LGU)
                        ld_dest_idx = arith.addi(tx, n_lgu_const)
                        ChannelGet("res1ToCons", l1_r, indices=[ld_dest_idx, ty])
                        # Swiglu chunks: receive into the full buffer at
                        # successive K_CHUNK offsets.
                        for k_chunk in range(K_LD_div):
                            ck_off = arith.ConstantOp.create_index(k_chunk * K_CHUNK)
                            ChannelGet(
                                "swigluToLd",
                                l1_swiglu,
                                indices=[tx, ty],
                                offsets=[ck_off],
                                sizes=[c_kchunk],
                                strides=[c_one],
                            )

                        cst0_bf16_ld = arith.ConstantOp(bf16_ty, 0.0)
                        vecTy_mt_ld = VectorType.get([M_TILE], bf16_ty)
                        id_map_ld = AffineMapAttr.get(AffineMap.get_identity(1))

                        for outer in for_(M_ld_div):
                            l1_partial_full = AllocOp(partial_full_ty, [], [])
                            l1_partial_full.attributes["air.shrinkage"] = BoolAttr.get(
                                False
                            )
                            l1_partial_strided = subview(
                                l1_partial_full.result, [0], [M_TILE], [1]
                            )
                            l1_partial = memref_cast(D_la_l1, l1_partial_strided)
                            l1_d_full = AllocOp(partial_full_ty, [], [])
                            l1_d_full.attributes["air.shrinkage"] = BoolAttr.get(False)
                            l1_d_strided = subview(l1_d_full.result, [0], [M_TILE], [1])
                            l1_d = memref_cast(D_la_l1, l1_d_strided)

                            linalg.fill(cst0_bf16_ld, outs=[l1_partial])
                            # Inner K loop: scf.for + per-iter PACKED alloc
                            # so the compiler auto-introduces ping-pong on
                            # ldL2ToL1 (DMA prefetch overlaps with compute).
                            for k_chunk in for_(K_LD_div):
                                l1_p = AllocOp(packed_ld_l1, [], [])
                                ChannelGet("ldL2ToL1", l1_p, indices=[tx])
                                k_chunk_i32 = arith.IndexCastOp(i32_ty, k_chunk).result
                                b_off_i32 = arith.muli(k_chunk_i32, c_kchunk_i32)
                                CallOp(
                                    matvec_offset_func,
                                    [l1_p, l1_swiglu, b_off_i32, l1_partial],
                                )
                                DeallocOp(l1_p)
                                yield_([])

                            local_off = arith.muli(outer, c_mt)
                            global_off = arith.addi(tx_base, local_off)
                            # Inline partial+r: one 8-wide bf16 vector add
                            # (patched aievec pads to 16-wide internally).
                            sub_r_ld = subview(l1_r.result, [global_off], [M_TILE], [1])
                            v_p_ld = transfer_read(
                                vecTy_mt_ld,
                                l1_partial,
                                [c0],
                                id_map_ld,
                                cst0_bf16_ld,
                                [True],
                            )
                            v_r_ld = transfer_read(
                                vecTy_mt_ld,
                                sub_r_ld,
                                [c0],
                                id_map_ld,
                                cst0_bf16_ld,
                                [True],
                            )
                            v_sum_ld = arith.addf(v_p_ld, v_r_ld)
                            transfer_write(
                                None, v_sum_ld, l1_d, [c0], id_map_ld, [True]
                            )
                            if not skip_inline:
                                for i in range(M_TILE):
                                    ci = arith.ConstantOp.create_index(i)
                                    abs_idx = arith.addi(local_off, ci)
                                    v = memref_load(l1_d, [ci])
                                    memref_store(v, l1_slab, [abs_idx])
                            DeallocOp(l1_partial_full)
                            DeallocOp(l1_d_full)
                            yield_([])

                        ChannelPut("ldOutD", l1_slab, indices=[tx])

                        DeallocOp(l1_swiglu)
                        DeallocOp(l1_r)
                        DeallocOp(l1_slab)

                    ld_herd_body.attributes["link_with"] = StringAttr.get(
                        KERNEL_OBJ_NAME
                    )
                    ld_herd_body.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    ld_herd_body.attributes["y_loc"] = IntegerAttr.get(T.i64(), 3)

    return build()


def ld_cpu_reference(A_q, A_s, A_z, swiglu, R):
    """LD: D = dequant(A_q,A_s,A_z) @ swiglu + R."""
    M_ = A_q.shape[0]
    K_ = swiglu.shape[0]
    n_groups = A_s.shape[0]
    gs = K_ // n_groups
    A_q_i = A_q.astype(np.int32)
    low = A_q_i & 0x0F
    high = (A_q_i >> 4) & 0x0F
    nibs = np.empty((M_, K_), dtype=np.int32)
    nibs[:, 0::2] = low
    nibs[:, 1::2] = high
    s_per_kk = np.repeat(A_s.astype(np.float32), gs, axis=0)
    z_per_kk = np.repeat(A_z.astype(np.int32), gs, axis=0)
    dequant = (nibs - z_per_kk.T) * s_per_kk.T
    raw = (dequant @ swiglu.astype(np.float32)).astype(bfloat16).astype(np.float32)
    return (raw + R.astype(np.float32)).astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="o_gemv_ffn_int4_fused.py",
        description="Full int4-AWQ ELF2 (post-attention block, 1-launch fused).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--m-tile", type=int, default=8, dest="m_tile")
    parser.add_argument("--k-chunk", type=int, default=2048, dest="k_chunk")
    parser.add_argument("--n-cores", type=int, default=8, dest="n_cores")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"], default="elf"
    )
    args = parser.parse_args()

    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    print(
        f"O GEMV + FFN full-int4 1-launch fused: "
        f"emb_dim={emb_dim}, hidden_dim={hidden_dim}, k_chunk={args.k_chunk}"
    )

    module = build_o_gemv_ffn_int4_fused_module(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        gs=args.gs,
        m_tile=args.m_tile,
        k_chunk=args.k_chunk,
        n_cores=args.n_cores,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="o_gemv_ffn_int4_fused",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    np.random.seed(42)
    K = emb_dim
    M_LA = emb_dim
    M_LGU = 2 * hidden_dim
    K_LD = hidden_dim
    n_groups_la = K // args.gs
    n_groups_lgu = K // args.gs
    n_groups_ld = K_LD // args.gs

    # LA: M_LA x K weights.
    A_q_la_unp = np.random.randint(0, 16, size=(M_LA, K), dtype=np.uint8)
    A_q_la = (A_q_la_unp[:, 0::2] | (A_q_la_unp[:, 1::2] << 4)).astype(np.uint8)
    A_s_la = np.random.uniform(0.005, 0.02, size=(n_groups_la, M_LA)).astype(bfloat16)
    A_z_la = np.random.randint(7, 9, size=(n_groups_la, M_LA), dtype=np.uint8)
    B_la = np.random.randn(K).astype(bfloat16)
    R_la = np.random.randn(M_LA).astype(bfloat16)

    # LGU: M_LGU x K weights.
    A_q_lgu_unp = np.random.randint(0, 16, size=(M_LGU, K), dtype=np.uint8)
    A_q_lgu = (A_q_lgu_unp[:, 0::2] | (A_q_lgu_unp[:, 1::2] << 4)).astype(np.uint8)
    A_s_lgu = np.random.uniform(0.005, 0.02, size=(n_groups_lgu, M_LGU)).astype(
        bfloat16
    )
    A_z_lgu = np.random.randint(7, 9, size=(n_groups_lgu, M_LGU), dtype=np.uint8)
    gamma = (np.random.randn(K) * 0.1 + 1.0).astype(bfloat16)

    # LD: M_LA x K_LD weights.
    A_q_ld_unp = np.random.randint(0, 16, size=(M_LA, K_LD), dtype=np.uint8)
    A_q_ld = (A_q_ld_unp[:, 0::2] | (A_q_ld_unp[:, 1::2] << 4)).astype(np.uint8)
    A_s_ld = np.random.uniform(0.005, 0.02, size=(n_groups_ld, M_LA)).astype(bfloat16)
    A_z_ld = np.random.randint(7, 9, size=(n_groups_ld, M_LA), dtype=np.uint8)
    assert K == M_LA, "Builder requires K == M_LA so res1 doubles as LD R"

    # Full chain: LA -> res1; LGU(res1, gamma) -> swiglu;
    # LD(swiglu, wd, res1) -> final
    res1_ref = la_cpu_reference(A_q_la, A_s_la, A_z_la, B_la, R_la)
    swiglu_ref = lgu_cpu_reference(A_q_lgu, A_s_lgu, A_z_lgu, res1_ref, gamma)
    D_ref = ld_cpu_reference(A_q_ld, A_s_ld, A_z_ld, swiglu_ref, res1_ref)

    PACKED_la = pack_inputs(
        A_q_la,
        A_s_la,
        A_z_la,
        M_LA,
        K,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        M_LA,
    )
    PACKED_lgu = pack_inputs(
        A_q_lgu,
        A_s_lgu,
        A_z_lgu,
        M_LGU,
        K,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        M_LGU,
    )
    PACKED_ld = pack_inputs(
        A_q_ld,
        A_s_ld,
        A_z_ld,
        M_LA,
        K_LD,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        M_LA,
    )

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="o_gemv_ffn_int4_fused",
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )
    sys.exit(
        runner.run_test(
            module,
            inputs=[PACKED_la, B_la, R_la, PACKED_lgu, gamma, PACKED_ld],
            expected_outputs=[D_ref, res1_ref],
            rtol=0.2,
            atol=2.0,
            min_correlation=0.99,
        )
    )
