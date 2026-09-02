# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""o_gemv_ffn_int4_fused — full-int4 1-launch ELF2 for the LLAMA decode block.

Single-launch alternative to o_gemv_ffn_int4_multi.py. Same post-attention
math (O proj + residual #1, RMSNorm, FFN gate/up + SwiGLU, FFN down +
residual #2), packed into ONE air.launch with three herd groups wired
together by two W->E cascade-shift chains, so res1 (LA->LGU/LD) and swiglu
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

Written on ``air.api``. Two things this kernel needs that most do not:

**Placement is pinned, not inferred.** ``air.herd(at=(column, row))``
sets the ``x_loc``/``y_loc`` that ``air-place-herds`` reads. Normally
letting the pass choose is right, but a cascade is a physical link
between *neighbouring* cores, so a W->E chain only exists if its herds
are actually laid out west to east along one row. Both chains here
depend on that, and the three rows are chosen so the packet broadcasts
and the shim sharing work out.

**Three channels are packet-switched rather than circuit-switched.**
``res1ToCons``, ``lguGAMMA`` and ``swigluToLd`` each fan one L1 buffer
out to many cores, and a packet broadcast reaches all of its
destinations over *one* flow where a circuit-switched channel needs one
per destination. That is not a tuning preference: res1ToCons has 16
destinations, and the LD stream switch arbiter has a 4-msel multicast
limit that separate flows would blow straight through.
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i8, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "matrix_vector_multiplication",
        "int4_awq",
    ),
)
from matvec_int4_packed import pack_inputs
from matvec_int4_packed_add import cpu_reference as la_cpu_reference

KERNEL_OBJ_NAME = "mv_int4_bf16.o"

# The rows the three herd groups sit on, west to east along each.
ROW_LGU, ROW_LD, ROW_LA = 2, 3, 4

SILU_VEC = 32
RMS_VEC = 16


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


def build_o_gemv_ffn_int4_fused_module(
    emb_dim=2048, hidden_dim=8192, gs=128, m_tile=8, k_chunk=2048, n_cores=8, **kwargs
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
        **kwargs,
    )


def build_launch(
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
    target="npu2",
):
    """Trace the kernel and return the launch; ``build_module`` lowers it."""
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
    half_M_per_core = M_lgu_per_core // 2  # 1024
    M_OUT = M_LGU // 2  # 8192 (assembled swiglu)
    assert half_M_per_core % SILU_VEC == 0
    assert K % RMS_VEC == 0

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

    # ---- L3 interface: inputs first, then the two outputs. ----
    P_la = air.tensor([N_LA * la_tiles_per_col, tile_bytes], i8)
    B_la = air.tensor([K], bf16)
    R_la = air.tensor([M_LA], bf16)
    P_lgu = air.tensor([total_lgu_tiles, tile_bytes], i8)
    G_l3 = air.tensor([K], bf16)
    P_ld = air.tensor([N_LD * ld_tiles_per_col, tile_bytes], i8)
    D_ld = air.tensor([M_LA], bf16)
    # DEBUG: capture LA's assembled res1 to L3 so it can be checked against
    # the CPU reference for LA on its own.
    D_dbg = air.tensor([M_LA], bf16)

    # ---- Channels ----
    # LA per-col coalesced inputs: B + R-slab + multi-dim PACKED all on ONE
    # packet channel per col.
    la_all = [
        air.channel(f"laAll_{c}", channel_type="npu_dma_packet") for c in range(N_LA)
    ]
    # W->E cascade chain across the LA row. N_LA-1 edges.
    casc_la = air.channel(
        "chan_cascade_la", size=[N_LA - 1], channel_type="npu_cascade"
    )
    # LGU: per-col packed via memtile, default circuit for non-shared flows.
    lgu_packed = air.channel("lguPACKED", size=[N_LGU])
    lgu_l2_l1 = air.channel("lguL2ToL1", size=[N_LGU])
    # Eastmost LA L1 -> all LGU + LD cores: 16-dest packet broadcast.
    res1_to_cons = air.channel(
        "res1ToCons",
        size=[1, 1],
        broadcast_shape=[N_LGU + N_LD, 1],
        channel_type="npu_dma_packet",
    )
    # DEBUG: also emit res1 to L3 so it can be inspected.
    la_res_debug = air.channel("laResDebug", size=[1])
    # Packet broadcast for gamma (shares LGU S2MM:0).
    lgu_gamma = air.channel(
        "lguGAMMA",
        size=[1, 1],
        broadcast_shape=[N_LGU, 1],
        channel_type="npu_dma_packet",
    )
    casc_lgu = air.channel(
        "chan_cascade_lgu", size=[N_LGU - 1], channel_type="npu_cascade"
    )
    # Eastmost LGU L1 -> all LD cores: ONE packet broadcast carrying
    # K_LD_div K_CHUNK-bf16 chunks in FIFO sequence. Collapsing the 4
    # chunks onto a single channel keeps the LD stream switch arbiter
    # under the 4-msel multicast limit (4 separate broadcast channels
    # + ldR1 would exceed it).
    swiglu_to_ld = air.channel(
        "swigluToLd",
        size=[1, 1],
        broadcast_shape=[N_LD, 1],
        channel_type="npu_dma_packet",
    )
    # LD: per-col packed via memtile (packet for shim sharing).
    ld_packed = air.channel("ldPACKED", size=[N_LD], channel_type="npu_dma_packet")
    ld_l2_l1 = air.channel("ldL2ToL1", size=[N_LD])
    ld_out = air.channel("ldOutD", size=[N_LD])

    # ---- Hand-written kernels ----
    matvec_store = air.extern(
        "matvec_int4_bf16_packed_store", link_with=KERNEL_OBJ_NAME
    )
    # LD path: a matvec taking a b-offset in elements, so one big swiglu
    # buffer can be kept and k iterated with a loop the compiler can
    # ping-pong the PACKED get against.
    matvec_offset = air.extern(
        "matvec_int4_bf16_packed_b_offset",
        link_with=KERNEL_OBJ_NAME,
        scalars=[i32],
    )

    with air.launch(name="o_gemv_ffn_int4_fused", target=target) as lch:

        @lch.body
        def _():
            # LA per-col puts (B + full R + multi-dim PACKED) on the same
            # per-col packet channel. Order matches the herd-side gets.
            for c in range(N_LA):
                la_all[c].put(B_la[0:K])
                # Push FULL R so each LA core reads its own per-core slab
                # out of it at the right offset (shared shape with LD).
                la_all[c].put(R_la[0:M_LA])
                lo = c * la_tiles_per_col
                la_all[c].put(P_la[lo : lo + la_tiles_per_col, :])

            # LGU per-col packed + gamma broadcast. res1 is supplied by LA
            # over res1ToCons, so there is no L3 res1 put here.
            for c in range(N_LGU):
                lo = c * M_lgu_div
                lgu_packed.put(P_lgu[lo : lo + M_lgu_div, :], indices=[c])
            lgu_gamma.put(G_l3[0:K])

            # DEBUG: catch LA's broadcast output to L3. The whole tensor,
            # not D_dbg[0:M_LA]: a full-extent slice says the same thing but
            # reaches the IR as an explicit [0] [2048] [1] access pattern,
            # and on a design this close to the shim's limits that is enough
            # to change the allocation ("failed to get MM2S tile for L3
            # allocation"). A bare tensor emits [] [] [], as it must here.
            la_res_debug.get(D_dbg)

            # LD per-col packed + per-col output. The residual addend is
            # res1, which LD receives from LA over res1ToCons.
            for c in range(N_LD):
                lo = c * ld_tiles_per_col
                ld_packed.put(P_ld[lo : lo + ld_tiles_per_col, :], indices=[c])
                d_lo = c * M_ld_per_core
                ld_out.get(D_ld[d_lo : d_lo + M_ld_per_core], indices=[c])

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    # ---- L2 staging ----
                    # One tile in flight per trip: the alloc is inside the
                    # loop so the compiler introduces ping-pong on it.
                    for c in range(N_LGU):
                        for _t in air.sequential(M_lgu_div):
                            l2 = air.alloc([tile_bytes], i8, scope=seg.private())
                            lgu_packed.get(l2, indices=[c])
                            lgu_l2_l1.put(l2, indices=[c])

                    for c in range(N_LD):
                        for _t in air.sequential(ld_tiles_per_col):
                            l2 = air.alloc([tile_bytes], i8, scope=seg.private())
                            ld_packed.get(l2, indices=[c])
                            ld_l2_l1.put(l2, indices=[c])

                    # ---- LA: N_LA 1x1 herds, west to east along ROW_LA ----
                    # Each LA core produces its M_la_per_core slab of res1
                    # into a full-M_LA scratch at col*M_la_per_core, then
                    # joins the W->E cascade. The eastmost one broadcasts
                    # the assembled res1 to every LGU and LD core.
                    for col in range(N_LA):
                        _la_herd(
                            col=col,
                            N_LA=N_LA,
                            K=K,
                            M_LA=M_LA,
                            M_TILE=M_TILE,
                            M_la_div=M_la_div,
                            M_la_per_core=M_la_per_core,
                            tile_bytes=tile_bytes,
                            skip_inline=skip_inline,
                            la_all=la_all,
                            casc_la=casc_la,
                            res1_to_cons=res1_to_cons,
                            la_res_debug=la_res_debug,
                            matvec_store=matvec_store,
                        )

                    _lgu_herd(
                        N_LGU=N_LGU,
                        K=K,
                        K_CHUNK=K_CHUNK,
                        K_LD_div=K_LD_div,
                        M_TILE=M_TILE,
                        M_OUT=M_OUT,
                        M_lgu_div=M_lgu_div,
                        half_M_per_core=half_M_per_core,
                        tile_bytes=tile_bytes,
                        skip_inline=skip_inline,
                        lgu_gamma=lgu_gamma,
                        res1_to_cons=res1_to_cons,
                        lgu_l2_l1=lgu_l2_l1,
                        casc_lgu=casc_lgu,
                        swiglu_to_ld=swiglu_to_ld,
                        matvec_store=matvec_store,
                    )

                    _ld_herd(
                        N_LGU=N_LGU,
                        N_LD=N_LD,
                        K_CHUNK=K_CHUNK,
                        K_LD=K_LD,
                        K_LD_div=K_LD_div,
                        M_LA=M_LA,
                        M_TILE=M_TILE,
                        M_ld_div=M_ld_div,
                        M_ld_per_core=M_ld_per_core,
                        tile_bytes=tile_bytes,
                        skip_inline=skip_inline,
                        res1_to_cons=res1_to_cons,
                        swiglu_to_ld=swiglu_to_ld,
                        ld_l2_l1=ld_l2_l1,
                        ld_out=ld_out,
                        matvec_offset=matvec_offset,
                    )

    return lch


def _la_herd(
    *,
    col,
    N_LA,
    K,
    M_LA,
    M_TILE,
    M_la_div,
    M_la_per_core,
    tile_bytes,
    skip_inline,
    la_all,
    casc_la,
    res1_to_cons,
    la_res_debug,
    matvec_store,
):
    """One LA core: O-proj GEMV + residual #1, then its cascade hop.

    Emitted as N_LA separate 1x1 herds rather than one 1-D herd because the
    cascade makes each column's body genuinely different -- the first only
    sends, the last broadcasts -- and because each has to be pinned to its
    own column for the W->E chain to be W->E.
    """
    is_first = col == 0
    is_last = col == N_LA - 1
    col_base = col * M_la_per_core

    with air.herd([range(1), range(1)], name=f"la_{col}", at=(col, ROW_LA)) as h:

        @h.body
        def _(_tx, _ty):
            l1_b = air.alloc([K], bf16, scope=h.private())
            l1_r = air.alloc([M_LA], bf16, scope=h.private())
            l1_local = air.alloc([M_LA], bf16, scope=h.private())
            if not skip_inline:
                # Only this core's slab is written below; the cascade adds
                # the buffers together, so everything else has to be zero.
                ops.fill(l1_local, 0.0)

            # B, then R, then the PACKED tiles inside the matvec loop.
            la_all[col].get(l1_b)
            la_all[col].get(l1_r)

            for outer in air.sequential(M_la_div):
                l1_p = air.alloc([tile_bytes], i8, scope=h.private())
                la_all[col].get(l1_p)
                # M_TILE wide, and handed to the kernel whole. The
                # predecessor allocates 32 and passes a subview of the head,
                # but air-shrink-memref-sizes-by-access narrows that to 8 and
                # folds the subview away, so 8 is what it actually compiles;
                # allocating it directly keeps a memref.subview out of the
                # dependency analysis, which does not see a write *through* a
                # subview as a write to the parent buffer.
                partial = air.alloc([M_TILE], bf16, scope=h.private(), vector=M_TILE)
                out = air.alloc([M_TILE], bf16, scope=h.private(), vector=M_TILE)

                global_off = col_base + outer * M_TILE
                matvec_store(l1_p, l1_b, partial)
                out[:] = partial[:] + l1_r[global_off : global_off + M_TILE]
                if not skip_inline:
                    l1_local[global_off : global_off + M_TILE] = out[:]

            # ---- W->E cascade shift ----
            if is_first:
                casc_la.put(l1_local, indices=[col])
            else:
                l1_recv = air.alloc([M_LA], bf16, scope=h.private())
                casc_la.get(l1_recv, indices=[col - 1])
                if not skip_inline:
                    l1_local[:] = l1_recv[:] + l1_local[:]
                if is_last:
                    res1_to_cons.put(l1_local)
                    # DEBUG: also send to L3.
                    la_res_debug.put(l1_local)
                else:
                    casc_la.put(l1_local, indices=[col])


def _lgu_herd(
    *,
    N_LGU,
    K,
    K_CHUNK,
    K_LD_div,
    M_TILE,
    M_OUT,
    M_lgu_div,
    half_M_per_core,
    tile_bytes,
    skip_inline,
    lgu_gamma,
    res1_to_cons,
    lgu_l2_l1,
    casc_lgu,
    swiglu_to_ld,
    matvec_store,
):
    """RMSNorm(res1, gamma) -> gate/up GEMV -> SwiGLU -> cascade -> LD."""
    # shape= pins the *physical* herd, one core per column. Without it the
    # DSL strip-mines the 8-wide grid onto its default 2 columns and loops
    # 4x, which is a different machine: the cascade needs eight cores laid
    # out W->E on one row, and the shim allocation is computed per column.
    with air.herd(
        [range(N_LGU), range(1)], shape=(N_LGU, 1), name="lgu_h", at=(0, ROW_LGU)
    ) as h:

        @h.body
        def _(tx, ty):
            l1_gamma = air.alloc([K], bf16, scope=h.private())
            l1_res1 = air.alloc([K], bf16, scope=h.private())
            # RMS writes back in place into res1 rather than into a second
            # buffer -- worth 4 KB of L1 here.
            l1_normed = l1_res1
            l1_gate = air.alloc([half_M_per_core], bf16, scope=h.private())
            l1_up = air.alloc([half_M_per_core], bf16, scope=h.private())
            # Per-core SwiGLU output (2 KB). The cascade step copies it
            # into l1_recv at the right offset, which is what lets the
            # 16 KB assembled buffer be allocated once rather than twice.
            l1_swiglu_out = air.alloc([half_M_per_core], bf16, scope=h.private())

            # The faster-arriving broadcast (gamma) first.
            lgu_gamma.get(l1_gamma, indices=[tx, ty])
            # LGU takes the first N_LGU destinations of res1ToCons.
            res1_to_cons.get(l1_res1, indices=[tx, ty])

            # ---- RMSNorm, in place ----
            if not skip_inline:
                acc = air.alloc([1], f32, scope=h.private())
                acc[:] = ops.reduce_add(ops.cast(l1_res1[:] * l1_res1[:], f32))
                rstd = ops.cast(ops.rsqrt(acc[:] / float(K) + 1.0e-5), bf16)
                l1_normed[:] = l1_res1[:] * rstd * l1_gamma[:]

            # ---- Hot int4 GEMV: one PACKED tile per trip, allocated in
            #      the loop so the compiler ping-pongs the get. ----
            # The result tile is hoisted *out* of the loop, and has to be:
            # air-label-scf-for-to-ping-pong only labels a loop whose body
            # allocates the one buffer the get targets, so a second alloc
            # beside it costs the loop its ping-pong (lguL2ToL1 stays at one
            # get instead of two) and the design then runs out of shim MM2S.
            partial = air.alloc([M_TILE], bf16, scope=h.private())
            # This fill is not needed for the arithmetic -- the kernel writes
            # every element of `partial` before anything reads it. It is here
            # to keep the alloc where it was written: air-fuse-alloc-dealloc
            # sinks an alloc into a loop when *every* user is inside it, and
            # sinking this one is what costs the loop its ping-pong. The fill
            # is a use outside the loop, so the alloc stays out. The
            # predecessor gets the same effect by accident, from a
            # memref.subview it happens to compute once above the loop.
            ops.fill(partial, 0.0)
            for outer in air.sequential(M_lgu_div):
                l1_p = air.alloc([tile_bytes], i8, scope=h.private())
                lgu_l2_l1.get(l1_p, indices=[tx])
                matvec_store(l1_p, l1_normed, partial)
                if not skip_inline:
                    # The kernel emits gate and up interleaved, so this
                    # de-interleaves them. A strided read would say it
                    # better, but a slice step is not a DMA access pattern
                    # and the DSL has no spelling for one.
                    pair_off = outer * (M_TILE // 2)
                    for i in range(M_TILE // 2):
                        l1_gate[pair_off + i] = partial[2 * i]
                        l1_up[pair_off + i] = partial[2 * i + 1]

            # ---- SwiGLU ----
            if not skip_inline:
                l1_swiglu_out[:] = ops.silu(l1_gate[:]) * l1_up[:]

            # ---- W->E cascade shift ----
            # l1_recv is the single 16 KB assembled buffer. Column 0 starts
            # it at zero and writes its own slab in; every other column
            # receives the assembled-so-far value and overwrites its own
            # slot, which is zero there because no earlier core writes it.
            l1_recv = air.alloc([M_OUT], bf16, scope=h.private())
            col_base = tx * half_M_per_core

            first = ops.branch(tx == 0)
            with first:
                ops.fill(l1_recv, 0.0)
            with first.otherwise():
                casc_lgu.get(l1_recv, indices=[tx - 1])

            if not skip_inline:
                # Both arms: copy this core's slab in at col_base.
                l1_recv[col_base : col_base + half_M_per_core] = l1_swiglu_out[:]

            last = ops.branch(tx == N_LGU - 1)
            with last:
                # Eastmost LGU: broadcast K_LD_div K_CHUNK chunks over one
                # packet channel, in FIFO order.
                for k_chunk in range(K_LD_div):
                    lo = k_chunk * K_CHUNK
                    swiglu_to_ld.put(l1_recv[lo : lo + K_CHUNK])
            with last.otherwise():
                casc_lgu.put(l1_recv, indices=[tx])


def _ld_herd(
    *,
    N_LGU,
    N_LD,
    K_CHUNK,
    K_LD,
    K_LD_div,
    M_LA,
    M_TILE,
    M_ld_div,
    M_ld_per_core,
    tile_bytes,
    skip_inline,
    res1_to_cons,
    swiglu_to_ld,
    ld_l2_l1,
    ld_out,
    matvec_offset,
):
    """FFN-down GEMV over the assembled swiglu, plus residual #2."""
    # shape= pins the *physical* herd, one core per column. Without it the
    # DSL strip-mines the 8-wide grid onto its default 2 columns and loops
    # 4x, which is a different machine: the cascade needs eight cores laid
    # out W->E on one row, and the shim allocation is computed per column.
    with air.herd(
        [range(N_LD), range(1)], shape=(N_LD, 1), name="ld_herd", at=(0, ROW_LD)
    ) as h:

        @h.body
        def _(tx, ty):
            # One full swiglu buffer; LGU sends K_LD_div K_CHUNK chunks in
            # FIFO order and each lands at its own offset.
            l1_swiglu = air.alloc([K_LD], bf16, scope=h.private())
            l1_r = air.alloc([M_LA], bf16, scope=h.private())
            l1_slab = air.alloc([M_ld_per_core], bf16, scope=h.private())

            # The residual addend first, matching the order the standalone
            # LGU+LD build used. LD takes the second half of res1ToCons'
            # destinations.
            res1_to_cons.get(l1_r, indices=[tx + N_LGU, ty])
            for k_chunk in range(K_LD_div):
                lo = k_chunk * K_CHUNK
                swiglu_to_ld.get(l1_swiglu[lo : lo + K_CHUNK], indices=[tx, ty])

            tx_base = tx * M_ld_per_core

            for outer in air.sequential(M_ld_div):
                partial = air.alloc([M_TILE], bf16, scope=h.private(), vector=M_TILE)
                out = air.alloc([M_TILE], bf16, scope=h.private(), vector=M_TILE)
                ops.fill(partial, 0.0)

                # Inner K loop, with the PACKED tile allocated per trip so
                # the compiler overlaps its DMA with the compute.
                for k_chunk in air.sequential(K_LD_div):
                    l1_p = air.alloc([tile_bytes], i8, scope=h.private())
                    ld_l2_l1.get(l1_p, indices=[tx])
                    matvec_offset(l1_p, l1_swiglu, k_chunk * K_CHUNK, partial)

                local_off = outer * M_TILE
                global_off = tx_base + local_off
                out[:] = partial[:] + l1_r[global_off : global_off + M_TILE]
                if not skip_inline:
                    l1_slab[local_off : local_off + M_TILE] = out[:]

            ld_out.put(l1_slab, indices=[tx])


def build_module(**kwargs):
    """Lower the kernel to a module, as the XRT backend and runner want it."""
    target = kwargs.get("target", "npu2")
    return build_launch(**kwargs).build(target=target)


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
        "--target",
        type=str,
        default="npu2",
        help="AIE generation. This kernel is npu2-only: it uses eight columns "
        "and three fixed rows, which npu1 does not have.",
    )
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
        target=args.target,
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
            target_device=args.target,
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
