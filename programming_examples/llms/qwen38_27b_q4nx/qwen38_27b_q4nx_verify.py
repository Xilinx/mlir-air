# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B two-token verify pass, the speculative decode's: the decode
# (qwen38_27b_q4nx_decode.py) on two tokens at positions L - 2 and L - 1 in
# one dispatch, each weight block read once for both (kernels/q4k_mvm2.cc).
# The same tile map; what changes per core:
#
#   GEMV array: x [2][K]; a packet carries both tokens' rows of its row
#     block; the down runs its K halves outermost (2 x 17408 does not fit L1)
#     with an accumulator per row group (mlp_block2.py).
#   RMS: both tokens' residuals [2][D], taken once per wave; the norm weights
#     one at a time on a channel of their own; the normed x [2][D] to the
#     assemble memtile, which also hands the first to the alpha / beta core.
#   conv: both tokens, token 1 after token 0's input in the cache; the cache
#     keeps [x2, in0, in1].
#   step cores: token 0's state (S1, for a rejected draft) and token 1's
#     (S2) each into a buffer of its own; both go out.
#   gate: z [2][DQ]; each token's mixer output in turn, gated into z in place.
#   RoPE: both tokens, q packed per token; the padding rounds dropped.
#   attention units: both tokens' q per KV head against each K/V block, read
#     once; token 0 masked at L - 1, token 1 at L.
#   GLU: the down's input a K half at a time.
#
# The ELF holds one wave (four layers); a step dispatches it once per wave,
# each run on its wave's share of the model's buffers. The launch repeated per
# wave in one dispatch, as the decode's is, does not route: with the volumes
# a wave loop leaves unknown, the router keeps more packet flows off a shared
# arbiter than the links hold.
#
# The host commits a verify: both tokens kept (SO, the new conv cache), or
# token 0 only (SO1, and the conv cache rebuilt from the old one and the
# new's first input).
import argparse
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
sys.path.insert(0, str(HERE.parents[1] / "fused_decode"))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import attn_layer as A  # noqa: E402
import common  # noqa: E402
import gdn_layer as G  # noqa: E402
import mlp_block2 as M2  # noqa: E402
import qwen38_27b_q4nx_decode as DEC  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from rope import rope_weights  # noqa: E402

R, RB, CB, BB, D, I = G.R, G.RB, G.CB, G.BB, G.D, G.I
NCOL, HD, NV, NCH, SLOT = G.NCOL, G.HD, G.NV, G.NCH, G.SLOT
DH, DQ, DK, NCU, KPU = A.DH, A.DQ, A.DK, A.NCU, A.KPU
QR, QHP, DQP = A.QR, A.QHP, A.DQP
NCB_X, NCB_H, NCB_O = G.NCB_X, G.NCB_H, G.NCB_O
PER_WAVE, NG_WAVE = DEC.PER_WAVE, DEC.NG_WAVE
CONV, GATE, RMS, GLU, ROPE = range(5)
MISC, MISC_LEN = DEC.MISC, DEC.MISC_LEN
ISSUE_AT, STEP_COL = DEC.ISSUE_AT, DEC.STEP_COL
QKV_G = G.QKV
UO = KPU * QR * DH
HALVES = M2.HALVES
IH = I // HALVES
NQKV = (A.RG_Q + A.RG_KV) * 512  # a token's q | k | v rows in the RoPE core
XR = 1024  # a round of the RMS core's x and norm weights
NXR = 2 * D // XR  # rounds of x
NWR = 2 * D // XR  # rounds of a layer's two norm weights
assert NQKV == DQP
# a unit's KV head's output is a step core's share of a layer's: og_p's size
assert QR * DH == NV * 32


def build_module(waves, maxl):
    """`waves` x (3 GDN layers, then an attention one), two tokens, for a
    context of up to `maxl` positions."""
    pw, ngw = PER_WAVE, NG_WAVE
    nl, ng = pw * waves, ngw * waves
    assert maxl % 16 == 0
    nb = maxl // 16
    k_ = lambda o, n, s=(): air.extern(  # noqa: E731
        n, link_with=f"{o}.o", scalars=list(s)
    )
    step2 = k_("q4k_mvm2", "q4k_step2", [i32] * 4)
    flush2 = k_("q4k_mvm2", "q4k_flush2", [i32])
    x_rd = k_("q38_ffn", "x_round", [i32])
    rms_inv = k_("q38_ffn", "rms_inv2", [i32])
    rms_app = k_("q38_ffn", "rms_apply2", [i32, i32])
    res2 = k_("q38_ffn", "residual_round2", [i32, i32])
    glu2 = k_("q38_ffn", "glu2_round", [i32, i32])
    conv2 = k_("gdn_prep2", "gdn_conv_cache2", [i32])
    l2n_t = k_("gdn_prep2", "gdn_l2norm_t", [i32, i32, i32])
    pack_t = k_("gdn_prep2", "gdn_pack_t", [i32, i32])
    ab_zero2 = k_("gdn_prep2", "ab_zero2")
    ab_mvm2 = k_("gdn_prep2", "ab_mvm2", [i32, i32, i32])
    ab_act2 = k_("gdn_prep2", "ab_activate2")
    r_slot = k_("router", "router_slot", [i32, i32, i32])
    gstep = k_("gdn_decode2", "gdn_step_ho", [i32, i32, i32])
    z_rd2 = k_("gate2", "z_round2", [i32, i32])
    dn = k_("gate2", "dn_norm_rhv")
    gsilu = k_("gate2", "gate_silu_tz", [i32, i32])
    gsig = k_("gate2", "gate_sigmoid_tz", [i32, i32])
    rope_rd2 = k_("rope2", "rope_round2", [i32, i32])
    rdrop = k_("rope2", "round_drop")
    rope_t = k_("rope2", "rope_compute_t", [i32, i32])
    qpack_t = k_("rope2", "q_pack_t", [i32, i32])
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    # the masked keys' values zeroed first (qwen36_35b_a3b_q4nx/kernels/attn_kv_q36.cc)
    kv = k_("attn_kv_q36", "attn_kv_blk_z", [i32, i32])
    fin = k_("attn_kv_q36", "attn_kv_fin")
    o_nat = k_("attn_kv_q36", "o_natural")

    arr = M2.TwoTokenArray(
        step2,
        flush2,
        row_block=2 * RB,
        block_bytes=BB,
        assemble_col=G.ASM,
        acc_words=4 * RB * G.RG_DN,
        columns=G.COLUMNS,
    )
    rnd = arr.round  # a two-token round: [column][core][token][32]

    X = air.tensor([waves + 1, pw, 2, D], bf16, inout=True)
    WD = air.tensor([nl, NCOL, 2, G.G_DENSE, 2, BB], ui8)
    M = air.tensor([nl, MISC_LEN], bf16)
    CWI = air.tensor([ng, NCH, 7 * HD], bf16)
    SI = air.tensor([ng, NV, R, HD * 32], f32)
    KV = air.tensor([waves, 2, maxl, DK], bf16, inout=True)
    RW = air.tensor([waves, 2, 3 * DH], bf16)
    CWO = air.tensor([ng, NCH, 7 * HD], bf16)
    SO = air.tensor([ng, NV, R, HD * 32], f32)  # after both tokens
    SO1 = air.tensor([ng, NV, R, HD * 32], f32)  # after token 0
    # debug probes (Q38V_DBG=ab,z,kv), one at a time: the design's routes
    # leave room for about one more flow
    dbg = set(filter(None, os.environ.get("Q38V_DBG", "").split(",")))
    DBG = air.tensor([waves, pw, 2 * D], bf16) if "ab" in dbg else None
    DBG_V = air.tensor([waves, pw, 4 * NV], f32) if "abv" in dbg else None
    DBG_Z = air.tensor([waves, pw, 2 * DQ], bf16) if "z" in dbg else None
    DBG_A = air.tensor([waves, 2 * UO], bf16) if dbg & {"kv", "kv1"} else None
    DBG_UNIT = 1 if "kv1" in dbg else 0
    # the context length after token 1: token 0 is at L - 2, token 1 at L - 1
    SEQ = air.tensor([], i32, name="L")

    ch = air.channel
    pkt = dict(channel_type="npu_dma_packet")
    m_in, to_rms = ch("MIn"), ch("ToRms")
    xn_ch, to_ab, ab_w, out_ch = ch("XN"), ch("ToAB"), ch("ABW"), ch("Out")
    to_asm = ch("ToAsm", **pkt)
    cw_in, cw_out = ch("CWIn"), ch("CWOut")
    kvq_ch = ch("KVQ", size=[1], broadcast_shape=[R])
    s_in, s_out = ch("SIn", size=[R]), ch("SOut", size=[R])
    s_l3, s_l3o, s_l3o1 = ch("SL3"), ch("SL3Out"), ch("SL3Out1")
    o_in, m_dn = ch("OIn"), ch("MDn")
    # each step core's outputs, both tokens, into memtile 3 on one channel
    og_p = ch("OGp", size=[R], **pkt)
    rw_in = ch("RwIn")
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    k_out, v_out = ch("KOut", **pkt), ch("VOut", **pkt)
    k_rd, v_rd = ch("KRd"), ch("VRd")
    k_cu, v_cu = ch("KCu", size=[NCU]), ch("VCu", size=[NCU])
    arr.channels(n_dests=5)
    dbg_ch = ch("Dbg") if DBG is not None else None
    dbg_v = ch("DbgV") if DBG_V is not None else None
    dbg_z = ch("DbgZ") if DBG_Z is not None else None
    dbg_a = ch("DbgA") if DBG_A is not None else None
    dbg_h = None

    with air.launch(name="verify", repeat=waves) as launch:

        def body(w):
            lbase = air.rtp(w * pw)
            Lp = air.rtp(SEQ.value)
            with air.segment(name="seg", params=[lbase, Lp]) as seg:

                @seg.body
                def _():
                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    def wave_loop(n):
                        return air.sequential(0, n)

                    ab_sh = air.alloc([4 * NV], f32, scope=seg.shared())

                    # ---- memtile 5: the RMS core's x, once a wave, and each
                    # layer's two norm weights; each token's mixer output to
                    # the gate core. A GDN layer's comes from the step cores,
                    # an attention layer's from the units, on the same packet
                    # channel (og_p[r]: step core r, or unit r // 2's KV head
                    # r % 2), so every layer moves the same way: a layer kind
                    # of its own would end a DMA chain that the next dispatch
                    # never restarts
                    #
                    # The RMS core's inputs a round at a time through one
                    # buffer: x once a wave, then each layer's two norm
                    # weights. Each source's output per token in a buffer of
                    # its own, filled whole (a memtile buffer filled in pieces
                    # and read by DMA can be read before its last piece).
                    r5 = l2([XR], 5)
                    for _k in air.sequential(0, NXR + pw * NWR):
                        m_in.get(r5)
                        to_rms.put(r5)
                    # and after each token's the layer's DeltaNet norm weight
                    # (a placeholder in an attention layer): a copy per token,
                    # so the gate core's input ring is [o, wn] each token and
                    # each buffer here appears once in its ring
                    ogs = [[l2([NV * 32], 5) for _t in range(2)] for _r in range(R)]
                    dn5 = [l2([HD], 5) for _t in range(2)]
                    # unrolled: a loop would ping-pong the ten buffers past
                    # the memtile's BD pool
                    for _j in range(pw):
                        for t in range(2):
                            m_dn.get(dn5[t])
                        for r in range(R):
                            for t in range(2):
                                og_p.get(ogs[r][t], indices=[r])
                        for t in range(2):
                            for r in range(R):
                                o_in.put(ogs[r][t])
                            o_in.put(dn5[t])

                    # ---- memtile 3: the KV cache to the units
                    km, vm = l2([16, DK], 3), l2([16, DK], 3)
                    for _b in wave_loop(nb):
                        k_rd.get(km)
                        v_rd.get(vm)
                        for u in range(NCU):
                            for kk in range(KPU):
                                hd = u * KPU + kk
                                kh = km.reshape(16, DK // 8, 8)[
                                    :, hd * DH // 8 : (hd + 1) * DH // 8, :
                                ]
                                k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                                vh = vm.reshape(2, 8, DK // 8, 8)[
                                    :, :, hd * DH // 8 : (hd + 1) * DH // 8, :
                                ]
                                v_cu.put(vh.transpose(0, 2, 1, 3), indices=[u])

                    # ---- memtile 4: the GDN state, head by head; S1 and S2 out
                    for _j in wave_loop(ngw):
                        for _h in wave_loop(NV):
                            sb = l2([R, HD * 32], 4, f32)
                            s_l3.get(sb)
                            for r in range(R):
                                s_in.put(sb[r, :], indices=[r])
                            air.dealloc(sb)
                            ob1 = l2([R, HD * 32], 4, f32)
                            ob2 = l2([R, HD * 32], 4, f32)
                            for r in range(R):
                                s_out.get(ob1[r, :], indices=[r])
                                s_out.get(ob2[r, :], indices=[r])
                            s_l3o1.put(ob1)
                            s_l3o.put(ob2)
                            air.dealloc(ob1)
                            air.dealloc(ob2)

                    # ---- the assemble memtile: the normed x (the first also to
                    # the alpha / beta core), the gated output, the GLU output a
                    # K half at a time
                    # a buffer per norm: the first's readers are the columns
                    # and the alpha / beta core, the second's the columns
                    # only, and a buffer's fills share one reader count
                    xa1, xa2 = l2([2 * D], G.ASM), l2([2 * D], G.ASM)
                    gyl, hs = l2([2 * DQ], G.ASM), l2([2 * IH], G.ASM)
                    for _j in wave_loop(pw):
                        xn_ch.get(xa1)
                        to_ab.put(xa1)
                        to_asm.get(gyl)
                        xn_ch.get(xa2)
                        for _h in range(HALVES):
                            to_asm.get(hs)
                        arr.broadcast(lambda i: [xa1, gyl, xa2] + [hs] * HALVES)
                    rounds = G.RG_QKV + G.RG_Z + G.RG_O + G.RG_GU + G.RG_DN
                    arr.gather(seg, pw * rounds)
                    arr.feed(seg, pw * G.G_DENSE)

                    def herd(name, at, params=()):
                        return air.herd(
                            [range(1)],
                            name=name,
                            shape=(1,),
                            at=at,
                            params=list(params),
                        )

                    # ---- RMS core
                    with herd("rms", G.RMS_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            # both tokens' x and normed x; x and the norm
                            # weights a round at a time into rw, the only
                            # buffer on its channel
                            x = air.alloc([2 * D], bf16, scope=p)
                            xn = air.alloc([2 * D], bf16, scope=p)
                            rw = air.alloc([XR], bf16, scope=p)
                            inv = air.alloc([16], f32, scope=p)
                            u = air.alloc([rnd], bf16, scope=p)
                            for k in air.sequential(0, NXR):
                                to_rms.get(rw)
                                x_rd(x, rw, k)
                            for j in wave_loop(pw):
                                for nrd in (G.RG_O, G.RG_DN):
                                    rms_inv(x, inv, D)
                                    for k in air.sequential(0, NWR // 2):
                                        to_rms.get(rw)
                                        rms_app(xn, x, inv, rw, k, D)
                                    xn_ch.put(xn)
                                    for jj in air.sequential(0, nrd):
                                        arr.out_y.get(u, indices=[0, RMS])
                                        res2(x, u, jj, D)
                            # the wave's output, once: x comes in once a wave
                            # and goes out once (xn sent on a second channel
                            # took the wrong channel's locks)
                            out_ch.put(x)

                    # ---- alpha / beta, both tokens
                    with herd("ab", G.AB_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x = air.alloc([2 * D], bf16, scope=p)
                            adt = air.alloc([4 * NV], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            ab = air.alloc([4 * NV], f32, scope=p)
                            for j in wave_loop(pw):
                                to_ab.get(x)
                                ab_w.get(blk)
                                adt[:] = blk[0 : 4 * NV]
                                ab_zero2(ab)
                                for rb in air.sequential(0, G.NRB_AB):
                                    for cb in air.sequential(0, NCB_X):
                                        ab_w.get(blk)
                                        ab_mvm2(blk, x, ab, rb, cb, D)
                                ab_act2(ab, adt)
                                if dbg_ch is not None:
                                    dbg_ch.put(x)
                                if dbg_v is not None:
                                    dbg_v.put(ab)
                                with air.ops.branch(j < ngw):
                                    ab_sh[:] = ab[:]

                    # ---- conv (GDN layers), both tokens
                    with herd("conv", G.CONV_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([rnd], bf16, scope=p)
                            hx_ = air.alloc([7 * HD], bf16, scope=p)
                            ho = air.alloc([3 * HD], bf16, scope=p)
                            cv = air.alloc([2 * QKV_G], bf16, scope=p)
                            ab = air.alloc([4 * NV], f32, scope=p)
                            kq = [
                                air.alloc([3 * SLOT], bf16, scope=p) for _ in range(2)
                            ]
                            for _j in wave_loop(ngw):
                                for lo, hi, norm in (
                                    (0, G.CONV_Q, 1),
                                    (G.CONV_Q, G.CONV_K, 0),
                                    (G.CONV_K, G.RG_QKV, None),
                                ):
                                    for rd in air.sequential(lo, hi):
                                        arr.out_y.get(u, indices=[0, CONV])
                                        for k in range(4):
                                            c = rd * 4 + k
                                            cw_in.get(hx_)
                                            conv2(hx_, u, cv, ho, c)
                                            cw_out.put(ho)
                                            if norm is not None:
                                                for t in range(2):
                                                    l2n_t(cv, c, norm, t)
                                ab[:] = ab_sh[:]
                                for hh in air.sequential(0, NV):
                                    for t in range(2):
                                        pack_t(cv, ab, kq[t], hh, t)
                                        kvq_ch.put(kq[t], indices=[0])

                    # ---- the recurrence column: token 0 from si into s1, token
                    # 1 from s1 into s2; each core's outputs,
                    # [token][head][32], to memtile 3
                    n = NV * 32

                    def step_core(r):
                        with herd(f"step{r}", (STEP_COL, 2 + r)) as h:

                            @h.body
                            def _(tx):
                                p = h.private()
                                kq = [
                                    air.alloc([3 * SLOT], bf16, scope=p)
                                    for _ in range(2)
                                ]
                                # the state in, after token 0 (S1), after token
                                # 1 (S2): stepping token 1 back into si sent it
                                # before the step wrote it
                                si = air.alloc([HD * 32], f32, scope=p)
                                s1 = air.alloc([HD * 32], f32, scope=p)
                                s2 = air.alloc([HD * 32], f32, scope=p)
                                og_ = air.alloc([2 * n], bf16, scope=p)
                                for _j in wave_loop(ngw):
                                    for hh in air.sequential(0, NV):
                                        for t in range(2):
                                            kvq_ch.get(kq[t], indices=[r])
                                        s_in.get(si, indices=[r])
                                        gstep(si, s1, kq[0], og_, r, hh, 0)
                                        gstep(s1, s2, kq[1], og_, r, hh, n)
                                        s_out.put(s1, indices=[r])
                                        s_out.put(s2, indices=[r])
                                    og_p.put(og_, indices=[r])

                    for r in range(R):
                        step_core(r)

                    # ---- gate: each token's mixer output gated into z
                    with herd("gate", G.GATE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([rnd], bf16, scope=p)
                            z = air.alloc([2 * DQ], bf16, scope=p)
                            wn = air.alloc([HD], bf16, scope=p)
                            o = air.alloc([DQ], bf16, scope=p)
                            on = air.alloc([DQ], bf16, scope=p)
                            for j in wave_loop(pw):
                                for jj in air.sequential(0, G.RG_Z):
                                    arr.out_y.get(u, indices=[0, GATE])
                                    z_rd2(z, u, jj, DQ)
                                for t in range(2):
                                    for r in range(R):
                                        o_in.get(o[r * NV * 32 : (r + 1) * NV * 32])
                                    o_in.get(wn)
                                    # the norm in every layer (unused in an
                                    # attention layer): wn used only in a
                                    # branch would never be handed back
                                    dn(on, o, wn)
                                    with air.ops.branch(j < ngw):
                                        gsilu(z, on, DQ, t)
                                    with air.ops.branch(j >= ngw):
                                        gsig(z, o, DQ, t)
                                to_asm.put(z)
                                if dbg_z is not None:
                                    dbg_z.put(z)

                    # ---- RoPE (attention layers), both tokens
                    with herd("rope", A.ROPE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([rnd], bf16, scope=p)
                            qkv = air.alloc([2 * NQKV], bf16, scope=p)
                            rw = air.alloc([2 * 3 * DH], bf16, scope=p)
                            q = air.alloc([DQP], bf16, scope=p)
                            k = air.alloc([2 * DK], bf16, scope=p)
                            v = air.alloc([2 * DK], bf16, scope=p)
                            rw_in.get(rw)
                            for jj in air.sequential(0, A.RG_KV):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd2(qkv, u, A.RG_Q + jj, NQKV)
                            for jj in air.sequential(0, A.RG_Q):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd2(qkv, u, jj, NQKV)
                            for jj in air.sequential(0, G.RG_QKV - A.RG_KV - A.RG_Q):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rdrop(u)
                            for t in range(2):
                                rope_t(q, k, v, qkv, rw, NQKV, t)
                                qpack_t(qkv, q, NQKV, t)
                            # K and V first: the units' KV read waits on them
                            k_out.put(k)
                            v_out.put(v)
                            # per token, per KV head: [8 query heads][DH]
                            for t in range(2):
                                for hd in range(NCU * KPU):
                                    o0 = t * NQKV + hd * QHP * DH
                                    q_bc.put(qkv[o0 : o0 + QHP * DH], indices=[0])

                    # ---- the attention units: both tokens against each block
                    s_blk = [
                        [
                            [
                                air.alloc([A.S_BLK], bf16, scope=seg.shared())
                                for _ in range(2)
                            ]
                            for _ in range(KPU)
                        ]
                        for _ in range(NCU)
                    ]

                    def unit(c, col, row):
                        with herd(f"qk{c}", (col, row), params=[Lp]) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                q = [
                                    [
                                        air.alloc([QHP * DH], bf16, scope=p)
                                        for _ in range(2)
                                    ]
                                    for _ in range(KPU)
                                ]
                                # the other unit's heads, each in a buffer of
                                # its own: a buffer filled twice in a row
                                # cannot share the gets' counting lock
                                qx = [
                                    [
                                        air.alloc([QHP * DH], bf16, scope=p)
                                        for _ in range(2)
                                    ]
                                    for _ in range(KPU)
                                ]
                                kb = air.alloc([A.BLK], bf16, scope=p)
                                mm = [
                                    [air.alloc([16], bf16, scope=p) for _ in range(2)]
                                    for _ in range(KPU)
                                ]
                                cc = [
                                    [air.alloc([8], f32, scope=p) for _ in range(2)]
                                    for _ in range(KPU)
                                ]
                                for t in range(2):
                                    for hd in range(NCU * KPU):
                                        mine = hd // KPU == c
                                        dst = (q if mine else qx)[hd % KPU][t]
                                        q_bc.get(dst, indices=[c])
                                for b in air.sequential(0, nb):
                                    for kk in range(KPU):
                                        k_cu.get(kb, indices=[c])
                                        for t in range(2):
                                            qk(
                                                q[kk][t],
                                                kb,
                                                mm[kk][t],
                                                cc[kk][t],
                                                s_blk[c][kk][t],
                                                b,
                                                Lp.as_index() - 1 + t,
                                            )

                        with herd(f"kv{c}", (col, row + 1), params=[Lp]) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([A.BLK], bf16, scope=p)
                                yy = [
                                    [
                                        air.alloc([QHP * DH], f32, scope=p)
                                        for _ in range(2)
                                    ]
                                    for _ in range(KPU)
                                ]
                                ll = [
                                    [air.alloc([16], f32, scope=p) for _ in range(2)]
                                    for _ in range(KPU)
                                ]
                                o = air.alloc([QR * DH], bf16, scope=p)
                                on = air.alloc([QR * DH], bf16, scope=p)
                                ou = air.alloc([2 * UO], bf16, scope=p)
                                for b in air.sequential(0, nb):
                                    for kk in range(KPU):
                                        v_cu.get(vb, indices=[c])
                                        # token 1 first: token 0's mask is
                                        # one key shorter, and its call
                                        # zeroes that key's value in vb
                                        for t in (1, 0):
                                            kv(
                                                s_blk[c][kk][t],
                                                vb,
                                                yy[kk][t],
                                                ll[kk][t],
                                                b,
                                                Lp.as_index() - 1 + t,
                                            )
                                for t in range(2):
                                    for kk in range(KPU):
                                        fin(yy[kk][t], ll[kk][t], o)
                                        o_nat(on, o)
                                        o0 = (kk * 2 + t) * QR * DH
                                        ou[o0 : o0 + QR * DH] = on[:]
                                # [KV head][token][QR x DH]: KV head kk on
                                # og_p[2 c + kk], as a step core's [token][n]
                                # (one run: a strided put's rows are too long
                                # for a core's BD)
                                n2 = 2 * QR * DH
                                for kk in range(KPU):
                                    og_p.put(
                                        ou[kk * n2 : (kk + 1) * n2],
                                        indices=[c * KPU + kk],
                                    )
                                if dbg_a is not None and c == DBG_UNIT:
                                    dbg_a.put(ou)

                    for c, (col, row) in enumerate(A.CU_LOC):
                        unit(c, col, row)

                    # ---- the issuer: each layer's dense weights
                    with herd("issue", ISSUE_AT, params=[lbase]) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            idx = air.alloc([16], i32, scope=p)

                            def issue(j):
                                r_slot(idx, 8, lbase, j)
                                arr.put_weights(
                                    lambda i, hh: [WD[idx[8], i, hh, :, :, :]]
                                )

                            for j in range(pw):
                                issue(j)

                    # ---- GLU, the down's input a K half at a time
                    with herd("glu", G.GLU_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([rnd], bf16, scope=p)
                            hh = air.alloc([2 * IH], bf16, scope=p)
                            nr = G.RG_GU // HALVES
                            for _j in wave_loop(pw):
                                for _h in range(HALVES):
                                    for jj in air.sequential(0, nr):
                                        arr.out_y.get(u, indices=[0, GLU])
                                        glu2(hh, u, jj, IH)
                                    to_asm.put(hh)
                                    if dbg_h is not None:
                                        dbg_h.put(hh)

                    # ---- the GEMV array
                    def phases(j):
                        is_att = j // ngw
                        dest = CONV + is_att * (ROPE - CONV)
                        yield M2.Phase2(D, G.RG_QKV, NCB_X, 1, dest)
                        yield M2.Phase2(D, G.RG_Z, NCB_X, 1, GATE, fresh=False)
                        yield M2.Phase2(DQ, G.RG_O, NCB_O, 1, RMS)
                        yield M2.Phase2(D, G.RG_GU, NCB_X, 1, GLU)
                        yield M2.Phase2(IH, G.RG_DN, NCB_H, HALVES, RMS)

                    arr.herd(2 * IH, None, phases, layers=pw)

                    # ---- host side
                    g0 = w * ngw
                    cw_out.get(CWO[g0 : g0 + ngw, :, 0 : 3 * HD])
                    m_in.put(X[w, pw - 1, :, :])
                    out_ch.get(X[w + 1, pw - 1, :, :])
                    s_l3.put(SI[g0 : g0 + ngw, :, :, :])
                    s_l3o.get(SO[g0 : g0 + ngw, :, :, :])
                    s_l3o1.get(SO1[g0 : g0 + ngw, :, :, :])
                    if dbg_ch is not None:
                        dbg_ch.get(DBG[w, :, :])
                    if dbg_v is not None:
                        dbg_v.get(DBG_V[w, :, :])
                    if dbg_z is not None:
                        dbg_z.get(DBG_Z[w, :, :])
                    if dbg_a is not None:
                        dbg_a.get(DBG_A[w, :])
                    for j in range(pw):
                        lay = w * pw + j
                        is_att = j == pw - 1
                        # in the order the layer takes them: transfers that
                        # share a shim channel run in program order, and the
                        # conv cache waits for the layer's qkv, which waits
                        # for the norm weights
                        m_in.put(M[lay, MISC["w1"][0] : MISC["w2"][1]])
                        for _t in range(2):
                            m_dn.put(M[lay, MISC["dnw"][0] : MISC["dnw"][1]])
                        ab_w.put(M[lay, MISC["adt"][0] : MISC["ab"][1]])
                        if j < ngw:
                            cw_in.put(CWI[g0 + j, :, :])
                        if is_att:
                            rw_in.put(RW[w, :, :])
                            pos = Lp.as_index() - 2
                            k_out.get(KV[w, 0, pos : pos + 2, :])
                            v_out.get(KV[w, 1, pos : pos + 2, :])
                            k_rd.put(KV[w, 0, 0:maxl, :])
                            v_rd.put(KV[w, 1, 0:maxl, :])

        launch.body(body)

    return launch.build(target="npu2")


def layer_weights2(w, l):
    """The decode's layer_weights with the down's blocks in the order the
    verify's cores take them: K half, row group, column block."""
    wd, misc, extra = DEC.layer_weights(w, l)
    ndn = G.RG_DN * NCB_H
    dn = wd[:, :, -ndn:].reshape(NCOL, 2, G.RG_DN, HALVES, NCB_H // HALVES, 2, BB)
    wd[:, :, -ndn:] = dn.transpose(0, 1, 3, 2, 4, 5, 6).reshape(NCOL, 2, ndn, 2, BB)
    return wd, misc, extra


def compile_all(maxl):
    """Kernels and the verify ELF (one wave), here."""
    for k in ("q4k_mvm2", "q38_ffn", "gdn_prep2", "gdn_decode2", "gate2", "router"):
        common.compile_kernel(k)
    for k in ("rope2", "attn_qk", "attn_kv_q36"):
        common.compile_kernel(k, engine=True)
    art = XRTBackend(instance_name="verify", output_format="elf").compile(
        build_module(1, maxl)
    )
    art.output_binary = os.path.abspath(art.output_binary)
    return art, Path(os.getcwd()) / "params.txt"


class Q38Verifier:
    """The verify ELF and the model's buffers. A step runs two tokens at
    positions L - 2 and L - 1, a dispatch per wave; commit() then keeps both,
    or token 0 only."""

    X, WD, M, CWI, SI, KV, RW, CWO, SO, SO1 = range(10)

    def __init__(self, art, params, w, layers, waves, maxl):
        import pyxrt as xrt

        from decode_elf import param_writers
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        self.xrt, self.waves, self.pw = xrt, waves, PER_WAVE
        pw, nl, ng = self.pw, len(layers), NG_WAVE * waves
        assert nl == pw * waves
        dev = xrt.device(0)
        to_dev = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        st = ng * NV * R * HD * 32 * 4
        sizes = [
            (waves + 1) * pw * 2 * D * 2,
            nl * NCOL * G.G_DENSE * R * BB,
            nl * MISC_LEN * 2,
            ng * NCH * 7 * HD * 2,
            st,
            waves * 2 * maxl * DK * 2,
            waves * 2 * 3 * DH * 2,
            ng * NCH * 7 * HD * 2,
            st,
            st,
        ]
        self.bos, self.maps = [], []
        for n in sizes:
            o = xrt.ext.bo(dev, n)
            m = np.frombuffer(o.map(), np.uint8)
            m[:] = 0
            self.bos.append(o)
            self.maps.append(m)
        mp = self.maps
        wd = mp[self.WD].reshape(nl, NCOL, 2, G.G_DENSE, 2, BB)
        misc = mp[self.M].view(bfloat16).reshape(nl, MISC_LEN)
        self.qkn, self.conv_w = [], []
        for i, l in enumerate(layers):
            wd[i], misc[i], extra = layer_weights2(w, l)
            if W.is_attention(l):
                self.qkn.append(extra)
            else:
                cw = G.conv_cache(np.zeros((3, QKV_G), np.float32), extra)
                for b in (self.CWI, self.CWO):
                    mp[b].view(bfloat16).reshape(ng, NCH, 7 * HD)[len(self.conv_w)] = cw
                self.conv_w.append(extra)
        for o in self.bos:
            o.sync(to_dev)
        # each argument's bytes per wave; X's run takes two rows of waves,
        # its input and its output
        self.share = [n // waves for n in sizes]
        self.share[self.X] = sizes[self.X] // (waves + 1)
        self.span = list(self.share)
        self.span[self.X] = 2 * self.share[self.X]
        self.subs = {}
        self.ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
        try:
            kern = xrt.ext.kernel(self.ctx, art.kernel)
        except RuntimeError:
            kern = xrt.ext.kernel(self.ctx, art.kernel.replace("main:", "seg:"))
        self.runs, self.params = [], []
        for wv in range(waves):
            r = xrt.run(kern)
            for i in range(len(self.bos)):
                r.set_arg(i, self._sub(i, wv))
            self.runs.append(r)
            self.params.append(ParameterScratchpad(r, str(params)))
        self.writers = param_writers(params, {"L": len(self.bos)})
        self.queue = True

    def _sub(self, i, wv):
        """Wave wv's share of argument i's buffer."""
        b = self.bos[i]
        key = (id(b), wv)
        if key not in self.subs:
            self.subs[key] = (
                b,
                self.xrt.bo(b, self.span[i], wv * self.share[i]),
            )
        return self.subs[key][1]

    def view(self, i, dt, shape):
        return self.maps[i].view(dt).reshape(shape)

    def step(self, x2, L):
        """x2 [2, D] at positions L - 2, L - 1; returns the layers' outputs
        [waves + 1, pw, 2, D] (row 0 the input)."""
        xrt = self.xrt
        xs = self.view(self.X, bfloat16, (self.waves + 1, self.pw, 2, D))
        xs[0, self.pw - 1] = x2
        rw = self.view(self.RW, bfloat16, (self.waves, 2, 3 * DH))
        for i, (qn, kn) in enumerate(self.qkn):
            for t in range(2):
                rw[i, t] = rope_weights(L - 2 + t, qn, kn).astype(bfloat16)
        for i in (self.X, self.RW):
            self.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for ps in self.params:
            for name, fn in self.writers:
                ps.write(name, np.int32(fn({len(self.bos): L})))
            ps.sync()
        # one context runs its commands in order: with queue, every wave's
        # is submitted before the first one's end is awaited
        for r in self.runs:
            r.start()
            if not self.queue:
                self._wait(r, L)
        if self.queue:
            for r in self.runs:
                self._wait(r, L)
        self.bos[self.X].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return xs.copy()

    def _wait(self, r, L):
        st = r.wait(120000)
        if st != self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"verify dispatch at L={L} ended in {st}")

    def _swap(self, a, b):
        self.bos[a], self.bos[b] = self.bos[b], self.bos[a]
        self.maps[a], self.maps[b] = self.maps[b], self.maps[a]
        for wv, r in enumerate(self.runs):
            r.set_arg(a, self._sub(a, wv))
            r.set_arg(b, self._sub(b, wv))

    def commit(self, both):
        """Keep both tokens of the last step, or token 0 only: then the
        states after token 0 (SO1) and the conv caches' inputs [x1, x2, a]
        from the old cache [x0, x1, x2] and the new [x2, a, b]."""
        xrt = self.xrt
        if not both:
            ng = len(self.conv_w)
            sh = (ng, NCH, 7, HD)
            for b in (self.CWO, self.CWI):
                self.bos[b].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            old, new = self.view(self.CWI, bfloat16, sh), self.view(
                self.CWO, bfloat16, sh
            )
            a = new[:, :, 1].copy()
            new[:, :, 0:2] = old[:, :, 1:3]
            new[:, :, 2] = a
            self.bos[self.CWO].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._swap(self.SO, self.SO1)
        self._swap(self.CWI, self.CWO)
        self._swap(self.SI, self.SO)


def main():
    """One wave (3 GDN layers, an attention layer) against the CPU reference:
    two tokens from empty, both kept; two more, token 0 kept (a rejected
    draft); two more after the rollback. Each step: the wave's output for
    both tokens, and after a step whose tokens are kept, the GDN states."""
    import filelock
    import torch

    from reference import Reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=1)
    ap.add_argument("--maxl", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is a city that")
    a = ap.parse_args()
    w = W.Qwen38Q4nx()
    ids = w.tokenizer().encode(a.prompt).ids
    layers = list(range(PER_WAVE * a.waves))
    art, params = compile_all(a.maxl)
    ok = True

    def chk(name, got, want, tol=0.999):
        nonlocal ok
        c, l2, _ = common.compare(got, want)
        good = c > tol
        ok &= good
        print(f"{name}: corr {c:.6f} rel L2 {l2:.2e} {'ok' if good else 'BAD'}")

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        v = Q38Verifier(art, params, w, layers, a.waves, a.maxl)
        emb = lambda s: np.asarray(w.embed(s)).astype(bfloat16)  # noqa: E731
        # ref_a: what both-kept steps see; ref_b: the rejected branch
        ref_a, ref_b = Reference(w, layers), Reference(w, layers)
        plan = [
            (ids[0:2], 2, True, [ref_a, ref_b], ref_a),
            (ids[2:4], 4, False, [ref_a], ref_a),
            (ids[3:5], 5, True, [ref_b], ref_b),
        ]
        for toks, L, both, refs, ref in plan:
            t0 = time.perf_counter()
            got = v.step(emb(toks), L)
            print(f"L={L} step: {(time.perf_counter() - t0) * 1e3:.1f} ms")
            for r in refs:
                if r is ref_b and L == 5:
                    r.forward(ids[2:3], lm=False)  # the kept token 0 of step 2
                out = r.forward(toks, lm=False)
            want = out.double().numpy()
            for wv in range(a.waves):
                g = got[1 + wv, PER_WAVE - 1].astype(np.float64)
                if wv == a.waves - 1:
                    for t in range(2):
                        chk(f"L={L} token {t} output", g[t], want[t])
            v.commit(both)
            if both:
                xrt = v.xrt
                v.bos[v.SI].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                s_dev = v.view(v.SI, np.float32, (-1, NV, R, HD, 32))
                g = 0
                for l in layers:
                    if W.is_attention(l):
                        continue
                    # [NV][R][HD keys][32 values] -> [NV][HD keys][R x 32 values]
                    sd = s_dev[g].transpose(0, 2, 1, 3).reshape(NV, HD, HD)
                    chk(f"L={L} layer {l} state", sd, ref.state[l].double().numpy())
                    g += 1
        del v
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
