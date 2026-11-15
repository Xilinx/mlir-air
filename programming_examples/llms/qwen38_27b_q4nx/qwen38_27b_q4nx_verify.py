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
#   step cores: token 0's state (S1, for a rejected draft) into the second
#     buffer, token 1's (S2) back into the first; both go out.
#   gate: z [2][DQ]; each token's mixer output in turn, gated into z in place.
#   RoPE: both tokens, q packed per token; the padding rounds dropped.
#   attention units: both tokens' q per KV head against each K/V block, read
#     once; token 0 masked at L - 1, token 1 at L.
#   GLU: the down's input a K half at a time.
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
assert NQKV == DQP


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
    rms_t = k_("q38_ffn", "rms_norm_tw", [i32, i32, i32])
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
    dn = k_("gate2", "dn_norm_wo")
    gsilu = k_("gate2", "gate_silu_tz", [i32, i32])
    gsig = k_("gate2", "gate_sigmoid_wo", [i32, i32])
    rope_rd2 = k_("rope2", "rope_round2", [i32, i32])
    rdrop = k_("rope2", "round_drop")
    rope_t = k_("rope2", "rope_compute_t", [i32, i32])
    qpack_t = k_("rope2", "q_pack_t", [i32, i32])
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    kv = k_("attn_kv_q36", "attn_kv_blk", [i32, i32])
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
    o_in, o_fwd = ch("OIn"), ch("OFwd")
    # each step core's outputs, both tokens, into memtile 3 on one channel
    og_p = ch("OGp", size=[R], **pkt)
    rw_in = ch("RwIn")
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    k_out, v_out = ch("KOut", **pkt), ch("VOut", **pkt)
    k_rd, v_rd = ch("KRd"), ch("VRd")
    k_cu, v_cu = ch("KCu", size=[NCU]), ch("VCu", size=[NCU])
    o_cu = ch("OCu", size=[NCU])
    arr.channels(n_dests=5)

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

                    # ---- memtile 5: the RMS core's x, once a wave; each
                    # layer's two norm weights; and per token [the DeltaNet
                    # norm weight | the mixer output], each one transfer to the
                    # gate core. A GDN layer's output comes from the step
                    # cores, an attention layer's from memtile 3 (the units'
                    # outputs there; with the KV relay this memtile's BDs and
                    # the units' channels would not fit memtile 3). An
                    # attention layer's slot for the norm weight stays unused.
                    # A buffer per kind of layer, as the decode's.
                    WO = HD + DQ
                    x5, w5 = l2([2 * D], 5), l2([2 * D], 5)
                    og_g, og_a = l2([2, WO], 5), l2([2, WO], 5)
                    m_in.get(x5)
                    to_rms.put(x5)
                    for _j in wave_loop(ngw):
                        m_in.get(w5)
                        to_rms.put(w5)
                        for t in range(2):
                            m_in.get(og_g[t, 0:HD])
                        for r in range(R):
                            o0 = HD + r * NV * 32
                            og_p.get(og_g[:, o0 : o0 + NV * 32], indices=[r])
                        for t in range(2):
                            o_in.put(og_g[t, :])
                    m_in.get(w5)
                    to_rms.put(w5)
                    for t in range(2):
                        o_fwd.get(og_a[t, HD:WO])
                    for t in range(2):
                        o_in.put(og_a[t, :])

                    # ---- memtile 3: the attention units' outputs, on to
                    # memtile 5 a token at a time; the KV cache to the units
                    oa3 = l2([2, DQ], 3)
                    for c in range(NCU):
                        o_cu.get(oa3[:, c * UO : (c + 1) * UO], indices=[c])
                    for t in range(2):
                        o_fwd.put(oa3[t, :])
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
                    xa = l2([2 * D], G.ASM)
                    gyl, hs = l2([2 * DQ], G.ASM), l2([2 * IH], G.ASM)
                    for _j in wave_loop(pw):
                        for t in range(2):
                            xn_ch.get(xa[t * D : (t + 1) * D])
                        to_ab.put(xa)
                        to_asm.get(gyl)
                        for t in range(2):
                            xn_ch.get(xa[t * D : (t + 1) * D])
                        for _h in range(HALVES):
                            to_asm.get(hs)
                        arr.broadcast(lambda i: [xa, gyl, xa] + [hs] * HALVES)
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
                            # both tokens' x; the layer's two norm weights;
                            # one token's normed x at a time
                            x = air.alloc([2 * D], bf16, scope=p)
                            wb = air.alloc([2 * D], bf16, scope=p)
                            xn = air.alloc([D], bf16, scope=p)
                            u = air.alloc([rnd], bf16, scope=p)
                            to_rms.get(x)
                            for j in wave_loop(pw):
                                to_rms.get(wb)
                                for k, nrd in ((0, G.RG_O), (1, G.RG_DN)):
                                    for t in range(2):
                                        rms_t(xn, x, wb, D, t, k)
                                        xn_ch.put(xn)
                                    for jj in air.sequential(0, nrd):
                                        arr.out_y.get(u, indices=[0, RMS])
                                        res2(x, u, jj, D)
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

                    # ---- the recurrence column: token 0 from si into so (S1),
                    # token 1 from so back into si (S2); each core's outputs,
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
                                si = air.alloc([HD * 32], f32, scope=p)
                                so = air.alloc([HD * 32], f32, scope=p)
                                og_ = air.alloc([2 * n], bf16, scope=p)
                                for _j in wave_loop(ngw):
                                    for hh in air.sequential(0, NV):
                                        for t in range(2):
                                            kvq_ch.get(kq[t], indices=[r])
                                        s_in.get(si, indices=[r])
                                        gstep(si, so, kq[0], og_, r, hh, 0)
                                        gstep(so, si, kq[1], og_, r, hh, n)
                                        s_out.put(so, indices=[r])
                                        s_out.put(si, indices=[r])
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
                            wo = air.alloc([HD + DQ], bf16, scope=p)
                            on = air.alloc([DQ], bf16, scope=p)
                            for j in wave_loop(pw):
                                for jj in air.sequential(0, G.RG_Z):
                                    arr.out_y.get(u, indices=[0, GATE])
                                    z_rd2(z, u, jj, DQ)
                                for t in range(2):
                                    o_in.get(wo)
                                    with air.ops.branch(j < ngw):
                                        dn(on, wo)
                                        gsilu(z, on, DQ, t)
                                    with air.ops.branch(j >= ngw):
                                        gsig(z, wo, DQ, t)
                                to_asm.put(z)

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
                            # per token, per KV head: [8 query heads][DH]
                            for t in range(2):
                                for hd in range(NCU * KPU):
                                    o0 = t * NQKV + hd * QHP * DH
                                    q_bc.put(qkv[o0 : o0 + QHP * DH], indices=[0])
                            k_out.put(k)
                            v_out.put(v)

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
                                        for t in range(2):
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
                                        o0 = t * UO + kk * QR * DH
                                        ou[o0 : o0 + QR * DH] = on[:]
                                o_cu.put(ou, indices=[c])

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
                    out_ch.get(X[w + 1, :, :, :])
                    s_l3.put(SI[g0 : g0 + ngw, :, :, :])
                    s_l3o.get(SO[g0 : g0 + ngw, :, :, :])
                    s_l3o1.get(SO1[g0 : g0 + ngw, :, :, :])
                    for j in range(pw):
                        lay = w * pw + j
                        is_att = j == pw - 1
                        if j < ngw:
                            cw_in.put(CWI[g0 + j, :, :])
                        m_in.put(M[lay, MISC["w1"][0] : MISC["w2"][1]])
                        if j < ngw:
                            for _t in range(2):
                                m_in.put(M[lay, MISC["dnw"][0] : MISC["dnw"][1]])
                        ab_w.put(M[lay, MISC["adt"][0] : MISC["ab"][1]])
                        if is_att:
                            rw_in.put(RW[w, :, :])
                            pos = Lp.as_index() - 2
                            k_out.get(KV[w, 0, pos : pos + 2, :])
                            v_out.get(KV[w, 1, pos : pos + 2, :])
                            k_rd.put(KV[w, 0, 0:maxl, :])
                            v_rd.put(KV[w, 1, 0:maxl, :])

        launch.body(body)

    return launch.build(target="npu2")
