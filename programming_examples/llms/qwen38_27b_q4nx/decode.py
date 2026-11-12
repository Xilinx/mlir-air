# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B decode: the 64 layers in one dispatch, then the LM head
# (lm_head.py) in a second, on the tile map of Qwen3.6-35B-A3B's decode
# (qwen36_35b_a3b_q4nx/decode.py) with its MoE block replaced by the dense MLP.
#
# The launch repeats once per four layers: three Gated DeltaNet layers, then a
# full-attention one, the model's pattern. Every core that both kinds of layer
# use (the GEMV array, RMS, gate, alpha / beta, GLU, the weight issuer) loops
# over the four with the same transfers each time; what differs between the
# two kinds is computed, not moved. The array sends its first projection to the
# conv core or to the RoPE core by a run-time destination. The cores of one
# kind of layer (conv and the recurrence column; RoPE and the attention column)
# run only in their layers. gdn_layer.py and attn_layer.py hold the two layers
# alone and describe each core.
#
#   GDN layers: the step cores pass their outputs down the column through the
#   L1 they share, and (4, 2) sends the layer's; the gate core applies the
#   DeltaNet norm and the SiLU gate.
#   Attention layers: two units, two KV heads each in turn, q broadcast to both
#   qk cores; each unit sends its output in head order to memtile 3.
#   Memtile 3 hands the gate core the norm weight (a placeholder in an
#   attention layer) and the layer's mixer output.
#   x stays in the RMS core between a wave's layers.
#   The issuer core (5, 4) starts each layer's dense weights, one transfer per
#   shim channel.
#
# The context length L is a run-time scalar: it places this token's K and V
# row, and the attention cores mask by it. Every attention layer reads the
# cache back whole, maxl positions.
import argparse
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import filelock
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
import mlp_block  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402
from rope import rope_weights  # noqa: E402

R, RB, CB, BB, D, I = G.R, G.RB, G.CB, G.BB, G.D, G.I
NCOL, HD, NV, NCH, SLOT = G.NCOL, G.HD, G.NV, G.NCH, G.SLOT
DH, DQ, DK, NCU, KPU = A.DH, A.DQ, A.DK, A.NCU, A.KPU
QR, QHP, DQP = A.QR, A.QHP, A.DQP
NCB_X, NCB_H, NCB_O = G.NCB_X, G.NCB_H, G.NCB_O
PER_WAVE, NG_WAVE = 4, 3  # layers per wave, GDN layers of them
CONV, GATE, RMS, GLU, ROPE = range(5)  # the array's consumers, RoPE last
assert DQ == G.DV  # the gate core and the o phase serve both kinds of layer
# one layout for every layer: a GDN layer's, behind a slot that stands in for
# x in the layers that do not take it from the host
MISC = {"xpad": (0, D), **{n: (a + D, b + D) for n, (a, b) in G.MISC.items()}}
MISC_LEN = G.MISC_LEN + D
ISSUE_AT = (5, 4)
STEP_COL = 4
QKV_G = G.QKV
UO = KPU * QR * DH  # an attention unit's output: its two KV heads' query heads


def build_module(waves, maxl):
    """`waves` x (3 GDN layers, then an attention one), for a context of up to
    `maxl` positions."""
    pw, ngw = PER_WAVE, NG_WAVE
    nl, ng = pw * waves, ngw * waves
    assert maxl % 16 == 0
    nb = maxl // 16
    k_ = lambda o, n, s=(): air.extern(  # noqa: E731
        n, link_with=f"{o}.o", scalars=list(s)
    )
    step = k_("q4k_mvm", "q4k_step", [i32])
    flush = k_("q4k_mvm", "q4k_flush")
    rms_norm = k_("q38_ffn", "rms_norm", [i32])
    res_rd = k_("q38_ffn", "residual_round", [i32])
    glu_s = k_("q38_ffn", "glu_shared_round", [i32])
    conv_k = k_("gdn_prep", "gdn_conv_cache", [i32])
    l2n = k_("gdn_prep", "gdn_l2norm", [i32, i32])
    pack = k_("gdn_prep", "gdn_pack", [i32])
    ab_zero, ab_mvm = k_("gdn_prep", "ab_zero"), k_("gdn_prep", "ab_mvm", [i32, i32])
    ab_act = k_("gdn_prep", "ab_activate")
    r_slot = k_("router", "router_slot", [i32, i32, i32])
    gstep = k_("gdn_decode", "gdn_step_h", [i32, i32])
    z_rd = k_("gate", "z_round", [i32])
    dn = k_("gate", "dn_norm_rhv")
    gsilu = k_("gate", "gate_silu", [i32])
    gsig = k_("gate", "gate_sigmoid", [i32])
    rope_rd = k_("rope_rounds", "rope_round", [i32])
    rope = k_("rope_rounds", "rope_compute", [i32])
    q_pack = k_("rope_rounds", "q_pack")
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    kv = k_("attn_kv_q36", "attn_kv_blk", [i32, i32])
    fin = k_("attn_kv_q36", "attn_kv_fin")
    o_nat = k_("attn_kv_q36", "o_natural")

    arr = MvmArray(
        row_block=RB,
        block_bytes=BB,
        assemble_col=G.ASM,
        acc_words=2 * RB,
        columns=G.COLUMNS,
    )

    # [wave][layer] outputs; a wave takes x from its predecessor's last layer
    X = air.tensor([waves + 1, pw, D], bf16, inout=True)
    # every layer's dense weights in one layout: an attention layer's k, v, q
    # rows are padded to a GDN layer's qkv rows, so the issuer starts them all
    # from one loop. A shim channel's share of a layer is one run of DDR
    # ([column][channel][G_DENSE][2 row blocks]), a linear transfer: strided
    # out of [column][G_DENSE][4 row blocks] it has no shim-legal tiling
    WD = air.tensor([nl, NCOL, 2, G.G_DENSE, 2, BB], ui8)
    M = air.tensor([nl, MISC_LEN], bf16)
    # the conv cache and the GDN state, read from one copy and written to the
    # other; the host swaps them each token, so a layer's reads and writes
    # need no ordering between them
    CWI = air.tensor([ng, NCH, 7 * HD], bf16)
    SI = air.tensor([ng, NV, R, HD * 32], f32)
    KV = air.tensor([waves, 2, maxl, DK], bf16, inout=True)
    RW = air.tensor([waves, 3 * DH], bf16)
    CWO = air.tensor([ng, NCH, 7 * HD], bf16)
    SO = air.tensor([ng, NV, R, HD * 32], f32)
    # the context length, this token's position + 1
    SEQ = air.tensor([], i32, name="L")

    ch = air.channel
    pkt = dict(channel_type="npu_dma_packet")
    m_in, to_rms, xn_ch, xs_ch = ch("MIn"), ch("ToRms"), ch("XN"), ch("XS")
    to_ab, ab_w, out_ch = ch("ToAB"), ch("ABW"), ch("Out")
    to_asm = ch("ToAsm", **pkt)
    cw_in, cw_out = ch("CWIn"), ch("CWOut")
    kvq_ch = ch("KVQ", size=[1], broadcast_shape=[R])
    s_in, s_out = ch("SIn", size=[R]), ch("SOut", size=[R])
    s_l3, s_l3o = ch("SL3"), ch("SL3Out")
    m_dn, o_in = ch("MDn"), ch("OIn")
    # a GDN layer's mixer output into memtile 3, from (4, 2)
    og_ch = ch("OG")
    rw_in = ch("RwIn")
    # q in the units' layout, to both qk cores
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    k_out, v_out = ch("KOut", **pkt), ch("VOut", **pkt)
    k_rd, v_rd = ch("KRd"), ch("VRd")
    k_cu, v_cu = ch("KCu", size=[NCU]), ch("VCu", size=[NCU])
    # each unit's output, into memtile 3
    o_cu = ch("OCu", size=[NCU])
    arr.channels(n_dests=5)

    with air.launch(name="decode", repeat=waves) as launch:

        def body(w):
            # the wave's first layer
            lbase = air.rtp(w * pw)
            Lp = air.rtp(SEQ.value)
            with air.segment(name="seg", params=[lbase, Lp]) as seg:

                @seg.body
                def _():
                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    def wave_loop(n):
                        return air.sequential(0, n)

                    # ---- shared L1
                    ab_sh = air.alloc([2 * NV], f32, scope=seg.shared())

                    # ---- memtile 5: the RMS core's inputs and side outputs.
                    # The memtiles' buffers serve every layer: allocated in the
                    # layer loop, each would be doubled for ping-pong
                    ins = [l2([D], 5) for _ in range(3)]
                    xs5, y5 = l2([D], 5), l2([D], 5)
                    for _j in wave_loop(pw):
                        for b in ins:
                            m_in.get(b)
                            to_rms.put(b)
                        xs_ch.get(xs5)
                        xs_ch.get(y5)
                        to_ab.put(xs5)
                        out_ch.put(y5)

                    # ---- memtile 3: the norm weight and the mixer output to
                    # the gate core: a GDN layer's from the recurrence column,
                    # the attention layer's from its two units; the KV cache to
                    # the units, each its two KV heads in turn per block
                    # A buffer per kind of layer: channels that fill one buffer
                    # take its free slots from one lock, so the units' channels
                    # would hold slots a GDN layer's output is waiting for
                    dnw3 = l2([HD], 3)
                    og_g, og_a = l2([DQ], 3), l2([DQ], 3)
                    for j in range(pw):
                        m_dn.get(dnw3)
                        if j < ngw:
                            og_ch.get(og_g)
                        else:
                            for c in range(NCU):
                                o_cu.get(og_a[c * UO : (c + 1) * UO], indices=[c])
                        o_in.put(dnw3)
                        o_in.put(og_g if j < ngw else og_a)
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

                    # ---- memtile 4: the GDN state, head by head
                    for _j in wave_loop(ngw):
                        for _h in wave_loop(NV):
                            sb = l2([R, HD * 32], 4, f32)
                            s_l3.get(sb)
                            for r in range(R):
                                s_in.put(sb[r, :], indices=[r])
                            air.dealloc(sb)
                            ob = l2([R, HD * 32], 4, f32)
                            for r in range(R):
                                s_out.get(ob[r, :], indices=[r])
                            s_l3o.put(ob)
                            air.dealloc(ob)

                    # ---- the assemble memtile: one buffer for both normed x
                    # of a layer, each broadcast before the next fills it
                    xa = l2([D], G.ASM)
                    gyl, hs = l2([DQ], G.ASM), l2([I], G.ASM)
                    for _j in wave_loop(pw):
                        xn_ch.get(xa)
                        xn_ch.get(xa)
                        to_asm.get(gyl)
                        to_asm.get(hs)
                        arr.broadcast(lambda i: [xa, gyl, xa, hs])
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
                            # x, its slot from the host, the two norm
                            # weights, the normed x of the mixer and of the
                            # MLP. Each buffer appears once a layer on each
                            # channel that carries it: a buffer received or
                            # sent twice a layer on a channel that carries
                            # others, or written after it is sent, shares a
                            # lock pair the core's accesses do not match. The
                            # projections' rows add into x as they arrive
                            x, xin, w1, w2, xab, xn = (
                                air.alloc([D], bf16, scope=p) for _ in range(6)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            for j in wave_loop(pw):
                                # x carries over from layer to layer; only the
                                # wave's first takes it from the host
                                for b in (xin, w1, w2):
                                    to_rms.get(b)
                                with air.ops.branch(j == 0):
                                    x[:] = xin[:]
                                rms_norm(xab, x, w1, D)
                                xn_ch.put(xab)
                                xs_ch.put(xab)
                                for jj in air.sequential(0, G.RG_O):
                                    arr.out_y.get(u, indices=[0, RMS])
                                    res_rd(x, u, jj)
                                rms_norm(xn, x, w2, D)
                                xn_ch.put(xn)
                                for jj in air.sequential(0, G.RG_DN):
                                    arr.out_y.get(u, indices=[0, RMS])
                                    res_rd(x, u, jj)
                                xs_ch.put(x)

                    # ---- alpha / beta (GDN layers; an attention layer's
                    # weights are zero and its result unused)
                    with herd("ab", G.AB_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x = air.alloc([D], bf16, scope=p)
                            adt = air.alloc([4 * NV], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            ab = air.alloc([2 * NV], f32, scope=p)
                            for j in wave_loop(pw):
                                to_ab.get(x)
                                ab_w.get(blk)
                                adt[:] = blk[0 : 4 * NV]
                                ab_zero(ab)
                                for rb in air.sequential(0, G.NRB_AB):
                                    for cb in air.sequential(0, NCB_X):
                                        ab_w.get(blk)
                                        ab_mvm(blk, x, ab, rb, cb)
                                ab_act(ab, adt)
                                with air.ops.branch(j < ngw):
                                    ab_sh[:] = ab[:]

                    # ---- conv (GDN layers)
                    with herd("conv", G.CONV_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hx_ = air.alloc([7 * HD], bf16, scope=p)
                            ho = air.alloc([3 * HD], bf16, scope=p)
                            cv = air.alloc([QKV_G], bf16, scope=p)
                            ab = air.alloc([2 * NV], f32, scope=p)
                            kvq = air.alloc([3 * SLOT], bf16, scope=p)
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
                                            conv_k(hx_, u, cv, ho, c)
                                            cw_out.put(ho)
                                            if norm is not None:
                                                l2n(cv, c, norm)
                                ab[:] = ab_sh[:]
                                for hh in air.sequential(0, NV):
                                    pack(cv, ab, kvq, hh)
                                    kvq_ch.put(kvq, indices=[0])

                    # ---- the recurrence column (GDN layers); outputs passed
                    # down the column through shared L1
                    relay = [
                        air.alloc([(R - r) * NV * 32], bf16, scope=seg.shared())
                        for r in range(1, R)
                    ]

                    def step_core(r):
                        with herd(f"step{r}", (STEP_COL, 2 + r)) as h:

                            @h.body
                            def _(tx):
                                p = h.private()
                                kvq = air.alloc([3 * SLOT], bf16, scope=p)
                                si = air.alloc([HD * 32], f32, scope=p)
                                so = air.alloc([HD * 32], f32, scope=p)
                                n = NV * 32
                                # this core's output, then those below it, sent
                                # on in one piece; the bottom core's alone
                                og_ = air.alloc([(R - r) * n], bf16, scope=p)
                                for _j in wave_loop(ngw):
                                    for hh in air.sequential(0, NV):
                                        kvq_ch.get(kvq, indices=[r])
                                        s_in.get(si, indices=[r])
                                        gstep(si, so, kvq, og_[0:n], r, hh)
                                        s_out.put(so, indices=[r])
                                    # [o_r | o_{r+1} ..] to the core below
                                    if r == R - 1:
                                        relay[r - 1][:] = og_[:]
                                    else:
                                        og_[n:] = relay[r][:]
                                        if r > 0:
                                            relay[r - 1][:] = og_[:]
                                        else:
                                            og_ch.put(og_)

                    for r in range(R):
                        step_core(r)

                    # ---- gate: DeltaNet norm and SiLU gate, or sigmoid gate
                    with herd("gate", G.GATE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            z = air.alloc([DQ], bf16, scope=p)
                            wn = air.alloc([HD], bf16, scope=p)
                            o = air.alloc([DQ], bf16, scope=p)
                            on = air.alloc([DQ], bf16, scope=p)
                            for j in wave_loop(pw):
                                for jj in air.sequential(0, G.RG_Z):
                                    arr.out_y.get(u, indices=[0, GATE])
                                    z_rd(z, u, jj)
                                o_in.get(wn)
                                o_in.get(o)
                                with air.ops.branch(j < ngw):
                                    dn(on, o, wn)
                                    gsilu(on, z, on, DQ)
                                with air.ops.branch(j >= ngw):
                                    gsig(on, z, o, DQ)
                                to_asm.put(on)

                    # ---- RoPE (attention layers)
                    with herd("rope", A.ROPE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            # [q | k | v] and room for the padded rows (a loop
                            # that only takes rounds loses its count); once
                            # roped, q in the units' layout
                            qkv = air.alloc([G.RG_QKV * arr.round], bf16, scope=p)
                            rw = air.alloc([3 * DH], bf16, scope=p)
                            q = air.alloc([DQP], bf16, scope=p)
                            k = air.alloc([DK], bf16, scope=p)
                            v = air.alloc([DK], bf16, scope=p)
                            rw_in.get(rw)
                            for jj in air.sequential(0, A.RG_KV):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, A.RG_Q + jj)
                            for jj in air.sequential(0, A.RG_Q):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, jj)
                            # the padded rows
                            for jj in air.sequential(0, G.RG_QKV - A.RG_KV - A.RG_Q):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, A.RG_KV + A.RG_Q + jj)
                            rope(q, k, v, qkv, rw, 0)
                            q_pack(qkv, q)
                            q_bc.put(qkv[0:DQP], indices=[0])
                            k_out.put(k)
                            v_out.put(v)

                    # ---- the attention units (attention layers), two KV heads
                    # each in turn; each unit's output goes to memtile 3
                    s_blk = [
                        [
                            air.alloc([A.S_BLK], bf16, scope=seg.shared())
                            for _ in range(KPU)
                        ]
                        for _ in range(NCU)
                    ]

                    def unit(c, col, row):
                        with herd(f"qk{c}", (col, row), params=[Lp]) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                qf = air.alloc([DQP], bf16, scope=p)
                                q = [
                                    air.alloc([QHP * DH], bf16, scope=p)
                                    for _ in range(KPU)
                                ]
                                kb = air.alloc([A.BLK], bf16, scope=p)
                                mm = [
                                    air.alloc([16], bf16, scope=p) for _ in range(KPU)
                                ]
                                cc = [air.alloc([8], f32, scope=p) for _ in range(KPU)]
                                # this unit's KV heads' q (a kernel takes one
                                # buffer shape, so not slices of qf)
                                q_bc.get(qf, indices=[c])
                                for kk in range(KPU):
                                    hd = c * KPU + kk
                                    q[kk][:] = qf[hd * QHP * DH : (hd + 1) * QHP * DH]
                                for b in air.sequential(0, nb):
                                    for kk in range(KPU):
                                        k_cu.get(kb, indices=[c])
                                        qk(
                                            q[kk],
                                            kb,
                                            mm[kk],
                                            cc[kk],
                                            s_blk[c][kk],
                                            b,
                                            Lp,
                                        )

                        with herd(f"kv{c}", (col, row + 1), params=[Lp]) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([A.BLK], bf16, scope=p)
                                yy = [
                                    air.alloc([QHP * DH], f32, scope=p)
                                    for _ in range(KPU)
                                ]
                                ll = [air.alloc([16], f32, scope=p) for _ in range(KPU)]
                                o = air.alloc([QR * DH], bf16, scope=p)
                                on = air.alloc([QR * DH], bf16, scope=p)
                                ou = air.alloc([UO], bf16, scope=p)
                                for b in air.sequential(0, nb):
                                    for kk in range(KPU):
                                        v_cu.get(vb, indices=[c])
                                        kv(s_blk[c][kk], vb, yy[kk], ll[kk], b, Lp)
                                for kk in range(KPU):
                                    fin(yy[kk], ll[kk], o)
                                    o_nat(on, o)
                                    ou[kk * QR * DH : (kk + 1) * QR * DH] = on[:]
                                o_cu.put(ou, indices=[c])

                    for c, (col, row) in enumerate(A.CU_LOC):
                        unit(c, col, row)

                    # ---- the issuer: each layer's dense weights, one transfer
                    # on each of the array's shim channels. A queued task reads
                    # its descriptor when it starts, so each layer has its own
                    # (the loop unrolled): one descriptor patched for every
                    # layer would send the queued layers the last one's address
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

                    # ---- GLU
                    with herd("glu", G.GLU_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hsl = air.alloc([I], bf16, scope=p)
                            for _j in wave_loop(pw):
                                for jj in air.sequential(0, G.RG_GU):
                                    arr.out_y.get(u, indices=[0, GLU])
                                    glu_s(hsl, u, jj)
                                to_asm.put(hsl)

                    # ---- the GEMV array: a GDN layer's qkv goes to the conv
                    # core, an attention layer's k, v, q to the RoPE core
                    def phases(j):
                        is_att = j // ngw
                        yield Phase(
                            D,
                            G.RG_QKV * NCB_X // 2,
                            NCB_X,
                            step,
                            CONV + is_att * (ROPE - CONV),
                        )
                        yield Phase(
                            D, G.RG_Z * NCB_X // 2, NCB_X, step, GATE, fresh=False
                        )
                        yield Phase(DQ, G.RG_O * NCB_O // 2, NCB_O, step, RMS)
                        yield Phase(D, G.RG_GU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, G.RG_DN * NCB_H // 2, NCB_H, step, RMS)

                    # one x buffer: every phase after z brings its own input
                    arr.herd(I, flush, phases, layers=pw)

                    # ---- host side. Per wave: the conv cache outputs, x and
                    # the layers' outputs, and the GDN states, each one run of
                    # DDR. Then each layer's transfers; a conv cache goes per
                    # layer, since its shim channel may also carry the alpha /
                    # beta weights, which wait on that layer's mixer
                    g0 = w * ngw
                    cw_out.get(CWO[g0 : g0 + ngw, :, 0 : 3 * HD])
                    # x in before the outputs: with the wave a loop index the
                    # two regions of X are not known apart, and a read after
                    # the write would wait for the whole wave
                    m_in.put(X[w, pw - 1, :])
                    out_ch.get(X[w + 1, :, :])
                    s_l3.put(SI[g0 : g0 + ngw, :, :, :])
                    s_l3o.get(SO[g0 : g0 + ngw, :, :, :])
                    for j in range(pw):
                        lay = w * pw + j
                        is_att = j == pw - 1
                        if j < ngw:
                            cw_in.put(CWI[g0 + j, :, :])
                        if j == 0:
                            m_in.put(M[lay, MISC["w1"][0] : MISC["w2"][1]])
                        else:
                            m_in.put(M[lay, 0 : MISC["w2"][1]])
                        ab_w.put(M[lay, MISC["adt"][0] : MISC["ab"][1]])
                        m_dn.put(M[lay, MISC["dnw"][0] : MISC["dnw"][1]])
                        if is_att:
                            rw_in.put(RW[w, :])
                            pos = Lp.as_index() - 1
                            k_out.get(KV[w, 0, pos, :])
                            v_out.get(KV[w, 1, pos, :])
                            k_rd.put(KV[w, 0, :, :])
                            v_rd.put(KV[w, 1, :, :])

        launch.body(body)

    return launch.build(target="npu2")


def layer_weights(w, l):
    """Layer l's (dense weights [NCOL, 2, G_DENSE, 2, BB], misc [MISC_LEN]), and
    its (q norm, k norm) or conv weights [4, QKV]."""
    misc = np.zeros(MISC_LEN, bfloat16)
    if W.is_attention(l):
        wd, m, qn, kn = A.weights(w, l)
        # k, v, q padded with zero groups to a GDN layer's qkv
        nkvq = (A.RG_KV + A.RG_Q) * NCB_X
        pad = np.zeros((NCOL, (G.RG_QKV - A.RG_KV - A.RG_Q) * NCB_X, R, BB), np.uint8)
        wd = np.concatenate([wd[:, :nkvq], pad, wd[:, nkvq:]], axis=1)
        for n in ("w1", "w2"):
            misc[MISC[n][0] : MISC[n][1]] = m[A.MISC[n][0] : A.MISC[n][1]]
        extra = (qn, kn)
    else:
        wqkv, wz, wo, m, cw = G.weights(w, l)
        gu, dn = mlp_block.weights(w, l)
        wd = np.concatenate(
            [t.reshape(NCOL, -1, R, BB) for t in (wqkv, wz, wo, gu, dn)], axis=1
        )
        misc[D:] = m
        extra = cw
    assert wd.shape == (NCOL, G.G_DENSE, R, BB), wd.shape
    # [column][channel][G_DENSE][2 row blocks]: channel h carries row blocks
    # 2h and 2h + 1
    wd = wd.reshape(NCOL, G.G_DENSE, 2, 2, BB).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(wd), misc, extra


class Q38Decoder:
    """The layers' ELF and the LM head's, their buffers, and a run of each.
    Per token the host writes x, the attention layers' RoPE rows and L, and
    swaps the conv caches' and the GDN states' in and out buffers.

    The dense weights (WD, about 14 GB for the model) go into their buffer
    layer by layer, without a host copy of the whole."""

    # the layers' arguments, in build_module's order
    X, WD, M, CWI, SI, KV, RW, CWO, SO = range(9)

    def __init__(self, arts, w, layers, waves, maxl, lm_args, params):
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

        def bo_of(nbytes):
            o = xrt.ext.bo(dev, nbytes)
            m = np.frombuffer(o.map(), np.uint8)
            m[:] = 0
            return o, m

        def bo(t):
            b = np.ascontiguousarray(t).view(np.uint8).ravel()
            o, m = bo_of(b.nbytes)
            m[:] = b
            o.sync(to_dev)
            return o

        sizes = {
            self.X: (waves + 1) * pw * D * 2,
            self.WD: nl * NCOL * G.G_DENSE * R * BB,
            self.M: nl * MISC_LEN * 2,
            self.CWI: ng * NCH * 7 * HD * 2,
            self.SI: ng * NV * R * HD * 32 * 4,
            self.KV: waves * 2 * maxl * DK * 2,
            self.RW: waves * 3 * DH * 2,
            self.CWO: ng * NCH * 7 * HD * 2,
            self.SO: ng * NV * R * HD * 32 * 4,
        }
        self.bos, maps = [], []
        for i in range(9):
            o, m = bo_of(sizes[i])
            self.bos.append(o)
            maps.append(m)
        wd = maps[self.WD].reshape(nl, NCOL, 2, G.G_DENSE, 2, BB)
        misc = maps[self.M].view(bfloat16).reshape(nl, MISC_LEN)
        cwi = maps[self.CWI].view(bfloat16).reshape(ng, NCH, 7 * HD)
        cwo = maps[self.CWO].view(bfloat16).reshape(ng, NCH, 7 * HD)
        self.qkn, self.conv_w = [], []
        for i, l in enumerate(layers):
            wd[i], misc[i], extra = layer_weights(w, l)
            if W.is_attention(l):
                self.qkn.append(extra)
            else:
                cw = G.conv_cache(np.zeros((3, QKV_G), np.float32), extra)
                cwi[len(self.conv_w)] = cw
                cwo[len(self.conv_w)] = cw
                self.conv_w.append(extra)
        for o in self.bos:
            o.sync(to_dev)

        def kernel(art):
            ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
            try:
                return ctx, xrt.ext.kernel(ctx, art.kernel)
            except RuntimeError:  # a repeated launch's kernel sits in the segment
                return ctx, xrt.ext.kernel(ctx, art.kernel.replace("main:", "seg:"))

        self.ctx, kern = kernel(arts[0])
        self.run = xrt.run(kern)
        for i, b in enumerate(self.bos):
            self.run.set_arg(i, b)
        self.params = ParameterScratchpad(self.run, str(params))
        self.writers = param_writers(params, {"L": len(self.bos)})
        self.xmap = maps[self.X].view(bfloat16).reshape(waves + 1, pw, D)
        self.rwmap = maps[self.RW].view(bfloat16).reshape(waves, 3 * DH)
        xsub = xrt.bo(self.bos[self.X], D * 2, (waves * pw + pw - 1) * D * 2)
        self.lbos = [xsub] + [bo(t) for t in lm_args]
        self.lctx, lkern = kernel(arts[1])
        self.lrun = xrt.run(lkern)
        for i, b in enumerate(self.lbos):
            self.lrun.set_arg(i, b)

    def _set(self, x, L):
        xrt = self.xrt
        self.xmap[0, self.pw - 1] = x
        for i, (qn, kn) in enumerate(self.qkn):
            self.rwmap[i] = rope_weights(L - 1, qn, kn).astype(bfloat16)
        for i in (self.X, self.RW):
            self.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for name, fn in self.writers:
            self.params.write(name, np.int32(fn({len(self.bos): L})))
        self.params.sync()

    def _swap(self):
        """This token's outputs are the next one's inputs."""
        for a, b in ((self.CWI, self.CWO), (self.SI, self.SO)):
            self.bos[a], self.bos[b] = self.bos[b], self.bos[a]
            self.run.set_arg(a, self.bos[a])
            self.run.set_arg(b, self.bos[b])

    def _go(self, runs, L):
        import lm_head as LM

        xrt = self.xrt
        for r in runs:
            r.start()
            st = r.wait(120000)
            if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch at L={L} ended in {st}")
        yb = self.lbos[-1]
        yb.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return LM.unpack_logits(np.frombuffer(yb.map(), bfloat16)).astype(np.float32)

    def seed(self, pf):
        """A prefill's state as the next token's inputs: each attention
        layer's K and V rows, each Gated DeltaNet layer's conv cache and
        state."""
        xrt, n, nv = self.xrt, pf.n, len(self.conv_w)
        mp = lambda i, dt: np.frombuffer(self.bos[i].map(), dt)  # noqa: E731
        kv = mp(self.KV, bfloat16).reshape(self.waves, 2, -1, DK)
        cw = mp(self.CWI, bfloat16).reshape(nv, NCH, 7 * HD)
        s = mp(self.SI, np.float32).reshape(nv, NV, R, HD * 32)
        g = 0
        for j in range(len(pf.layers)):
            if j in pf.kv:
                for h in range(2):
                    kv[j // self.pw, h, :n] = np.concatenate(pf.kv[j][h])
            else:
                cw[g] = G.conv_cache(pf.conv[j], self.conv_w[g])
                s[g] = G.state_chunks(pf.state[j])
                g += 1
        for i in (self.KV, self.CWI, self.SI):
            self.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def reset(self):
        """A new sequence: every Gated DeltaNet layer's state and conv history
        zero, in both buffers of each pair; the conv weights stay. The KV rows
        need no reset, as a step reads only rows below its position."""
        nv, to = len(self.conv_w), self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        for i in (self.CWI, self.CWO):
            cw = np.frombuffer(self.bos[i].map(), bfloat16).reshape(nv, NCH, 7, HD)
            cw[:, :, :3] = 0
            self.bos[i].sync(to)
        for i in (self.SI, self.SO):
            np.frombuffer(self.bos[i].map(), np.float32)[:] = 0
            self.bos[i].sync(to)

    def head(self, x, L):
        """The LM head alone on the last layer's output x, at position L - 1."""
        self.xmap[self.waves, self.pw - 1] = x
        self.bos[self.X].sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return self._go((self.lrun,), L)

    def step(self, x, L):
        """Decode x at position L - 1; returns the logits."""
        self._set(x, L)
        logits = self._go((self.run, self.lrun), L)
        self._swap()
        return logits

    def bench(self, x, L, n):
        """n tokens at context length L, the layers' run then the LM head's;
        the state is not carried (the same inputs every time)."""
        self._set(x, L)
        t = []
        done = self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        for _ in range(n + 1):
            t0 = time.perf_counter()
            self.run.start()
            st = self.run.wait(120000)
            t1 = time.perf_counter()
            if st != done:
                raise RuntimeError(f"layers' dispatch at L={L} ended in {st}")
            self.lrun.start()
            st = self.lrun.wait(120000)
            t.append((t1 - t0, time.perf_counter() - t1))
            if st != done:
                raise RuntimeError(f"LM head's dispatch at L={L} ended in {st}")
        t = np.array(t[1:]) * 1e3
        tot = t.sum(1)
        med = float(np.median(tot))
        print(
            f"L={L}, {n} tokens: layers {np.median(t[:, 0]):.3f} ms + LM head"
            f" {np.median(t[:, 1]):.3f} ms = {med:.3f} ms median"
        )
        print(
            f"[bench] decode ctx={L} median {med:.2f} ms min {tot.min():.2f} ms "
            f"({1e3 / med:.2f} tok/s)"
        )


def compile_all(waves, maxl):
    """Kernels, the layers' ELF here and the LM head's in lm/."""
    import lm_head as LM

    for k in ("q4k_mvm", "q38_ffn", "gate", "router", "gdn_prep", "gdn_decode"):
        common.compile_kernel(k)
    for k in ("rope_rounds", "attn_qk", "attn_kv_q36"):
        common.compile_kernel(k, engine=True)
    art = XRTBackend(instance_name="decode", output_format="elf").compile(
        build_module(waves, maxl)
    )
    art.output_binary = os.path.abspath(art.output_binary)
    here = os.getcwd()
    os.makedirs("lm", exist_ok=True)
    os.chdir("lm")
    lart = LM.compile_lm_head()
    lart.output_binary = os.path.abspath(lart.output_binary)
    os.chdir(here)
    return art, lart, Path(here) / "params.txt"


def load(w, maxl, waves=W.N_LAYERS // PER_WAVE, arts=None):
    """The decode built for context maxl (compile_all, here) and loaded: a
    Q38Decoder over the first PER_WAVE * waves layers."""
    import lm_head as LM

    arts = arts or compile_all(waves, maxl)
    layers = list(range(PER_WAVE * waves))
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    y = np.zeros((LM.NCOL, LM.ROUNDS, LM.NROW * LM.RB), bfloat16)
    lm_args = [nw, LM.pack_weights(w), y]
    return Q38Decoder(arts, w, layers, waves, maxl, lm_args, arts[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--prompt", default="The capital of France")
    ap.add_argument(
        "--prompt-len", type=int, default=0, help="repeat the prompt to N tokens"
    )
    ap.add_argument(
        "--fused-dir",
        type=Path,
        default=os.environ.get("FUSED_PREFILL_DIR") or None,
        help="prefill on this fused prefill build (fused_build.py)",
    )
    ap.add_argument(
        "--warm", action="store_true", help="prefill once untimed before timing"
    )
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate")
    ap.add_argument("--maxl", type=int, default=64, help="context the build holds")
    ap.add_argument("--waves", type=int, default=W.N_LAYERS // PER_WAVE)
    ap.add_argument("--no-ref", action="store_true", help="skip the CPU reference")
    ap.add_argument("--bench", type=int, default=0, help="then time n tokens")
    ap.add_argument("--bench-l", type=int, default=0, help="at this context length")
    a = ap.parse_args()
    if a.print_module_only:
        print(build_module(a.waves, a.maxl))
        return 0
    if a.compile_only:
        compile_all(a.waves, a.maxl)
        print("Compilation passed.")
        return 0
    from reference import Reference

    w = W.Qwen38Q4nx()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    if a.prompt_len:
        ids = (ids * (a.prompt_len // len(ids) + 1))[: a.prompt_len]
    pw = PER_WAVE
    layers = list(range(pw * a.waves))
    assert len(ids) + a.gen <= a.maxl
    # FLM prefills the whole prompt and takes the first token from the LM head
    # on its last row; the fused prefill takes prompts up to its build's length.
    # A bench times the decode alone.
    fused = a.fused_dir is not None and len(layers) == W.N_LAYERS and not a.bench
    if fused:
        import prefill as P

        manifest = a.fused_dir / "manifest.json"
        if not manifest.is_file():
            sys.exit(f"{a.fused_dir}: no fused prefill build (fused_build.py)")
        fused = len(ids) <= json.loads(manifest.read_text())["max_len"]
    arts = compile_all(a.waves, a.maxl)
    # the device is held from the first context to the last dispatch
    lock = filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock"))
    with lock:
        t0 = time.time()
        npu = load(w, a.maxl, a.waves, arts)
        print(f"weights loaded in {time.time() - t0:.0f}s")

        def embed(t):
            return np.asarray(w.embed([t])[0]).astype(bfloat16)

        ref = None if a.no_ref else Reference(w, layers=layers)
        toks, ok, times = list(ids), True, []

        def check(i, got, new):
            nonlocal ok
            nxt = int(np.argmax(got))
            if ref is not None:
                want = ref.forward(new).float().numpy()
                ok &= nxt == int(np.argmax(want))
                print(
                    f"L={i + 1}: next {nxt} (reference {int(np.argmax(want))}),"
                    f" logits corr {G.corr(got, want)[0]:.4f}"
                )
            return nxt

        first, t_prefill = 0, 0.0
        if fused:
            pf = P.Qwen38Prefill(a.fused_dir, w, layers)
            if a.warm:
                pf.prefill(ids)
            t0 = time.perf_counter()
            x = pf.prefill(ids)
            npu.seed(pf)
            got = npu.head(x[-1].astype(bfloat16), len(ids))
            t_prefill = time.perf_counter() - t0
            first = len(ids)
            print(f"prefill of {first} tokens: {t_prefill:.3f}s")
            nxt = check(first - 1, got, ids)
            if a.gen > 0:
                toks.append(nxt)
        # without the prefill the prompt goes through the decode a token at a time
        for i in range(first, len(ids) + a.gen - 1):
            t0 = time.perf_counter()
            got = npu.step(embed(toks[i]), i + 1)
            times.append(time.perf_counter() - t0)
            nxt = check(i, got, [toks[i]])
            if i + 1 >= len(ids):
                toks.append(nxt)
        print(repr(a.prompt), "->", repr(tok.decode(toks[len(ids) :])))
        n_prompt = max(0, len(ids) - first)
        print(f"Inference: prompt_len={len(ids)}, n_tokens={len(toks) - len(ids)}")
        print(f"Time to first token (TTFT): {t_prefill + sum(times[:n_prompt]):.3f}s")
        if len(times) > n_prompt:
            gen = np.array(times[n_prompt:]) * 1e3
            print(
                f"Decode: median {np.median(gen):.2f} ms ({1e3 / np.median(gen):.2f} tok/s)"
            )
        if a.bench:
            npu.bench(embed(toks[-1]), a.bench_l or len(toks), a.bench)
        del npu
        if fused:
            del pf
    if ref is not None:
        print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
