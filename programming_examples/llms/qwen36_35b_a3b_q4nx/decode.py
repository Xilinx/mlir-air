# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.6-35B-A3B decode on FastFlowLM's tile map: the 40 layers in one
# dispatch, then the LM head (lm_head.py) in a second.
#
# The launch repeats once per four layers: three Gated DeltaNet layers, then a
# full-attention one, the model's pattern. Every core that both kinds of layer
# use (the GEMV array, RMS, gate, alpha/beta and router GEMV, router, GLU) loops
# over the four with the same transfers each time; what differs between the
# two kinds is computed, not moved. The array sends its first projection to the
# conv core or to the RoPE core by a run-time destination, as FastFlowLM's
# projection cores pick theirs. The cores of one kind of layer (conv and the
# recurrence column; RoPE and the attention column) run only in their layers.
# gdn_layer.py and attn_layer.py hold the two layers alone and describe each
# core.
#
#   GDN layers: the step cores pass their outputs down the column through the
#   L1 they share, and (4, 2) sends the layer's; the gate core applies the
#   DeltaNet norm and the SiLU gate.
#   Attention layers: the units' outputs, in head order, reach (3, 3) the same
#   way. Both kinds send their mixer output to memtile 3 on one packet channel,
#   and memtile 3 hands the gate core the norm weight (a placeholder in an
#   attention layer) and the output.
#   x stays in the RMS core between a wave's layers.
#   The router core starts each layer's dense weights and its experts. An
#   expert's gate/up and down rows are one run of DDR, one transfer per shim
#   channel per turn.
#
# The context length L is a run-time scalar: it places this token's K and V
# row, and the attention cores mask by it. Every attention layer reads the
# cache back whole, maxl positions.
import argparse
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
import moe_block  # noqa: E402
import qwen36_35b_a3b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402
from rope import rope_weights  # noqa: E402

R, RB, CB, BB, D, I, E = G.R, G.RB, G.CB, G.BB, G.D, G.I, G.E
NCOL, TURNS, HD, NV, NCH, SLOT = G.NCOL, G.TURNS, G.HD, G.NV, G.NCH, G.SLOT
DH, DQ, DK, QKV, NCU, QH = A.DH, A.DQ, A.DK, A.QKV, A.NCU, A.QH
NCB_X, NCB_H, NCB_O = G.NCB_X, G.NCB_H, G.NCB_O
PER_WAVE, NG_WAVE = 4, 3  # layers per wave, GDN layers of them
CONV, GATE, RMS, GLU, ROPE = range(5)  # the array's consumers, RoPE last
# an expert's rows: gate/up, then down, per column
G_EXP = G.RG_GU * NCB_X + G.RG_DN * NCB_H
# one layout for every layer: a GDN layer's, behind a slot that stands in for
# x in the layers that do not take it from the host
MISC = {"xpad": (0, D), **{n: (a + D, b + D) for n, (a, b) in G.MISC.items()}}
MISC_LEN = G.MISC_LEN + D
ROPE_AT = A.ROPE_AT
STEP_COL = 4
QKV_G = G.QKV


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
    rms_norm = k_("moe_glu", "rms_norm", [i32])
    res_add = k_("moe_glu", "residual_add", [i32])
    copy_r = k_("moe_glu", "copy_round", [i32])
    sg_dot = k_("moe_glu", "shared_gate_dot", [i32])
    sg_apply = k_("moe_glu", "shared_gate_apply", [i32])
    glu_s = k_("moe_glu", "glu_shared_round", [i32])
    glu_e = k_("moe_glu", "glu_expert_round", [i32])
    acc_r = k_("moe_glu", "moe_acc_round", [i32, i32])
    copy_up = k_("moe_glu", "copy_up")
    acc_zero = k_("moe_glu", "acc_zero", [i32])
    out_k = k_("moe_glu", "moe_out", [i32])
    conv_k = k_("gdn_prep", "gdn_conv_cache", [i32])
    l2n = k_("gdn_prep", "gdn_l2norm", [i32, i32])
    pack = k_("gdn_prep", "gdn_pack", [i32])
    ab_zero, ab_mvm = k_("ab_router", "ab_zero"), k_("ab_router", "ab_mvm", [i32, i32])
    ab_act = k_("ab_router", "ab_activate")
    r_zero = k_("ab_router", "router_zero")
    r_mvm = k_("ab_router", "router_mvm", [i32, i32])
    r_topk = k_("router", "router_topk_at", [i32, i32])
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

    # every layer's experts, gate/up then down: expert e of layer l is E l + e
    WE = air.tensor([nl * E, G_EXP, R, BB], ui8)
    # [wave][layer] outputs; a wave takes x from its predecessor's last layer
    X = air.tensor([waves + 1, pw, D], bf16, inout=True)
    # every layer's dense weights in one layout: an attention layer's k, v, q
    # rows are padded to a GDN layer's qkv rows, so the router starts them all
    # from one loop (its program memory holds one copy of the issue code)
    WD = air.tensor([nl, NCOL, G.G_DENSE, R, BB], ui8)
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
    to_glu, glu_y = ch("ToGlu"), ch("GluY")
    to_asm = ch("ToAsm", **pkt)
    cw_in, cw_out = ch("CWIn"), ch("CWOut")
    kvq_ch = ch("KVQ", size=[1], broadcast_shape=[R])
    s_in, s_out = ch("SIn", size=[R]), ch("SOut", size=[R])
    s_l3, s_l3o = ch("SL3"), ch("SL3Out")
    m_dn, o_in = ch("MDn"), ch("OIn")
    # a layer's mixer output into memtile 3: GDN from (4, 2), attention from
    # (3, 3); one per layer, in layer order
    og_ch = ch("OG", **pkt)
    rw_in = ch("RwIn")
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    k_out, v_out = ch("KOut", **pkt), ch("VOut", **pkt)
    kv_rd = ch("KVRd")
    k_cu, v_cu = ch("KCu", size=[NCU]), ch("VCu", size=[NCU])
    arr.channels(n_dests=5)

    with air.launch(name="decode", repeat=waves) as launch:

        def body(w):
            # the wave's first expert row in WE, and its first layer
            ebase = air.rtp(w * (pw * E))
            lbase = air.rtp(w * pw)
            Lp = air.rtp(SEQ.value)
            with air.segment(name="seg", params=[ebase, lbase, Lp]) as seg:

                @seg.body
                def _():
                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    def wave_loop(n):
                        return air.sequential(0, n)

                    # ---- shared L1
                    ab_sh = air.alloc([2 * NV], f32, scope=seg.shared())
                    logits = air.alloc([E], f32, scope=seg.shared())
                    scores = air.alloc([16], f32, scope=seg.shared())
                    turn_go = air.alloc([16], f32, scope=seg.shared())

                    # ---- memtile 5: the RMS core's inputs and side outputs.
                    # The memtiles' buffers serve every layer: allocated in the
                    # layer loop, each would be doubled for ping-pong
                    ins = [l2([D], 5) for _ in range(4)]
                    moe5 = l2([D], 5)
                    xs5, x25, sd5, y5 = (l2([D], 5) for _ in range(4))
                    for _j in wave_loop(pw):
                        for b in ins:
                            m_in.get(b)
                            to_rms.put(b)
                        glu_y.get(moe5)
                        to_rms.put(moe5)
                        for b in (xs5, x25, sd5, y5):
                            xs_ch.get(b)
                        to_ab.put(xs5)
                        to_ab.put(x25)
                        to_glu.put(sd5)
                        out_ch.put(y5)

                    # ---- memtile 3: the norm weight and the mixer output to
                    # the gate core; the KV cache to the attention units
                    dnw3, og = l2([HD], 3), l2([DQ], 3)
                    for _j in wave_loop(pw):
                        m_dn.get(dnw3)
                        og_ch.get(og)
                        o_in.put(dnw3)
                        o_in.put(og)
                    kvm = l2([2, 16, DK], 3)
                    km, vm = kvm[0, :, :], kvm[1, :, :]
                    for _b in wave_loop(nb):
                        kv_rd.get(kvm)
                        for u in range(NCU):
                            kh = km.reshape(16, DK // 8, 8)[
                                :, u * DH // 8 : (u + 1) * DH // 8, :
                            ]
                            k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                            vh = vm.reshape(2, 8, DK // 8, 8)[
                                :, :, u * DH // 8 : (u + 1) * DH // 8, :
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

                    # ---- the assemble memtile
                    # one buffer for both normed x of a layer and one for both
                    # turns' GLU outputs: each is broadcast before the next
                    # fills it (and a buffer filled twice a layer keeps the
                    # loop from being doubled for ping-pong, which would take
                    # more locks than the memtile has)
                    xa = l2([D], G.ASM)
                    gyl, hs = l2([DQ], G.ASM), l2([I], G.ASM)
                    hx = l2([NCOL, I], G.ASM)
                    for _j in wave_loop(pw):
                        xn_ch.get(xa)
                        xn_ch.get(xa)
                        to_asm.get(gyl)
                        to_asm.get(hs)
                        for _t in range(TURNS):
                            to_asm.get(hx)
                        arr.broadcast(lambda i: [xa, gyl, xa, hs] + [hx[i, :]] * TURNS)
                    rd_gdn = G.RG_QKV + G.RG_Z + G.RG_O
                    rd_moe = G.RG_SGU + G.RG_SDN + TURNS * (G.RG_GU + G.RG_DN)
                    arr.gather(seg, pw * (rd_gdn + rd_moe))
                    arr.feed(
                        seg,
                        pw * (G.G_DENSE + TURNS * G_EXP),
                    )

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
                            x, w1, w2, wsg, xn, xn2, mix, sdn, moe, y, xin = (
                                air.alloc([D], bf16, scope=p) for _ in range(11)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            gsc = air.alloc([16], bf16, scope=p)
                            # x carries over from layer to layer; only the
                            # wave's first takes it from the host
                            for j in wave_loop(pw):
                                for b in (xin, w1, w2, wsg):
                                    to_rms.get(b)
                                with air.ops.branch(j == 0):
                                    x[:] = xin[:]
                                rms_norm(xn, x, w1, D)
                                xn_ch.put(xn)
                                xs_ch.put(xn)
                                for jj in air.sequential(0, G.RG_O):
                                    arr.out_y.get(u, indices=[0, RMS])
                                    copy_r(mix, u, jj)
                                res_add(y, x, mix, D)
                                rms_norm(xn2, x, w2, D)
                                # done with xn2 before it goes
                                sg_dot(xn2, wsg, gsc, D)
                                xn_ch.put(xn2)
                                xs_ch.put(xn2)
                                for jj in air.sequential(0, G.RG_SDN):
                                    arr.out_y.get(u, indices=[0, RMS])
                                    copy_r(sdn, u, jj)
                                sg_apply(gsc, sdn, D)
                                xs_ch.put(sdn)
                                to_rms.get(moe)
                                res_add(y, x, moe, D)
                                xs_ch.put(y)

                    # ---- alpha / beta (GDN layers) and the router GEMV
                    with herd("ab", G.AB_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x = air.alloc([D], bf16, scope=p)
                            adt = air.alloc([4 * NV], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            ab = air.alloc([2 * NV], f32, scope=p)
                            lg = air.alloc([E], f32, scope=p)
                            for j in wave_loop(pw):
                                to_ab.get(x)
                                ab_w.get(blk)
                                adt[:] = blk[0 : 4 * NV]
                                ab_zero(ab)
                                for rb in air.sequential(0, 2):
                                    for cb in air.sequential(0, NCB_X):
                                        ab_w.get(blk)
                                        ab_mvm(blk, x, ab, rb, cb)
                                ab_act(ab, adt)
                                with air.ops.branch(j < ngw):
                                    ab_sh[:] = ab[:]
                                to_ab.get(x)
                                r_zero(lg)
                                for rb in air.sequential(0, G.NRB_R):
                                    for cb in air.sequential(0, NCB_X):
                                        ab_w.get(blk)
                                        r_mvm(blk, x, lg, rb, cb)
                                logits[:] = lg[:]

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
                                    (0, 4, 1),
                                    (4, 8, 0),
                                    (8, 16, None),
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
                                o = air.alloc([NV * 32], bf16, scope=p)
                                n = NV * 32
                                # this core's output and those below it, sent
                                # on in one piece
                                if r < R - 1:
                                    og = air.alloc([(R - r) * n], bf16, scope=p)
                                for _j in wave_loop(ngw):
                                    for hh in air.sequential(0, NV):
                                        kvq_ch.get(kvq, indices=[r])
                                        s_in.get(si, indices=[r])
                                        gstep(si, so, kvq, o, r, hh)
                                        s_out.put(so, indices=[r])
                                    # [o_r | o_{r+1} ..] to the core below
                                    if r == R - 1:
                                        relay[r - 1][:] = o[:]
                                    else:
                                        og[0:n] = o[:]
                                        og[n:] = relay[r][:]
                                        if r > 0:
                                            relay[r - 1][:] = og[:]
                                        else:
                                            og_ch.put(og)

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
                    with herd("rope", ROPE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            # room for the padded rows too: a loop that
                            # only takes rounds loses its count
                            qkv = air.alloc([G.RG_QKV * arr.round], bf16, scope=p)
                            rw = air.alloc([3 * DH], bf16, scope=p)
                            q = air.alloc([DQ], bf16, scope=p)
                            qp = air.alloc([DQ], bf16, scope=p)
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
                            q_pack(qp, q)
                            q_bc.put(qp, indices=[0])
                            k_out.put(k)
                            v_out.put(v)

                    # ---- the attention units (attention layers); unit 1's
                    # output reaches (3, 3) through (3, 4)
                    s_blk = [
                        air.alloc([A.S_BLK], bf16, scope=seg.shared())
                        for _ in range(NCU)
                    ]
                    qsh = [
                        air.alloc([QH * DH], bf16, scope=seg.shared())
                        for _ in range(NCU)
                    ]
                    o_k1 = air.alloc([QH * DH], bf16, scope=seg.shared())  # kv1 -> qk1
                    o_q1 = air.alloc([QH * DH], bf16, scope=seg.shared())  # qk1 -> kv0

                    def unit(c, col, row):
                        with herd(f"qk{c}", (col, row), params=[Lp]) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                q = air.alloc([QH * DH], bf16, scope=p)
                                kb = air.alloc([A.BLK], bf16, scope=p)
                                mm = air.alloc([16], bf16, scope=p)
                                cc = air.alloc([8], f32, scope=p)
                                q[:] = qsh[c][:]
                                for b in air.sequential(0, nb):
                                    k_cu.get(kb, indices=[c])
                                    qk(q, kb, mm, cc, s_blk[c], b, Lp)
                                if c == 1:
                                    o_q1[:] = o_k1[:]

                        with herd(f"kv{c}", (col, row + 1), params=[Lp]) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([A.BLK], bf16, scope=p)
                                yy = air.alloc([QH * DH], f32, scope=p)
                                ll = air.alloc([16], f32, scope=p)
                                o = air.alloc([QH * DH], bf16, scope=p)
                                on = air.alloc([QH * DH], bf16, scope=p)
                                qf = air.alloc([DQ], bf16, scope=p)
                                q_bc.get(qf, indices=[c])
                                qsh[c][:] = qf[c * QH * DH : (c + 1) * QH * DH]
                                for b in air.sequential(0, nb):
                                    v_cu.get(vb, indices=[c])
                                    kv(s_blk[c], vb, yy, ll, b, Lp)
                                fin(yy, ll, o)
                                o_nat(on, o)
                                if c == 1:
                                    o_k1[:] = on[:]
                                else:
                                    # q is in qsh by now: qf takes the output
                                    qf[0 : QH * DH] = on[:]
                                    qf[QH * DH :] = o_q1[:]
                                    og_ch.put(qf)

                    for c, (col, row) in enumerate(A.CU_LOC):
                        unit(c, col, row)

                    # ---- router: each layer's dense weights, then its two
                    # turns of experts, on the array's shim channels
                    with herd("router", G.ROUTER_AT, params=[ebase, lbase]) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            lg = air.alloc([E], f32, scope=p)
                            idx = air.alloc([16], i32, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            go = air.alloc([16], f32, scope=p)

                            for j in wave_loop(pw):
                                r_slot(idx, 8, lbase, j)
                                arr.put_weights(
                                    lambda i, hh: [
                                        WD[idx[8], i, :, 2 * hh : 2 * hh + 2, :]
                                    ]
                                )
                                lg[:] = logits[:]
                                r_topk(lg, idx, sc, ebase, j)
                                scores[:] = sc[:]
                                for t in range(TURNS):
                                    # turn 1 waits for the GLU core to finish turn 0's gate/up
                                    if t:
                                        go[:] = turn_go[:]
                                    arr.put_weights(
                                        lambda i, hh, t=t: [
                                            WE[
                                                idx[NCOL * t + i],
                                                :,
                                                2 * hh : 2 * hh + 2,
                                                :,
                                            ]
                                        ]
                                    )

                    # ---- GLU
                    with herd("glu", G.GLU_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            g = air.alloc([arr.round], bf16, scope=p)
                            hsl = air.alloc([I], bf16, scope=p)
                            hh = air.alloc([NCOL * I], bf16, scope=p)
                            sdn = air.alloc([D], bf16, scope=p)
                            acc = air.alloc([D], f32, scope=p)
                            y = air.alloc([D], bf16, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            for _j in wave_loop(pw):
                                for jj in air.sequential(0, G.RG_SGU):
                                    arr.out_y.get(u, indices=[0, GLU])
                                    glu_s(hsl, u, jj)
                                to_asm.put(hsl)
                                acc_zero(acc, D)
                                sc[:] = scores[:]
                                for t in range(TURNS):
                                    for s_ in air.sequential(0, G.RG_GU // 2):
                                        # every round lands in u
                                        arr.out_y.get(u, indices=[0, GLU])
                                        copy_up(g, u)
                                        arr.out_y.get(u, indices=[0, GLU])
                                        glu_e(hh, g, u, s_)
                                    to_asm.put(hh)
                                    if t + 1 < TURNS:
                                        turn_go[:] = sc[:]
                                    for rg in air.sequential(0, G.RG_DN):
                                        arr.out_y.get(u, indices=[0, GLU])
                                        acc_r(acc, u, sc, t, rg)
                                to_glu.get(sdn)
                                out_k(y, sdn, acc, D)
                                glu_y.put(y)

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
                        yield Phase(DQ, G.RG_O * NCB_O // 2, NCB_O, step, RMS, xbuf=1)
                        yield Phase(D, G.RG_SGU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, G.RG_SDN * NCB_H // 2, NCB_H, step, RMS, xbuf=1)
                        for _t in range(TURNS):
                            yield Phase(
                                D, G.RG_GU * NCB_X // 2, NCB_X, step, GLU, fresh=False
                            )
                            yield Phase(
                                I, G.RG_DN * NCB_H // 2, NCB_H, step, GLU, xbuf=1
                            )

                    arr.herd([DQ, DQ], flush, phases, layers=pw)

                    # ---- host side. Per wave: the conv cache outputs, x and
                    # the layers' outputs, and the GDN states, each one run of
                    # DDR. Then each layer's transfers; a conv cache goes per
                    # layer, since its shim channel may also carry the alpha /
                    # beta and router weights, which wait on that layer's
                    # mixer
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
                            m_in.put(M[lay, MISC["w1"][0] : MISC["wsg"][1]])
                        else:
                            m_in.put(M[lay, 0 : MISC["wsg"][1]])
                        ab_w.put(M[lay, MISC["adt"][0] : MISC["router"][1]])
                        m_dn.put(M[lay, MISC["dnw"][0] : MISC["dnw"][1]])
                        if is_att:
                            rw_in.put(RW[w, :])
                            pos = Lp.as_index() - 1
                            k_out.get(KV[w, 0, pos, :])
                            v_out.get(KV[w, 1, pos, :])
                            # back in 16-position blocks, K then V of each
                            kv_rd.put(
                                KV.reshape(waves, 2, nb, 16, DK)[
                                    w : w + 1, :, :, :, :
                                ].transpose(0, 2, 1, 3, 4)
                            )

        launch.body(body)

    return launch.build(target="npu2")


def layer_inputs(w, layers, waves):
    """Host arrays for `layers`, `waves` of them from layer 0."""
    nl = len(layers)
    pw = nl // waves
    WE, WD, M = [], [], np.zeros((nl, MISC_LEN), bfloat16)
    qkn = []
    conv_w = []
    for l in layers:
        wgu, wdn, sgu, sdn, wr, sg = moe_block.weights(w, l)
        WE.append(
            np.concatenate(
                [wgu.reshape(E, -1, R, BB), wdn.reshape(E, -1, R, BB)], axis=1
            )
        )
        if W.is_attention(l):
            wd, misc, *_, qn, kn = A.weights(w, l)
            # k, v, q padded with zero groups to a GDN layer's qkv
            nkvq = (A.RG_KV + A.RG_Q) * NCB_X
            pad = np.zeros(
                (NCOL, (G.RG_QKV - A.RG_KV - A.RG_Q) * NCB_X, R, BB), np.uint8
            )
            WD.append(np.concatenate([wd[:, :nkvq], pad, wd[:, nkvq:]], axis=1))
            for n in ("w1", "w2", "wsg", "router"):
                M[l, MISC[n][0] : MISC[n][1]] = misc[A.MISC[n][0] : A.MISC[n][1]]
            qkn.append((qn, kn))
        else:
            wqkv, wz, wo, misc, cw = G.weights(w, l)
            WD.append(
                np.concatenate(
                    [t.reshape(NCOL, -1, R, BB) for t in (wqkv, wz, wo, sgu, sdn)],
                    axis=1,
                )
            )
            M[l, D:] = misc
            M[l, MISC["wsg"][0] : MISC["wsg"][1]] = sg
            M[l, MISC["router"][0] : MISC["router"][1]] = wr.ravel()
            conv_w.append(cw)
    return np.ascontiguousarray(np.concatenate(WE)), np.stack(WD), M, qkn, conv_w


class Q36Decoder:
    """The layers' ELF and the LM head's, their buffers, and a run of each.
    Per token the host writes x, the attention layers' RoPE rows and L, and
    swaps the conv caches' and the GDN states' in and out buffers."""

    def __init__(self, arts, args, lm_args, waves, pw, qkn, params):
        import pyxrt as xrt

        from decode_elf import param_writers
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        self.xrt, self.waves, self.pw, self.qkn = xrt, waves, pw, qkn
        dev = xrt.device(0)

        def bo(t):
            b = np.ascontiguousarray(t).view(np.uint8).ravel()
            o = xrt.ext.bo(dev, b.nbytes)
            np.frombuffer(o.map(), np.uint8)[:] = b
            o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            return o

        def kernel(art):
            ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
            try:
                return ctx, xrt.ext.kernel(ctx, art.kernel)
            except RuntimeError:  # a repeated launch's kernel sits in the segment
                return ctx, xrt.ext.kernel(ctx, art.kernel.replace("main:", "seg:"))

        self.bos = [bo(t) for t in args]
        self.ctx, kern = kernel(arts[0])
        self.run = xrt.run(kern)
        for i, b in enumerate(self.bos):
            self.run.set_arg(i, b)
        self.params = ParameterScratchpad(self.run, str(params))
        self.writers = param_writers(params, {"L": len(args)})
        self.xmap = np.frombuffer(self.bos[1].map(), bfloat16).reshape(waves + 1, pw, D)
        self.rwmap = np.frombuffer(self.bos[7].map(), bfloat16).reshape(waves, 3 * DH)
        xsub = xrt.bo(self.bos[1], D * 2, (waves * pw + pw - 1) * D * 2)
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
        for i in (1, 7):
            self.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for name, fn in self.writers:
            self.params.write(name, np.int32(fn({len(self.bos): L})))
        self.params.sync()

    def _swap(self):
        """This token's outputs are the next one's inputs."""
        for a, b in ((4, 8), (5, 9)):
            self.bos[a], self.bos[b] = self.bos[b], self.bos[a]
            self.run.set_arg(a, self.bos[a])
            self.run.set_arg(b, self.bos[b])

    def step(self, x, L):
        """Decode x at position L - 1; returns the logits."""
        import lm_head as LM

        xrt = self.xrt
        self._set(x, L)
        for r in (self.run, self.lrun):
            r.start()
            st = r.wait(120000)
            if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch at L={L} ended in {st}")
        self._swap()
        yb = self.lbos[-1]
        yb.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return LM.unpack_logits(np.frombuffer(yb.map(), bfloat16)).astype(np.float32)

    def bench(self, x, L, n):
        """n tokens at context length L, the layers' run then the LM head's;
        the state is not carried (the same inputs every time)."""
        self._set(x, L)
        t = []
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            for _ in range(n + 1):
                t0 = time.perf_counter()
                self.run.start()
                self.run.wait(120000)
                t1 = time.perf_counter()
                self.lrun.start()
                self.lrun.wait(120000)
                t.append((t1 - t0, time.perf_counter() - t1))
        t = np.array(t[1:]) * 1e3
        print(
            f"L={L}, {n} tokens: layers {np.median(t[:, 0]):.3f} ms + LM head"
            f" {np.median(t[:, 1]):.3f} ms = {np.median(t.sum(1)):.3f} ms median"
        )


def compile_all(waves, maxl):
    """Kernels, the layers' ELF here and the LM head's in lm/."""
    import lm_head as LM

    for k in (
        "q4k_mvm",
        "moe_glu",
        "gate",
        "router",
        "gdn_prep",
        "gdn_decode",
        "ab_router",
    ):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--prompt", default="The capital of France")
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
    import lm_head as LM
    from reference import Reference

    w = W.Qwen36Q4nx()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    pw = PER_WAVE
    layers = list(range(pw * a.waves))
    assert len(ids) + a.gen <= a.maxl
    arts = compile_all(a.waves, a.maxl)
    t0 = time.time()
    we, wd, M, qkn, conv_w = layer_inputs(w, layers, a.waves)
    CW = np.stack([G.conv_cache(np.zeros((3, QKV_G), np.float32), cw) for cw in conv_w])
    S = np.zeros((len(conv_w), NV, R, HD * 32), np.float32)
    KV = np.zeros((a.waves, 2, a.maxl, DK), bfloat16)
    xs = np.zeros((a.waves + 1, pw, D), bfloat16)
    rw = np.zeros((a.waves, 3 * DH), bfloat16)
    args = [we, xs, wd, M, CW, S, KV, rw, CW.copy(), S.copy()]
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    y = np.zeros((LM.NCOL, LM.ROUNDS, LM.NROW * LM.RB), bfloat16)
    lm_args = [nw, LM.pack_weights(w), y]
    npu = Q36Decoder(arts, args, lm_args, a.waves, pw, qkn, arts[2])
    del we, wd, args, lm_args
    print(f"weights loaded in {time.time() - t0:.0f}s")

    def embed(t):
        return np.asarray(w.embed([t])[0]).astype(bfloat16)

    # the prompt goes through the decode one token at a time
    ref = None if a.no_ref else Reference(w, layers=layers)
    toks, ok = list(ids), True
    for i in range(len(ids) + a.gen - 1):
        got = npu.step(embed(toks[i]), i + 1)
        nxt = int(np.argmax(got))
        if ref is not None:
            want = ref.forward([toks[i]]).float().numpy()
            ok &= nxt == int(np.argmax(want))
            print(
                f"L={i + 1}: next {nxt} (reference {int(np.argmax(want))}),"
                f" logits corr {G.corr(got, want)[0]:.4f}"
            )
        if i + 1 >= len(ids):
            toks.append(nxt)
    print(repr(a.prompt), "->", repr(tok.decode(toks[len(ids) :])))
    if a.bench:
        npu.bench(embed(toks[-1]), a.bench_l or len(toks), a.bench)
    if ref is not None:
        print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
