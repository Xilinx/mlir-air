# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Qwen3.6 Gated DeltaNet decode layer in one dispatch, on FastFlowLM's tile
# map (FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/iron/layer_aie.py):
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array (shared MvmArray): qkv to the conv
#                 core, z to the gate core, ssm_out to the RMS core, then the
#                 MoE block's shared expert and two turns of four routed
#                 experts, as moe_block.py runs them
#   memtile 2     the array's assemble memtile: the normed x, the gated output,
#                 the post-attention normed x, the GLU outputs. A core keeps x
#                 in L1 for the next phase that reads the same input
#   (2, 2)        RMS: the input norm; the residual and the post-attention
#                 norm; the shared expert's sigmoid gate; the MoE residual
#   (5, 2)        conv: the causal conv4 over the cached inputs, SiLU, the q/k
#                 l2-norms; the cache's inputs shift in place; each value
#                 head's [k, alpha | v, beta | q] to the step cores
#   (5, 3)        alpha / beta (bf16 weights from DDR), to the conv core; then
#                 the router GEMV into float32 logits
#   (5, 4)        router: the top 8 of 256 and the experts' weight transfers
#   (5, 5)        GLU: the shared and routed GLUs and the weighted sum
#   column 4      the recurrence, a 32-wide value slice of every head per core;
#                 the state streams through memtile 4 head by head, in and back
#                 out in place
#   memtile 3     joins the step cores' outputs for the norm core
#   (2, 5)        DeltaNet norm, to the gate core
#   (2, 4)        gate: SiLU(z) * norm(o), the input of ssm_out
#   memtile 5     the RMS core's inputs (x, the norm weights, the MoE output)
#                 and its side outputs: the alpha / beta and router input, the
#                 gated shared expert to the GLU core, the layer's output
import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import common  # noqa: E402
import qwen36_35b_a3b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402
import moe_block  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.04

R, RB, CB, BB = 4, 32, 256, 4736
D, HD = W.D, 128
NK, NV = 16, 32  # key heads, value heads
QKV, DV = W.QKV, NV * HD  # 8192, 4096
NCH = QKV // HD  # conv chunks
SLOT = HD + 32
NCOL = 4
COLUMNS, ASM = [0, 1, 6, 7], 2
RMS_AT, GATE_AT, DN_AT = (2, 2), (2, 4), (2, 5)
CONV_AT, AB_AT, STEP_AT = (5, 2), (5, 3), (4, 2)
ROUTER_AT, GLU_AT = (5, 4), (5, 5)
CONV, GATE, RMS, GLU = range(4)  # the array's consumers
I, E, TURNS = W.I_EXP, W.N_EXPERTS, W.TOP_K // 4
# rounds of 512 rows, one 128-row group per column each: qkv, z, ssm_out, the
# shared expert's gate/up and down; an expert's (one column's) row groups
RG_QKV, RG_Z, RG_O = QKV // 512, DV // 512, D // 512
RG_SGU, RG_SDN = 2 * I // 512, D // 512
RG_GU, RG_DN = 2 * I // 128, D // 128
NCB_X, NCB_O, NCB_H = D // CB, DV // CB, I // CB
NRB_R = E // RB
# a column's dense weights in phase order, [row group x col block][4][block]:
# qkv, z, ssm_out, the shared expert's gate/up and down; one transfer per
# shim channel
G_DENSE = (RG_QKV + RG_Z + RG_SGU) * NCB_X + RG_O * NCB_O + RG_SDN * NCB_H
# the misc bf16 tensor
MISC = {}
_at = 0
for _n, _sz in (
    ("w1", D),
    ("w2", D),
    ("wsg", D),
    ("dnw", HD),
    # A and the dt bias (float32), padded to a weight block: the alpha / beta
    # core takes them as the first block of its one weight stream
    ("adt", CB * RB),
    ("ab", 2 * NCB_X * CB * RB),
    ("router", NRB_R * NCB_X * CB * RB),
):
    MISC[_n] = (_at, _at + _sz)
    _at += _sz
MISC_LEN = _at


def build_module():
    q4k = dict(link_with="q4k_mvm.o")
    step = air.extern("q4k_step", scalars=[i32], **q4k)
    flush = air.extern("q4k_flush", **q4k)
    k_ = lambda o, n, s=(): air.extern(  # noqa: E731
        n, link_with=f"{o}.o", scalars=list(s)
    )
    rms_norm, res_add = k_("moe_glu", "rms_norm", [i32]), k_(
        "moe_glu", "residual_add", [i32]
    )
    copy_r = k_("moe_glu", "copy_round", [i32])
    conv_k = k_("gdn_prep", "gdn_conv_cache", [i32])
    l2n = k_("gdn_prep", "gdn_l2norm", [i32, i32])
    pack = k_("gdn_prep", "gdn_pack", [i32])
    ab_zero, ab_mvm = k_("ab_router", "ab_zero"), k_("ab_router", "ab_mvm", [i32, i32])
    ab_act = k_("ab_router", "ab_activate")
    r_zero, r_mvm = k_("ab_router", "router_zero"), k_(
        "ab_router", "router_mvm", [i32, i32]
    )
    r_topk = k_("router", "router_topk", [i32])
    glu_s, glu_e = k_("moe_glu", "glu_shared_round", [i32]), k_(
        "moe_glu", "glu_expert_round", [i32]
    )
    acc_r, copy_up = k_("moe_glu", "moe_acc_round", [i32, i32]), k_(
        "moe_glu", "copy_up"
    )
    acc_zero = k_("moe_glu", "acc_zero", [i32])
    sg_dot = k_("moe_glu", "shared_gate_dot", [i32])
    sg_apply = k_("moe_glu", "shared_gate_apply", [i32])
    out_k = k_("moe_glu", "moe_out", [i32])
    gstep = k_("gdn_decode", "gdn_step_h", [i32, i32])
    dn = k_("gate", "dn_norm_rhv")
    z_rd, gsilu = k_("gate", "z_round", [i32]), k_("gate", "gate_silu", [i32])

    arr = MvmArray(
        row_block=RB,
        block_bytes=BB,
        assemble_col=ASM,
        acc_words=2 * RB,
        columns=COLUMNS,
    )

    WGU = air.tensor([E, RG_GU, NCB_X, R, BB], ui8)
    WDN = air.tensor([E, RG_DN, NCB_H, R, BB], ui8)
    X = air.tensor([D], bf16)
    WD = air.tensor([NCOL, G_DENSE, R, BB], ui8)
    M = air.tensor([MISC_LEN], bf16)
    CW = air.tensor([NCH, 7 * HD], bf16, inout=True)
    S = air.tensor([NV, R, HD * 32], f32, inout=True)
    OUT = air.tensor([D], bf16)

    def m(n):
        return M[MISC[n][0] : MISC[n][1]]

    ch = air.channel
    m_in, to_rms, xn_ch, xs_ch = ch("MIn"), ch("ToRms"), ch("XN"), ch("XS")
    to_ab, ab_w, out_ch = ch("ToAB"), ch("ABW"), ch("Out")
    to_glu, glu_y = ch("ToGlu"), ch("GluY")
    # the gate and GLU cores' outputs to the assemble memtile, in an order the
    # data dependences fix: gated output, shared GLU, each turn's GLU
    to_asm = ch("ToAsm", channel_type="npu_dma_packet")
    cw_in, cw_out = ch("CWIn"), ch("CWOut")
    kvq_ch = ch("KVQ", size=[1, 1], broadcast_shape=[1, R])
    s_in, s_out = ch("SIn", size=[1, R]), ch("SOut", size=[1, R])
    s_l3, s_l3o = ch("SL3"), ch("SL3Out")
    o_ch = ch("OStep", size=[1, R], channel_type="npu_dma_packet")
    m_dn, to_dn, dn_ch = ch("MDn"), ch("ToDn"), ch("DnOut")
    arr.channels(n_dests=4)

    with air.launch(name="gdn") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    def phases():
                        yield Phase(D, RG_QKV * NCB_X // 2, NCB_X, step, CONV)
                        yield Phase(
                            D, RG_Z * NCB_X // 2, NCB_X, step, GATE, fresh=False
                        )
                        yield Phase(DV, RG_O * NCB_O // 2, NCB_O, step, RMS, xbuf=1)
                        yield Phase(D, RG_SGU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, RG_SDN * NCB_H // 2, NCB_H, step, RMS, xbuf=1)
                        for _t in range(TURNS):
                            yield Phase(
                                D, RG_GU * NCB_X // 2, NCB_X, step, GLU, fresh=False
                            )
                            yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, GLU, xbuf=1)

                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    ab_sh = air.alloc([2 * NV], f32, scope=seg.shared())
                    logits = air.alloc([E], f32, scope=seg.shared())
                    scores = air.alloc([16], f32, scope=seg.shared())
                    turn_go = air.alloc([16], f32, scope=seg.shared())
                    # memtile 5: the RMS core's inputs, then its side outputs
                    ins = [l2([D], 5) for _ in range(4)]
                    m_in.put(X)
                    for n in ("w1", "w2", "wsg"):
                        m_in.put(m(n))
                    for b in ins:
                        m_in.get(b)
                        to_rms.put(b)
                    moe5 = l2([D], 5)
                    to_rms.put(moe5)
                    xs5, x25, sd5, y5 = (l2([D], 5) for _ in range(4))
                    for b in (xs5, x25, sd5, y5):
                        xs_ch.get(b)
                    glu_y.get(moe5)
                    to_ab.put(xs5)
                    to_ab.put(x25)
                    to_glu.put(sd5)
                    out_ch.put(y5)
                    out_ch.get(OUT)
                    for n in ("adt", "ab", "router"):
                        ab_w.put(m(n))
                    # the conv cache: each chunk in, its shifted inputs back
                    cw_in.put(CW)
                    cw_out.get(CW[:, 0 : 3 * HD])
                    # memtile 4: the state, head by head, each core its chunk
                    s_l3.put(S)
                    s_l3o.get(S)
                    for _h in air.sequential(0, NV):
                        sb = l2([R, HD * 32], 4, f32)
                        s_l3.get(sb)
                        for r in range(R):
                            s_in.put(sb[r, :], indices=[0, r])
                        air.dealloc(sb)
                        ob = l2([R, HD * 32], 4, f32)
                        for r in range(R):
                            s_out.get(ob[r, :], indices=[0, r])
                        s_l3o.put(ob)
                        air.dealloc(ob)
                    # memtile 3: the norm weight, then the step cores' outputs
                    dnw3, o3 = l2([HD], 3), l2([R * NV * 32], 3)
                    m_dn.put(m("dnw"))
                    m_dn.get(dnw3)
                    for r in range(R):
                        o_ch.get(o3[r * NV * 32 : (r + 1) * NV * 32], indices=[0, r])
                    to_dn.put(dnw3)
                    to_dn.put(o3)
                    # the assemble memtile: the normed x (qkv, z), the gated
                    # output (ssm_out), the post-attention normed x (the
                    # experts' gate/up), the GLU outputs (their down)
                    xa, x2 = l2([D], ASM), l2([D], ASM)
                    gyl, hs = l2([DV], ASM), l2([I], ASM)
                    hx = [l2([NCOL, I], ASM) for _ in range(TURNS)]
                    xn_ch.get(xa)
                    xn_ch.get(x2)
                    for b in [gyl, hs] + hx:
                        to_asm.get(b)
                    arr.put_weights(lambda i, h: [WD[i, :, 2 * h : 2 * h + 2, :]])
                    arr.broadcast(
                        lambda i: [xa, gyl, x2, hs]
                        + [hx[t][i, :] for t in range(TURNS)]
                    )
                    arr.gather(
                        seg,
                        RG_QKV
                        + RG_Z
                        + RG_O
                        + RG_SGU
                        + RG_SDN
                        + TURNS * (RG_GU + RG_DN),
                    )
                    arr.feed(seg, G_DENSE + TURNS * (RG_GU * NCB_X + RG_DN * NCB_H))

                    with air.herd([range(1)], name="rms", shape=(1,), at=RMS_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x, w1, w2, wsg, xn, xn2, mix, sdn, moe, y = (
                                air.alloc([D], bf16, scope=p) for _ in range(10)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            gsc = air.alloc([16], bf16, scope=p)
                            for b in (x, w1, w2, wsg):
                                to_rms.get(b)
                            rms_norm(xn, x, w1, D)
                            xn_ch.put(xn)
                            xs_ch.put(xn)
                            for j in air.sequential(0, RG_O):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(y, x, mix, D)
                            rms_norm(xn2, x, w2, D)
                            # done with xn2 before it goes: the core gives a
                            # buffer up to its DMA until the transfer is over
                            sg_dot(xn2, wsg, gsc, D)
                            xn_ch.put(xn2)
                            xs_ch.put(xn2)
                            # the shared expert's down rows, gated
                            for j in air.sequential(0, RG_SDN):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(sdn, u, j)
                            sg_apply(gsc, sdn, D)
                            xs_ch.put(sdn)
                            to_rms.get(moe)
                            res_add(y, x, moe, D)
                            xs_ch.put(y)

                    with air.herd([range(1)], name="ab", shape=(1,), at=AB_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x = air.alloc([D], bf16, scope=p)
                            adt = air.alloc([4 * NV], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            ab = air.alloc([2 * NV], f32, scope=p)
                            to_ab.get(x)
                            ab_w.get(blk)
                            adt[:] = blk[0 : 4 * NV]
                            ab_zero(ab)
                            for rb in air.sequential(0, 2):
                                for cb in air.sequential(0, NCB_X):
                                    ab_w.get(blk)
                                    ab_mvm(blk, x, ab, rb, cb)
                            ab_act(ab, adt)
                            ab_sh[:] = ab[:]
                            lg = air.alloc([E], f32, scope=p)
                            to_ab.get(x)
                            r_zero(lg)
                            for rb in air.sequential(0, NRB_R):
                                for cb in air.sequential(0, NCB_X):
                                    ab_w.get(blk)
                                    r_mvm(blk, x, lg, rb, cb)
                            logits[:] = lg[:]

                    with air.herd([range(1)], name="conv", shape=(1,), at=CONV_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hx = air.alloc([7 * HD], bf16, scope=p)
                            ho = air.alloc([3 * HD], bf16, scope=p)
                            cv = air.alloc([QKV], bf16, scope=p)
                            ab = air.alloc([2 * NV], f32, scope=p)
                            kvq = air.alloc([3 * SLOT], bf16, scope=p)
                            # q chunks (normed, scaled), k (normed), then v
                            for lo, hi, norm in ((0, 4, 1), (4, 8, 0), (8, 16, None)):
                                for rd in air.sequential(lo, hi):
                                    arr.out_y.get(u, indices=[0, CONV])
                                    for k in range(4):
                                        c = rd * 4 + k
                                        cw_in.get(hx)
                                        conv_k(hx, u, cv, ho, c)
                                        cw_out.put(ho)
                                        if norm is not None:
                                            l2n(cv, c, norm)
                            ab[:] = ab_sh[:]
                            for hh in air.sequential(0, NV):
                                pack(cv, ab, kvq, hh)
                                kvq_ch.put(kvq)

                    with air.herd(
                        [range(1), range(R)], name="step", shape=(1, R), at=STEP_AT
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            p = h.private()
                            kvq = air.alloc([3 * SLOT], bf16, scope=p)
                            si = air.alloc([HD * 32], f32, scope=p)
                            so = air.alloc([HD * 32], f32, scope=p)
                            o = air.alloc([NV * 32], bf16, scope=p)
                            for hh in air.sequential(0, NV):
                                kvq_ch.get(kvq, indices=[tx, ty])
                                s_in.get(si, indices=[tx, ty])
                                gstep(si, so, kvq, o, ty, hh)
                                s_out.put(so, indices=[tx, ty])
                            o_ch.put(o, indices=[tx, ty])

                    with air.herd([range(1)], name="dn", shape=(1,), at=DN_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            w = air.alloc([HD], bf16, scope=p)
                            o = air.alloc([R * NV * 32], bf16, scope=p)
                            y = air.alloc([DV], bf16, scope=p)
                            to_dn.get(w)
                            to_dn.get(o)
                            dn(y, o, w)
                            dn_ch.put(y)

                    with air.herd([range(1)], name="gate", shape=(1,), at=GATE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            z = air.alloc([DV], bf16, scope=p)
                            on = air.alloc([DV], bf16, scope=p)
                            for j in air.sequential(0, RG_Z):
                                arr.out_y.get(u, indices=[0, GATE])
                                z_rd(z, u, j)
                            dn_ch.get(on)
                            gsilu(on, z, on, DV)
                            to_asm.put(on)

                    with air.herd(
                        [range(1)], name="router", shape=(1,), at=ROUTER_AT
                    ) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            lg = air.alloc([E], f32, scope=p)
                            idx = air.alloc([16], i32, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            go = air.alloc([16], f32, scope=p)
                            lg[:] = logits[:]
                            r_topk(lg, idx, sc, 0)
                            scores[:] = sc[:]
                            for t in range(TURNS):
                                # turn 1 waits for the GLU core to finish turn 0's gate/up
                                if t:
                                    go[:] = turn_go[:]
                                for Wt in (WGU, WDN):
                                    arr.put_weights(
                                        lambda i, hh, Wt=Wt, t=t: [
                                            Wt[
                                                idx[NCOL * t + i],
                                                :,
                                                :,
                                                2 * hh : 2 * hh + 2,
                                                :,
                                            ]
                                        ]
                                    )

                    with air.herd([range(1)], name="glu", shape=(1,), at=GLU_AT) as h:

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
                            for j in air.sequential(0, RG_SGU):
                                arr.out_y.get(u, indices=[0, GLU])
                                glu_s(hsl, u, j)
                            to_asm.put(hsl)
                            acc_zero(acc, D)
                            sc[:] = scores[:]
                            for t in range(TURNS):
                                for s_ in air.sequential(0, RG_GU // 2):
                                    # every round lands in u: the core's DMA
                                    # repeats one block, whatever the loop
                                    arr.out_y.get(u, indices=[0, GLU])
                                    copy_up(g, u)
                                    arr.out_y.get(u, indices=[0, GLU])
                                    glu_e(hh, g, u, s_)
                                to_asm.put(hh)
                                if t + 1 < TURNS:
                                    turn_go[:] = sc[:]
                                for rg in air.sequential(0, RG_DN):
                                    arr.out_y.get(u, indices=[0, GLU])
                                    acc_r(acc, u, sc, t, rg)
                            to_glu.get(sdn)
                            out_k(y, sdn, acc, D)
                            glu_y.put(y)

                    arr.herd([DV, DV], flush, phases)

    return launch.build(target="npu2")


def by_column(blocks, ncb):
    """A Q4_K projection's row blocks [rows / 32, ncb, BB] -> [column, row
    groups / 4, ncb, 4, BB]: column i takes row groups i, i + 4, ..."""
    t = blocks.reshape(-1, R, ncb, BB).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t.reshape(-1, NCOL, ncb, R, BB).swapaxes(0, 1))


def weights(w, layer):
    """(WQKV, WZ, WO, M) of a GDN layer, and its conv weights [4, QKV]."""
    raw = lambda n: np.asarray(w.raw(w.name(layer, n)))  # noqa: E731
    wqkv = by_column(raw("linear_attn.qkv_proj.weight"), NCB_X)
    wz = by_column(raw("self_attn.gate_proj.weight"), NCB_X)
    wo = by_column(raw("linear_attn.ssm_out_proj.weight"), NCB_O)
    _, _, _, al, be, conv, norm, a, dt = w.gdn(layer)
    abw = np.concatenate([al, be]).reshape(2, RB, NCB_X, CB).transpose(0, 2, 3, 1)
    adt = np.concatenate([a, dt]).astype(np.float32).view(bfloat16)
    misc = np.zeros(MISC_LEN, bfloat16)
    for n, v in (
        ("w1", w.norms(layer)[0]),
        ("w2", w.norms(layer)[1]),
        ("dnw", norm),
        ("adt", np.pad(np.asarray(adt), (0, CB * RB - adt.size))),
        ("ab", abw.astype(bfloat16).ravel()),
    ):
        misc[MISC[n][0] : MISC[n][1]] = np.asarray(v).astype(bfloat16).ravel()
    return wqkv, wz, wo, misc, np.asarray(conv, np.float32).T


def conv_cache(hist, conv_w):
    """The device cache [NCH, 7 * 128]: [x0, x1, x2, w0, w1, w2, w3] per chunk,
    from the last three inputs hist [3, QKV] and the weights [4, QKV]."""
    t = np.concatenate([hist, conv_w]).astype(bfloat16)  # [7, QKV]
    return np.ascontiguousarray(t.reshape(7, NCH, HD).transpose(1, 0, 2)).reshape(
        NCH, 7 * HD
    )


def state_chunks(s):
    """S [NV, 128 keys, 128 values] -> [NV, R, 128 * 32]: core r's values."""
    s = np.asarray(s, np.float32).reshape(NV, HD, R, 32).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(s).reshape(NV, R, HD * 32)


def run_debug(art, inputs):
    """One dispatch through pyxrt; whatever state it ends in, read every
    buffer back."""
    import pyxrt as xrt

    # the context is made and released under the lock, like the dispatch
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        dev = xrt.device(0)
        ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
        kern = xrt.ext.kernel(ctx, art.kernel)
        bos = []
        for t in inputs:
            b = np.ascontiguousarray(t).view(np.uint8).ravel()
            bo = xrt.ext.bo(dev, b.nbytes)
            np.frombuffer(bo.map(), np.uint8)[:] = b
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bos.append(bo)
        run = xrt.run(kern)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        print("dispatch:", run.wait(20000))
        out = []
        for t, bo in zip(inputs, bos):
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            out.append(
                np.frombuffer(bo.map(), t.dtype)[: t.size].reshape(t.shape).copy()
            )
        del run, kern, ctx
    return out


def corr(a, b):
    return common.compare(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="time n dispatches")
    ap.add_argument(
        "--tokens", default="760,6511,314,9338", help="history, then the token"
    )
    a = ap.parse_args()
    assert not W.is_attention(a.layer)
    module = build_module()
    if a.print_module_only:
        print(module)
        return 0
    for k in (
        "q4k_mvm",
        "moe_glu",
        "gdn_prep",
        "gdn_decode",
        "gate",
        "ab_router",
        "router",
    ):
        common.compile_kernel(k)

    from reference import Reference, rmsnorm, t

    w = W.Qwen36Q4nx()
    ref = Reference(w, layers=[a.layer])
    ids = [int(v) for v in a.tokens.split(",")]
    xs = t(np.asarray(w.embed(ids), np.float32))
    n1 = ref.weights(a.layer)["norms"][0]
    if len(ids) > 1:  # the history: the state and the conv inputs
        ref.gdn(a.layer, rmsnorm(xs[:-1], n1))
        ref.pos = len(ids) - 1
    S0 = ref.state.get(a.layer, torch.zeros(NV, HD, HD)).numpy().copy()
    H0 = ref.conv.get(a.layer, torch.zeros(3, QKV)).numpy().copy()
    x = xs[-1:]
    want = ref.layer(a.layer, x)[0].numpy()

    wqkv, wz, wo, misc, conv_w = weights(w, a.layer)
    wgu, wdn, sgu, sdn, wr, sg = moe_block.weights(w, a.layer)
    for n, v in (("wsg", sg), ("router", wr.ravel())):
        misc[MISC[n][0] : MISC[n][1]] = v
    cw = conv_cache(H0, conv_w)
    s = state_chunks(S0)
    be = XRTBackend(instance_name="gdn", output_format="elf")
    art = be.compile(module)
    xb = x[0].numpy().astype(bfloat16)
    wd = np.concatenate(
        [t.reshape(NCOL, -1, R, BB) for t in (wqkv, wz, wo, sgu, sdn)], axis=1
    )
    assert wd.shape[1] == G_DENSE
    args = [wgu, wdn, xb, wd, misc, cw, s, np.zeros(D, bfloat16)]
    if a.bench:
        return moe_block.bench(art, args, a.bench)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        res = run(*args)
        be.unload()
    out, cw2, s2 = res[7], res[5], res[6]
    ok = True
    checks = [
        (
            "conv inputs",
            cw2.reshape(NCH, 7, HD)[:, :3],
            conv_cache(ref.conv[a.layer].numpy(), conv_w)
            .reshape(NCH, 7, HD)[:, :3]
            .astype(np.float32),
        ),
        ("state", s2, state_chunks(ref.state[a.layer].numpy())),
        ("layer out", out.astype(np.float32), want),
    ]
    for name, got, ref_v in checks:
        c, l2, e = corr(got.astype(np.float32), ref_v)
        print(f"{name}: corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
        ok &= c > 0.999 and l2 < TOL
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
