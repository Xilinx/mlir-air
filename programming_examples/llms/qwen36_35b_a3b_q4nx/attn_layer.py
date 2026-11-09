# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Qwen3.6 full-attention decode layer in one dispatch, on FastFlowLM's tile
# map (FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/iron/layer_aie.py), with the same
# GEMV array, RMS core and MoE block as gdn_layer.py:
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: k and v, then q, to the RoPE core;
#                 the output gate to the gate core; o to the RMS core; then the
#                 MoE block
#   (2, 3)        RoPE: q / k RMSNorm and partial RoPE; this token's k and v go
#                 to the DDR cache at its position, q in the units' layout to
#                 both kv cores, which hand their unit's half to the qk core
#                 through the L1 they share
#   memtile 3     the cache back in 16-key blocks in the kernels' layout; the
#                 units' outputs back in head order, to the gate core
#   column 3      two attention units, one KV head and its 8 query heads each:
#                 a qk core scores each block into L1 it shares with the kv core
#                 above it, which accumulates the values
#   (2, 4)        gate: sigmoid(gate) * o, the input of o_proj
#
# The context length is fixed at build time here.
import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

import filelock
import numpy as np
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
from gdn_layer import (  # noqa: E402
    BB,
    CB,
    COLUMNS,
    D,
    E,
    GATE_AT,
    GLU_AT,
    I,
    NCB_H,
    NCB_X,
    NCOL,
    NRB_R,
    R,
    RB,
    RG_DN,
    RG_GU,
    RG_SDN,
    RG_SGU,
    RMS_AT,
    ROUTER_AT,
    TURNS,
    by_column,
    corr,
)
from rope import rope_weights  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.06

ASM = 2
DH, NQ, NKV = W.DH, W.NQ, W.NKV
DQ, DK = NQ * DH, NKV * DH
QKV = DQ + 2 * DK
NCU, QH = NKV, NQ // NKV  # attention units, query heads per unit
BLK = 16 * DH  # a unit's K or V block: 16 keys of its KV head
S_BLK = ((QH * 16 + 16 + 63) // 64) * 64
ROPE_AT, LOGITS_AT = (2, 3), (5, 3)
CU_LOC = [(3, 2), (3, 4)]  # each unit's qk core; its kv core one row up
ROPE, GATE, RMS, GLU = range(4)  # the array's consumers
# rounds of 512 rows: k and v, q, the output gate, o
RG_KV, RG_Q, RG_G, RG_O = 2 * DK // 512, DQ // 512, DQ // 512, D // 512
NCB_O = DQ // CB
G_DENSE = (RG_KV + RG_Q + RG_G + RG_SGU) * NCB_X + RG_O * NCB_O + RG_SDN * NCB_H
MISC = {}
_at = 0
for _n, _sz in (
    ("w1", D),
    ("w2", D),
    ("wsg", D),
    ("router", NRB_R * NCB_X * CB * RB),
):
    MISC[_n] = (_at, _at + _sz)
    _at += _sz
MISC_LEN = _at


def build_module(L):
    """Decode at position L - 1: L keys, this token's the last."""
    nb = (L + 15) // 16
    slot = L - 1
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
    rope_rd, rope = k_("rope_rounds", "rope_round", [i32]), k_(
        "rope_rounds", "rope_compute", [i32]
    )
    q_pack = k_("rope_rounds", "q_pack")
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    kv = k_("attn_kv", "attn_kv_blk", [i32, i32])
    fin = k_("attn_kv", "attn_kv_fin")
    z_rd, gsig = k_("gate", "z_round", [i32]), k_("gate", "gate_sigmoid", [i32])
    r_zero, r_mvm = k_("router", "router_zero"), k_("router", "router_mvm", [i32, i32])
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
    RW = air.tensor([3 * DH], bf16)
    # K, then V: [position][KV head][DH]; this token's row is written in place
    KV = air.tensor([2, nb * 16, DK], bf16, inout=True)
    OUT = air.tensor([D], bf16)

    def m(n):
        return M[MISC[n][0] : MISC[n][1]]

    ch = air.channel
    m_in, to_rms, xn_ch, xs_ch = ch("MIn"), ch("ToRms"), ch("XN"), ch("XS")
    to_ab, ab_w, out_ch = ch("ToAB"), ch("ABW"), ch("Out")
    to_glu, glu_y = ch("ToGlu"), ch("GluY")
    to_asm = ch("ToAsm", channel_type="npu_dma_packet")
    rw_in = ch("RwIn")
    # q in the units' layout, to both kv cores
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    # k and v share the RoPE core's second MM2S as two packet flows
    k_out = ch("KOut", channel_type="npu_dma_packet")
    v_out = ch("VOut", channel_type="npu_dma_packet")
    k_rd, v_rd = ch("KRd"), ch("VRd")
    k_cu, v_cu, o_cu = (
        ch("KCu", size=[NCU]),
        ch("VCu", size=[NCU]),
        ch("OCu", size=[NCU]),
    )
    o_gate = ch("OGate")
    arr.channels(n_dests=4)

    with air.launch(name="attn") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    def phases():
                        yield Phase(D, RG_KV * NCB_X // 2, NCB_X, step, ROPE)
                        yield Phase(
                            D, RG_Q * NCB_X // 2, NCB_X, step, ROPE, fresh=False
                        )
                        yield Phase(
                            D, RG_G * NCB_X // 2, NCB_X, step, GATE, fresh=False
                        )
                        yield Phase(DQ, RG_O * NCB_O // 2, NCB_O, step, RMS, xbuf=1)
                        yield Phase(D, RG_SGU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, RG_SDN * NCB_H // 2, NCB_H, step, RMS, xbuf=1)
                        for _t in range(TURNS):
                            yield Phase(
                                D, RG_GU * NCB_X // 2, NCB_X, step, GLU, fresh=False
                            )
                            yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, GLU, xbuf=1)

                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

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
                    x25, sd5, y5 = (l2([D], 5) for _ in range(3))
                    for b in (x25, sd5, y5):
                        xs_ch.get(b)
                    glu_y.get(moe5)
                    to_ab.put(x25)
                    to_glu.put(sd5)
                    out_ch.put(y5)
                    out_ch.get(OUT)
                    ab_w.put(m("router"))
                    rw_in.put(RW)
                    # this token's k and v into the cache, then the cache back
                    k_out.get(KV[0, slot, :])
                    v_out.get(KV[1, slot, :])
                    k_rd.put(KV[0, :, :])
                    v_rd.put(KV[1, :, :])
                    # memtile 3: the cache, o
                    om = l2([DQ], 3)
                    km, vm = l2([16, DK], 3), l2([16, DK], 3)
                    for _b in air.sequential(0, nb):
                        k_rd.get(km)
                        v_rd.get(vm)
                        for u in range(NCU):
                            # [key][head u's DH / 8][8] -> [DH / 8][key][8]
                            kh = km.reshape(16, DK // 8, 8)[
                                :, u * DH // 8 : (u + 1) * DH // 8, :
                            ]
                            k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                            # -> [key half][DH / 8][key][8]
                            vh = vm.reshape(2, 8, DK // 8, 8)[
                                :, :, u * DH // 8 : (u + 1) * DH // 8, :
                            ]
                            v_cu.put(vh.transpose(0, 2, 1, 3), indices=[u])
                    for c in range(NCU):
                        o_cu.get(
                            om.reshape(NCU, QH, DH // 8, 8)[
                                c : c + 1, :, :, :
                            ].transpose(0, 2, 1, 3),
                            indices=[c],
                        )
                    o_gate.put(om)
                    # the assemble memtile: the normed x (k, v, q, gate), the
                    # gated output (o), the post-attention normed x, the GLU
                    # outputs
                    xa, x2 = l2([D], ASM), l2([D], ASM)
                    gyl, hs = l2([DQ], ASM), l2([I], ASM)
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
                        RG_KV
                        + RG_Q
                        + RG_G
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
                            for j in air.sequential(0, RG_O):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(y, x, mix, D)
                            rms_norm(xn2, x, w2, D)
                            # done with xn2 before it goes
                            sg_dot(xn2, wsg, gsc, D)
                            xn_ch.put(xn2)
                            xs_ch.put(xn2)
                            for j in air.sequential(0, RG_SDN):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(sdn, u, j)
                            sg_apply(gsc, sdn, D)
                            xs_ch.put(sdn)
                            to_rms.get(moe)
                            res_add(y, x, moe, D)
                            xs_ch.put(y)

                    with air.herd([range(1)], name="rope", shape=(1,), at=ROPE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            qkv = air.alloc([QKV], bf16, scope=p)
                            rw = air.alloc([3 * DH], bf16, scope=p)
                            q = air.alloc([DQ], bf16, scope=p)
                            qp = air.alloc([DQ], bf16, scope=p)
                            k = air.alloc([DK], bf16, scope=p)
                            v = air.alloc([DK], bf16, scope=p)
                            rw_in.get(rw)
                            # k and v come first, into [q | k | v]
                            for j in air.sequential(0, RG_KV):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, RG_Q + j)
                            for j in air.sequential(0, RG_Q):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, j)
                            rope(q, k, v, qkv, rw, 0)
                            q_pack(qp, q)
                            q_bc.put(qp, indices=[0])
                            k_out.put(k)
                            v_out.put(v)

                    def unit(c, col, row):
                        s = air.alloc([S_BLK], bf16, scope=seg.shared())
                        qsh = air.alloc([QH * DH], bf16, scope=seg.shared())

                        with air.herd(
                            [range(1)], name=f"qk{c}", shape=(1,), at=(col, row)
                        ) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                q = air.alloc([QH * DH], bf16, scope=p)
                                kb = air.alloc([BLK], bf16, scope=p)
                                mm = air.alloc([16], bf16, scope=p)
                                cc = air.alloc([8], f32, scope=p)
                                q[:] = qsh[:]
                                for b in air.sequential(0, nb):
                                    k_cu.get(kb, indices=[c])
                                    qk(q, kb, mm, cc, s, b, L)

                        with air.herd(
                            [range(1)], name=f"kv{c}", shape=(1,), at=(col, row + 1)
                        ) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([BLK], bf16, scope=p)
                                yy = air.alloc([QH * DH], f32, scope=p)
                                ll = air.alloc([16], f32, scope=p)
                                o = air.alloc([QH * DH], bf16, scope=p)
                                qf = air.alloc([DQ], bf16, scope=p)
                                q_bc.get(qf, indices=[c])
                                qsh[:] = qf[c * QH * DH : (c + 1) * QH * DH]
                                for b in air.sequential(0, nb):
                                    v_cu.get(vb, indices=[c])
                                    kv(s, vb, yy, ll, b, L)
                                fin(yy, ll, o)
                                o_cu.put(o, indices=[c])

                    for c, (col, row) in enumerate(CU_LOC):
                        unit(c, col, row)

                    with air.herd([range(1)], name="gate", shape=(1,), at=GATE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            z = air.alloc([DQ], bf16, scope=p)
                            on = air.alloc([DQ], bf16, scope=p)
                            for j in air.sequential(0, RG_G):
                                arr.out_y.get(u, indices=[0, GATE])
                                z_rd(z, u, j)
                            o_gate.get(on)
                            gsig(on, z, on, DQ)
                            to_asm.put(on)

                    with air.herd(
                        [range(1)], name="logits", shape=(1,), at=LOGITS_AT
                    ) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x = air.alloc([D], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            lg = air.alloc([E], f32, scope=p)
                            to_ab.get(x)
                            r_zero(lg)
                            for rb in air.sequential(0, NRB_R):
                                for cb in air.sequential(0, NCB_X):
                                    ab_w.get(blk)
                                    r_mvm(blk, x, lg, rb, cb)
                            logits[:] = lg[:]

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
                                    # every round lands in u
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

                    arr.herd([DQ, DQ], flush, phases)

    return launch.build(target="npu2")


def weights(w, layer):
    """(WD, M) of an attention layer, and its q / k norms."""
    raw = lambda n: np.asarray(w.raw(w.name(layer, n)))  # noqa: E731
    sa = "self_attn."
    qg = raw(sa + "q_proj.weight").reshape(2, -1, NCB_X, BB)  # q, then gate
    kvb = np.concatenate([raw(sa + "k_proj.weight"), raw(sa + "v_proj.weight")])
    wgu, wdn, sgu, sdn, wr, sg = moe_block.weights(w, layer)
    parts = [
        by_column(kvb, NCB_X),
        by_column(qg[0], NCB_X),
        by_column(qg[1], NCB_X),
        by_column(raw(sa + "o_proj.weight"), NCB_O),
        sgu,
        sdn,
    ]
    wd = np.concatenate([t.reshape(NCOL, -1, R, BB) for t in parts], axis=1)
    assert wd.shape[1] == G_DENSE
    misc = np.zeros(MISC_LEN, bfloat16)
    n1, n2 = w.norms(layer)
    for n, v in (("w1", n1), ("w2", n2), ("wsg", sg), ("router", wr.ravel())):
        misc[MISC[n][0] : MISC[n][1]] = np.asarray(v).astype(bfloat16).ravel()
    *_, qn, kn = w.attn(layer)
    return wd, misc, wgu, wdn, qn, kn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument(
        "--tokens", default="760,6511,314,9338", help="history, then the token"
    )
    ap.add_argument("--bench", type=int, default=0, help="time n dispatches")
    a = ap.parse_args()
    assert W.is_attention(a.layer)
    ids = [int(v) for v in a.tokens.split(",")]
    L = len(ids)
    module = build_module(L)
    if a.print_module_only:
        print(module)
        return 0
    for k in ("q4k_mvm", "moe_glu", "gate", "router"):
        common.compile_kernel(k)
    for k in ("rope_rounds", "attn_qk", "attn_kv"):
        common.compile_kernel(k, engine=True)

    from reference import Reference, rmsnorm, t

    w = W.Qwen36Q4nx()
    ref = Reference(w, layers=[a.layer])
    xs = t(np.asarray(w.embed(ids), np.float32))
    n1 = ref.weights(a.layer)["norms"][0]
    if L > 1:  # the history: K and V of the earlier positions
        ref.attention(a.layer, rmsnorm(xs[:-1], n1))
        ref.pos = L - 1
    kv = np.zeros((2, (L + 15) // 16 * 16, DK), bfloat16)
    if L > 1:
        k0, v0 = ref.kv[a.layer]
        kv[0, : L - 1] = k0.reshape(L - 1, DK).numpy().astype(bfloat16)
        kv[1, : L - 1] = v0.reshape(L - 1, DK).numpy().astype(bfloat16)
    x = xs[-1:]
    want = ref.layer(a.layer, x)[0].numpy()
    k1, v1 = ref.kv[a.layer]

    wd, misc, wgu, wdn, qn, kn = weights(w, a.layer)
    rw = rope_weights(L - 1, qn, kn)
    be = XRTBackend(instance_name="attn", output_format="elf")
    art = be.compile(module)
    xb = x[0].numpy().astype(bfloat16)
    args = [wgu, wdn, xb, wd, misc, rw, kv, np.zeros(D, bfloat16)]
    if a.bench:
        return moe_block.bench(art, args, a.bench)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        res = run(*args)
        be.unload()
    out, kv2 = res[7], res[6].reshape(kv.shape)
    ok = True
    for name, got, ref_v in (
        ("new k", kv2[0, L - 1].astype(np.float32), k1[-1].reshape(-1).numpy()),
        ("new v", kv2[1, L - 1].astype(np.float32), v1[-1].reshape(-1).numpy()),
        ("layer out", out.astype(np.float32), want),
    ):
        c, l2, e = corr(got, ref_v)
        print(f"{name}: corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
        ok &= c > 0.999 and l2 < TOL
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
