# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Qwen3-Coder decode layer (attention, then the MoE block) in one dispatch,
# on the Qwen3.6 tile map with four attention units:
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: k and v, then q, to the RoPE core;
#                 o to the RMS core; then the MoE block
#   (2, 2)        RMS: the layer's norms and residual adds
#   (2, 3)        RoPE: q / k RMSNorm and RoPE; this token's k and v go to the
#                 DDR cache at its position, q in the units' layout to the kv
#                 cores, which hand their unit's share to the qk core through
#                 the L1 they share
#   memtiles 3, 4 the cache of KV heads 0-1 and 2-3 back in 16-key blocks in
#                 the kernels' layout; memtile 3 also gathers the units'
#                 outputs in head order for o_proj
#   columns 3, 4  four attention units, one KV head and its 8 query heads each:
#                 a qk core scores each block into L1 it shares with the kv
#                 core above it, which accumulates the values
#   column 5      the router GEMV, top 8 of 128 and the expert transfers, GLU
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
import moe_block as MB  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.06

R, RB, CB, BB = MB.R, MB.RB, MB.CB, MB.BB
D, I, E, TURNS, NCOL = MB.D, MB.I, MB.E, MB.TURNS, MB.NCOL
NCB_X, NCB_H, RG_GU, RG_DN, NRB_R = MB.NCB_X, MB.NCB_H, MB.RG_GU, MB.RG_DN, MB.NRB_R
SLICES, COLUMNS, ASM = MB.SLICES, MB.COLUMNS, MB.ASM
LOGITS_AT, ROUTER_AT, GLU_AT = MB.LOGITS, MB.ROUTER, MB.GLU_AT
RMS_AT, ROPE_AT = (2, 2), (2, 3)
DH, NQ, NKV = W.DH, W.NQ, W.NKV
DQ, DK = NQ * DH, NKV * DH
QKV = DQ + 2 * DK
NCU, QH = NKV, NQ // NKV  # attention units, query heads per unit
BLK = 16 * DH  # a unit's K or V block: 16 keys of its KV head
S_BLK = ((QH * 16 + 16 + 63) // 64) * 64
# each unit's qk core, its kv core one row up; units 2c, 2c + 1 on column
# CU_COLS[c], fed by that column's memtile
CU_COLS = [3, 4]
CU_LOC = [(col, row) for col in CU_COLS for row in (2, 4)]
UPM = NCU // len(CU_COLS)  # units per memtile
HPM = UPM * DH  # a memtile's KV heads' width
O_COL = 3  # the memtile that gathers the units' outputs
ROPE, RMS, GLU = range(3)  # the array's consumers
# rounds of 512 rows: k and v, q, o
RG_KV, RG_Q, RG_O = 2 * DK // 512, DQ // 512, D // 512
NCB_O = DQ // CB
G_DENSE = (RG_KV + RG_Q) * NCB_X + RG_O * NCB_O
MISC = {}
_at = 0
for _n, _sz in (("w1", D), ("w2", D), ("router", NRB_R * NCB_X * CB * RB)):
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
    rms_norm = k_("moe_glu", "rms_norm", [i32])
    res_add = k_("moe_glu", "residual_add", [i32])
    copy_r = k_("moe_glu", "copy_round", [i32])
    rope_rd = k_("rope_rounds", "rope_round", [i32])
    rope = k_("rope_rounds", "rope_compute", [i32])
    q_pack = k_("rope_rounds", "q_pack")
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    kv = k_("attn_kv_q36", "attn_kv_blk", [i32, i32])
    fin = k_("attn_kv_q36", "attn_kv_fin")
    o_nat = k_("attn_kv_q36", "o_natural")
    r_zero, r_mvm = k_("router", "router_zero"), k_("router", "router_mvm", [i32, i32])
    r_topk = k_("router", "router_topk", [i32])
    glu_e = k_("moe_glu", "glu_expert_round", [i32])
    acc_r = k_("moe_glu", "moe_acc_round", [i32, i32])
    copy_up, acc_zero = k_("moe_glu", "copy_up"), k_("moe_glu", "acc_zero", [i32])
    out_k = k_("moe_glu", "moe_out_routed", [i32])

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
    glu_y = ch("GluY")
    to_asm = ch("ToAsm", channel_type="npu_dma_packet")
    rw_in = ch("RwIn")
    # q in the units' layout, to every kv core
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    # k and v share the RoPE core's second MM2S as two packet flows
    k_out = ch("KOut", channel_type="npu_dma_packet")
    v_out = ch("VOut", channel_type="npu_dma_packet")
    k_rd = ch("KRd", size=[len(CU_COLS)])
    v_rd = ch("VRd", size=[len(CU_COLS)])
    k_cu, v_cu, o_cu = (
        ch("KCu", size=[NCU]),
        ch("VCu", size=[NCU]),
        ch("OCu", size=[NCU]),
    )
    arr.channels(n_dests=3)

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
                        yield Phase(DQ, RG_O * NCB_O // 2, NCB_O, step, RMS, xbuf=1)
                        for _t in range(TURNS):
                            yield Phase(D, RG_GU * NCB_X // 2, NCB_X, step, GLU)
                            yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, GLU, xbuf=1)

                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    logits = air.alloc([E], f32, scope=seg.shared())
                    scores = air.alloc([16], f32, scope=seg.shared())
                    turn_go = air.alloc([16], f32, scope=seg.shared())
                    # memtile 5: the RMS core's inputs, then its side outputs
                    ins = [l2([D], 5) for _ in range(3)]
                    m_in.put(X)
                    for n in ("w1", "w2"):
                        m_in.put(m(n))
                    for b in ins:
                        m_in.get(b)
                        to_rms.put(b)
                    moe5 = l2([D], 5)
                    x25, y5 = l2([D], 5), l2([D], 5)
                    for b in (x25, y5):
                        xs_ch.get(b)
                    glu_y.get(moe5)
                    to_rms.put(moe5)
                    to_ab.put(x25)
                    out_ch.put(y5)
                    out_ch.get(OUT)
                    ab_w.put(m("router"))
                    rw_in.put(RW)
                    # this token's k and v into the cache, then each memtile's
                    # KV heads of the cache back
                    k_out.get(KV[0, slot, :])
                    v_out.get(KV[1, slot, :])
                    for c in range(len(CU_COLS)):
                        k_rd.put(KV[0, :, c * HPM : (c + 1) * HPM], indices=[c])
                        v_rd.put(KV[1, :, c * HPM : (c + 1) * HPM], indices=[c])
                    for c, col in enumerate(CU_COLS):
                        km, vm = l2([16, HPM], col), l2([16, HPM], col)
                        for _b in air.sequential(0, nb):
                            k_rd.get(km, indices=[c])
                            v_rd.get(vm, indices=[c])
                            for j in range(UPM):
                                u = c * UPM + j
                                # [key][head j's DH / 8][8] -> [DH / 8][key][8]
                                kh = km.reshape(16, HPM // 8, 8)[
                                    :, j * DH // 8 : (j + 1) * DH // 8, :
                                ]
                                k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                                # -> [key half][DH / 8][key][8]
                                vh = vm.reshape(2, 8, HPM // 8, 8)[
                                    :, :, j * DH // 8 : (j + 1) * DH // 8, :
                                ]
                                v_cu.put(vh.transpose(0, 2, 1, 3), indices=[u])
                    # the units' outputs, in head order
                    om = l2([DQ], O_COL)
                    for u in range(NCU):
                        o_cu.get(
                            om.reshape(NCU, QH, DH // 8, 8)[
                                u : u + 1, :, :, :
                            ].transpose(0, 2, 1, 3),
                            indices=[u],
                        )
                    to_asm.put(om)
                    # the assemble memtile: the normed x (k, v, q), the
                    # attention output (o), the post-attention normed x, the
                    # GLU outputs
                    xa = l2([D], ASM)
                    x2 = [l2([D], ASM) for _ in range(TURNS)]
                    gyl = l2([DQ], ASM)
                    hx = [l2([NCOL, I], ASM) for _ in range(TURNS)]
                    xn_ch.get(xa)
                    for b in x2:
                        xn_ch.get(b)
                    for b in [gyl] + hx:
                        to_asm.get(b)
                    arr.put_weights(lambda i, h: [WD[i, :, 2 * h : 2 * h + 2, :]])
                    arr.broadcast(
                        lambda i: [xa, gyl]
                        + [b for t in range(TURNS) for b in (x2[t], hx[t][i, :])]
                    )
                    arr.gather(seg, RG_KV + RG_Q + RG_O + TURNS * (RG_GU + RG_DN))
                    arr.feed(seg, G_DENSE + TURNS * (RG_GU * NCB_X + RG_DN * NCB_H))

                    with air.herd([range(1)], name="rms", shape=(1,), at=RMS_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x, w1, w2, xn, xn2, mix, moe, y = (
                                air.alloc([D], bf16, scope=p) for _ in range(8)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            for b in (x, w1, w2):
                                to_rms.get(b)
                            rms_norm(xn, x, w1, D)
                            xn_ch.put(xn)
                            for j in air.sequential(0, RG_O):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(y, x, mix, D)
                            rms_norm(xn2, x, w2, D)
                            for _t in range(TURNS):
                                xn_ch.put(xn2)
                            xs_ch.put(xn2)
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
                            hh = air.alloc([NCOL * I], bf16, scope=p)
                            acc = air.alloc([D], f32, scope=p)
                            y = air.alloc([D], bf16, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            acc_zero(acc, D)
                            sc[:] = scores[:]
                            for t in range(TURNS):
                                for s_ in air.sequential(0, SLICES):
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
                            out_k(y, acc, D)
                            glu_y.put(y)

                    arr.herd([DQ, DQ], flush, phases)

    return launch.build(target="npu2")


def by_column(t, ncb):
    """A projection's row blocks [rows / 32, ncb, BB] -> [column, row groups /
    4, ncb, 4, BB]: column i takes row groups i, i + 4, ..."""
    t = t.reshape(-1, NCOL, R, ncb, BB).transpose(1, 0, 3, 2, 4)
    return np.ascontiguousarray(t)


def weights(w, layer):
    """(WD, M, WGU, WDN, q_norm, k_norm) of a layer."""
    raw = lambda n: np.asarray(w.raw(W.name(layer, n)))  # noqa: E731
    sa = "self_attn."
    kvb = np.concatenate([raw(sa + "k_proj.weight"), raw(sa + "v_proj.weight")])
    parts = [
        by_column(kvb, NCB_X),
        by_column(raw(sa + "q_proj.weight"), NCB_X),
        by_column(raw(sa + "o_proj.weight"), NCB_O),
    ]
    wd = np.concatenate([t.reshape(NCOL, -1, R, BB) for t in parts], axis=1)
    assert wd.shape[1] == G_DENSE
    wgu, wdn, wr = MB.weights(w, layer)
    misc = np.zeros(MISC_LEN, bfloat16)
    n1, n2 = w.norms(layer)
    for n, v in (("w1", n1), ("w2", n2), ("router", wr.ravel())):
        misc[MISC[n][0] : MISC[n][1]] = np.asarray(v).astype(bfloat16).ravel()
    qn, kn = (w.vec(layer, f"self_attn.{n}.weight") for n in ("q_norm", "k_norm"))
    return wd, misc, wgu, wdn, qn, kn


def rope_weights(pos, qn, kn):
    """[cos DH / 2 | sin DH / 2 | q_norm DH | k_norm DH] at position pos."""
    inv = 1.0 / (W.THETA ** (np.arange(0, DH, 2) / DH))
    f = pos * inv
    rw = np.zeros(3 * DH, np.float32)
    rw[: DH // 2], rw[DH // 2 : DH] = np.cos(f), np.sin(f)
    rw[DH : 2 * DH], rw[2 * DH :] = qn, kn
    return rw.astype(bfloat16)


def compile_kernels():
    for k in ("q4k_mvm", "moe_glu", "router"):
        common.compile_kernel(k)
    for k in ("rope_rounds", "attn_qk"):
        common.compile_kernel(k, engine=True)
    # skip the rescale of a block that moves no running max
    common.compile_kernel("attn_kv_q36", ["-DATTN_LAZY_RESCALE=1"], engine=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--tokens", default="785,6722,315,9625", help="history, then the token"
    )
    ap.add_argument("--bench", type=int, default=0, help="time n dispatches")
    a = ap.parse_args()
    ids = [int(v) for v in a.tokens.split(",")]
    L = len(ids)
    module = build_module(L)
    if a.print_module_only:
        print(module)
        return 0
    compile_kernels()

    from reference import Reference, rmsnorm, t

    w = W.Qwen3CoderQ4k()
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
        return MB.bench(art, args, a.bench)
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
        c, l2, e = common.compare(got, ref_v)
        print(f"{name}: cos {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
        ok &= c > 0.999 and l2 < TOL
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
