# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Qwen3.8 full-attention decode layer in one dispatch, on the tile map of
# Qwen3.6-35B-A3B's (qwen36_35b_a3b_q4nx/attn_layer.py), with the GEMV array,
# RMS core and dense MLP of gdn_layer.py:
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: k and v, then q, to the RoPE core;
#                 the output gate to the gate core; o to the RMS core; then the
#                 MLP
#   (2, 3)        RoPE: q / k RMSNorm and partial RoPE; this token's k and v go
#                 to the DDR cache at its position, q in the units' layout to
#                 both kv cores, which hand their unit's share to the qk core
#                 through the L1 they share
#   memtile 3     the cache back in 16-key blocks in the kernels' layout; the
#                 units' outputs back in head order, to the gate core
#   column 3      two attention units, two KV heads each, in turn: per 16-key
#                 block a qk core scores each head's 6 query heads (padded to
#                 the kernels' 8) into L1 it shares with the kv core above it,
#                 which accumulates the values
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
import mlp_block  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402
from gdn_layer import (  # noqa: E402
    BB,
    CB,
    COLUMNS,
    D,
    GATE_AT,
    GLU_AT,
    I,
    NCB_H,
    NCB_X,
    NCOL,
    R,
    RB,
    RG_DN,
    RG_GU,
    RMS_AT,
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
# the kernels take 8 query heads per KV head: each KV head's 6 padded with
# zero heads (rope_compute writes q so); attn_kv gives o back packed, the 6
QR, QHP = NQ // NKV, 8
DQP = NKV * QHP * DH
NCU, KPU = 2, NKV // 2  # attention units, KV heads per unit
BLK = 16 * DH  # a K or V block: 16 keys of one KV head
S_BLK = ((QHP * 16 + 16 + 63) // 64) * 64
ROPE_AT = (2, 3)
CU_LOC = [(3, 2), (3, 4)]  # each unit's qk core; its kv core one row up
ROPE, GATE, RMS, GLU = range(4)  # the array's consumers
# rounds of 512 rows: k and v, q, the output gate, o
RG_KV, RG_Q, RG_G, RG_O = 2 * DK // 512, DQ // 512, DQ // 512, D // 512
NCB_O = DQ // CB
G_DENSE = (RG_KV + RG_Q + RG_G + RG_GU) * NCB_X + RG_O * NCB_O + RG_DN * NCB_H
# a transfer per phase, as in gdn_layer.py
G_PARTS = []
_at = 0
for _n in (
    RG_KV * NCB_X,
    RG_Q * NCB_X,
    RG_G * NCB_X,
    RG_O * NCB_O,
    RG_GU * NCB_X,
    RG_DN * NCB_H,
):
    G_PARTS.append((_at, _at + _n))
    _at += _n
assert _at == G_DENSE
MISC = {}
_at = 0
for _n, _sz in (("w1", D), ("w2", D)):
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
    rms_norm, res_add = k_("q38_ffn", "rms_norm", [i32]), k_(
        "q38_ffn", "residual_inplace", [i32]
    )
    copy_r = k_("q38_ffn", "copy_round", [i32])
    glu_s = k_("q38_ffn", "glu_shared_round", [i32])
    rope_rd, rope = k_("rope_rounds", "rope_round", [i32]), k_(
        "rope_rounds", "rope_compute", [i32]
    )
    q_pack = k_("rope_rounds", "q_pack")
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    kv = k_("attn_kv", "attn_kv_blk", [i32, i32])
    fin = k_("attn_kv", "attn_kv_fin")
    z_rd, gsig = k_("gate", "z_round", [i32]), k_("gate", "gate_sigmoid", [i32])

    arr = MvmArray(
        row_block=RB,
        block_bytes=BB,
        assemble_col=ASM,
        acc_words=2 * RB,
        columns=COLUMNS,
    )

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
    out_ch = ch("Out")
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
                        yield Phase(DQ, RG_O * NCB_O // 2, NCB_O, step, RMS)
                        yield Phase(D, RG_GU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, RMS)

                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    # memtile 5: the RMS core's inputs, then its output
                    ins = [l2([D], 5) for _ in range(3)]
                    m_in.put(X)
                    for n in ("w1", "w2"):
                        m_in.put(m(n))
                    for b in ins:
                        m_in.get(b)
                        to_rms.put(b)
                    y5 = l2([D], 5)
                    xs_ch.get(y5)
                    out_ch.put(y5)
                    out_ch.get(OUT)
                    rw_in.put(RW)
                    # this token's k and v into the cache, then the cache back
                    k_out.get(KV[0, slot, :])
                    v_out.get(KV[1, slot, :])
                    k_rd.put(KV[0, :, :])
                    v_rd.put(KV[1, :, :])
                    # memtile 3: the cache, each unit its two KV heads in
                    # turn per block; o
                    om = l2([DQ], 3)
                    km, vm = l2([16, DK], 3), l2([16, DK], 3)
                    for _b in air.sequential(0, nb):
                        k_rd.get(km)
                        v_rd.get(vm)
                        for u in range(NCU):
                            for kk in range(KPU):
                                hd = u * KPU + kk
                                # [key][head's DH / 8][8] -> [DH / 8][key][8]
                                kh = km.reshape(16, DK // 8, 8)[
                                    :, hd * DH // 8 : (hd + 1) * DH // 8, :
                                ]
                                k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                                # -> [key half][DH / 8][key][8]
                                vh = vm.reshape(2, 8, DK // 8, 8)[
                                    :, :, hd * DH // 8 : (hd + 1) * DH // 8, :
                                ]
                                v_cu.put(vh.transpose(0, 2, 1, 3), indices=[u])
                    for c in range(NCU):
                        for kk in range(KPU):
                            hd = c * KPU + kk
                            # [DH / 8][6 heads][8] -> head order
                            o_cu.get(
                                om.reshape(NKV, QR, DH // 8, 8)[
                                    hd : hd + 1, :, :, :
                                ].transpose(0, 2, 1, 3),
                                indices=[c],
                            )
                    o_gate.put(om)
                    # the assemble memtile: the normed x (k, v, q, gate), the
                    # gated output (o), the post-attention normed x, the GLU
                    # output
                    xa, x2 = l2([D], ASM), l2([D], ASM)
                    gyl, hs = l2([DQ], ASM), l2([I], ASM)
                    xn_ch.get(xa)
                    xn_ch.get(x2)
                    to_asm.get(gyl)
                    to_asm.get(hs)
                    arr.put_weights(
                        lambda i, h: [
                            WD[i, a:b, 2 * h : 2 * h + 2, :] for a, b in G_PARTS
                        ]
                    )
                    arr.broadcast(lambda i: [xa, gyl, x2, hs])
                    arr.gather(seg, RG_KV + RG_Q + RG_G + RG_O + RG_GU + RG_DN)
                    arr.feed(seg, G_DENSE)

                    with air.herd([range(1)], name="rms", shape=(1,), at=RMS_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            # x, a norm weight (w1, then w2), the normed x
                            # of the mixer and of the MLP, a projection's rows.
                            # A buffer sent on a channel that also carries
                            # another is written once per layer: written again
                            # after its send, the compiler holds the send back
                            x, w, xab, xn, mix = (
                                air.alloc([D], bf16, scope=p) for _ in range(5)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            to_rms.get(x)
                            to_rms.get(w)
                            rms_norm(xab, x, w, D)
                            xn_ch.put(xab)
                            for j in air.sequential(0, RG_O):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(x, mix, D)
                            to_rms.get(w)
                            rms_norm(xn, x, w, D)
                            xn_ch.put(xn)
                            for j in air.sequential(0, RG_DN):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(x, mix, D)
                            xs_ch.put(x)

                    with air.herd([range(1)], name="rope", shape=(1,), at=ROPE_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            # [q | k | v]; once roped, q in the units' layout
                            qkv = air.alloc([QKV], bf16, scope=p)
                            rw = air.alloc([3 * DH], bf16, scope=p)
                            q = air.alloc([DQP], bf16, scope=p)
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
                            q_pack(qkv, q)
                            q_bc.put(qkv, indices=[0])
                            k_out.put(k)
                            v_out.put(v)

                    def unit(c, col, row):
                        s = [
                            air.alloc([S_BLK], bf16, scope=seg.shared())
                            for _ in range(KPU)
                        ]
                        qsh = [
                            air.alloc([QHP * DH], bf16, scope=seg.shared())
                            for _ in range(KPU)
                        ]

                        with air.herd(
                            [range(1)], name=f"qk{c}", shape=(1,), at=(col, row)
                        ) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                q = [
                                    air.alloc([QHP * DH], bf16, scope=p)
                                    for _ in range(KPU)
                                ]
                                kb = air.alloc([BLK], bf16, scope=p)
                                mm = [
                                    air.alloc([16], bf16, scope=p) for _ in range(KPU)
                                ]
                                cc = [air.alloc([8], f32, scope=p) for _ in range(KPU)]
                                for kk in range(KPU):
                                    q[kk][:] = qsh[kk][:]
                                for b in air.sequential(0, nb):
                                    for kk in range(KPU):
                                        k_cu.get(kb, indices=[c])
                                        qk(q[kk], kb, mm[kk], cc[kk], s[kk], b, L)

                        with air.herd(
                            [range(1)], name=f"kv{c}", shape=(1,), at=(col, row + 1)
                        ) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([BLK], bf16, scope=p)
                                yy = [
                                    air.alloc([QHP * DH], f32, scope=p)
                                    for _ in range(KPU)
                                ]
                                ll = [air.alloc([16], f32, scope=p) for _ in range(KPU)]
                                o = air.alloc([QR * DH], bf16, scope=p)
                                qf = air.alloc([DQP], bf16, scope=p)
                                q_bc.get(qf, indices=[c])
                                for kk in range(KPU):
                                    hd = c * KPU + kk
                                    qsh[kk][:] = qf[hd * QHP * DH : (hd + 1) * QHP * DH]
                                for b in air.sequential(0, nb):
                                    for kk in range(KPU):
                                        v_cu.get(vb, indices=[c])
                                        kv(s[kk], vb, yy[kk], ll[kk], b, L)
                                for kk in range(KPU):
                                    fin(yy[kk], ll[kk], o)
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

                    with air.herd([range(1)], name="glu", shape=(1,), at=GLU_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hsl = air.alloc([I], bf16, scope=p)
                            for j in air.sequential(0, RG_GU):
                                arr.out_y.get(u, indices=[0, GLU])
                                glu_s(hsl, u, j)
                            to_asm.put(hsl)

                    arr.herd(I, flush, phases)

    return launch.build(target="npu2")


def weights(w, layer):
    """(WD, M) of an attention layer, and its q / k norms."""
    raw = lambda n: np.asarray(w.raw(w.name(layer, n)))  # noqa: E731
    sa = "self_attn."
    qg = raw(sa + "q_proj.weight").reshape(2, -1, NCB_X, BB)  # q, then gate
    kvb = np.concatenate([raw(sa + "k_proj.weight"), raw(sa + "v_proj.weight")])
    gu, dn = mlp_block.weights(w, layer)
    parts = [
        by_column(kvb, NCB_X),
        by_column(qg[0], NCB_X),
        by_column(qg[1], NCB_X),
        by_column(raw(sa + "o_proj.weight"), NCB_O),
        gu,
        dn,
    ]
    wd = np.concatenate([t.reshape(NCOL, -1, R, BB) for t in parts], axis=1)
    assert wd.shape[1] == G_DENSE
    misc = np.zeros(MISC_LEN, bfloat16)
    n1, n2 = w.norms(layer)
    for n, v in (("w1", n1), ("w2", n2)):
        misc[MISC[n][0] : MISC[n][1]] = np.asarray(v).astype(bfloat16).ravel()
    *_, qn, kn = w.attn(layer)
    return wd, misc, qn, kn


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
    for k in ("q4k_mvm", "q38_ffn", "gate"):
        common.compile_kernel(k)
    for k in ("rope_rounds", "attn_qk", "attn_kv"):
        common.compile_kernel(k, engine=True)

    from reference import Reference, rmsnorm, t

    w = W.Qwen38Q4nx()
    ref = Reference(w, layers=[a.layer])
    xs = t(np.asarray(w.embed(ids), np.float32))
    p = ref.weights(a.layer)
    n1 = p["norms"][0]
    if L > 1:  # the history: K and V of the earlier positions
        ref.attention(a.layer, rmsnorm(xs[:-1], n1), p)
        ref.pos = L - 1
    kv = np.zeros((2, (L + 15) // 16 * 16, DK), bfloat16)
    if L > 1:
        k0, v0 = ref.kv[a.layer]
        kv[0, : L - 1] = k0.reshape(L - 1, DK).numpy().astype(bfloat16)
        kv[1, : L - 1] = v0.reshape(L - 1, DK).numpy().astype(bfloat16)
    x = xs[-1:]
    want = ref.layer(a.layer, x)[0].numpy()
    k1, v1 = ref.kv[a.layer]

    wd, misc, qn, kn = weights(w, a.layer)
    rw = rope_weights(L - 1, qn, kn)
    be = XRTBackend(instance_name="attn", output_format="elf")
    art = be.compile(module)
    xb = x[0].numpy().astype(bfloat16)
    args = [xb, wd, misc, rw, kv, np.zeros(D, bfloat16)]
    if a.bench:
        return mlp_block.bench(art, args, a.bench)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        res = run(*args)
        be.unload()
    out, kv2 = res[5], res[4].reshape(kv.shape)
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
