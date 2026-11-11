# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Qwen3.8 Gated DeltaNet decode layer in one dispatch, on the tile map of
# Qwen3.6-35B-A3B's (qwen36_35b_a3b_q4nx/gdn_layer.py) with the MoE block
# replaced by the dense MLP (mlp_block.py):
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array (shared MvmArray): qkv to the conv
#                 core, z to the gate core, ssm_out to the RMS core, then the
#                 MLP's gate/up to the GLU core and its down to the RMS core
#   memtile 2     the array's assemble memtile: the normed x, the gated output,
#                 the post-attention normed x, the GLU output. A core keeps x
#                 in L1 for the next phase that reads the same input
#   (2, 2)        RMS: the input norm; the residual and the post-attention
#                 norm; the MLP residual
#   (5, 2)        conv: the causal conv4 over the cached inputs, SiLU, the q/k
#                 l2-norms; the cache's inputs shift in place; each value
#                 head's [k, alpha | v, beta | q] to the step cores
#   (5, 3)        alpha / beta (bf16 weights from DDR), to the conv core
#   (5, 5)        GLU: SiLU(gate) * up, the input of down
#   column 4      the recurrence, a 32-wide value slice of every head per core;
#                 the state streams through memtile 4 head by head, in and back
#                 out in place
#   memtile 3     joins the step cores' outputs for the norm core
#   (2, 5)        DeltaNet norm, to the gate core
#   (2, 4)        gate: SiLU(z) * norm(o), the input of ssm_out
#   memtile 5     the RMS core's inputs (x, the norm weights) and its side
#                 outputs: the alpha / beta input, the layer's output
#
# At D = 5120 the RMS core holds five [D] buffers (x, a norm weight, the two
# normed x, a projection's rows), where Qwen3.6's held ten at D = 2048.
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
import mlp_block  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.04

R, RB, CB, BB = 4, 32, 256, 4736
D, HD, I = W.D, 128, W.I
NK, NV = W.NK, W.NVH  # key heads, value heads
QKV, DV = W.QKV, NV * HD  # 10240, 6144
NCH = QKV // HD  # conv chunks
SLOT = HD + 32
NCOL = 4
COLUMNS, ASM = [0, 1, 6, 7], 2
RMS_AT, GATE_AT, DN_AT = (2, 2), (2, 4), (2, 5)
CONV_AT, AB_AT, STEP_AT = (5, 2), (5, 3), (4, 2)
GLU_AT = (5, 5)
CONV, GATE, RMS, GLU = range(4)  # the array's consumers
# rounds of 512 rows, one 128-row group per column each: qkv, z, ssm_out, the
# MLP's gate/up and down
RG_QKV, RG_Z, RG_O = QKV // 512, DV // 512, D // 512
RG_GU, RG_DN = 2 * I // 512, D // 512
NCB_X, NCB_O, NCB_H = D // CB, DV // CB, I // CB
# the conv core's rounds: q, k, then v
CONV_Q, CONV_K = NK * HD // 512, 2 * NK * HD // 512
# alpha / beta rows [alpha NV | beta NV] in 32-row blocks
NRB_AB = 2 * NV // RB
# a column's dense weights in phase order, [row group x col block][4][block]:
# qkv, z, ssm_out, the MLP's gate/up and down; one transfer per shim channel
G_DENSE = (RG_QKV + RG_Z + RG_GU) * NCB_X + RG_O * NCB_O + RG_DN * NCB_H
# two transfers per shim channel, the mixer's projections and the MLP's: one
# of all G_DENSE blocks (2920 = 40 x 73) has no shim-legal tiling, and the
# decode's core-issued transfers per channel bound a lock's count (63)
G_MIX = (RG_QKV + RG_Z) * NCB_X + RG_O * NCB_O
G_PARTS = [(0, G_MIX), (G_MIX, G_DENSE)]
# the misc bf16 tensor
MISC = {}
_at = 0
for _n, _sz in (
    ("w1", D),
    ("w2", D),
    ("dnw", HD),
    # A and the dt bias (float32), padded to a weight block: the alpha / beta
    # core takes them as the first block of its one weight stream
    ("adt", CB * RB),
    ("ab", NRB_AB * NCB_X * CB * RB),
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
    rms_norm, res_add = k_("q38_ffn", "rms_norm", [i32]), k_(
        "q38_ffn", "residual_inplace", [i32]
    )
    copy_r = k_("q38_ffn", "copy_round", [i32])
    glu_s = k_("q38_ffn", "glu_shared_round", [i32])
    conv_k = k_("gdn_prep", "gdn_conv_cache", [i32])
    l2n = k_("gdn_prep", "gdn_l2norm", [i32, i32])
    pack = k_("gdn_prep", "gdn_pack", [i32])
    ab_zero, ab_mvm = k_("gdn_prep", "ab_zero"), k_("gdn_prep", "ab_mvm", [i32, i32])
    ab_act = k_("gdn_prep", "ab_activate")
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
    # the gate and GLU cores' outputs to the assemble memtile, in the order the
    # data dependences fix: the gated output, then the GLU output
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
                        yield Phase(DV, RG_O * NCB_O // 2, NCB_O, step, RMS)
                        yield Phase(D, RG_GU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, RMS)

                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    ab_sh = air.alloc([2 * NV], f32, scope=seg.shared())
                    # memtile 5: the RMS core's inputs, then its side outputs
                    ins = [l2([D], 5) for _ in range(3)]
                    m_in.put(X)
                    for n in ("w1", "w2"):
                        m_in.put(m(n))
                    for b in ins:
                        m_in.get(b)
                        to_rms.put(b)
                    xs5, y5 = l2([D], 5), l2([D], 5)
                    xs_ch.get(xs5)
                    xs_ch.get(y5)
                    to_ab.put(xs5)
                    out_ch.put(y5)
                    out_ch.get(OUT)
                    for n in ("adt", "ab"):
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
                    # output (ssm_out), the post-attention normed x (gate/up),
                    # the GLU output (down)
                    xa, x2 = l2([D], ASM), l2([D], ASM)
                    gyl, hs = l2([DV], ASM), l2([I], ASM)
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
                    arr.gather(seg, RG_QKV + RG_Z + RG_O + RG_GU + RG_DN)
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
                            xs_ch.put(xab)
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
                            for rb in air.sequential(0, NRB_AB):
                                for cb in air.sequential(0, NCB_X):
                                    ab_w.get(blk)
                                    ab_mvm(blk, x, ab, rb, cb)
                            ab_act(ab, adt)
                            ab_sh[:] = ab[:]

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
                            for lo, hi, norm in (
                                (0, CONV_Q, 1),
                                (CONV_Q, CONV_K, 0),
                                (CONV_K, RG_QKV, None),
                            ):
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

                    # one x buffer: every phase after z brings its own input
                    arr.herd(I, flush, phases)

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
    abw = np.concatenate([al, be]).reshape(NRB_AB, RB, NCB_X, CB).transpose(0, 2, 3, 1)
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
    for k in ("q4k_mvm", "q38_ffn", "gdn_prep", "gdn_decode", "gate"):
        common.compile_kernel(k)

    from reference import Reference, rmsnorm, t

    w = W.Qwen38Q4nx()
    ref = Reference(w, layers=[a.layer])
    ids = [int(v) for v in a.tokens.split(",")]
    xs = t(np.asarray(w.embed(ids), np.float32))
    p = ref.weights(a.layer)
    n1 = p["norms"][0]
    if len(ids) > 1:  # the history: the state and the conv inputs
        ref.gdn(a.layer, rmsnorm(xs[:-1], n1), p)
        ref.pos = len(ids) - 1
    S0 = ref.state.get(a.layer, torch.zeros(NV, HD, HD)).numpy().copy()
    H0 = ref.conv.get(a.layer, torch.zeros(3, QKV)).numpy().copy()
    x = xs[-1:]
    want = ref.layer(a.layer, x)[0].numpy()

    wqkv, wz, wo, misc, conv_w = weights(w, a.layer)
    gu, dnw = mlp_block.weights(w, a.layer)
    cw = conv_cache(H0, conv_w)
    s = state_chunks(S0)
    be = XRTBackend(instance_name="gdn", output_format="elf")
    art = be.compile(module)
    xb = x[0].numpy().astype(bfloat16)
    wd = np.concatenate(
        [t_.reshape(NCOL, -1, R, BB) for t_ in (wqkv, wz, wo, gu, dnw)], axis=1
    )
    assert wd.shape[1] == G_DENSE
    args = [xb, wd, misc, cw, s, np.zeros(D, bfloat16)]
    if a.bench:
        return mlp_block.bench(art, args, a.bench)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        res = run(*args)
        be.unload()
    out, cw2, s2 = res[5], res[3], res[4]
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
