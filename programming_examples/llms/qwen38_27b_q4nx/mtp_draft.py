# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8's multi-token-prediction (MTP) head in one dispatch, the draft of
# the speculative decode: from the main model's output h at position i and the
# token at i + 1, x = fc([rms(embed) * w_e | rms(h) * w_h]), then one gated
# attention decoder layer with its own KV cache, on the tile map of
# attn_layer.py with the decode's attention units (run-time context length).
# The host computes fc's input (two norms of D values); the fc is one more
# projection on the GEMV array, its rows to the RMS core as the layer's x. The
# head's final norm is the LM head's (lm_head.py with the MTP's norm weight).
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: fc, then k and v, q, the output
#                 gate, o, gate/up, down
#   (2, 2)        RMS: x from fc's rows; the two norms and residuals
#   (2, 3)        RoPE; this token's k and v to the DDR cache at L - 1
#   memtile 3     the cache back in 16-key blocks; the units' outputs
#   column 3      two attention units, two KV heads each
#   (2, 4)        gate: sigmoid(gate) * o
#   (5, 5)        GLU
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

R, BB, D, I, NCOL = G.R, G.BB, G.D, G.I, G.NCOL
DH, DQ, DK, NKV = A.DH, A.DQ, A.DK, A.NKV
QR, QHP, DQP, NCU, KPU = A.QR, A.QHP, A.DQP, A.NCU, A.KPU
NCB_X, NCB_O, NCB_H = A.NCB_X, A.NCB_O, G.NCB_H
UO = KPU * QR * DH  # an attention unit's output
NCB_FC = 2 * D // G.CB
RG_FC = D // 512
G_FC = RG_FC * NCB_FC
G_ALL = G_FC + A.G_DENSE
# the array's weights in two transfers per shim channel: fc and the mixer,
# then the MLP
G_PARTS = [(0, G_FC + A.G_MIX), (G_FC + A.G_MIX, G_ALL)]
ROPE, GATE, RMS, GLU = range(4)  # the array's consumers
MISC, MISC_LEN = A.MISC, A.MISC_LEN
# main()'s elementwise gate on the head's normed output, a fraction of
# max |want| at common's rtol (measured: at most 1.0e-2)
ATOL = 3e-2


def build_module(maxl):
    """The head at position L - 1 (L a run-time scalar), its cache up to maxl
    positions."""
    assert maxl % 16 == 0
    nb = maxl // 16
    k_ = lambda o, n, s=(): air.extern(  # noqa: E731
        n, link_with=f"{o}.o", scalars=list(s)
    )
    step = k_("q4k_mvm", "q4k_step", [i32])
    flush = k_("q4k_mvm", "q4k_flush")
    rms_norm = k_("q38_ffn", "rms_norm", [i32])
    res_add = k_("q38_ffn", "residual_inplace", [i32])
    copy_r = k_("q38_ffn", "copy_round", [i32])
    glu_s = k_("q38_ffn", "glu_shared_round", [i32])
    rope_rd = k_("rope_rounds", "rope_round", [i32])
    rope = k_("rope_rounds", "rope_compute", [i32])
    q_pack = k_("rope_rounds", "q_pack")
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    # the masked keys' values zeroed first (qwen36_35b_a3b_q4nx/kernels/attn_kv_q36.cc)
    kv = k_("attn_kv_q36", "attn_kv_blk_z", [i32, i32])
    fin = k_("attn_kv_q36", "attn_kv_fin")
    o_nat = k_("attn_kv_q36", "o_natural")
    z_rd, gsig = k_("gate", "z_round", [i32]), k_("gate", "gate_sigmoid", [i32])

    arr = MvmArray(
        row_block=G.RB,
        block_bytes=BB,
        assemble_col=A.ASM,
        acc_words=2 * G.RB,
        columns=G.COLUMNS,
    )

    XF = air.tensor([2 * D], bf16)  # fc's input
    # a shim channel's share is one run of DDR, as in the decode
    WD = air.tensor([NCOL, 2, G_ALL, 2, BB], ui8)
    M = air.tensor([MISC_LEN], bf16)
    RW = air.tensor([3 * DH], bf16)
    KV = air.tensor([2, maxl, DK], bf16, inout=True)
    OUT = air.tensor([D], bf16)
    SEQ = air.tensor([], i32, name="L")

    ch = air.channel
    pkt = dict(channel_type="npu_dma_packet")
    m_in, to_rms, xn_ch, xs_ch = ch("MIn"), ch("ToRms"), ch("XN"), ch("XS")
    out_ch = ch("Out")
    # into the assemble memtile as packets: fc's input from the host, then
    # the gate's and the GLU's outputs, on one port (its six take the gather's
    # four, the normed x and this)
    to_asm = ch("ToAsm", **pkt)
    rw_in = ch("RwIn")
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    k_out, v_out = ch("KOut", **pkt), ch("VOut", **pkt)
    k_rd, v_rd = ch("KRd"), ch("VRd")
    k_cu, v_cu = ch("KCu", size=[NCU]), ch("VCu", size=[NCU])
    o_cu, o_gate = ch("OCu", size=[NCU]), ch("OGate")
    arr.channels(n_dests=4)

    with air.launch(name="mtp", repeat=1) as launch:

        def body(_w):
            Lp = air.rtp(SEQ.value)
            with air.segment(name="seg", params=[Lp]) as seg:

                @seg.body
                def _():
                    def phases():
                        yield Phase(2 * D, G_FC // 2, NCB_FC, step, RMS)
                        yield Phase(D, A.RG_KV * NCB_X // 2, NCB_X, step, ROPE)
                        yield Phase(
                            D, A.RG_Q * NCB_X // 2, NCB_X, step, ROPE, fresh=False
                        )
                        yield Phase(
                            D, A.RG_G * NCB_X // 2, NCB_X, step, GATE, fresh=False
                        )
                        yield Phase(DQ, A.RG_O * NCB_O // 2, NCB_O, step, RMS)
                        yield Phase(D, G.RG_GU * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, G.RG_DN * NCB_H // 2, NCB_H, step, RMS)

                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    # memtile 5: the norm weights to the RMS core, its output
                    ins = [l2([D], 5) for _ in range(2)]
                    for b in ins:
                        m_in.get(b)
                        to_rms.put(b)
                    y5 = l2([D], 5)
                    xs_ch.get(y5)
                    out_ch.put(y5)
                    # memtile 3: the cache, each unit its two KV heads in
                    # turn per block; the units' outputs to the gate core
                    om = l2([DQ], 3)
                    km, vm = l2([16, DK], 3), l2([16, DK], 3)
                    for _b in air.sequential(0, nb):
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
                    for c in range(NCU):
                        o_cu.get(om[c * UO : (c + 1) * UO], indices=[c])
                    o_gate.put(om)
                    # the assemble memtile: fc's input, the normed x (k, v,
                    # q, gate), the gated output (o), the post-attention
                    # normed x, the GLU output
                    xf = l2([2 * D], A.ASM)
                    xa, x2 = l2([D], A.ASM), l2([D], A.ASM)
                    gyl, hs = l2([DQ], A.ASM), l2([I], A.ASM)
                    to_asm.get(xf)
                    xn_ch.get(xa)
                    xn_ch.get(x2)
                    to_asm.get(gyl)
                    to_asm.get(hs)
                    arr.broadcast(lambda i: [xf, xa, gyl, x2, hs])
                    arr.gather(
                        seg,
                        RG_FC + A.RG_KV + A.RG_Q + A.RG_G + A.RG_O + G.RG_GU + G.RG_DN,
                    )
                    arr.feed(seg, G_ALL)

                    with air.herd([range(1)], name="rms", shape=(1,), at=G.RMS_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            # x (fc's rows), a norm weight, the normed x of
                            # the mixer and of the MLP, a projection's rows
                            x, w, xab, xn, mix = (
                                air.alloc([D], bf16, scope=p) for _ in range(5)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            for j in air.sequential(0, RG_FC):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(x, u, j)
                            to_rms.get(w)
                            rms_norm(xab, x, w, D)
                            xn_ch.put(xab)
                            for j in air.sequential(0, A.RG_O):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(x, mix, D)
                            to_rms.get(w)
                            rms_norm(xn, x, w, D)
                            xn_ch.put(xn)
                            for j in air.sequential(0, G.RG_DN):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(mix, u, j)
                            res_add(x, mix, D)
                            xs_ch.put(x)

                    with air.herd(
                        [range(1)], name="rope", shape=(1,), at=A.ROPE_AT
                    ) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            qkv = air.alloc([A.QKV], bf16, scope=p)
                            rw = air.alloc([3 * DH], bf16, scope=p)
                            q = air.alloc([DQP], bf16, scope=p)
                            k = air.alloc([DK], bf16, scope=p)
                            v = air.alloc([DK], bf16, scope=p)
                            rw_in.get(rw)
                            # k and v come first, into [q | k | v]
                            for j in air.sequential(0, A.RG_KV):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, A.RG_Q + j)
                            for j in air.sequential(0, A.RG_Q):
                                arr.out_y.get(u, indices=[0, ROPE])
                                rope_rd(qkv, u, j)
                            rope(q, k, v, qkv, rw, 0)
                            q_pack(qkv, q)
                            q_bc.put(qkv[0:DQP], indices=[0])
                            k_out.put(k)
                            v_out.put(v)

                    s_blk = [
                        [
                            air.alloc([A.S_BLK], bf16, scope=seg.shared())
                            for _ in range(KPU)
                        ]
                        for _ in range(NCU)
                    ]

                    def unit(c, col, row):
                        with air.herd(
                            [range(1)],
                            name=f"qk{c}",
                            shape=(1,),
                            at=(col, row),
                            params=[Lp],
                        ) as hq:

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

                        with air.herd(
                            [range(1)],
                            name=f"kv{c}",
                            shape=(1,),
                            at=(col, row + 1),
                            params=[Lp],
                        ) as hk:

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

                    with air.herd(
                        [range(1)], name="gate", shape=(1,), at=G.GATE_AT
                    ) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            z = air.alloc([DQ], bf16, scope=p)
                            on = air.alloc([DQ], bf16, scope=p)
                            for j in air.sequential(0, A.RG_G):
                                arr.out_y.get(u, indices=[0, GATE])
                                z_rd(z, u, j)
                            o_gate.get(on)
                            gsig(on, z, on, DQ)
                            to_asm.put(on)

                    with air.herd([range(1)], name="glu", shape=(1,), at=G.GLU_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hsl = air.alloc([I], bf16, scope=p)
                            for j in air.sequential(0, G.RG_GU):
                                arr.out_y.get(u, indices=[0, GLU])
                                glu_s(hsl, u, j)
                            to_asm.put(hsl)

                    arr.herd(I, flush, phases)

                # host transfers, in the order the dispatch takes them
                to_asm.put(XF)
                m_in.put(M[MISC["w1"][0] : MISC["w2"][1]])
                arr.put_weights(lambda i, h: [WD[i, h, a:b, :, :] for a, b in G_PARTS])
                rw_in.put(RW)
                pos = Lp.as_index() - 1
                k_out.get(KV[0, pos, :])
                v_out.get(KV[1, pos, :])
                k_rd.put(KV[0, :, :])
                v_rd.put(KV[1, :, :])
                out_ch.get(OUT)

        launch.body(body)

    return launch.build(target="npu2")


def weights(w):
    """(WD [NCOL, 2, G_ALL, 2, BB], M, q norm, k norm, (w_e, w_h)) of the MTP
    head: fc's blocks ahead of its layer's (attn_layer.weights)."""
    fc = G.by_column(np.asarray(w.raw("model.mtp.fc.weight")), NCB_FC)
    wd, misc, qn, kn = A.weights(w, "mtp")
    wd = np.concatenate([fc.reshape(NCOL, -1, R, BB), wd], axis=1)
    assert wd.shape[1] == G_ALL
    wd = wd.reshape(NCOL, G_ALL, 2, 2, BB).transpose(0, 2, 1, 3, 4)
    _, ne, nh, nf = w.mtp_head()
    return np.ascontiguousarray(wd), misc, qn, kn, (ne, nh, nf)


def rms(x, wt):
    x = np.asarray(x, np.float32)
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + W.EPS) * wt


def fc_input(w, h, tok, norms):
    """fc's input [rms(embed(tok)) * w_e | rms(rms(h) * final norm) * w_h], as
    the reference: h the main model's last layer output (before its norm)."""
    ne, nh, _ = (np.asarray(n, np.float32) for n in norms)
    fn = np.asarray(w.final_norm(), np.float32)
    e = np.asarray(w.embed([tok])[0], np.float32)
    hn = rms(h, fn).astype(bfloat16).astype(np.float32)
    return np.concatenate([rms(e, ne), rms(hn, nh)]).astype(bfloat16)


def compile_mtp(maxl):
    """Kernels and the head's ELF, built in cwd; (artifact, params.txt)."""
    for k in ("q4k_mvm", "q38_ffn", "gate"):
        common.compile_kernel(k)
    for k in ("rope_rounds", "attn_qk", "attn_kv_q36"):
        common.compile_kernel(k, engine=True)
    art = XRTBackend(instance_name="mtp", output_format="elf").compile(
        build_module(maxl)
    )
    art.output_binary = os.path.abspath(art.output_binary)
    return art, Path(os.getcwd()) / "params.txt"


class MtpDraft:
    """The head's dispatch with its weights and KV cache on the device."""

    XF, WD, M, RW, KV, OUT = range(6)

    def __init__(self, art, params, w, maxl):
        import pyxrt as xrt

        from decode_elf import param_writers
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        self.xrt, self.w, self.maxl = xrt, w, maxl
        wd, misc, self.qn, self.kn, self.norms = weights(w)
        dev = xrt.device(0)
        tensors = [
            np.zeros(2 * D, bfloat16),
            wd,
            misc,
            np.zeros(3 * DH, bfloat16),
            np.zeros((2, maxl, DK), bfloat16),
            np.zeros(D, bfloat16),
        ]
        self.bos = []
        for t in tensors:
            b = np.ascontiguousarray(t).view(np.uint8).ravel()
            o = xrt.ext.bo(dev, b.nbytes)
            np.frombuffer(o.map(), np.uint8)[:] = b
            o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.bos.append(o)
        self.ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
        try:
            kern = xrt.ext.kernel(self.ctx, art.kernel)
        except RuntimeError:  # a repeated launch's kernel sits in the segment
            kern = xrt.ext.kernel(self.ctx, art.kernel.replace("main:", "seg:"))
        self.run = xrt.run(kern)
        for i, b in enumerate(self.bos):
            self.run.set_arg(i, b)
        self.params = ParameterScratchpad(self.run, str(params))
        self.writers = param_writers(params, {"L": len(self.bos)})
        self.core = {
            ln.split()[0]
            for ln in Path(params).read_text().splitlines()
            if len(ln.split()) > 3 and ln.split()[3] == "core"
        }

    def map(self, i, dt):
        return np.frombuffer(self.bos[i].map(), dt)

    def step(self, h, tok, L, lo=0):
        """The head at position L - 1 on the main model's output h there and
        the next token; returns its output before the final norm. With lo,
        its attention skips the keys below lo (a window of whole 16-key
        blocks, so up to 15 keys from lo up as well): the rows a caller never
        ran the head on."""
        from rope import rope_weights

        xrt = self.xrt
        to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.map(self.XF, bfloat16)[:] = fc_input(self.w, h, tok, self.norms)
        self.map(self.RW, bfloat16)[:] = rope_weights(L - 1, self.qn, self.kn)
        for i in (self.XF, self.RW):
            self.bos[i].sync(to)
        # the attention cores' L carries the window in bits 20-30
        # (aie_kernel_utils.h attn_window_lo); the cache offsets take L
        win = min((L - lo) // 16, 0x7FF) << 20 if lo > 0 else 0
        for name, fn in self.writers:
            v = fn({len(self.bos): L})
            if name in self.core:
                v |= win
            self.params.write(name, np.int32(v))
        self.params.sync()
        self.run.start()
        st = self.run.wait(120000)
        if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"MTP dispatch at L={L} ended in {st}")
        self.bos[self.OUT].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return self.map(self.OUT, bfloat16).copy()


def main():
    """The head over a few positions of a prompt against the reference's
    Mtp, h from the reference's main model."""
    import torch

    from reference import Mtp, Reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--maxl", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is a city that")
    ap.add_argument("--layers", type=int, default=4, help="main-model layers for h")
    a = ap.parse_args()
    w = W.Qwen38Q4nx()
    ids = w.tokenizer().encode(a.prompt).ids
    n = len(ids) - 1
    # h: any hidden state will do for the head's check; a few layers' output
    hs = Reference(w, list(range(a.layers))).forward(ids[:n], lm=False)
    hs = hs.numpy().astype(bfloat16)
    mtp = Mtp(w)
    art, params = compile_mtp(a.maxl)
    ok = True
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        d = MtpDraft(art, params, w, a.maxl)
        for i in range(n):
            got = d.step(hs[i], ids[i + 1], i + 1)
            # both through the head's final norm, the reference's output
            got = rms(got, np.asarray(d.norms[2], np.float32))
            h = torch.from_numpy(hs[i : i + 1].astype(np.float32))
            want = mtp.forward(h, ids[i + 1 : i + 2])[0].double().numpy()
            ok &= common.check(f"L={i + 1}", got, want, ATOL)
        del d
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
