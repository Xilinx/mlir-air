# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-Coder-30B-A3B decode: the 48 layers in one dispatch, then the LM head
# in a second. One air.launch, repeated once per wave of PER_WAVE layers; every
# core loops over the wave's layers:
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: each layer's k and v, q, o, then
#                 two turns of four experts
#   (2, 2)        RMS: the norms and residual adds; x stays here between a
#                 wave's layers
#   (2, 3)        RoPE: q / k RMSNorm and RoPE; k and v to the DDR cache
#   columns 3, 4  four attention units, fed by memtiles 3 and 4 (KV heads 0-1
#                 and 2-3); memtile 3 gathers their outputs
#   column 5      the router GEMV; router: each layer's dense weights, top 8 of
#                 128 and the experts' transfers; GLU
#
# The context length L is a run-time scalar: it places this token's K and V
# row, and the attention cores mask by it. Every layer reads the cache back
# whole, maxl positions.
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
import moe_block as MB  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402

R, RB, CB, BB, D, I, E = A.R, A.RB, A.CB, A.BB, A.D, A.I, A.E
NCOL, TURNS, SLICES = A.NCOL, A.TURNS, A.SLICES
DH, DQ, DK, QKV, NCU, QH = A.DH, A.DQ, A.DK, A.QKV, A.NCU, A.QH
NCB_X, NCB_H, NCB_O = A.NCB_X, A.NCB_H, A.NCB_O
RG_KV, RG_Q, RG_O, RG_GU, RG_DN = A.RG_KV, A.RG_Q, A.RG_O, A.RG_GU, A.RG_DN
CU_COLS, CU_LOC, UPM, HPM, O_COL = A.CU_COLS, A.CU_LOC, A.UPM, A.HPM, A.O_COL
ROPE, RMS, GLU = A.ROPE, A.RMS, A.GLU
PER_WAVE = 4
# an expert's rows: gate/up, then down, per column
G_EXP = RG_GU * NCB_X + RG_DN * NCB_H
# a slot that stands in for x in the layers that do not take it from the host
MISC = {"xpad": (0, D), **{n: (a + D, b + D) for n, (a, b) in A.MISC.items()}}
MISC_LEN = A.MISC_LEN + D


def build_module(waves, maxl):
    """`waves` x PER_WAVE layers, for a context of up to `maxl` positions."""
    pw = PER_WAVE
    nl = pw * waves
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
    glu_e = k_("moe_glu", "glu_expert_round", [i32])
    acc_r = k_("moe_glu", "moe_acc_round", [i32, i32])
    copy_up = k_("moe_glu", "copy_up")
    acc_zero = k_("moe_glu", "acc_zero", [i32])
    out_k = k_("moe_glu", "moe_out_routed", [i32])
    r_zero = k_("router", "router_zero")
    r_mvm = k_("router", "router_mvm", [i32, i32])
    r_topk = k_("router", "router_topk_at", [i32, i32])
    r_slot = k_("router", "router_slot", [i32, i32, i32])
    rope_rd = k_("rope_rounds", "rope_round", [i32])
    rope = k_("rope_rounds", "rope_compute", [i32])
    q_pack = k_("rope_rounds", "q_pack")
    qk = k_("attn_qk", "attn_qk_blk", [i32, i32])
    kv = k_("attn_kv_q36", "attn_kv_blk", [i32, i32])
    fin = k_("attn_kv_q36", "attn_kv_fin")

    arr = MvmArray(
        row_block=RB,
        block_bytes=BB,
        assemble_col=MB.ASM,
        acc_words=2 * RB,
        columns=MB.COLUMNS,
    )

    WE = air.tensor([nl * E, G_EXP, R, BB], ui8)
    # x of each wave's last layer: [0] the token's embedding, [w + 1] wave w's
    X = air.tensor([waves + 1, pw, D], bf16)
    WD = air.tensor([nl, NCOL, A.G_DENSE, R, BB], ui8)
    M = air.tensor([nl, MISC_LEN], bf16)
    KV = air.tensor([nl, 2, maxl, DK], bf16, inout=True)
    RW = air.tensor([nl, 3 * DH], bf16)
    # the context length, this token's position + 1
    SEQ = air.tensor([], i32, name="L")

    ch = air.channel
    pkt = dict(channel_type="npu_dma_packet")
    m_in, to_rms, xn_ch, xs_ch = ch("MIn"), ch("ToRms"), ch("XN"), ch("XS")
    to_ab, ab_w, out_ch = ch("ToAB"), ch("ABW"), ch("Out")
    glu_y = ch("GluY")
    to_asm = ch("ToAsm", **pkt)
    rw_in = ch("RwIn")
    q_bc = ch("QBc", size=[1], broadcast_shape=[NCU])
    k_out, v_out = ch("KOut", **pkt), ch("VOut", **pkt)
    k_rd = ch("KRd", size=[len(CU_COLS)])
    v_rd = ch("VRd", size=[len(CU_COLS)])
    k_cu, v_cu = ch("KCu", size=[NCU]), ch("VCu", size=[NCU])
    o_cu = ch("OCu", size=[NCU])
    arr.channels(n_dests=3)

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

                    logits = air.alloc([E], f32, scope=seg.shared())
                    scores = air.alloc([16], f32, scope=seg.shared())
                    turn_go = air.alloc([16], f32, scope=seg.shared())

                    # ---- memtile 5: the RMS core's inputs and side outputs.
                    # The memtiles' buffers serve every layer: allocated in the
                    # layer loop, each would be doubled for ping-pong
                    ins = [l2([D], 5) for _ in range(3)]
                    moe5, x25, y5 = l2([D], 5), l2([D], 5), l2([D], 5)
                    for _j in wave_loop(pw):
                        for b in ins:
                            m_in.get(b)
                            to_rms.put(b)
                        glu_y.get(moe5)
                        to_rms.put(moe5)
                        for b in (x25, y5):
                            xs_ch.get(b)
                        to_ab.put(x25)
                        out_ch.put(y5)

                    # ---- memtiles 3 and 4: the cache of their KV heads back
                    # in 16-key blocks, to their units
                    for c, col in enumerate(CU_COLS):
                        km, vm = l2([16, HPM], col), l2([16, HPM], col)
                        for _j in wave_loop(pw):
                            for _b in wave_loop(nb):
                                k_rd.get(km, indices=[c])
                                v_rd.get(vm, indices=[c])
                                for j in range(UPM):
                                    u = c * UPM + j
                                    kh = km.reshape(16, HPM // 8, 8)[
                                        :, j * DH // 8 : (j + 1) * DH // 8, :
                                    ]
                                    k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                                    vh = vm.reshape(2, 8, HPM // 8, 8)[
                                        :, :, j * DH // 8 : (j + 1) * DH // 8, :
                                    ]
                                    v_cu.put(vh.transpose(0, 2, 1, 3), indices=[u])
                    # the units' outputs in head order, the input of o_proj
                    om = l2([DQ], O_COL)
                    for _j in wave_loop(pw):
                        for u in range(NCU):
                            o_cu.get(
                                om.reshape(NCU, QH, DH // 8, 8)[
                                    u : u + 1, :, :, :
                                ].transpose(0, 2, 1, 3),
                                indices=[u],
                            )
                        to_asm.put(om)

                    # ---- the assemble memtile. One buffer for the normed x
                    # and one for both turns' GLU outputs: each is broadcast
                    # before the next fills it (and a buffer filled more than
                    # once a layer keeps the loop from being doubled for
                    # ping-pong, which would take more locks than the memtile
                    # has)
                    xa = l2([D], MB.ASM)
                    gyl = l2([DQ], MB.ASM)
                    hx = l2([NCOL, I], MB.ASM)
                    for _j in wave_loop(pw):
                        for _ in range(1 + TURNS):
                            xn_ch.get(xa)
                        to_asm.get(gyl)
                        for _t in range(TURNS):
                            to_asm.get(hx)
                        arr.broadcast(lambda i: [xa, gyl] + [xa, hx[i, :]] * TURNS)
                    rd = RG_KV + RG_Q + RG_O + TURNS * (RG_GU + RG_DN)
                    arr.gather(seg, pw * rd)
                    arr.feed(seg, pw * (A.G_DENSE + TURNS * G_EXP))

                    def herd(name, at, params=()):
                        return air.herd(
                            [range(1)],
                            name=name,
                            shape=(1,),
                            at=at,
                            params=list(params),
                        )

                    # ---- RMS core
                    with herd("rms", A.RMS_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x, w1, w2, xn, xn2, mix, moe, y, xin = (
                                air.alloc([D], bf16, scope=p) for _ in range(9)
                            )
                            u = air.alloc([arr.round], bf16, scope=p)
                            # x carries over from layer to layer; only the
                            # wave's first takes it from the host
                            for j in wave_loop(pw):
                                for b in (xin, w1, w2):
                                    to_rms.get(b)
                                with air.ops.branch(j == 0):
                                    x[:] = xin[:]
                                rms_norm(xn, x, w1, D)
                                xn_ch.put(xn)
                                for jj in air.sequential(0, RG_O):
                                    arr.out_y.get(u, indices=[0, RMS])
                                    copy_r(mix, u, jj)
                                res_add(y, x, mix, D)
                                rms_norm(xn2, x, w2, D)
                                for _t in range(TURNS):
                                    xn_ch.put(xn2)
                                xs_ch.put(xn2)
                                to_rms.get(moe)
                                res_add(y, x, moe, D)
                                xs_ch.put(y)

                    # ---- RoPE
                    with herd("rope", A.ROPE_AT) as h:

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
                            for _j in wave_loop(pw):
                                rw_in.get(rw)
                                for jj in air.sequential(0, RG_KV):
                                    arr.out_y.get(u, indices=[0, ROPE])
                                    rope_rd(qkv, u, RG_Q + jj)
                                for jj in air.sequential(0, RG_Q):
                                    arr.out_y.get(u, indices=[0, ROPE])
                                    rope_rd(qkv, u, jj)
                                rope(q, k, v, qkv, rw, 0)
                                q_pack(qp, q)
                                q_bc.put(qp, indices=[0])
                                k_out.put(k)
                                v_out.put(v)

                    # ---- the attention units
                    def unit(c, col, row):
                        s = air.alloc([A.S_BLK], bf16, scope=seg.shared())
                        qsh = air.alloc([QH * DH], bf16, scope=seg.shared())

                        with herd(f"qk{c}", (col, row), params=[Lp]) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                q = air.alloc([QH * DH], bf16, scope=p)
                                kb = air.alloc([A.BLK], bf16, scope=p)
                                mm = air.alloc([16], bf16, scope=p)
                                cc = air.alloc([8], f32, scope=p)
                                for _j in wave_loop(pw):
                                    q[:] = qsh[:]
                                    for b in air.sequential(0, nb):
                                        k_cu.get(kb, indices=[c])
                                        qk(q, kb, mm, cc, s, b, Lp)

                        with herd(f"kv{c}", (col, row + 1), params=[Lp]) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([A.BLK], bf16, scope=p)
                                yy = air.alloc([QH * DH], f32, scope=p)
                                ll = air.alloc([16], f32, scope=p)
                                o = air.alloc([QH * DH], bf16, scope=p)
                                qf = air.alloc([DQ], bf16, scope=p)
                                for _j in wave_loop(pw):
                                    q_bc.get(qf, indices=[c])
                                    qsh[:] = qf[c * QH * DH : (c + 1) * QH * DH]
                                    for b in air.sequential(0, nb):
                                        v_cu.get(vb, indices=[c])
                                        kv(s, vb, yy, ll, b, Lp)
                                    fin(yy, ll, o)
                                    o_cu.put(o, indices=[c])

                    for c, (col, row) in enumerate(CU_LOC):
                        unit(c, col, row)

                    # ---- the router GEMV
                    with herd("logits", MB.LOGITS) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x = air.alloc([D], bf16, scope=p)
                            blk = air.alloc([CB * RB], bf16, scope=p)
                            lg = air.alloc([E], f32, scope=p)
                            for _j in wave_loop(pw):
                                to_ab.get(x)
                                r_zero(lg)
                                for rb in air.sequential(0, A.NRB_R):
                                    for cb in air.sequential(0, NCB_X):
                                        ab_w.get(blk)
                                        r_mvm(blk, x, lg, rb, cb)
                                logits[:] = lg[:]

                    # ---- router: each layer's dense weights, then its two
                    # turns of experts, on the array's shim channels
                    with herd("router", MB.ROUTER, params=[ebase, lbase]) as h:

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
                    with herd("glu", MB.GLU_AT) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            g = air.alloc([arr.round], bf16, scope=p)
                            hh = air.alloc([NCOL * I], bf16, scope=p)
                            acc = air.alloc([D], f32, scope=p)
                            y = air.alloc([D], bf16, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            for _j in wave_loop(pw):
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

                    # ---- the GEMV array
                    def phases(j):
                        yield Phase(D, RG_KV * NCB_X // 2, NCB_X, step, ROPE)
                        yield Phase(
                            D, RG_Q * NCB_X // 2, NCB_X, step, ROPE, fresh=False
                        )
                        yield Phase(DQ, RG_O * NCB_O // 2, NCB_O, step, RMS, xbuf=1)
                        for _t in range(TURNS):
                            yield Phase(D, RG_GU * NCB_X // 2, NCB_X, step, GLU)
                            yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, GLU, xbuf=1)

                    arr.herd([DQ, DQ], flush, phases, layers=pw)

                    # ---- host side. Per wave: x in, then the layers'
                    # outputs, each one run of DDR; then each layer's
                    # transfers
                    # x in before the outputs: with the wave a loop index the
                    # two regions of X are not known apart, and a read after
                    # the write would wait for the whole wave
                    m_in.put(X[w, pw - 1, :])
                    out_ch.get(X[w + 1, :, :])
                    for j in range(pw):
                        lay = w * pw + j
                        if j == 0:
                            m_in.put(M[lay, MISC["w1"][0] : MISC["w2"][1]])
                        else:
                            m_in.put(M[lay, 0 : MISC["w2"][1]])
                        ab_w.put(M[lay, MISC["router"][0] : MISC["router"][1]])
                        rw_in.put(RW[lay, :])
                        pos = Lp.as_index() - 1
                        k_out.get(KV[lay, 0, pos, :])
                        v_out.get(KV[lay, 1, pos, :])
                        for c in range(len(CU_COLS)):
                            hs = slice(c * HPM, (c + 1) * HPM)
                            k_rd.put(KV[lay, 0, :, hs], indices=[c])
                            v_rd.put(KV[lay, 1, :, hs], indices=[c])

        launch.body(body)

    return launch.build(target="npu2")


def layer_inputs(w, layers):
    """Host arrays for `layers`: (WE, WD, M, each layer's (q_norm, k_norm))."""
    WE, WD, M, qkn = [], [], np.zeros((len(layers), MISC_LEN), bfloat16), []
    for i, l in enumerate(layers):
        wd, misc, wgu, wdn, qn, kn = A.weights(w, l)
        WE.append(
            np.concatenate(
                [wgu.reshape(E, -1, R, BB), wdn.reshape(E, -1, R, BB)], axis=1
            )
        )
        WD.append(wd)
        M[i, D:] = misc
        qkn.append((qn, kn))
    return np.ascontiguousarray(np.concatenate(WE)), np.stack(WD), M, qkn


class Q3CDecoder:
    """The layers' ELF and the LM head's, their buffers, and a run of each.
    Per token the host writes x, the layers' RoPE rows and L."""

    def __init__(self, arts, args, lm_args, waves, pw, qkn, params, we_slack=0):
        """we_slack: bytes past the experts (args[0]) that a reader of WE
        other than this decode may touch."""
        import pyxrt as xrt

        from decode_elf import param_writers
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        self.xrt, self.waves, self.pw, self.qkn = xrt, waves, pw, qkn
        dev = xrt.device(0)

        def bo(t, extra=0):
            b = np.ascontiguousarray(t).view(np.uint8).ravel()
            o = xrt.ext.bo(dev, b.nbytes + extra)
            m = np.frombuffer(o.map(), np.uint8)
            m[: b.nbytes] = b
            m[b.nbytes :] = 0
            o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            return o

        def kernel(art):
            ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
            try:
                return ctx, xrt.ext.kernel(ctx, art.kernel)
            except RuntimeError:  # a repeated launch's kernel sits in the segment
                return ctx, xrt.ext.kernel(ctx, art.kernel.replace("main:", "seg:"))

        self.bos = [bo(t, we_slack if i == 0 else 0) for i, t in enumerate(args)]
        self.ctx, kern = kernel(arts[0])
        self.run = xrt.run(kern)
        for i, b in enumerate(self.bos):
            self.run.set_arg(i, b)
        self.params = ParameterScratchpad(self.run, str(params))
        self.writers = param_writers(params, {"L": len(args)})
        self.xmap = np.frombuffer(self.bos[1].map(), bfloat16).reshape(waves + 1, pw, D)
        self.rwmap = np.frombuffer(self.bos[5].map(), bfloat16).reshape(-1, 3 * DH)
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
            self.rwmap[i] = A.rope_weights(L - 1, qn, kn)
        for i in (1, 5):
            self.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for name, fn in self.writers:
            self.params.write(name, np.int32(fn({len(self.bos): L})))
        self.params.sync()

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

    def step(self, x, L):
        """Decode x at position L - 1; returns the logits."""
        self._set(x, L)
        return self._go((self.run, self.lrun), L)

    def seed(self, kv, n):
        """A prefill's K and V rows of its n positions, per layer:
        kv[layer] = (K [n, DK], V [n, DK])."""
        cache = np.frombuffer(self.bos[4].map(), bfloat16).reshape(len(kv), 2, -1, DK)
        for l, (k, v) in enumerate(kv):
            cache[l, 0, :n], cache[l, 1, :n] = k, v
        self.bos[4].sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def head(self, x, L):
        """The LM head alone on the last layer's output x, at position L - 1."""
        self.xmap[self.waves, self.pw - 1] = x
        self.bos[1].sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return self._go((self.lrun,), L)

    def bench(self, x, L, n):
        """n tokens at context length L, the layers' run then the LM head's;
        the state is not carried (the same inputs every time)."""
        self._set(x, L)
        t = []
        for _ in range(n + 1):
            t0 = time.perf_counter()
            self.run.start()
            self.run.wait(120000)
            t1 = time.perf_counter()
            self.lrun.start()
            self.lrun.wait(120000)
            t.append((t1 - t0, time.perf_counter() - t1))
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

    A.compile_kernels()
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


def load(w, maxl, waves=W.N_LAYERS // PER_WAVE, we_slack=0):
    """Compile (into the current directory) and load the decode: a
    Q3CDecoder of every layer, at a context of up to maxl. Hold the NPU lock
    while it lives. we_slack: as Q3CDecoder's."""
    import lm_head as LM

    arts = compile_all(waves, maxl)
    layers = list(range(PER_WAVE * waves))
    we, wd, M, qkn = layer_inputs(w, layers)
    KV = np.zeros((len(layers), 2, maxl, DK), bfloat16)
    xs = np.zeros((waves + 1, PER_WAVE, D), bfloat16)
    rw = np.zeros((len(layers), 3 * DH), bfloat16)
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    y = np.zeros((LM.NCOL, LM.ROUNDS, LM.NROW * LM.RB), bfloat16)
    return Q3CDecoder(
        arts,
        [we, xs, wd, M, KV, rw],
        [nw, LM.pack_weights(w), y],
        waves,
        PER_WAVE,
        qkn,
        arts[2],
        we_slack,
    )


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

    w = W.Qwen3CoderQ4k()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    if a.prompt_len:
        ids = (ids * (a.prompt_len // len(ids) + 1))[: a.prompt_len]
    layers = list(range(PER_WAVE * a.waves))
    assert len(ids) + a.gen <= a.maxl
    # the prefill takes the whole prompt and the first token comes from the LM
    # head on its last row; a bench times the decode alone
    fused = a.fused_dir is not None and len(layers) == W.N_LAYERS and not a.bench
    if fused:
        import prefill as P

        manifest = a.fused_dir / "manifest.json"
        if not manifest.is_file():
            sys.exit(f"{a.fused_dir}: no fused prefill build (fused_build.py)")
        fused = len(ids) <= json.loads(manifest.read_text())["max_len"]
    # the device is held from the first context to the last dispatch
    lock = filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock"))
    with lock:
        t0 = time.time()
        npu = load(w, a.maxl, a.waves, P.WE_SLACK if fused else 0)
        print(f"compiled and loaded in {time.time() - t0:.0f}s")

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
                    f" logits cos {common.compare(got, want)[0]:.4f}"
                )
            return nxt

        first, t_prefill = 0, 0.0
        if fused:
            pf = P.Q3CPrefill(a.fused_dir, w, layers, we=(npu.bos[0], 0))
            if a.warm:
                pf.prefill(ids)
            t0 = time.perf_counter()
            x = pf.prefill(ids)
            npu.seed(pf.kv_rows(), len(ids))
            got = npu.head(x[-1].astype(bfloat16), len(ids))
            t_prefill = time.perf_counter() - t0
            # its hw contexts slow the decode's dispatches; prefill() resumes
            pf.suspend()
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
