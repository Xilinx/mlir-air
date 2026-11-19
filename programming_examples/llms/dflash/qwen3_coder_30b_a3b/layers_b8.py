# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The verify pass's layers for 8 token rows in one dispatch, routed on the
# device. One launch, run once per layer (a wave): layer l's attention,
# then its MoE. Each core runs its body once a wave, so each of its DMA
# channels has to be one repeating unit:
#
#   columns 0, 1, 6, 7  the Q4_K array: K and V, each unit's q, o_proj,
#                 then the MoE's turns of 4 experts, one a column, each core
#                 a row block of its column's weights. Every x it takes
#                 (the normed x, o_proj's halves, the MoE's) arrives as H
#                 messages, three each
#   memtile 2     the egress, and a ring of 4 H buffers, slot i to column i
#   (2, 2)        feeder: the only writer of the ring. Per wave the normed
#                 x, o_proj's x, the MoE's plan and x, the turns' H
#                 messages; the plan and the x's from the planner, o_proj's
#                 from memtile 5
#   (5, 3)        planner: the normed x (RMSNorm 1) as bfp16 tiles to the
#                 feeder; RMSNorm 2's x routes the layer: router logits, top
#                 8, the plan, sent with x as tiles
#   (5, 4)        router: the attention's weights, then the experts'
#                 transfers
#   (2, 4), (2, 5)  residual cores, 4 token rows each: RMSNorm 1, + o_proj,
#                 RMSNorm 2, + the MoE sum, one code path for the phases
#   (2, 3)        front: k RMSNorm and RoPE; the 8 new K and V rows to the
#                 DDR cache at P
#   memtiles 3, 4 the cache back in 16-key blocks, ahead of it RW (K) and
#                 a zero block (V)
#   columns 3, 4  four units, one KV head and its 8 query heads each: a qk
#                 core over a pv core, the scores by DMA. A unit's first K
#                 block is RW, which norms and rotates its q
#   memtile 5     the residual cores' normed rows to the planner; the units'
#                 outputs to the feeder
import argparse
import os
import subprocess
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
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[2])
]

import model_b8 as M  # noqa: E402
import q4k_layout as QL  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.api._index import coerce_index  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.decode.array import MvmArray  # noqa: E402

W = M.W
R, BB, D, E = M.R, M.BB, M.D, M.E
TOK, NCOL, COLUMNS, ASM = M.TOK, M.NCOL, M.COLUMNS, M.ASM
NCB_X, NCB_H, RG_GU, RG_DN, G_EXP = M.NCB_X, M.NCB_H, M.RG_GU, M.RG_DN, M.G_EXP
SLICES, HMSG, PLAN_B, TOK_W, EGR = M.SLICES, M.HMSG, M.PLAN_B, M.TOK_W, M.EGR
RG_KV, RG_Q, RG_O, G_DENSE = M.RG_KV, M.RG_Q, M.RG_O, M.G_DENSE
DH, NKV, DK = M.DH, M.NKV, M.DK
CU_COLS, CU_LOC, UPM, HPM = M.CU_COLS, M.CU_LOC, M.UPM, M.HPM
XB, OB, OT = M.XB, M.OB, M.OT
SSZ, YT, QH_DQ = M.SSZ, M.YT, M.QH_DQ
XPARTS = 3  # H messages that carry x
FEEDER_AT, FRONT_AT, PLANNER_AT, ROUTER_AT = (2, 2), (2, 3), (5, 3), (5, 4)
RES_AT = [(2, 4), (2, 5)]
X_COL = 5
HT = D // 8 * 32  # a residual core's f32 words: [dim tile][4 tokens][8]
STG = 256  # f32 words of a residual way-in stage
HSTG = HT // STG  # stages of h
NW = D // (2 * STG)  # stages of a norm weight
NSTG = HSTG + 2 * NW
BLKW = 16 * DH  # a unit's K or V block, bf16; RW is padded to it
FEEDER, RES0, FRONT, UNIT0 = 0, 1, 3, 4
N_DESTS = UNIT0 + NKV
XPC = TOK * D // 4  # bf16 of a normed-x piece: 64 dim tiles x 8 rows
PXB = PLAN_B + XB  # a message to the feeder: [plan | x as bfp16 tiles]
# a count-free ring: the bound only has to be past what any dispatch runs
RING = 1 << 20


def build_module(nl, reads, cap=None, causal=True):
    """nl layers (a wave each), attention reading `reads` cache positions
    of a cache of `cap`."""
    cap = cap or reads
    if reads <= 0 or reads % 16 or reads > cap:
        raise ValueError(
            f"reads {reads}: a positive multiple of 16 up to the cache's {cap}"
        )
    nb = reads // 16
    waves = nl
    k_ = lambda n, s=(): air.extern(  # noqa: E731
        n, link_with="layers_b8.o", scalars=list(s)
    )
    # MoE
    scores = k_("moe_scores", [i32])
    zero, block = k_("moe_zero", [i32]), k_("moe_block", [i32, i32, i32])
    block_h = k_("moe_block_h", [i32, i32, i32])
    block_o = k_("block_o", [i32, i32, i32])
    clear = k_("moe_clear", [i32])
    glu = k_("moe_glu")
    acc_k, out_k = k_("moe_acc", [i32]), k_("moe_out", [i32])
    fd_header, fd_scatter = k_("fd_header", [i32]), k_("fd_scatter", [i32])
    fd_copy, fd_turns = k_("fd_copy", [i32]), k_("fd_turns")
    mx_part = k_("mx_part", [i32])
    # attention
    ab_zero, ab_block, ab_flush = k_("ab_zero"), k_("ab_block", [i32]), k_("ab_flush")
    fr_k, fr_v = k_("fr_k"), k_("fr_v")
    qk_raw = k_("qk_raw", [i32])
    qk_step = k_("qk_step", [i32, i32, i32])
    pv_zero, pv_step, pv_fin2 = (
        k_("pv_zero"),
        k_("pv_step", [i32, i32, i32]),
        k_("pv_fin2"),
    )
    # residual
    res_in2 = k_("res_in2", [i32, i32, i32])
    res_add = k_("res_add", [i32, i32, i32])
    res_xn = k_("res_xn", [i32, i32])
    res_counts = k_("res_counts", [i32])
    # planner, router
    lg_zero, lg_block = k_("lg_zero"), k_("lg_block", [i32])
    plan_make = k_("plan_make", [i32, i32])
    px_tiles, fd_xpart2 = k_("px_tiles", [i32]), k_("fd_xpart2", [i32])
    slot_set = k_("slot_set", [i32])

    arr = MvmArray(row_block=EGR, block_bytes=BB, assemble_col=ASM, columns=COLUMNS)
    ch = air.channel
    c_, r_ = arr.cols, arr.rows
    arr.channels(n_dests=N_DESTS)
    pkt = dict(channel_type="npu_dma_packet")
    xh, tok = ch("XH"), ch("Tok")
    xc = ch("XC", size=[c_, 1], broadcast_shape=[c_, r_])
    f_in = ch("FIn", **pkt)  # the feeder's way in: planner and memtile 5
    xp, wr_in = ch("XP"), ch("WRIn")
    rw_m, rw_f, kv_out = ch("RWM"), ch("RWF"), ch("KVOut")
    k_rd, v_rd = ch("KRd", size=[len(CU_COLS)]), ch("VRd", size=[len(CU_COLS)])
    to_s = ch("ToS", size=[NKV])
    k_cu, v_cu, o_cu = (
        ch("KCu", size=[NKV]),
        ch("VCu", size=[NKV]),
        ch("OCu", size=[NKV]),
    )
    h_in, h_out, x_out = ch("HIn", size=[2]), ch("HOut", size=[2]), ch("XOut", size=[2])

    WE = air.tensor([nl * E, G_EXP, R, BB], ui8)
    WD = air.tensor([waves, NCOL, G_DENSE, R, BB], ui8)
    WR = air.tensor([waves, E * D], bf16)
    RW = air.tensor([waves, BLKW], bf16)
    KV = air.tensor([waves, 2, cap, DK], bf16, inout=True)
    # each layer's K block 0: RW as a unit's K block view reads it; V's
    # block 0 is 0
    RWB = air.tensor([waves, 16, DK], bf16)
    ZB = air.tensor([16, DK], bf16)
    # per wave and residual core: h (wave 0's only), then the norm weights
    HIN = air.tensor([waves, 2, NSTG * STG], f32)
    # per wave and residual core: h after the layer
    HO = air.tensor([waves, 2, 1, HT], f32)
    POS = air.tensor([], i32, name="P")

    with air.launch(name="layers", repeat=waves) as launch:

        def body(w):
            lay = air.rtp(w)
            Pp = air.rtp(POS.value)
            with air.segment(name="seg", params=[Pp, lay]) as seg:

                @seg.body
                def _():
                    def l2(shape, col, dt=ui8):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    def herd(name, at, params=()):
                        return air.herd(
                            [range(1)],
                            name=name,
                            shape=(1,),
                            at=at,
                            params=list(params),
                        )

                    # ---- memtile 2: the ring, the egress, the weights
                    hbuf = l2([NCOL, HMSG], ASM)
                    for _k in air.sequential(0, RING):
                        for i in range(NCOL):
                            xh.get(hbuf[i, :])
                            xc.put(hbuf[i, :], indices=[i, 0])
                    arr.gather(seg, RING)
                    arr.feed(seg, RING)

                    # ---- memtiles 3, 4: the cache back in 16-key blocks.
                    # Block 0 is RW (K) and 0 (V), sent ahead of the
                    # cache, so every channel here is one repeating unit,
                    # as waves need
                    for cc, col in enumerate(CU_COLS):
                        km, vm = l2([16, HPM], col, bf16), l2([16, HPM], col, bf16)
                        for _b in air.sequential(0, RING):
                            k_rd.get(km, indices=[cc])
                            v_rd.get(vm, indices=[cc])
                            for j in range(UPM):
                                u = cc * UPM + j
                                kh = km.reshape(16, HPM // 8, 8)[
                                    :, j * DH // 8 : (j + 1) * DH // 8, :
                                ]
                                k_cu.put(kh.transpose(1, 0, 2), indices=[u])
                                vh = vm.reshape(2, 8, HPM // 8, 8)[
                                    :, :, j * DH // 8 : (j + 1) * DH // 8, :
                                ]
                                v_cu.put(vh.transpose(0, 2, 1, 3), indices=[u])
                    # the front's RW through memtile 3: a host channel to
                    # a core would be shared with the residual cores'
                    rwm = l2([BLKW], CU_COLS[0], bf16)
                    for _r in air.sequential(0, RING):
                        rw_m.get(rwm)
                        rw_f.put(rwm)

                    # ---- memtile 5: each phase's normed rows of both residual
                    # cores, piece by piece, to the planner; the units'
                    # outputs, half by half, to the feeder
                    xm = l2([XPC], X_COL, bf16)
                    for _ph in air.sequential(0, 2):
                        for _q in air.sequential(0, 4):
                            for c in range(2):
                                x_out.get(
                                    xm.reshape(D // 32, 2, 32)[:, c : c + 1, :],
                                    indices=[c],
                                )
                            xp.put(xm)
                    # o_proj's halves as messages to the feeder, their
                    # plan part unused: units 0, 1 in om[0], 2, 3 in om[1]
                    om = [l2([PXB], X_COL) for _ in range(2)]
                    for _r in air.sequential(0, RING):
                        for u in range(NKV):
                            lo = PLAN_B + (u % 2) * OT
                            o_cu.get(om[u // 2][lo : lo + OT], indices=[u])
                        for half in om:
                            f_in.put(half)

                    # ---- planner. Each message is sent right after it is
                    # written: a send of what the wave before wrote would
                    # wait for this wave's writes
                    with herd("planner", PLANNER_AT, params=[lay]) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            xq = air.alloc([XPC], bf16, scope=p)
                            wbs = [air.alloc([16 * E], bf16, scope=p) for _ in range(2)]
                            lg = air.alloc([TOK * E], f32, scope=p)
                            px = air.alloc([PXB], ui8, scope=p)
                            # RMSNorm 1's x, to the array
                            for q in range(4):
                                xp.get(xq)
                                px_tiles(xq, px, q)
                            f_in.put(px)
                            # RMSNorm 2's x: the route, then the plan and x
                            lg_zero(lg)
                            for q in range(4):
                                xp.get(xq)
                                px_tiles(xq, px, q)
                                for k in air.sequential(0, D // 128):
                                    for half, wb in enumerate(wbs):
                                        wr_in.get(wb)
                                        lg_block(xq, wb, lg, k * 2 + half)
                            plan_make(lg, px, 0, lay)
                            f_in.put(px)

                    # ---- feeder
                    with herd("feeder", FEEDER_AT, params=[]) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            # every message in: [plan | x]
                            plan = air.alloc([PXB], ui8, scope=p)
                            rnd = air.alloc([arr.round], bf16, scope=p)
                            hm = air.alloc([NCOL * HMSG], ui8, scope=p)
                            tk = air.alloc([TOK_W * 4], ui8, scope=p)
                            cnt = air.alloc([16], i32, scope=p)

                            def token():
                                fd_copy(plan, tk, TOK_W * 4)
                                tok.put(tk)

                            def send():
                                for i in range(NCOL):
                                    xh.put(hm[i * HMSG : (i + 1) * HMSG])

                            def relay():
                                for q in range(XPARTS):
                                    fd_xpart2(hm, plan, q)
                                    send()

                            # attention: the normed x, o_proj's halves
                            for _x in range(3):
                                f_in.get(plan)
                                relay()
                            # the MoE
                            f_in.get(plan)
                            fd_turns(plan, cnt)
                            token()
                            relay()
                            for t in air.sequential(0, coerce_index(cnt[0])):
                                fd_header(hm, plan, t)
                                for s_ in range(SLICES):
                                    arr.out_y.get(rnd, indices=[0, FEEDER])
                                    fd_scatter(rnd, hm, s_)
                                token()
                                send()

                    # ---- router: the attention's weights, then the experts'
                    # transfers, a turn's gate and up two turns ahead
                    with herd("router", ROUTER_AT, params=[lay]) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            idx = air.alloc([TOK_W], i32, scope=p)
                            sl = air.alloc([16], i32, scope=p)

                            def put(t, lo, hi):
                                arr.put_weights(
                                    lambda i, hh: [
                                        WE[
                                            idx[8 + NCOL * t + i],
                                            lo:hi,
                                            2 * hh : 2 * hh + 2,
                                            :,
                                        ]
                                    ]
                                )

                            def gu(t):
                                put(t, 0, RG_GU * NCB_X)

                            def dn(t):
                                put(t, RG_GU * NCB_X, G_EXP)

                            def parity(k, f):
                                with air.ops.branch(k % 2 == 0) as even:
                                    f()
                                with even.otherwise():
                                    f()

                            # an index read on the core keeps these
                            # device-issued, ahead of the experts
                            slot_set(sl, lay)
                            arr.put_weights(
                                lambda i, hh: [WD[sl[0], i, :, 2 * hh : 2 * hh + 2, :]]
                            )
                            tok.get(idx)
                            gu(0)
                            gu(1)
                            n = coerce_index(idx[0])
                            for t in air.sequential(0, n):
                                tok.get(idx)
                                parity(t, lambda t=t: dn(t))
                                with air.ops.branch(t + 2 < n):
                                    parity(t, lambda t=t: gu(t + 2))

                    # ---- residual cores: one loop over the phases, so each
                    # channel has one site. Phase 0: h, RMSNorm 1's weight,
                    # x1 out; 1: + o_proj, RMSNorm 2's weight, x2 out; 2:
                    # + the MoE
                    def res(c, at):
                        with herd(f"res{c}", at, params=[lay]) as h:

                            @h.body
                            def _(tx):
                                p = h.private()
                                hh = air.alloc([HT], f32, scope=p)
                                stg = air.alloc([STG], f32, scope=p)
                                wn = air.alloc([2 * D], bf16, scope=p)
                                rnd = air.alloc([arr.round], bf16, scope=p)
                                xs = air.alloc([XPC // 2], bf16, scope=p)
                                cnt = air.alloc([16], i32, scope=p)
                                for ph in air.sequential(0, 3):
                                    res_counts(cnt, ph)
                                    for k in air.sequential(0, coerce_index(cnt[0])):
                                        h_in.get(stg, indices=[c])
                                        res_in2(stg, hh, wn, k, ph, lay)
                                    for k in air.sequential(0, coerce_index(cnt[1])):
                                        arr.out_y.get(rnd, indices=[0, RES0 + c])
                                        res_add(rnd, hh, k, c, ph)
                                    for q in air.sequential(0, coerce_index(cnt[2])):
                                        res_xn(hh, wn, xs, q, ph)
                                        x_out.put(xs, indices=[c])
                                h_out.put(hh, indices=[c])

                    for c, at in enumerate(RES_AT):
                        res(c, at)

                    # ---- front
                    with herd("front", FRONT_AT, params=[]) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            rw = air.alloc([BLKW], bf16, scope=p)
                            rnd = air.alloc([arr.round], bf16, scope=p)
                            kv = air.alloc([2 * TOK * DK], bf16, scope=p)
                            rw_f.get(rw)
                            arr.out_y.get(rnd, indices=[0, FRONT])
                            fr_k(rnd, rw, kv)
                            arr.out_y.get(rnd, indices=[0, FRONT])
                            fr_v(rnd, kv)
                            kv_out.put(kv[0 : TOK * DK])
                            kv_out.put(kv[TOK * DK : 2 * TOK * DK])

                    # ---- the units
                    def unit(u, col, row):
                        with herd(f"qk{u}", (col, row), params=[Pp]) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                rnd = air.alloc([arr.round], bf16, scope=p)
                                qs = air.alloc([TOK * QH_DQ], bf16, scope=p)
                                qb = air.alloc([TOK * 16 * 72], ui8, scope=p)
                                kb = air.alloc([BLKW], bf16, scope=p)
                                kf = air.alloc([32 * 72], ui8, scope=p)
                                mm = air.alloc([TOK * 16], bf16, scope=p)
                                for hh in range(2):
                                    arr.out_y.get(rnd, indices=[0, UNIT0 + u])
                                    qk_raw(rnd, qs, hh)
                                for b in air.sequential(0, nb + 1):
                                    k_cu.get(kb, indices=[u])
                                    s = air.alloc([TOK * SSZ], bf16, scope=p)
                                    qk_step(qs, kb, qb, kf, mm, s, b, Pp, int(causal))
                                    to_s.put(s, indices=[u])
                                    air.dealloc(s)

                        with herd(f"pv{u}", (col, row + 1), params=[Pp]) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([BLKW], bf16, scope=p)
                                vf = air.alloc([32 * 72], ui8, scope=p)
                                yy = air.alloc([TOK * YT], f32, scope=p)
                                ot = air.alloc([OT], ui8, scope=p)
                                pv_zero(yy)
                                for b in air.sequential(0, nb + 1):
                                    v_cu.get(vb, indices=[u])
                                    s = air.alloc([TOK * SSZ], bf16, scope=p)
                                    to_s.get(s, indices=[u])
                                    pv_step(s, vb, vf, yy, b, Pp, int(causal))
                                    air.dealloc(s)
                                pv_fin2(yy, ot)
                                o_cu.put(ot, indices=[u])

                    for u, (col, row) in enumerate(CU_LOC):
                        unit(u, col, row)

                    # ---- the array
                    def array_herd(base, col, n):
                        with air.herd(
                            [range(n), range(r_)],
                            name=f"mvm{base}",
                            shape=(n, r_),
                            at=(col, arr.row0),
                            params=[],
                        ) as he:

                            @he.body
                            def _(tx, ty):
                                cx = tx + base if base else tx
                                p = he.private()
                                xb = air.alloc([XB], ui8, scope=p)
                                hb = air.alloc([HMSG], ui8, scope=p)
                                wab = [air.alloc([BB], ui8, scope=p) for _ in range(2)]
                                sp = air.alloc([4096], ui8, scope=p)
                                acc = air.alloc([512], f32, scope=p)
                                moe = air.alloc([RG_DN * 256], f32, scope=p)
                                yb = air.alloc([16 + EGR], bf16, scope=p)
                                sc = air.alloc([16], f32, scope=p)
                                cnt = air.alloc([16], i32, scope=p)

                                def egress(dest):
                                    arr.out_a.put(
                                        yb[14 : 16 + EGR], indices=[cx, ty], dest=dest
                                    )

                                def weights(wb):
                                    arr.w_l1.get(wb, indices=[cx, ty])

                                def x_parts():
                                    for q in range(XPARTS):
                                        xc.get(hb, indices=[cx, ty])
                                        mx_part(hb, xb, cnt, q)

                                def gate_up():
                                    for _s in air.sequential(0, SLICES):
                                        zero(acc, 512)
                                        for k in air.sequential(0, NCB_X):
                                            for half, wb in enumerate(wab):
                                                b = k * 2 + half
                                                weights(wb)
                                                block(
                                                    xb,
                                                    wb,
                                                    acc,
                                                    sp,
                                                    b % NCB_X,
                                                    b // NCB_X,
                                                    0,
                                                )
                                        glu(acc, yb)
                                        egress(FEEDER)

                                def down():
                                    xc.get(hb, indices=[cx, ty])
                                    scores(hb, sc, cx)
                                    zero(acc, 256)
                                    for k in air.sequential(0, RG_DN * NCB_H // 2):
                                        for half, wb in enumerate(wab):
                                            b = k * 2 + half
                                            weights(wb)
                                            block_h(
                                                hb, wb, acc, sp, b % NCB_H, 0, M.HDR
                                            )
                                            with air.ops.branch(b % NCB_H == NCB_H - 1):
                                                acc_k(moe, acc, sc, b // NCB_H)
                                                zero(acc, 256)

                                def rounds(n, ncb, dest):
                                    for _j in air.sequential(0, n):
                                        ab_zero(acc)
                                        for k in air.sequential(0, ncb // 2):
                                            for half, wb in enumerate(wab):
                                                weights(wb)
                                                ab_block(xb, wb, acc, sp, k * 2 + half)
                                        ab_flush(acc, yb)
                                        egress(dest)

                                def to_res(n):
                                    for c in range(2):
                                        for rg in air.sequential(0, n):
                                            out_k(moe, yb, rg)
                                            egress(RES0 + c)

                                # ---- attention
                                x_parts()
                                rounds(RG_KV, NCB_X, FRONT)
                                for u in range(NKV):
                                    rounds(RG_Q // NKV, NCB_X, UNIT0 + u)
                                clear(moe, RG_O * 256)
                                for _hf in range(2):
                                    x_parts()
                                    for j in air.sequential(0, RG_O):
                                        for k in air.sequential(0, NCB_X // 2):
                                            for half, wb in enumerate(wab):
                                                weights(wb)
                                                block_o(
                                                    xb,
                                                    wb,
                                                    moe,
                                                    sp,
                                                    k * 2 + half,
                                                    j,
                                                    0,
                                                )
                                to_res(RG_O)

                                # ---- MoE
                                x_parts()
                                clear(moe, RG_DN * 256)
                                gate_up()
                                for _t in air.sequential(1, coerce_index(cnt[0])):
                                    gate_up()
                                    down()
                                down()
                                to_res(RG_DN)

                    array_herd(0, 0, 2)
                    array_herd(2, 6, 2)

                    # ---- host side, per wave
                    rw_m.put(RW[w, :])
                    for c in range(2):
                        h_in.put(HIN[w, c, :], indices=[c])
                        h_out.get(HO[w, c, :, :], indices=[c])
                    wr_in.put(WR[w, :])
                    pos = Pp.as_index()
                    kv_out.get(KV[w, 0, pos : pos + TOK, :])
                    kv_out.get(KV[w, 1, pos : pos + TOK, :])
                    for cc in range(len(CU_COLS)):
                        hs = slice(cc * HPM, (cc + 1) * HPM)
                        k_rd.put(RWB[w, :, hs], indices=[cc])
                        k_rd.put(KV[w, 0, 0:reads, hs], indices=[cc])
                        v_rd.put(ZB[:, hs], indices=[cc])
                        v_rd.put(KV[w, 1, 0:reads, hs], indices=[cc])

        launch.body(body)

    return launch.build(target="npu2")


def compile_kernel():
    peano = os.environ["PEANO_INSTALL_DIR"]
    inc = Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"
    subprocess.run(
        [
            f"{peano}/bin/clang++",
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-DNDEBUG",
            "-D__AIE_API_AIE_ADF_HPP__",
            "-Wno-parentheses",
            "-Wno-deprecated-declarations",
            f"-I{inc}",
            f"-I{HERE / 'kernels'}",
            "-c",
            str(HERE / "kernels" / "layers_b8.cc"),
            "-o",
            "layers_b8.o",
        ],
        check=True,
    )


def rw_block(rw):
    """RW as the cache's 16 rows before its first: each unit's K block view of
    them (dim tile, key, 8) reads RW in order."""
    r = np.asarray(rw, np.float32).reshape(16, 16, 8).transpose(1, 0, 2)
    return np.tile(r.reshape(16, DH), (1, NKV))


def h_way_in(h, w1, w2):
    """A residual core's way in: h tiles (4 rows), then w1 and w2 as bf16
    pairs in f32 words."""
    n = np.concatenate([np.asarray(w1, bfloat16), np.asarray(w2, bfloat16)])
    return np.concatenate([np.asarray(h, np.float32).ravel(), n.view(np.float32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--layer0", type=int, default=0)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--weights", default=None, help="dir of cached we<l>/wdf<l>.npy")
    a = ap.parse_args()
    nl = a.layers
    module = build_module(nl, a.reads)
    if a.print_module_only:
        print(module)
        return 0
    compile_kernel()
    from air_examples.llms.shared.fused_prefill import hostops as H
    import torch

    H.build(".", 4)
    H.load(".")
    w = W.Qwen3CoderQ4k()
    layers = [a.layer0 + i for i in range(nl)]
    cache_dir = Path(a.weights) if a.weights else None

    def cached(name, make):
        f = cache_dir / f"{name}.npy" if cache_dir else None
        if f is not None and f.is_file():
            return np.load(f, mmap_mode="r")
        return make()

    waves = nl
    we = np.empty((nl * E, G_EXP, R, BB), np.uint8)
    wd = np.zeros((waves, NCOL, G_DENSE, R, BB), np.uint8)
    wr = np.zeros((waves, E * D), bfloat16)
    rw = np.zeros((waves, BLKW), bfloat16)
    hin = np.zeros((waves, 2, NSTG * STG), np.float32)
    rng = np.random.default_rng(a.seed)
    h0 = (rng.standard_normal((TOK, D)) * 0.5).astype(np.float32)
    kv = np.zeros((waves, 2, a.reads, DK), np.float32)
    rwb = np.zeros((waves, 16, DK), bfloat16)
    for i, l in enumerate(layers):
        we[i * E : (i + 1) * E] = cached(
            f"we{l}", lambda: QL.tiled(M.expert_weights(w, l))
        )
        wd[i] = cached(
            f"wdf{l}",
            lambda: QL.tiled(M.dense_weights(w, l)),
        )
        wr[i] = M.router_blocks(np.asarray(w.router(l), np.float32))
        qn, kn = (w.vec(l, f"self_attn.{n}.weight") for n in ("q_norm", "k_norm"))
        rw[i, : M.RW_LEN] = M.rope_table(a.pos, qn, kn)
        n1, n2 = w.norms(l)
        for c in range(2):
            ht = M.to_res(h0).reshape(D // 8, TOK, 8)[:, 4 * c : 4 * c + 4, :]
            ht = ht if i == 0 else np.zeros_like(ht)
            hin[i, c] = h_way_in(ht, n1, n2)
        kv[i, :, : a.pos] = rng.standard_normal((2, a.pos, DK)) * [[[3.0]], [[1.0]]]
        rwb[i] = rw_block(rw[i])
    kv = kv.astype(bfloat16)
    ho = np.zeros((waves, 2, 1, HT), np.float32)
    zb = np.zeros((16, DK), bfloat16)
    args = [we, wd, wr, rw, kv.copy(), rwb, zb, hin, ho, np.int32(a.pos)]
    opts = dict(stack_size=4096)
    be = XRTBackend(instance_name="layers", output_format="elf", **opts)
    art = be.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        try:
            run = be.load(art)
        except Exception:
            # with run-time parameters, a repeated launch's kernel sits in
            # the segment
            art.kernel = art.kernel.replace("main:", "seg:")
            run = be.load(art)
        t0 = time.perf_counter()
        res = run(*args)
        print(f"[run] first dispatch {1e3 * (time.perf_counter() - t0):.2f} ms")
        be.unload()
        if a.bench:
            tb = XRTBackend(
                instance_name="layers",
                output_format="elf",
                n_perf_iters=a.bench,
                n_warmup_iters=3,
                **opts,
            )
            trun = tb.load(art)
            trun(*args)
            print(f"[bench] {nl} layers: {tb.last_latency_us / 1e3:.3f} ms a dispatch")
            tb.unload()
    got = np.asarray(res[-2]).reshape(waves, 2, -1, D // 8, 4, 8)[:, :, -1]

    def h_at(wv):
        return (
            np.concatenate([got[wv, 0], got[wv, 1]], axis=1)
            .transpose(1, 0, 2)
            .reshape(TOK, D)
        )

    # the reference, float64, routed on the host
    h = h0.astype(np.float64)
    ok = True
    for i, l in enumerate(layers):
        n1, n2 = (np.asarray(v, np.float32) for v in w.norms(l))
        x1 = M.rmsnorm(h, n1).astype(bfloat16)
        att = M.attention_reference(w, l, x1, kv[i].astype(np.float64), a.pos)
        h_att = h + att
        x2 = M.rmsnorm(h_att, n2).astype(bfloat16)
        ex, sc = M.route(
            x2.astype(np.float32),
            torch.from_numpy(np.asarray(w.router(l), np.float32)),
        )
        h_moe = h_att + M.moe_reference(w, l, x2, ex, sc)
        d = M.compare(h_at(i) - h, h_moe - h)
        print(
            f"layer {l} ({len(ex)} experts): its update cos {d[0]:.6f} rel L2 {d[1]:.2e}"
        )
        ok &= d[1] < 4e-2
        # the next layer from the device's h, so each layer is checked alone
        h = h_at(i).astype(np.float64)
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
