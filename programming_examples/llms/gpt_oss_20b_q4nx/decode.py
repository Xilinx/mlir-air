# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS-20B decode: all 24 layers and the LM head in one dispatch.
#
# The tile map is FastFlowLM's layer.xclbin
# (FLM_Xclbin/GPTOSS/gpt_npu_bin/iron/flowArch_aie.py):
#
#   (1, 2) RMS core: the input norm; then residual + o bias + the post-attention
#          norm; then the MoE residual, written back over x.
#   (1, 3) relay to the assemble memtile of every array input (FLM's
#          remove-padding core): both norms' outputs and the attention output.
#   (1, 4) RoPE; the attention units in columns 0 and 7.
#   columns 2..5: the shared GEMV array runs every projection
#          in turn -- qkv and o (Q4_1) and the experts' gate/up and down (MXFP4,
#          one expert per column) -- and its egress on the assemble memtile
#          (column 1) delivers each round to RoPE, the RMS core, the GLU core
#          or the router slave.
#   column 6: the MoE router and GLU.
#
# The launch repeats once per layer (air.launch(repeat=)), each wave re-arming
# the router slave's core-started expert fetches; the per-layer weights, router
# block and KV cache are addressed by the wave index. Every wave also takes the final norm of its
# output and runs an LM-head phase on the MVM cores whose round count is zero
# except in the last wave, so every DMA ring sees the same pattern each wave.
#
# The context length L is a runtime scalar (fused_decode's DYNSEQ): it places
# this token's KV slot and RoPE row, and the attention cores mask by it. The
# cache is read back whole (MAXL keys) every layer; sliding layers keep a linear
# cache and mask to their last 128 keys, as fused_decode does.
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "fused_decode"))

import common  # noqa: E402
import gpt_oss_20b_q4nx_weights as g  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from attn_block import BLK, CU_LOC, DK, DQ, NCU, QH, QKV, SLAB, S_BLK  # noqa: E402
from shared.moe.array import MvmArray, Phase  # noqa: E402
from shared.moe.ffn import Moe, MoeShape, router_kernels  # noqa: E402

DH = g.DH
R = 4
NCB = g.D_PAD // g.COL_BLOCK  # 23
NCB_O = DQ // g.COL_BLOCK  # 32
NRG_QKV = (DQ + 2 * DK) // (R * g.ROW_BLOCK)  # 40 row groups
NRG_O = 24  # o_proj rows 2880 padded to 3072
NRG_GU, NRG_DN = 46, 23
NRG_EX = NRG_GU + NRG_DN
NB_ROUTER = g.D // 64
COLY = R * g.ROW_BLOCK  # a column's rows of one round
ROUND = 4 * COLY  # 512
VEC = 3072  # the o rows (padded) and the o bias
UNIT_Q = 16 * DH + QH  # one attention unit's q, then its sinks
# LM head: 1571 row groups of 128, padded to an even number of rounds (the GLU
# core forwards the logits two rounds at a time)
NRD_LM = 394
LM_RG = 4 * NRD_LM
LM_ROWS = LM_RG * COLY
ARRAY = MvmArray(row_block=g.ROW_BLOCK, block_bytes=g.BLOCK_BYTES)
SHAPE = MoeShape(
    n_experts=g.N_EXPERTS,
    top_k=g.TOP_K,
    d_pad=g.D_PAD,
    router_blocks=NB_ROUTER,
    rg_gu=NRG_GU,
    rg_dn=NRG_DN,
    col_block=g.COL_BLOCK,
)
# egress destinations
ROPE_DEST, RMS_DEST, GLU_DEST, SLAVE_DEST = range(4)
PAD_DEST, ROUTER_DEST = range(2)

# one layer's packed bf16 block; the final norm follows the last layer's
MISC = {}
_at = 0
for _n, _sz in (
    ("router", (NB_ROUTER + 1) * 2048),
    ("w1", g.D),
    ("w2", g.D),
    ("bo", g.D),
    ("bias", QKV),
    ("sinks", DH),
):
    MISC[_n] = (_at, _at + _sz)
    _at += _sz
MISC_LEN = _at
WD_QKV = NRG_QKV * NCB * R * g.BLOCK_BYTES
WD_O = NRG_O * NCB_O * R * g.BLOCK_BYTES
WD_LAYER = WD_QKV + WD_O
ARGS = ("X", "WD", "WL", "WE", "M", "RP", "KV", "LG", "L")


def build_module(maxl, n_layers=g.N_LAYERS, lm=True, lm_rounds=NRD_LM):
    assert maxl % 16 == 0
    nb = maxl // 16
    NL = n_layers
    ex = lambda n, s=(), o="mvm.o": air.extern(
        n, link_with=o, scalars=list(s)
    )  # noqa: E731
    mx_flush = ex("mxfp4_flush_hdr")
    q_step, mx_step = ex("proj_qmm_step", [i32]), ex("mxfp4_step", [i32])
    # writes the norm at y+16: a packet put sends from y+14, header first
    rms = ex("rms_norm_aie_hdr", [i32], "rms_residual.o")
    res = ex("residual_add_aie", (), "rms_residual.o")
    rope = ex("rope_compute", [i32], "rope_q.o")
    pack_q = ex("pack_q_sinks", (), "rope_q.o")
    qk = ex("gpt_attn_qk_blk", [i32, i32, i32], "attn.o")
    kv = ex("gpt_attn_kv_blk", [i32], "attn.o")
    fin = ex("gpt_attn_kv_fin", (), "attn.o")
    rk = router_kernels()
    glu = ex("swiglu_slice", (), "swiglu.o")
    arr = ARRAY
    moe = Moe(SHAPE, arr, rk)

    X = air.tensor([g.D], bf16, inout=True)
    WD = air.tensor([NL * WD_LAYER], ui8)
    # LM head, one contiguous run per (column, half): a linear shim BD
    WL = air.tensor([4, 2, NRD_LM * NCB * 2 * g.BLOCK_BYTES], ui8)
    # every layer's experts: expert e of layer l is row 32 l + e
    WE = air.tensor([NL * g.N_EXPERTS, NRG_EX, NCB, R, g.BLOCK_BYTES], ui8)
    M = air.tensor([NL * MISC_LEN + g.D], bf16)
    RP = air.tensor([maxl, DH], bf16)  # cos|sin of each position
    KV = air.tensor([NL, 4, maxl, SLAB], bf16, inout=True)
    LG = air.tensor([LM_ROWS], bf16)
    SEQ = air.tensor([], i32, name="L")

    ch = air.channel
    pkt = dict(channel_type="npu_dma_packet")
    rms_in = ch("RmsIn")
    # the norms' outputs: the relay, and (post-attention) the router
    rms_x = ch("RmsX", size=[1, 1], broadcast_shape=[1, 2], **pkt)
    mlp_out = ch("MlpOut", **pkt)
    layer_out = ch("LayerOut")
    attn_o = ch("AttnO")
    pad_x = ch("PadX")
    wr_m = ch("WRM")
    arr.channels(n_dests=4)
    glu_out = ch("GluOut")
    lg_out = ch("LgOut")
    rw_in = ch("RwIn")
    q_out = ch("QOut")
    k_out = ch("KOut", **pkt)
    v_out = ch("VOut", **pkt)
    q_cu = ch("QCu", size=[NCU])
    k_rd = ch("KRd", size=[2])
    v_rd = ch("VRd", size=[2])
    k_cu = ch("KCu", size=[NCU])
    v_cu = ch("VCu", size=[NCU])
    o_cu = ch("OCu", size=[NCU])

    with air.launch(name="decode", repeat=NL) as launch:

        @launch.body
        def _(w):
            Lp = air.rtp(SEQ.value)
            last = air.rtp(w >= (NL - 1 if lm else NL))
            # sliding layers (the even ones) attend their last 128 keys
            win = air.rtp((1 - w % 2) * g.SLIDING_WINDOW)
            ebase = air.rtp(w * g.N_EXPERTS)  # this layer's row of WE
            ml = w * MISC_LEN

            def m_(n):
                return M[ml + MISC[n][0] : ml + MISC[n][1]]

            with air.segment(name="seg", params=[Lp, last, win, ebase]) as seg:

                @seg.body
                def _():
                    # ---- host side
                    pos = Lp.as_index() - 1
                    rms_in.put(X)
                    for n in ("w1", "w2", "bo"):
                        rms_in.put(m_(n))
                    rms_in.put(M[NL * MISC_LEN : NL * MISC_LEN + g.D])
                    layer_out.get(X)
                    wr_m.put(m_("router"))
                    # the position's RoPE row, then the biases and the sinks
                    rw_in.put(RP[pos, :])
                    rw_in.put(M[ml + MISC["bias"][0] : ml + MISC["sinks"][1]])
                    for half in range(2):
                        k_out.get(KV[w, 2 * half, pos, :])
                        v_out.get(KV[w, 2 * half + 1, pos, :])
                    for half in range(2):
                        k_rd.put(KV[w, 2 * half, :, :], indices=[half])
                        v_rd.put(KV[w, 2 * half + 1, :, :], indices=[half])
                    # dense weights; the experts' come from the router slave.
                    # Column i takes row groups 4r + i.
                    wd = w * WD_LAYER
                    wq = WD[wd : wd + WD_QKV].reshape(
                        NRG_QKV // 4, 4, NCB, R, g.BLOCK_BYTES
                    )
                    wo = WD[wd + WD_QKV : wd + WD_LAYER].reshape(
                        NRG_O // 4, 4, NCB_O, R, g.BLOCK_BYTES
                    )
                    arr.put_weights(
                        lambda i, h: [
                            v[:, i, :, 2 * h : 2 * h + 2, :] for v in (wq, wo)
                        ]
                    )

                    # ---- assemble memtile (column 1)
                    def l2(shape, col=1, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    x1, xo, x2, xf = (
                        l2([g.D_PAD]),
                        l2([DQ]),
                        l2([g.D_PAD]),
                        l2([g.D_PAD]),
                    )
                    hid = l2([NRG_GU // 2, ROUND])
                    pad_x.get(x1)
                    pad_x.get(xo)
                    pad_x.get(x2)
                    for sl in air.sequential(0, NRG_GU // 2):
                        glu_out.get(hid[sl, :])
                    pad_x.get(xf)
                    arr.broadcast(
                        lambda i: [
                            x1,
                            xo,
                            x2,
                            hid[:, i * COLY : i * COLY + g.COL_BLOCK],
                            xf,
                        ]
                    )
                    # the memtile rings are count-free: the LM rounds of the last
                    # wave go round the same buffers
                    arr.gather(seg, (NRG_QKV + NRG_O) // 4 + NRG_EX)
                    arr.feed(
                        seg, (NRG_QKV * NCB + NRG_O * NCB_O) // 4 + NRG_EX * NCB + 1
                    )

                    # ---- aux memtile (column 6): q to the units, o back
                    qm, om = l2([NCU * UNIT_Q], col=6), l2([DQ], col=6)
                    q_out.get(qm)
                    for c in range(NCU):
                        q_cu.put(qm[c * UNIT_Q : (c + 1) * UNIT_Q], indices=[c])
                    for c in range(NCU):
                        o_cu.get(
                            om.reshape(NCU, 2, 8, 8, 8)[
                                c : c + 1, :, :, :, :
                            ].transpose(0, 1, 3, 2, 4),
                            indices=[c],
                        )
                    attn_o.put(om)

                    # ---- KV cache memtiles (columns 0, 7)
                    for half, col in ((0, 0), (1, 7)):
                        km, vm = l2([16, SLAB], col=col), l2([16, SLAB], col=col)
                        for b in air.sequential(0, nb):
                            k_rd.get(km, indices=[half])
                            v_rd.get(vm, indices=[half])
                            for u in range(2):
                                c = 2 * half + u
                                kh = km.reshape(16, 32, 8)[:, 16 * u : 16 * u + 16, :]
                                k_cu.put(kh.transpose(1, 0, 2), indices=[c])
                                vh = vm.reshape(2, 8, 32, 8)[
                                    :, :, 16 * u : 16 * u + 16, :
                                ]
                                v_cu.put(vh.transpose(0, 2, 1, 3), indices=[c])

                    moe.shared_buffers(seg)

                    def herd(name, at, params=()):
                        return air.herd(
                            [range(1)], name=name, shape=(1,), at=at, params=params
                        )

                    # ---- RMS core
                    with herd("rms", (1, 2)) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            x, w1, w2, wf = (
                                air.alloc([g.D_PAD], bf16, scope=p) for _ in range(4)
                            )
                            bo = air.alloc([VEC], bf16, scope=p)
                            o = air.alloc([VEC], bf16, scope=p)  # o rows, MoE out
                            y1, y2, yr, yf = (
                                air.alloc([16 + g.D_PAD], bf16, scope=p)
                                for _ in range(4)
                            )

                            def y_(y):
                                return y[14 : 16 + g.D_PAD]

                            rms_in.get(x[0 : g.D])
                            rms_in.get(w1[0 : g.D])
                            rms_in.get(w2[0 : g.D])
                            rms_in.get(bo[0 : g.D])
                            rms_in.get(wf[0 : g.D])
                            # the array reads D_PAD: the norms leave a zero tail
                            for yy in (y1, y2, yr, yf):
                                air.ops.fill(yy, 0.0)
                            rms(y1, x, w1, 0)
                            rms_x.put(y_(y1), dest=PAD_DEST)
                            for r in air.sequential(0, NRG_O // 4):
                                arr.out_y.get(
                                    o[r * ROUND : (r + 1) * ROUND],
                                    indices=[0, RMS_DEST],
                                )
                            # in place (elementwise): x = h = x + o + bo
                            res(x, x, o)
                            res(x, x, bo)
                            rms(y2, x, w2, 0)
                            rms(yr, x, w2, 0)
                            rms_x.put(y_(y2), dest=PAD_DEST)
                            rms_x.put(y_(yr), dest=ROUTER_DEST)
                            mlp_out.get(o[0 : g.D_PAD])
                            res(x, x, o)  # h + mlp
                            # the final norm, for the LM head of the last wave
                            rms(yf, x, wf, 0)
                            rms_x.put(y_(yf), dest=PAD_DEST)
                            layer_out.put(x[0 : g.D])

                    # ---- relay to the assemble memtile
                    with herd("relay", (1, 3)) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            a = [air.alloc([g.D_PAD], bf16, scope=p) for _ in range(3)]
                            b = [air.alloc([g.D_PAD], bf16, scope=p) for _ in range(3)]
                            oa, ob = air.alloc([DQ], bf16, scope=p), air.alloc(
                                [DQ], bf16, scope=p
                            )
                            # receive and send from separate buffers
                            rms_x.get(a[0], indices=[0, PAD_DEST])
                            b[0][:] = a[0][:]
                            pad_x.put(b[0])
                            attn_o.get(oa)
                            ob[:] = oa[:]
                            pad_x.put(ob)
                            for k in (1, 2):
                                rms_x.get(a[k], indices=[0, PAD_DEST])
                                b[k][:] = a[k][:]
                                pad_x.put(b[k])

                    # ---- RoPE
                    with herd("rope", (1, 4)) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            qkv = air.alloc([QKV], bf16, scope=p)
                            rw = air.alloc([DH + QKV + DH], bf16, scope=p)
                            q = air.alloc([DQ], bf16, scope=p)
                            qp = air.alloc([NCU * UNIT_Q], bf16, scope=p)
                            k, v = air.alloc([DK], bf16, scope=p), air.alloc(
                                [DK], bf16, scope=p
                            )
                            rw_in.get(rw[0:DH])
                            rw_in.get(rw[DH : DH + QKV + DH])
                            for r in air.sequential(0, NRG_QKV // 4):
                                arr.out_y.get(
                                    qkv[r * ROUND : (r + 1) * ROUND],
                                    indices=[0, ROPE_DEST],
                                )
                            rope(q, k, v, qkv, rw, 0)
                            pack_q(q, rw, qp)
                            q_out.put(qp)
                            k_out.put(k)
                            v_out.put(v)

                    # ---- attention units
                    def unit(c, col, row):
                        s = air.alloc([S_BLK], bf16, scope=seg.shared())
                        sink = air.alloc([QH], bf16, scope=seg.shared())
                        with herd(f"qk{c}", (col, row), params=[Lp, win]) as hq:

                            @hq.body
                            def _(tx):
                                p = hq.private()
                                q = air.alloc([UNIT_Q], bf16, scope=p)
                                kb = air.alloc([BLK], bf16, scope=p)
                                m = air.alloc([16], f32, scope=p)
                                cc = air.alloc([16], f32, scope=p)
                                q_cu.get(q, indices=[c])
                                sink[:] = q[16 * DH : UNIT_Q]
                                for b in air.sequential(0, nb):
                                    k_cu.get(kb, indices=[c])
                                    qk(q, kb, m, cc, s, b, Lp, win)

                        with herd(f"kv{c}", (col, row + 1)) as hk:

                            @hk.body
                            def _(tx):
                                p = hk.private()
                                vb = air.alloc([BLK], bf16, scope=p)
                                y = air.alloc([16 * DH], f32, scope=p)
                                l = air.alloc([16], f32, scope=p)
                                ms = air.alloc([16], f32, scope=p)
                                sk = air.alloc([QH], bf16, scope=p)
                                o = air.alloc([16 * DH], bf16, scope=p)
                                sk[:] = sink[:]
                                for b in air.sequential(0, nb):
                                    v_cu.get(vb, indices=[c])
                                    kv(s, vb, y, l, ms, b)
                                fin(y, l, sk, ms, o)
                                o_cu.put(o, indices=[c])

                    for c, (col, row) in enumerate(CU_LOC):
                        unit(c, col, row)

                    # ---- MoE router, GLU
                    moe.master(lambda x: rms_x.get(x, indices=[0, ROUTER_DEST]), wr_m)
                    moe.slave(WE, ebase, SLAVE_DEST, mlp_out)

                    def lm_forward(u, lg):
                        """The LM head's logits, two rounds at a time."""
                        for r2 in air.sequential(0, last.as_index() * (lm_rounds // 2)):
                            arr.out_y.get(u[0:ROUND], indices=[0, GLU_DEST])
                            arr.out_y.get(u[ROUND : 2 * ROUND], indices=[0, GLU_DEST])
                            lg[:] = u[:]
                            lg_out.put(lg)

                    moe.glu(glu, GLU_DEST, glu_out, params=[last], tail=lm_forward)

                    # ---- MVM cores: qkv, o, gate/up, down, LM head
                    def q4(x, wb, yacc, cb):
                        q_step(x, cb, wb, yacc)

                    def mx(x, wb, yacc, cb):
                        mx_step(x, wb, yacc, cb)

                    def phases():
                        yield Phase(
                            NCB * 128, NRG_QKV // 4 * NCB // 2, NCB, q4, ROPE_DEST
                        )
                        yield Phase(DQ, NRG_O // 4 * NCB_O // 2, NCB_O, q4, RMS_DEST)
                        yield Phase(g.D_PAD, NRG_GU * NCB // 2, NCB, mx, GLU_DEST)
                        # the down phase's odd block count, plus the one filler
                        # block the host sends after it
                        yield Phase(
                            g.D_PAD, (NRG_DN * NCB + 1) // 2, NCB, mx, SLAVE_DEST
                        )
                        yield Phase(
                            g.D_PAD,
                            last.as_index() * (lm_rounds * NCB // 2),
                            NCB,
                            q4,
                            GLU_DEST,
                        )

                    arr.herd(DQ, mx_flush, phases, params=[last])

            # one filler block per core after the experts, which keeps every
            # phase's block count even for the cores' two weight buffers
            arr.put_weights(lambda i, h: [WD[0 : 2 * g.BLOCK_BYTES]])
            # the LM head's weights queue behind the experts on the same
            # channels, and only the last wave has logits to drain
            if not lm:
                return
            with air.ops.switch(w >= NL - 1):
                arr.put_weights(
                    lambda i, h: [WL[i, h, 0 : lm_rounds * NCB * 2 * g.BLOCK_BYTES]]
                )
                lg_out.get(LG[0 : lm_rounds * ROUND])

    return launch.build(target="npu2")


BACKEND = dict(
    stack_size=4096,
    omit_while_true_loop=False,
    use_lock_race_condition_fix_v2=True,
)


def compile_decode(maxl, n_layers, out_dir, lm=True, lm_rounds=NRD_LM):
    """Build the kernels and the full ELF in out_dir; returns (elf, kernel)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    common.compile_kernel("mvm", model=True)
    common.compile_kernel("rope_q", model=True)
    common.compile_kernel("rms_residual", shared=True)
    common.compile_kernel("router", moe=True)
    for k in ("swiglu", "attn"):
        common.compile_kernel(k)
    backend = XRTBackend(output_format="elf", instance_name="decode", **BACKEND)
    art = backend.compile(build_module(maxl, n_layers, lm, lm_rounds))
    return out_dir / art.output_binary, art.kernel


def lm_head_blocks(w):
    """FLM's 64-row LM-head groups -> [column, half, round, col block, 2 row
    blocks, block], padded: column i takes row groups 4r + i, half h its row
    blocks 2h and 2h + 1."""
    t = w.raw("lm_head.weight").view(torch.uint8)  # [3142, 23, 2, 2560]
    rb = t.permute(0, 2, 1, 3).reshape(-1, NCB, g.BLOCK_BYTES)
    rb = torch.cat((rb, rb.new_zeros(LM_RG * R - rb.shape[0], NCB, g.BLOCK_BYTES)))
    rb = rb.reshape(NRD_LM, 4, 2, 2, NCB, g.BLOCK_BYTES)
    return rb.permute(1, 2, 0, 4, 3, 5).contiguous()


def misc_layer(w, layer):
    m = torch.zeros(MISC_LEN)

    def put(n, t):
        a, b = MISC[n]
        m[a:b] = t.float().ravel()

    wr = (
        w.raw(f"model.layers.{layer}.mlp.router.weight")
        .reshape(NB_ROUTER, 2048)
        .float()
    )
    rb = torch.zeros(1, 2048)
    rb[0, :32] = w.router(layer)[1]
    put("router", torch.cat((rb, wr)))
    n1, n2 = w.norms(layer)
    att = w.attn(layer)
    put("w1", n1)
    put("w2", n2)
    put("bo", att["bo"])
    put("bias", torch.cat((att["bq"], att["bk"], att["bv"])))
    put("sinks", att["sinks"])
    return m


def _bf16(t):
    return t.float().numpy().astype(bfloat16).view(np.uint8).ravel()


class NpuDecoder:
    """The ELF, its buffers and one persistent run; L goes to the scratchpad."""

    def __init__(self, elf, kernel, maxl, layers, w):
        import pyxrt as xrt
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )
        from decode_elf import param_writers

        self.xrt, self.maxl, self.w = xrt, maxl, w
        nl = len(layers)
        sizes = dict(
            X=g.D * 2,
            WD=nl * WD_LAYER,
            WL=LM_RG * NCB * R * g.BLOCK_BYTES,
            WE=nl * g.N_EXPERTS * g.EXPERT_BYTES,
            M=(nl * MISC_LEN + g.D) * 2,
            RP=maxl * DH * 2,
            KV=nl * 4 * maxl * SLAB * 2,
            LG=LM_ROWS * 2,
        )
        self.dev = xrt.device(0)
        self.ctx = xrt.hw_context(self.dev, xrt.elf(str(elf)))
        self.kern = xrt.ext.kernel(self.ctx, kernel)
        self.bo = {n: xrt.ext.bo(self.dev, s) for n, s in sizes.items()}
        self.map = {n: np.frombuffer(b.map(), np.uint8) for n, b in self.bo.items()}
        t0 = time.time()
        for j, l in enumerate(layers):
            p = f"model.layers.{l}.self_attn."
            wd = torch.cat(
                [
                    w.raw(p + n + ".weight").view(torch.uint8).ravel()
                    for n in ("q_proj", "k_proj", "v_proj", "o_proj")
                ]
            )
            self.map["WD"][j * WD_LAYER : (j + 1) * WD_LAYER] = wd.numpy()
            we = w.raw(f"model.layers.{l}.ffn_gate_up_down_exps.weight")
            n = g.N_EXPERTS * g.EXPERT_BYTES
            self.map["WE"][j * n : (j + 1) * n] = we.view(torch.uint8).numpy().ravel()
        self.map["WL"][:] = lm_head_blocks(w).numpy().ravel()
        misc = torch.cat([misc_layer(w, l) for l in layers] + [w.final_norm()])
        self.map["M"][:] = _bf16(misc)
        cos, sin = g.rope_cos_sin(list(range(maxl)))
        self.map["RP"][:] = _bf16(torch.cat((cos, sin), dim=1))
        self.map["KV"][:] = 0
        for b in self.bo.values():
            b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        print(f"weights in {time.time() - t0:.0f}s")
        self.run = xrt.run(self.kern)
        for i, n in enumerate(ARGS[:-1]):
            self.run.set_arg(i, self.bo[n])
        params = elf.parent / "params.txt"
        self.params = ParameterScratchpad(self.run, str(params))
        self.writers = param_writers(params, {"L": len(ARGS) - 1})

    def step(self, token, L):
        """Decode `token` at position L-1; returns (hidden out, logits)."""
        xrt = self.xrt
        assert 1 <= L <= self.maxl
        self.map["X"][:] = _bf16(self.w.embed(torch.tensor([token]))[0])
        self.bo["X"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for name, fn in self.writers:
            self.params.write(name, np.int32(fn({len(ARGS) - 1: L})))
        self.params.sync()
        self.run.start()
        state = self.run.wait(120000)
        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"dispatch at L={L} ended in {state}")
        for n in ("X", "LG"):
            self.bo[n].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        def f32(n, k):
            return torch.from_numpy(
                self.map[n][: 2 * k].view(bfloat16).astype(np.float32)
            )

        return f32("X", g.D), f32("LG", g.VOCAB)

    def bench(self, L, n):
        """Wall time of n dispatches at context length L, nothing else."""
        for name, fn in self.writers:
            self.params.write(name, np.int32(fn({len(ARGS) - 1: L})))
        self.params.sync()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.run.start()
            self.run.wait(120000)
            times.append(time.perf_counter() - t0)
        return times


def cos_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


class ChainRef:
    """The CPU reference through an arbitrary sequence of layers."""

    def __init__(self, w, layers):
        self.w, self.layers = w, layers
        self.kv = [[torch.zeros(g.N_KV_HEADS, 0, DH)] * 2 for _ in layers]

    def step(self, token, pos):
        x = self.w.embed(torch.tensor([token]))
        for j, l in enumerate(self.layers):
            n1, n2 = self.w.norms(l)
            x = x + g.attention(self.w.attn(l), l, g.rmsnorm(x, n1), self.kv[j], [pos])
            x = x + g.moe(self.w, l, g.rmsnorm(x, n2))[0]
        return x[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--maxl", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=g.N_LAYERS)
    ap.add_argument(
        "--layers", type=int, nargs="*", help="model layer of each wave (debug)"
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=8, help="greedy tokens")
    ap.add_argument("--build-dir", type=Path, default=Path("build_decode"))
    ap.add_argument("--no-lm", action="store_true", help="skip the LM head")
    ap.add_argument("--bench", type=int, default=0, help="time N dispatches")
    a = ap.parse_args()
    layers = a.layers or list(range(a.n_layers))
    if a.print_module_only:
        print(build_module(a.maxl, len(layers), not a.no_lm))
        return 0
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(g.flm_model_dir() / "tokenizer.json"))
    ids = tok.encode(a.prompt).ids
    elf, kernel = compile_decode(
        a.maxl, len(layers), a.build_dir.resolve(), lm=not a.no_lm
    )
    w = g.GptOssQ4nx()
    full = layers == list(range(g.N_LAYERS))
    ref = g.Reference(w) if full else ChainRef(w, layers)
    ok, head = True, None
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        npu = NpuDecoder(elf, kernel, a.maxl, layers, w)
        if a.bench:
            npu.step(ids[0], 1)
            t = np.array(npu.bench(len(ids), a.bench)) * 1e3
            print(
                f"{len(layers)} layers{'' if a.no_lm else ' + LM head'}: "
                f"{t.mean():.2f} ms mean, {t.min():.2f} min, {t.max():.2f} max "
                f"over {a.bench} dispatches ({1e3 / t.mean():.2f} tok/s)"
            )
            return 0
        seq, out = list(ids), []
        for t in range(len(ids) + a.steps - 1):
            t0 = time.time()
            x, logits = npu.step(seq[t], t + 1)
            dt = time.time() - t0
            if not full:
                # a debug build: its last wave's output, and the LM head on it
                xr = ref.step(seq[t], t)
                c = cos_sim(x, xr)
                ok &= c > 0.99
                msg = f"L={t + 1}: last wave out cos {c:.5f}"
                if a.no_lm:
                    print(f"{msg} ({dt*1e3:.0f} ms)")
                    # no logits to pick the next token from
                    if t + 1 >= len(ids):
                        break
                    continue
                if head is None:
                    head = w.lm_head()
                want = g.rmsnorm(xr[None], w.final_norm())[0] @ head.T
                print(f"{msg} ({dt*1e3:.0f} ms)")
            else:
                want = ref.forward([seq[t]])
            if t + 1 >= len(ids):
                nxt = int(logits.argmax())
                c = cos_sim(logits, want)
                print(
                    f"L={t + 1}: npu {nxt} {tok.decode([nxt])!r} vs ref "
                    f"{int(want.argmax())}  logits cos {c:.5f} ({dt*1e3:.0f} ms)"
                )
                ok &= nxt == int(want.argmax()) and c > 0.99
                seq.append(nxt)
                out.append(nxt)
            elif full:
                print(f"L={t + 1}: prompt ({dt*1e3:.0f} ms)")
    if out:
        print(f"{a.prompt!r} -> {tok.decode(out)!r}")
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
