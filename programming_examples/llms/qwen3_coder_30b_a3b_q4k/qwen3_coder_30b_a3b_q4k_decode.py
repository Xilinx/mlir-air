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
# row, and the attention cores mask by it. A layers' ELF reads a fixed number
# of cache positions back each layer: maxl, or one of --reads. Each token runs
# the ELF with the fewest that covers L.
import os
import sys
import types
from pathlib import Path

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
import moe_block as MB  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.decode import layer as DL  # noqa: E402
from air_examples.llms.shared.moe import decode as MD  # noqa: E402
from air_examples.llms.shared.decode.array import MvmArray, Phase  # noqa: E402
from kv_bos import kv_groups  # noqa: E402

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


def kv_layout(cap, nl=W.N_LAYERS):
    """KV buffers for a cache of `cap` positions, as fused_decode.kv_layout:
    (elements per layer, layers per buffer, elements per buffer, buffer-arg
    index per buffer). Buffers past the first follow RW."""
    layer = 2 * cap * DK
    group, sizes = kv_groups(nl, layer)
    return layer, group, sizes, [4] + [6 + i for i in range(len(sizes) - 1)]


def build_module(waves, maxl, cap=None):
    """`waves` x PER_WAVE layers, for a context of up to `maxl` positions, on a
    KV cache of `cap` (default maxl) positions."""
    pw = PER_WAVE
    nl = pw * waves
    cap = cap or maxl
    assert maxl % 16 == 0 and maxl <= cap
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
    kv = k_("attn_kv_q36", "attn_kv_blk_z", [i32, i32])
    fin = k_("attn_kv_q36", "attn_kv_fin")
    K = types.SimpleNamespace(
        rms_norm=rms_norm,
        res_add=res_add,
        copy_r=copy_r,
        glu_e=glu_e,
        acc_r=acc_r,
        copy_up=copy_up,
        acc_zero=acc_zero,
        out_k=out_k,
        r_zero=r_zero,
        r_mvm=r_mvm,
        r_topk=r_topk,
        r_slot=r_slot,
    )

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
    _, kv_group, kv_sizes, _ = kv_layout(cap, nl)
    kv_rows = [n // (2 * cap * DK) for n in kv_sizes]
    KV = air.tensor([kv_rows[0], 2, cap, DK], bf16, inout=True)
    RW = air.tensor([nl, 3 * DH], bf16)
    KVS = [KV] + [air.tensor([n, 2, cap, DK], bf16, inout=True) for n in kv_rows[1:]]
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

            def kv_traffic(i, lay, pos):
                kv, l = KVS[i], (lay - i * kv_group if i else lay)
                k_out.get(kv[l, 0, pos, :])
                v_out.get(kv[l, 1, pos, :])
                for c in range(len(CU_COLS)):
                    hs = slice(c * HPM, (c + 1) * HPM)
                    k_rd.put(kv[l, 0, 0:maxl, hs], indices=[c])
                    v_rd.put(kv[l, 1, 0:maxl, hs], indices=[c])

            def kv_split_traffic(lay, pos):
                # one case per KV buffer, keyed by the layer's; folds once the
                # waves unroll
                if len(KVS) == 1:
                    return kv_traffic(0, lay, pos)
                last = len(KVS) - 1
                with air.ops.switch(lay // kv_group, cases=list(range(last))) as arm:
                    kv_traffic(last, lay, pos)
                for i in range(last):
                    with arm.case(i):
                        kv_traffic(i, lay, pos)

            with air.segment(name="seg", params=[ebase, lbase, Lp]) as seg:

                @seg.body
                def _():
                    def l2(shape, col, dt=bf16):
                        return DL.l2(seg, shape, col, dt)

                    wave_loop, herd = DL.wave_loop, DL.herd
                    l1 = MD.moe_l1(seg, E)
                    logits = l1[0]

                    # ---- memtile 5: the RMS core's inputs and side outputs
                    MD.memtile5(
                        seg, pw, D, 3, (to_ab, out_ch), m_in, to_rms, glu_y, xs_ch
                    )

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

                    # ---- the assemble memtile: the normed x, o, and each
                    # turn's GLU outputs
                    MD.assemble(
                        seg,
                        arr,
                        pw,
                        MB.ASM,
                        (D, DQ, [NCOL, I]),
                        1 + TURNS,
                        TURNS,
                        xn_ch,
                        to_asm,
                        lambda xa, b, i: [xa, b[0]] + [xa, b[1][i, :]] * TURNS,
                    )
                    rd = RG_KV + RG_Q + RG_O + TURNS * (RG_GU + RG_DN)
                    arr.gather(seg, pw * rd)
                    arr.feed(seg, pw * (A.G_DENSE + TURNS * G_EXP))

                    # ---- RMS core
                    MD.rms_core(
                        arr,
                        pw,
                        A.RMS_AT,
                        D,
                        RG_O,
                        TURNS,
                        (to_rms, xn_ch, xs_ch, RMS),
                        K,
                    )

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
                                MD.router_gemv(
                                    x, blk, lg, logits, to_ab, ab_w, K, A.NRB_R, NCB_X
                                )

                    # ---- router: each layer's dense weights, then its two
                    # turns of experts
                    MD.router_core(
                        arr, pw, MB.ROUTER, ebase, lbase, E, NCOL, TURNS, WD, WE, l1, K
                    )

                    # ---- GLU
                    MD.glu_core(
                        arr,
                        pw,
                        MB.GLU_AT,
                        (D, I, NCOL),
                        (SLICES, RG_DN, TURNS),
                        l1,
                        (to_asm, glu_y, None),
                        K,
                        GLU,
                    )

                    # ---- the GEMV array
                    def phases(j):
                        yield Phase(D, RG_KV * NCB_X // 2, NCB_X, step, ROPE)
                        yield Phase(
                            D, RG_Q * NCB_X // 2, NCB_X, step, ROPE, fresh=False
                        )
                        yield Phase(DQ, RG_O * NCB_O // 2, NCB_O, step, RMS, xbuf=1)
                        yield from MD.moe_phases(
                            step,
                            (D, I),
                            (RG_GU, RG_DN, TURNS, NCB_X, NCB_H),
                            (GLU, RMS),
                        )

                    arr.herd([DQ, DQ], flush, phases, layers=pw)

                    # ---- host side. Per wave: x in, then the layers'
                    # outputs; then each layer's transfers
                    DL.wave_x(m_in, out_ch, X, w, pw)
                    for j in range(pw):
                        lay = w * pw + j
                        MD.layer_misc(m_in, ab_w, M, MISC, lay, j, "w2", "router")
                        rw_in.put(RW[lay, :])
                        pos = Lp.as_index() - 1
                        kv_split_traffic(lay, pos)

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


class Q3CDecoder(DL.Decoder):
    """The layers' ELFs and the LM head's, their buffers, and a run of each.
    Per token the host writes x, the layers' RoPE rows and L, and runs the
    layers' ELF with the shortest cache read that covers L."""

    def __init__(self, decs, lart, args, lm_args, waves, pw, qkn, cap, we_slack=0):
        """decs: (positions read, ELF artifact, params.txt) per layers' ELF,
        all on the same buffers. args: the buffer args but the KV cache, which
        is allocated here for `cap` positions. we_slack: bytes past the experts
        (args[0]) that a reader of WE other than this decode may touch."""
        import lm_head as LM
        from decode_elf import param_writers
        from kv_bos import KVBuffers
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        super().__init__(waves, pw, D, LM.unpack_logits)
        xrt, self.qkn = self.xrt, qkn

        def zeros(n, _arg):
            o = xrt.ext.bo(self.dev, n)
            np.frombuffer(o.map(), np.uint8)[:] = 0
            o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            return o

        bos = [self.bo(t, we_slack if i == 0 else 0) for i, t in enumerate(args)]
        layout = types.SimpleNamespace(kv_layout=lambda c: kv_layout(c, waves * pw))
        self.cap, self.kv = cap, KVBuffers(layout, cap, zeros)
        self.bos = bos[:4] + [self.kv.kvc] + bos[4:] + list(self.kv.extra)
        self.decs = []
        for read, art, params in sorted(decs, key=lambda d: d[0]):
            ctx, kern = self.kernel(art)
            run = xrt.run(kern)
            for i, b in enumerate(self.bos):
                run.set_arg(i, b)
            self.decs.append(
                dict(
                    read=read,
                    ctx=ctx,
                    run=run,
                    params=ParameterScratchpad(run, str(params)),
                    writers=param_writers(params, {"L": len(self.bos)}),
                )
            )
        self.rwmap = np.frombuffer(self.bos[5].map(), bfloat16).reshape(-1, 3 * DH)
        self.attach_lm(lart, lm_args)

    def _set(self, x, L):
        xrt = self.xrt
        self.xmap[0, self.pw - 1] = x
        for i, (qn, kn) in enumerate(self.qkn):
            self.rwmap[i] = A.rope_weights(L - 1, qn, kn)
        for i in (1, 5):
            self.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        dec = next((d for d in self.decs if d["read"] >= L), None)
        if dec is None:
            raise ValueError(f"L={L} is past the cache ({self.decs[-1]['read']})")
        for name, fn in dec["writers"]:
            dec["params"].write(name, np.int32(fn({len(self.bos): L})))
        dec["params"].sync()
        self.run = dec["run"]

    def step(self, x, L):
        """Decode x at position L - 1; returns the logits."""
        self._set(x, L)
        return self._go((self.run, self.lrun), L)

    def seed(self, kv, n):
        """A prefill's K and V rows of its n positions, per layer:
        kv[layer] = (K [n, DK], V [n, DK])."""
        to_dev = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        for l, (k, v) in enumerate(kv):
            bo, off = self.kv.locate(l)
            for r, t in enumerate((k, v)):
                rows = np.ascontiguousarray(np.asarray(t[:n], bfloat16)).view(np.uint8)
                at = off + r * self.cap * DK * 2
                bo.write(rows.ravel(), at)
                bo.sync(to_dev, rows.nbytes, at)


def compile_all(waves, maxl, reads=()):
    """Kernels, the layers' ELFs and the LM head's in lm/: one layers' ELF
    reading the whole cache of maxl positions, here, and one per entry of
    reads reading only that many, in read<n>/. Returns ([(positions read,
    ELF artifact, params.txt)], the LM head's artifact)."""
    import lm_head as LM

    A.compile_kernels()
    here = os.getcwd()
    decs = []
    for read in sorted({*reads, maxl}):
        d = here if read == maxl else os.path.join(here, f"read{read}")
        os.makedirs(d, exist_ok=True)
        os.chdir(d)
        for o in Path(here).glob("*.o"):
            if d != here and not Path(o.name).exists():
                os.symlink(o, o.name)
        art = XRTBackend(instance_name="decode", output_format="elf").compile(
            build_module(waves, read, maxl)
        )
        art.output_binary = os.path.abspath(art.output_binary)
        decs.append((read, art, Path(d) / "params.txt"))
        os.chdir(here)
    os.makedirs("lm", exist_ok=True)
    os.chdir("lm")
    lart = LM.compile_lm_head()
    lart.output_binary = os.path.abspath(lart.output_binary)
    os.chdir(here)
    return decs, lart


def load(w, maxl, waves=W.N_LAYERS // PER_WAVE, we_slack=0, reads=()):
    """Compile (into the current directory) and load the decode: a
    Q3CDecoder of every layer, at a context of up to maxl, with compile_all's
    reads. Hold the NPU lock while it lives. we_slack: as Q3CDecoder's."""
    import lm_head as LM

    decs, lart = compile_all(waves, maxl, reads)
    layers = list(range(PER_WAVE * waves))
    we, wd, M, qkn = layer_inputs(w, layers)
    xs = np.zeros((waves + 1, PER_WAVE, D), bfloat16)
    rw = np.zeros((len(layers), 3 * DH), bfloat16)
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    y = np.zeros((LM.NCOL, LM.ROUNDS, LM.NROW * LM.RB), bfloat16)
    return Q3CDecoder(
        decs,
        lart,
        [we, xs, wd, M, rw],
        [nw, LM.pack_weights(w), y],
        waves,
        PER_WAVE,
        qkn,
        maxl,
        we_slack,
    )
