# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.6 MoE decode block alone, in one dispatch, on FastFlowLM's tile map
# (FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/iron/layer_aie.py):
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array (shared MvmArray): the shared
#                 expert as a dense projection over all four columns, then two
#                 turns of four routed experts, the expert of rank i on array
#                 column i
#   column 2      the array's assemble memtile (FLM's output router), which
#                 also broadcasts each column its input: x, or the GLU output
#                 its expert needs
#   (5, 3)        the router GEMV into float32 logits (FLM's alpha/beta core
#                 does it between its own projections)
#   (5, 4)        router: the top 8 of 256 and the experts' weight transfers
#                 (WGU[idx[i], ...], an L3 slice at an offset the core chose);
#                 the second turn's once the GLU core has the first turn's
#                 gate/up
#   (2, 2)        RMS: the shared expert's down rows, gated by sigmoid(x . w)
#                 (FLM's rms_residual core does this), to the GLU core
#   (5, 5)        GLU: the shared and routed GLUs, the score-weighted sum of
#                 the routed experts' down rows, the gated shared expert
#
# Weights as FLM's loader arranges them: an expert's up and gate rows
# interleaved in 128-row slices, [row group][col block][4 row blocks][block].
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

# a check's bound on the relative L2 error
TOL = 0.1

R, RB, CB, BB = 4, 32, 256, 4736
D, I, E, K = W.D, W.I_EXP, W.N_EXPERTS, W.TOP_K
NCB_X, NCB_H = D // CB, I // CB  # col blocks of x (8) and of a GLU output (2)
RG_GU, RG_DN = 2 * I // (R * RB), D // (R * RB)  # 8, 16 row groups
TURNS, NCOL = K // 4, 4
NRB_R = E // RB  # router row blocks
GLU, RMS = 0, 1  # the array's consumers
COLUMNS, ASM, MOE_COL = [0, 1, 6, 7], 2, 5
LOGITS, ROUTER, GLU_AT, RMS_AT = (MOE_COL, 3), (MOE_COL, 4), (MOE_COL, 5), (2, 2)


def build_module():
    q4k = dict(link_with="q4k_mvm.o")
    step = air.extern("q4k_step", scalars=[i32], **q4k)
    flush = air.extern("q4k_flush", **q4k)
    r_ = lambda n, s=(): air.extern(  # noqa: E731
        n, link_with="router.o", scalars=list(s)
    )
    r_zero, r_mvm, r_topk = (
        r_("router_zero"),
        r_("router_mvm", [i32, i32]),
        r_("router_topk", [i32]),
    )
    g_ = lambda n, s=(): air.extern(  # noqa: E731
        n, link_with="moe_glu.o", scalars=list(s)
    )
    glu_s, glu_e = g_("glu_shared_round", [i32]), g_("glu_expert_round", [i32])
    acc_r, copy_r = g_("moe_acc_round", [i32, i32]), g_("copy_round", [i32])
    copy_up = g_("copy_up")
    acc_zero, sgate, out_k = (
        g_("acc_zero", [i32]),
        g_("shared_gate", [i32]),
        g_("moe_out", [i32]),
    )

    arr = MvmArray(
        row_block=RB,
        block_bytes=BB,
        assemble_col=ASM,
        acc_words=2 * RB,
        columns=COLUMNS,
    )

    WGU = air.tensor([E, RG_GU, NCB_X, R, BB], ui8)
    WDN = air.tensor([E, RG_DN, NCB_H, R, BB], ui8)
    SGU = air.tensor([NCOL, RG_GU // NCOL, NCB_X, R, BB], ui8)
    SDN = air.tensor([NCOL, RG_DN // NCOL, NCB_H, R, BB], ui8)
    WR = air.tensor([NRB_R, NCB_X, CB * RB], bf16)
    X = air.tensor([D], bf16)
    WSG = air.tensor([D], bf16)
    OUT = air.tensor([D], bf16)

    x_r, wr_r = air.channel("XR"), air.channel("WRR")
    x_l2 = air.channel("XL2")
    # memtile 5: x to the logits core; x, then the shared-expert gate weight,
    # to the RMS core
    x_m5, g_in = air.channel("XM5"), air.channel("GIn")
    shared_ch = air.channel("Shared")  # RMS core to GLU core
    arr.channels(n_dests=2)
    glu_out = air.channel("GluOut")
    out_ch = air.channel("Out")

    with air.launch(name="moe") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    wr_r.put(WR)
                    # one copy of x per gate/up phase: a put empties an L2
                    # buffer, so each broadcast of x needs its own
                    for _ in range(1 + TURNS):
                        x_l2.put(X)
                    x_m5.put(X)
                    x_m5.put(WSG)
                    out_ch.get(OUT)
                    # the shared expert, a dense projection: column i has row
                    # groups i, i + 4, ...
                    arr.put_weights(lambda i, h: [SGU[i, :, :, 2 * h : 2 * h + 2, :]])
                    arr.put_weights(lambda i, h: [SDN[i, :, :, 2 * h : 2 * h + 2, :]])
                    logits = air.alloc([E], f32, scope=seg.shared())
                    scores = air.alloc([16], f32, scope=seg.shared())
                    turn_go = air.alloc([16], f32, scope=seg.shared())

                    def l2(shape):
                        return air.alloc(shape, bf16, scope=seg.private(), column=ASM)

                    # assemble memtile: x, then the GLU outputs each column needs
                    l2x, hs = [l2([D]) for _ in range(1 + TURNS)], l2([I])
                    m5x = air.alloc([D], bf16, scope=seg.private(), column=MOE_COL)
                    m5sg = air.alloc([D], bf16, scope=seg.private(), column=MOE_COL)
                    hx = [l2([NCOL, I]) for _ in range(TURNS)]
                    for b in l2x:
                        x_l2.get(b)
                    x_m5.get(m5x)
                    x_m5.get(m5sg)
                    x_r.put(m5x)
                    g_in.put(m5x)
                    g_in.put(m5sg)
                    glu_out.get(hs)
                    for t in range(TURNS):
                        glu_out.get(hx[t])
                    arr.broadcast(
                        lambda i: [l2x[0], hs]
                        + [b for t in range(TURNS) for b in (l2x[1 + t], hx[t][i, :])]
                    )
                    nrd = RG_GU // NCOL + RG_DN // NCOL + TURNS * (RG_GU + RG_DN)
                    arr.gather(seg, nrd)
                    blk = (
                        2 * NCB_X + 4 * NCB_H + TURNS * (RG_GU * NCB_X + RG_DN * NCB_H)
                    )
                    arr.feed(seg, blk)

                    with air.herd(
                        [range(1)], name="logits", shape=(1,), at=LOGITS
                    ) as hl:

                        @hl.body
                        def _(tx):
                            p = hl.private()
                            x = air.alloc([D], bf16, scope=p)
                            w = air.alloc([CB * RB], bf16, scope=p)
                            lg = air.alloc([E], f32, scope=p)
                            x_r.get(x)
                            r_zero(lg)
                            for rb in air.sequential(0, NRB_R):
                                for cb in air.sequential(0, NCB_X):
                                    wr_r.get(w)
                                    r_mvm(w, x, lg, rb, cb)
                            logits[:] = lg[:]

                    with air.herd(
                        [range(1)], name="router", shape=(1,), at=ROUTER
                    ) as hr:

                        @hr.body
                        def _(tx):
                            p = hr.private()
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

                                def fetch(Wt, t=t):
                                    arr.put_weights(
                                        lambda i, h: [
                                            Wt[
                                                idx[NCOL * t + i],
                                                :,
                                                :,
                                                2 * h : 2 * h + 2,
                                                :,
                                            ]
                                        ]
                                    )

                                fetch(WGU)
                                fetch(WDN)

                    with air.herd([range(1)], name="glu", shape=(1,), at=GLU_AT) as hg:

                        @hg.body
                        def _(tx):
                            p = hg.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            g = air.alloc([arr.round], bf16, scope=p)
                            hsl = air.alloc([I], bf16, scope=p)
                            h = air.alloc([NCOL * I], bf16, scope=p)
                            sdn = air.alloc([D], bf16, scope=p)
                            acc = air.alloc([D], f32, scope=p)
                            y = air.alloc([D], bf16, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            # the shared expert
                            for j in air.sequential(0, RG_GU // NCOL):
                                arr.out_y.get(u, indices=[0, GLU])
                                glu_s(hsl, u, j)
                            glu_out.put(hsl)
                            # the routed experts, a turn of four at a time
                            acc_zero(acc, D)
                            sc[:] = scores[:]
                            for t in range(TURNS):
                                for s in air.sequential(0, RG_GU // 2):
                                    # every round lands in u: the core's DMA
                                    # repeats one block, whatever the loop
                                    arr.out_y.get(u, indices=[0, GLU])
                                    copy_up(g, u)
                                    arr.out_y.get(u, indices=[0, GLU])
                                    glu_e(h, g, u, s)
                                glu_out.put(h)
                                if t + 1 < TURNS:
                                    turn_go[:] = sc[:]
                                for rg in air.sequential(0, RG_DN):
                                    arr.out_y.get(u, indices=[0, GLU])
                                    acc_r(acc, u, sc, t, rg)
                            shared_ch.get(sdn)
                            out_k(y, sdn, acc, D)
                            out_ch.put(y)

                    with air.herd([range(1)], name="rms", shape=(1,), at=RMS_AT) as hm:

                        @hm.body
                        def _(tx):
                            p = hm.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            sdn = air.alloc([D], bf16, scope=p)
                            x = air.alloc([D], bf16, scope=p)
                            wsg = air.alloc([D], bf16, scope=p)
                            g_in.get(x)
                            g_in.get(wsg)
                            for j in air.sequential(0, RG_DN // NCOL):
                                arr.out_y.get(u, indices=[0, RMS])
                                copy_r(sdn, u, j)
                            sgate(x, wsg, sdn, D)
                            shared_ch.put(sdn)

                    def phases():
                        yield Phase(D, NCB_X, NCB_X, step, GLU)  # shared gate/up
                        yield Phase(I, 2 * NCB_H, NCB_H, step, RMS)  # shared down
                        for _t in range(TURNS):
                            yield Phase(D, RG_GU * NCB_X // 2, NCB_X, step, GLU)
                            yield Phase(I, RG_DN * NCB_H // 2, NCB_H, step, GLU)

                    arr.herd(D, flush, phases)

    return launch.build(target="npu2")


def interleave(up, gate, n):
    """n experts' up / gate row blocks [n * 16, NCB, BB] -> [n, 8 row groups
    (up slice s, then gate slice s), NCB, 4, BB]."""
    u = up.reshape(n, 4, R, NCB_X, BB).transpose(0, 1, 3, 2, 4)
    g = gate.reshape(n, 4, R, NCB_X, BB).transpose(0, 1, 3, 2, 4)
    return np.stack([u, g], 2).reshape(n, RG_GU, NCB_X, R, BB)


def down(dn, n):
    """n experts' down row blocks [n * 64, 2, BB] -> [n, 16, 2, 4, BB]."""
    return dn.reshape(n, RG_DN, R, NCB_H, BB).transpose(0, 1, 3, 2, 4)


def per_column(t):
    """A dense projection's row groups [rg, ...] -> [column, rg / 4, ...]:
    column i takes row groups i, i + 4, ..."""
    return np.ascontiguousarray(t.reshape(-1, NCOL, *t.shape[1:]).swapaxes(0, 1))


def weights(w, layer):
    m = lambda n: np.asarray(w.raw(w.name(layer, f"mlp.{n}.weight")))  # noqa: E731
    wgu = np.ascontiguousarray(interleave(m("up_exps_proj"), m("gate_exps_proj"), E))
    wdn = np.ascontiguousarray(down(m("down_exps_proj"), E))
    sgu = per_column(
        interleave(m("share_up_exps_proj"), m("share_gate_exps_proj"), 1)[0]
    )
    sdn = per_column(down(m("share_down_exps_proj"), 1)[0])
    router, sg = w.router(layer)
    wr = router.reshape(NRB_R, RB, NCB_X, CB).transpose(0, 2, 3, 1)
    wr = np.ascontiguousarray(wr.astype(bfloat16)).reshape(NRB_R, NCB_X, CB * RB)
    return wgu, wdn, sgu, sdn, wr, sg.astype(bfloat16)


def hw_scores(z):
    """Softmax of the top logits (largest first) through the AIE2P exp2."""
    t = (z - z[0]) * np.log2(np.e)
    fl = np.floor(t)
    e = np.ldexp(1 + (t - fl), fl.astype(int))
    return e / e.sum()


def bench(art, inputs, n):
    """Wall time of n dispatches, the weights already on the device."""
    import time

    import pyxrt as xrt

    dev = xrt.device(0)
    ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
    kern = xrt.ext.kernel(ctx, art.kernel)
    bos = []
    for t in inputs:
        b = t.view(np.uint8).ravel()
        bo = xrt.ext.bo(dev, b.nbytes)
        np.frombuffer(bo.map(), np.uint8)[:] = b
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos.append(bo)
    run = xrt.run(kern)
    for i, bo in enumerate(bos):
        run.set_arg(i, bo)
    times = []
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        for _ in range(n):
            t0 = time.perf_counter()
            run.start()
            st = run.wait(120000)
            if st is not None and st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch ended in {st}")
            times.append(time.perf_counter() - t0)
    t = np.array(times[1:]) * 1e3
    p10, p50, p90 = np.percentile(t, [10, 50, 90])
    print(
        f"{n} dispatches: median {p50:.3f} ms, p10 {p10:.3f}, p90 {p90:.3f},"
        f" min {t.min():.3f} ms"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=9338)
    ap.add_argument("--bench", type=int, default=0, help="time n dispatches")
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        return 0
    for k in ("q4k_mvm", "router", "moe_glu"):
        common.compile_kernel(k)
    w = W.Qwen36Q4nx()
    x = w.embed([a.token])[0]
    x = (x / np.sqrt((x * x).mean() + W.EPS) * w.norms(a.layer)[1]).astype(bfloat16)
    wgu, wdn, sgu, sdn, wr, sg = weights(w, a.layer)
    # eight buffers: a full ELF (the xclbin path binds at most five)
    be = XRTBackend(instance_name="moe", output_format="elf")
    art = be.compile(module)
    if a.bench:
        return bench(
            art, [wgu, wdn, sgu, sdn, wr, x, sg, np.zeros(D, bfloat16)], a.bench
        )
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = run(wgu, wdn, sgu, sdn, wr, x, sg, np.zeros(D, bfloat16))[7]
        be.unload()

    from reference import Reference, t

    ref = Reference(w, layers=[a.layer])
    xt = t(x.astype(np.float32))[None]
    want, (idx, top) = ref.moe(a.layer, xt)
    lg = (t(w.router(a.layer)[0].astype(bfloat16).astype(np.float32)) @ xt[0]).numpy()
    hw = hw_scores(np.sort(lg)[::-1][:K])
    # the same block with the hardware's routing weights
    p = ref.weights(a.layer)
    routed = sum(
        float(s)
        * ((torch.nn.functional.silu(xt[0] @ t(eg).T) * (xt[0] @ t(eu).T)) @ t(ed).T)
        for e, s in zip(idx[0].tolist(), hw)
        for eg, eu, ed in [w.expert(a.layer, e)]
    )
    sgw, sup, sdw = p["shared"]
    shared = (torch.nn.functional.silu(xt[0] @ sgw.T) * (xt[0] @ sup.T)) @ sdw.T
    want_hw = routed + torch.sigmoid(xt[0] @ p["router"][1]) * shared
    g = out.astype(np.float64)
    ok = True
    refs = [("exact softmax", want[0]), ("hw exp2 softmax", want_hw)]
    for name, wv in refs:
        wv = wv.double().numpy()
        c, l2, e = common.compare(g, wv)
        print(f"vs {name}: corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
        ok &= c > 0.999 and l2 < TOL
    print("experts", idx[0].tolist())
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
