# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8 dense MLP decode block alone, in one dispatch, on the tile map of
# Qwen3.6-35B-A3B's MoE block (qwen36_35b_a3b_q4nx/moe_block.py), whose shared
# expert is this block at Qwen3.6's width:
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array (shared MvmArray): gate/up, then
#                 down, each a dense projection over all four columns
#   column 2      the array's assemble memtile, which also broadcasts each
#                 column its input: x, then the GLU output
#   (5, 5)        GLU: SiLU(gate) * up of each round, into the down input
#   (2, 2)        the down rows, to the output
#
# Weights as FLM's loader arranges them: up and gate rows interleaved in
# 128-row slices, [row group][col block][4 row blocks][block], row group g on
# column g % 4.
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
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from air_examples.llms.shared.moe.array import MvmArray, Phase  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.1

R, RB, CB, BB = 4, 32, 256, 4736
D, I = W.D, W.I
NCB_X, NCB_H = D // CB, I // CB  # col blocks of x (20) and of the GLU output (68)
SL = R * RB  # a row group: 128 rows, one column's share of a round
RG_GU, RG_DN = 2 * I // SL, D // SL  # 272, 40 row groups
NCOL = 4
GLU, OUT = 0, 1  # the array's consumers
COLUMNS, ASM = [0, 1, 6, 7], 2
GLU_AT, OUT_AT = (5, 5), (2, 2)


def build_module():
    q4k = dict(link_with="q4k_mvm.o")
    step = air.extern("q4k_step", scalars=[i32], **q4k)
    flush = air.extern("q4k_flush", **q4k)
    g_ = lambda n, s=(): air.extern(  # noqa: E731
        n, link_with="q38_ffn.o", scalars=list(s)
    )
    glu_s, copy_r = g_("glu_shared_round", [i32]), g_("copy_round", [i32])

    arr = MvmArray(
        row_block=RB,
        block_bytes=BB,
        assemble_col=ASM,
        acc_words=2 * RB,
        columns=COLUMNS,
    )

    WGU = air.tensor([NCOL, RG_GU // NCOL, NCB_X, R, BB], ui8)
    WDN = air.tensor([NCOL, RG_DN // NCOL, NCB_H, R, BB], ui8)
    X = air.tensor([D], bf16)
    OUT_T = air.tensor([D], bf16)

    x_l2 = air.channel("XL2")
    arr.channels(n_dests=2)
    glu_out = air.channel("GluOut")
    out_ch = air.channel("Out")

    with air.launch(name="mlp") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    x_l2.put(X)
                    out_ch.get(OUT_T)
                    # column i has row groups i, i + 4, ...
                    arr.put_weights(lambda i, h: [WGU[i, :, :, 2 * h : 2 * h + 2, :]])
                    arr.put_weights(lambda i, h: [WDN[i, :, :, 2 * h : 2 * h + 2, :]])

                    def l2(shape):
                        return air.alloc(shape, bf16, scope=seg.private(), column=ASM)

                    # assemble memtile: x, then the GLU output
                    l2x, hs = l2([D]), l2([I])
                    x_l2.get(l2x)
                    glu_out.get(hs)
                    arr.broadcast(lambda i: [l2x, hs])
                    arr.gather(seg, RG_GU // NCOL + RG_DN // NCOL)
                    arr.feed(seg, RG_GU // NCOL * NCB_X + RG_DN // NCOL * NCB_H)

                    with air.herd([range(1)], name="glu", shape=(1,), at=GLU_AT) as hg:

                        @hg.body
                        def _(tx):
                            p = hg.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            hsl = air.alloc([I], bf16, scope=p)
                            for j in air.sequential(0, RG_GU // NCOL):
                                arr.out_y.get(u, indices=[0, GLU])
                                glu_s(hsl, u, j)
                            glu_out.put(hsl)

                    with air.herd([range(1)], name="out", shape=(1,), at=OUT_AT) as ho:

                        @ho.body
                        def _(tx):
                            p = ho.private()
                            u = air.alloc([arr.round], bf16, scope=p)
                            y = air.alloc([D], bf16, scope=p)
                            for j in air.sequential(0, RG_DN // NCOL):
                                arr.out_y.get(u, indices=[0, OUT])
                                copy_r(y, u, j)
                            out_ch.put(y)

                    def phases():
                        yield Phase(D, RG_GU // NCOL * NCB_X // 2, NCB_X, step, GLU)
                        yield Phase(I, RG_DN // NCOL * NCB_H // 2, NCB_H, step, OUT)

                    arr.herd(I, flush, phases)

    return launch.build(target="npu2")


def interleave(up, gate):
    """up / gate row blocks [I / 32, NCB_X, BB] -> [RG_GU (up slice s, then
    gate slice s), NCB_X, 4, BB]."""
    n = I // SL
    u = up.reshape(n, R, NCB_X, BB).transpose(0, 2, 1, 3)
    g = gate.reshape(n, R, NCB_X, BB).transpose(0, 2, 1, 3)
    return np.stack([u, g], 1).reshape(RG_GU, NCB_X, R, BB)


def down(dn):
    """down row blocks [D / 32, NCB_H, BB] -> [RG_DN, NCB_H, 4, BB]."""
    return dn.reshape(RG_DN, R, NCB_H, BB).transpose(0, 2, 1, 3)


def per_column(t):
    """A dense projection's row groups [rg, ...] -> [column, rg / 4, ...]:
    column i takes row groups i, i + 4, ..."""
    return np.ascontiguousarray(t.reshape(-1, NCOL, *t.shape[1:]).swapaxes(0, 1))


def weights(w, layer):
    m = lambda n: np.asarray(w.raw(w.name(layer, f"mlp.{n}.weight")))  # noqa: E731
    gu = per_column(interleave(m("up_proj"), m("gate_proj")))
    dn = per_column(down(m("down_proj")))
    return gu, dn


def bench(art, inputs, n, weights):
    """Wall time of n dispatches, the weights already on the device; weights
    are the indices of the inputs counted as the dispatch's weights."""
    import time

    import pyxrt as xrt

    times = []
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
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
        for _ in range(n):
            t0 = time.perf_counter()
            run.start()
            st = run.wait(120000)
            if st is not None and st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch ended in {st}")
            times.append(time.perf_counter() - t0)
        del run, kern, ctx
    t = np.array(times[1:]) * 1e3
    p10, p50, p90 = np.percentile(t, [10, 50, 90])
    gb = sum(inputs[i].nbytes for i in weights) / 1e9
    print(
        f"{n} dispatches: median {p50:.3f} ms, p10 {p10:.3f}, p90 {p90:.3f},"
        f" min {t.min():.3f} ms; {gb / p50 * 1e3:.1f} GB/s of weights at the median"
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
    for k in ("q4k_mvm", "q38_ffn"):
        common.compile_kernel(k)
    w = W.Qwen38Q4nx()
    x = w.embed([a.token])[0]
    x = (x / np.sqrt((x * x).mean() + W.EPS) * w.norms(a.layer)[1]).astype(bfloat16)
    gu, dn = weights(w, a.layer)
    be = XRTBackend(instance_name="mlp", output_format="elf")
    art = be.compile(module)
    if a.bench:
        return bench(art, [gu, dn, x, np.zeros(D, bfloat16)], a.bench, (0, 1))
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = run(gu, dn, x, np.zeros(D, bfloat16))[3]
        be.unload()

    import torch

    from reference import Reference, t

    xt = t(x.astype(np.float32))[None]
    want = Reference.mlp(xt, Reference(w, layers=[a.layer]).weights(a.layer))[0]
    c, l2, e = common.compare(out, want.double().numpy())
    print(f"corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
    ok = c > 0.999 and l2 < TOL
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
