# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Two tokens' Q4_K projection from the bundle through q4k_step2 / q4k_flush2
# (kernels/q4k_mvm2.cc) on one column of four cores, as q4k_mvm.py runs one
# token's; each token against its float64 product. --bench also times the
# one-token kernel on the same blocks.
import argparse
import os
import tempfile
import time

import filelock
import numpy as np
from ml_dtypes import bfloat16

import common
import qwen38_27b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32, ui8
from air.backend.xrt import XRTBackend

R, RB, CB, BLOCK = 4, 32, 256, 4736
# the element-wise atol, over max |want|, as q4k_mvm.py's
ATOL = 2e-2


def build_module(nrg, ncb, two=True):
    nt = 2 if two else 1
    obj = "q4k_mvm2.o"
    sfx = "2" if two else ""
    step = air.extern(
        f"q4k_step{sfx}", link_with=obj, scalars=[i32] * (4 if two else 1)
    )
    flush = air.extern(f"q4k_flush{sfx}", link_with=obj, scalars=[i32] if two else [])
    K = ncb * CB
    Wt = air.tensor([nrg, ncb, R, BLOCK], ui8)
    X = air.tensor([nt * K], bf16)
    YW = 16 + nt * RB  # the packet header slot, then each token's rows
    Y = air.tensor([nrg, R, YW], bf16)
    w_l2 = air.channel("WL2")
    w_l1 = air.channel("WL1", size=[1, R])
    x_l1 = air.channel("XL1", size=[1, 1], broadcast_shape=[1, R])
    y_out = air.channel("YOut", size=[1, R])

    with air.launch(name="mvm2") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    x_l1.put(X)
                    l2w = air.alloc([R, BLOCK], ui8, scope=seg.private())
                    for rg in air.sequential(0, nrg):
                        for cb in air.sequential(0, ncb):
                            w_l2.put(Wt[rg, cb, :, :])
                            w_l2.get(l2w)
                            for r in range(R):
                                w_l1.put(l2w[r, :], indices=[0, r])
                    for r in range(R):
                        y_out.get(Y[:, r, :], indices=[0, r])

                    with air.herd([range(1), range(R)], name="mvm", shape=(1, R)) as h:

                        @h.body
                        def _(tx, ty):
                            p = h.private()
                            w = air.alloc([BLOCK], ui8, scope=p)
                            x = air.alloc([nt * K], bf16, scope=p)
                            acc = air.alloc([nt * 2 * RB], f32, scope=p)
                            y = air.alloc([YW], bf16, scope=p)
                            x_l1.get(x, indices=[tx, ty])
                            for rg in air.sequential(0, nrg):
                                for cb in air.sequential(0, ncb):
                                    w_l1.get(w, indices=[tx, ty])
                                    if two:
                                        step(x, w, acc, cb, K, 0, 0)
                                    else:
                                        step(x, w, acc, cb)
                                if two:
                                    flush(acc, y, 0)
                                else:
                                    flush(acc, y)
                                y_out.put(y, indices=[tx, ty])

    return launch.build(target="npu2")


def run(two, blocks, x, nrg, ncb, n):
    """The module's y and the median of n timed dispatches."""
    import pyxrt as xrt

    be = XRTBackend(instance_name="mvm2")
    art = be.compile(build_module(nrg, ncb, two))
    nt = 2 if two else 1
    y = np.zeros((nrg, R, 16 + nt * RB), bfloat16)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        f = be.load(art)
        out = f(blocks, x[: nt * ncb * CB].copy(), y)[2]
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            f(blocks, x[: nt * ncb * CB].copy(), y)
            ts.append(time.perf_counter() - t0)
        be.unload()
    out = np.asarray(out).reshape(nrg, R, 16 + nt * RB)[:, :, 16:]
    return out.reshape(nrg, R, nt, RB), float(np.median(ts)) if ts else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="mlp.gate_proj.weight")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=1024, help="first N rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    common.compile_kernel("q4k_mvm2", extra=os.environ.get("MVM2_FLAGS", "").split())
    w = W.Qwen38Q4nx()
    name = w.name(a.layer, a.tensor)
    _, ncb, _ = w.shape(name)
    nrg = a.rows // RB // R
    raw = np.asarray(w.raw(name)).reshape(-1, ncb, BLOCK)[: nrg * R]
    blocks = raw.reshape(nrg, R, ncb, BLOCK).transpose(0, 2, 1, 3).copy()
    K = ncb * CB
    x = np.random.default_rng(a.seed).standard_normal(2 * K).astype(bfloat16)
    wf = w.dequant_rows(name, 0, nrg * R * RB, K).astype(np.float64)
    out, t2 = run(True, blocks, x, nrg, ncb, a.bench)
    ok = True
    for tk in range(2):
        want = wf @ x[tk * K : (tk + 1) * K].astype(np.float64)
        got = out[:, :, tk, :].reshape(-1).astype(np.float64)
        ok &= common.check(f"token {tk}", got, want, ATOL)
    if a.bench:
        _, t1 = run(False, blocks, x, nrg, ncb, a.bench)
        print(
            f"{nrg * R * RB} rows x {K}: one token {t1 * 1e3:.3f} ms, two {t2 * 1e3:.3f} ms ({t2 / t1:.2f}x)"
        )
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
