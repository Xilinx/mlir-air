# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The router on one core: a layer's router weights in 32 x 256 column-major
# blocks against x, then the top 8 experts and their softmax weights.
import argparse
import os
import tempfile

import filelock
import numpy as np
from ml_dtypes import bfloat16

import common
import qwen36_35b_a3b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt import XRTBackend

E, D, RB, CB = W.N_EXPERTS, W.D, 32, 256
NRB, NCB = E // RB, D // CB


def build_module():
    zero = air.extern("router_zero", link_with="router.o")
    mvm = air.extern("router_mvm", link_with="router.o", scalars=[i32, i32])
    topk = air.extern("router_topk", link_with="router.o", scalars=[i32])
    Wt = air.tensor([NRB, NCB, CB * RB], bf16)
    X = air.tensor([D], bf16)
    IDX = air.tensor([16], i32)
    SC = air.tensor([16], f32)
    w_in = air.channel("WIn")
    x_in = air.channel("XIn")
    i_out = air.channel("IOut")
    s_out = air.channel("SOut")

    with air.launch(name="router") as launch:

        @launch.body
        def _():
            x_in.put(X)
            for rb in range(NRB):
                for cb in range(NCB):
                    w_in.put(Wt[rb, cb, :])
            i_out.get(IDX)
            s_out.get(SC)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="router", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            p = h.private()
                            w = air.alloc([CB * RB], bf16, scope=p)
                            x = air.alloc([D], bf16, scope=p)
                            logits = air.alloc([E], f32, scope=p)
                            idx = air.alloc([16], i32, scope=p)
                            sc = air.alloc([16], f32, scope=p)
                            x_in.get(x)
                            zero(logits)
                            for rb in air.sequential(0, NRB):
                                for cb in air.sequential(0, NCB):
                                    w_in.get(w)
                                    mvm(w, x, logits, rb, cb)
                            topk(logits, idx, sc, 0)
                            i_out.put(idx)
                            s_out.put(sc)

    return launch.build(target="npu2")


def hw_softmax(z):
    """The softmax of z (largest first) as the kernel computes it: the AIE2P
    exp2 instruction is linear between powers of two, 2^x = 2^floor(x) *
    (1 + frac(x))."""
    t = (z - z[0]) * np.log2(np.e)
    fl = np.floor(t)
    e = np.ldexp(1 + (t - fl), fl.astype(int))
    return e / e.sum()


def pack(router):
    """router [E, D] -> [NRB, NCB, CB columns, RB rows] blocks."""
    r = router.reshape(NRB, RB, NCB, CB).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(r.astype(bfloat16)).reshape(NRB, NCB, CB * RB)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("router")
    w = W.Qwen36Q4nx()
    router = w.router(a.layer)[0]
    blocks = pack(router)
    be = XRTBackend(instance_name="router")
    art = be.compile(module)
    ok = True
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        for t in range(a.trials):
            rng = np.random.default_rng(t)
            # a post-norm hidden state: a token's embedding, RMS-normalized
            x = w.embed([int(rng.integers(0, W.VOCAB))])[0]
            x = (x / np.sqrt((x * x).mean() + W.EPS)).astype(bfloat16)
            out = run(blocks, x, np.zeros(16, np.int32), np.zeros(16, np.float32))
            idx, sc = out[2][: W.TOP_K], out[3][: W.TOP_K]
            lg = router.astype(np.float64) @ x.astype(np.float64)
            want = np.argsort(-lg, kind="stable")[: W.TOP_K]
            p = hw_softmax(lg[want])
            good = list(idx) == list(want) and np.allclose(sc, p, rtol=1e-2, atol=2e-3)
            ok &= good
            print(f"trial {t}: experts {list(idx)} {'ok' if good else 'MISMATCH'}")
            if not good:
                print("  want", list(want), "scores", sc, "want", p)
        be.unload()
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
