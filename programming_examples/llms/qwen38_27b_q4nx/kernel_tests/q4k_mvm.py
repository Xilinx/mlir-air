# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Q4_K projection (or the Q8 LM head) from the bundle through
# q4k_step / q4k_flush (q8_*) on one column of four cores: each core takes one
# 32-row block of a group of four, its blocks as stored, and x whole.
import argparse

import numpy as np
from ml_dtypes import bfloat16

import common
import qwen38_27b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32, ui8
from air.backend.xrt_runner import XRTRunner

R, RB, CB = 4, 32, 256


def build_module(nrg, ncb, kernel="q4k", BLOCK=4736):
    obj = f"{kernel}_mvm.o"
    step = air.extern(f"{kernel}_step", link_with=obj, scalars=[i32])
    flush = air.extern(f"{kernel}_flush", link_with=obj)
    Wt = air.tensor([nrg, ncb, R, BLOCK], ui8)
    X = air.tensor([ncb * CB], bf16)
    Y = air.tensor([nrg, R, RB], bf16)
    w_l2 = air.channel("WL2")
    w_l1 = air.channel("WL1", size=[1, R])
    x_l1 = air.channel("XL1", size=[1, 1], broadcast_shape=[1, R])
    y_out = air.channel("YOut", size=[1, R])

    with air.launch(name=f"{kernel}_mvm") as launch:

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
                            x = air.alloc([ncb * CB], bf16, scope=p)
                            acc = air.alloc([2 * RB], f32, scope=p)
                            y = air.alloc([16 + RB], bf16, scope=p)
                            x_l1.get(x, indices=[tx, ty])
                            for rg in air.sequential(0, nrg):
                                for cb in air.sequential(0, ncb):
                                    w_l1.get(w, indices=[tx, ty])
                                    step(x, w, acc, cb)
                                flush(acc, y)
                                y_out.put(y[16 : 16 + RB], indices=[tx, ty])

    return launch.build(target="npu2")


TENSORS = {
    "k": ("self_attn.k_proj.weight", 3),
    "o": ("self_attn.o_proj.weight", 3),
    "z": ("self_attn.gate_proj.weight", 0),
    "gate": ("mlp.gate_proj.weight", 0),
    "down": ("mlp.down_proj.weight", 0),
    "lm_head": ("lm_head.weight", None),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--tensor", choices=list(TENSORS), default="k")
    ap.add_argument("--rows", type=int, default=0, help="first N rows only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    part, layer = TENSORS[a.tensor]
    w = W.Qwen38Q4nx()
    name = part if layer is None else w.name(layer, part)
    nbi, ncb, BLOCK = w.shape(name)
    kernel = {4736: "q4k", 8704: "q8"}[BLOCK]
    if a.rows:
        nbi = a.rows // RB
    nrg = nbi // R
    module = build_module(nrg, ncb, kernel, BLOCK)
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel(f"{kernel}_mvm")
    raw = np.asarray(w.raw(name)).reshape(-1, ncb, BLOCK)[: nrg * R]
    blocks = raw.reshape(nrg, R, ncb, BLOCK).transpose(0, 2, 1, 3).copy()
    K = ncb * CB
    x = np.random.default_rng(a.seed).standard_normal(K).astype(bfloat16)
    wf = w.dequant_rows(name, 0, nrg * R * RB, K).astype(np.float64)
    y = (wf @ x.astype(np.float64)).reshape(nrg, R, RB)
    runner = XRTRunner(instance_name=f"{kernel}_mvm")
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[blocks, x],
            expected_outputs=[y.astype(bfloat16)],
            rtol=2e-2,
            atol=2e-2 * float(np.abs(y).max()),
            min_correlation=0.9999,
        )
    )
