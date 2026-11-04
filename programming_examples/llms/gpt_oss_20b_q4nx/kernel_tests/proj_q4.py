# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A dense Q4_1 projection from the bundle (32x128 blocks, four row blocks per
# group) through the proj_qmm kernel the other LLMs use, on one column of four
# cores fed like the expert GEMV: one 4-block transfer per (row group, column
# block) split in the memtile, x broadcast.
import argparse

import torch
from ml_dtypes import bfloat16

import common  # noqa: F401  (sets sys.path)
import gpt_oss_20b_q4nx_weights as g
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt_runner import XRTRunner

R = 4
HALF = g.BLOCK_BYTES // 2  # a block as bf16 elements


def build_module(nrg, ncb):
    zero = air.extern("proj_qmm_zero", link_with="proj_qmm.o", scalars=[i32])
    acc = air.extern("proj_qmm_acc256", link_with="proj_qmm.o")
    flush = air.extern("proj_qmm_flush", link_with="proj_qmm.o")
    W = air.tensor([nrg, ncb, R, HALF], bf16)
    X = air.tensor([ncb * g.COL_BLOCK], bf16)
    Y = air.tensor([nrg, R, g.ROW_BLOCK], bf16)
    w_l2 = air.channel("WL2")
    w_l1 = air.channel("WL1", size=[1, R])
    x_l1 = air.channel("XL1", size=[1, 1], broadcast_shape=[1, R])
    y_out = air.channel("YOut", size=[1, R])

    with air.launch(name="proj_q4") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2w = air.alloc([R, HALF], bf16, scope=seg.private())
                    for rg in air.sequential(0, nrg):
                        for cb in air.sequential(0, ncb):
                            w_l2.put(W[rg, cb, :, :])
                            w_l2.get(l2w)
                            for r in range(R):
                                w_l1.put(l2w[r, :], indices=[0, r])
                            x_l1.put(X[cb * g.COL_BLOCK : (cb + 1) * g.COL_BLOCK])
                    for r in range(R):
                        y_out.get(Y[:, r, :], indices=[0, r])

                    with air.herd([range(1), range(R)], name="proj", shape=(1, R)) as h:

                        @h.body
                        def _(tx, ty):
                            w = air.alloc([HALF], bf16, scope=h.private())
                            x = air.alloc([g.COL_BLOCK], bf16, scope=h.private())
                            yacc = air.alloc([g.ROW_BLOCK], f32, scope=h.private())
                            y = air.alloc([g.ROW_BLOCK], bf16, scope=h.private())
                            for rg in air.sequential(0, nrg):
                                zero(yacc, 0)
                                for cb in air.sequential(0, ncb):
                                    w_l1.get(w, indices=[tx, ty])
                                    x_l1.get(x, indices=[tx, ty])
                                    acc(x, w, yacc)
                                flush(yacc, y)
                                y_out.put(y, indices=[tx, ty])

    return launch.build(target="npu2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--proj", choices=["q", "k", "v", "o"], default="q")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    w = g.GptOssQ4nx()
    raw = w.raw(f"model.layers.{a.layer}.self_attn.{a.proj}_proj.weight").view(
        torch.uint8
    )
    nrg, ncb = raw.shape[0], raw.shape[1]
    module = build_module(nrg, ncb)
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("proj_qmm", shared=True)
    wf = g.dequant_q4_1(g._ungroup_rows(raw, R))  # padded rows/cols kept
    x = torch.randn(ncb * g.COL_BLOCK, generator=torch.Generator().manual_seed(a.seed))
    x = x.to(torch.bfloat16).float()
    y = (wf @ x).reshape(nrg, R, g.ROW_BLOCK)
    runner = XRTRunner(instance_name="proj_q4")
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[
                raw.contiguous()
                .view(torch.int16)
                .numpy()
                .view(bfloat16)
                .reshape(nrg, ncb, R, HALF),
                x.numpy().astype(bfloat16),
            ],
            expected_outputs=[y.numpy().astype(bfloat16)],
            rtol=2e-2,
            atol=2e-2 * float(y.abs().max()),
            min_correlation=0.9999,
        )
    )
