# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Clamped SwiGLU slices (gate[512] ++ up[512] -> 512) against the HF formula.
import argparse

import torch
from ml_dtypes import bfloat16

import common  # noqa: F401  (sets sys.path)
import gpt_oss_20b_q4nx_weights as g
from air import api as air
from air.api import bf16
from air.backend.xrt_runner import XRTRunner

N = 23  # slices, one MoE layer's worth per expert column


def build_module():
    glu = air.extern("swiglu_slice", link_with="swiglu.o")
    X = air.tensor([N, 1024], bf16)
    Y = air.tensor([N, 512], bf16)
    with air.launch(name="swiglu") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="glu", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            x = air.alloc([1024], bf16, scope=h.private())
                            y = air.alloc([512], bf16, scope=h.private())
                            for i in air.sequential(0, N):
                                air.ops.load(x, X[i, :])
                                glu(x, y)
                                air.ops.store(y, Y[i, :])

    return launch.build(target="npu2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("swiglu")
    # wide enough that both clamps engage
    x = 4 * torch.randn(N, 2, 512, generator=torch.Generator().manual_seed(a.seed))
    x = x.to(torch.bfloat16).float()
    y = g.swiglu(x[:, 0], x[:, 1])
    # FLM's glu LUT spans [-4, 4) in steps of 1/8 and its first segment is
    # zero, so glu is 0 below -3.875
    y = torch.where(x[:, 0] < -3.875, torch.zeros_like(y), y)
    runner = XRTRunner(instance_name="swiglu")
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[x.reshape(N, 1024).numpy().astype(bfloat16)],
            expected_outputs=[y.numpy().astype(bfloat16)],
            rtol=3e-2,
            atol=3e-2,
        )
    )
