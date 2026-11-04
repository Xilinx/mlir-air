# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One expert's full GEMV (gate/up/down row groups, 8832 x 2944) on one column
# of four cores, laid out as FLM streams it: row group rg, column block cb sends
# block (rg, cb, r) to core r, and the 128-element slice cb of x to all four.
# Each row block starts from the biases carried in its column-block-0 block.
import argparse

import torch
from ml_dtypes import bfloat16

import common  # noqa: F401  (sets sys.path)
import gpt_oss_20b_q4nx_weights as g
from air import api as air
from air.api import bf16, f32, ui8
from air.backend.xrt_runner import XRTRunner

NRG, NCB, R = g.EXPERT_ROW_GROUPS, g.D_PAD // g.COL_BLOCK, 4


def build_module():
    bias_init = air.extern("mxfp4_bias_init", link_with="mvm_mxfp4.o")
    acc = air.extern("mxfp4_acc128", link_with="mvm_mxfp4.o")
    flush = air.extern("mxfp4_flush", link_with="mvm_mxfp4.o")

    W = air.tensor([NRG, NCB, R, g.BLOCK_BYTES], ui8)
    X = air.tensor([g.D_PAD], bf16)
    Y = air.tensor([NRG, R, g.ROW_BLOCK], bf16)
    # FLM's dataflow: one 4-block transfer per (rg, cb) into the memtile, split
    # there to the four cores; x broadcast to all four.
    w_l2 = air.channel("WL2")
    w_l1 = air.channel("WL1", size=[1, R])
    x_l1 = air.channel("XL1", size=[1, 1], broadcast_shape=[1, R])
    y_out = air.channel("YOut", size=[1, R])

    with air.launch(name="mvm_mxfp4") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2w = air.alloc([R, g.BLOCK_BYTES], ui8, scope=seg.private())
                    for rg in air.sequential(0, NRG):
                        for cb in air.sequential(0, NCB):
                            w_l2.put(W[rg, cb, :, :])
                            w_l2.get(l2w)
                            for r in range(R):
                                w_l1.put(l2w[r, :], indices=[0, r])
                            x_l1.put(X[cb * g.COL_BLOCK : (cb + 1) * g.COL_BLOCK])
                    for r in range(R):
                        y_out.get(Y[:, r, :], indices=[0, r])

                    with air.herd(
                        [range(1), range(R)], name="expert", shape=(1, R)
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            w = air.alloc([g.BLOCK_BYTES], ui8, scope=h.private())
                            x = air.alloc([g.COL_BLOCK], bf16, scope=h.private())
                            yacc = air.alloc([g.ROW_BLOCK], f32, scope=h.private())
                            y = air.alloc([g.ROW_BLOCK], bf16, scope=h.private())
                            for rg in air.sequential(0, NRG):
                                w_l1.get(w, indices=[tx, ty])
                                x_l1.get(x, indices=[tx, ty])
                                bias_init(w, yacc)
                                acc(x, w, yacc)
                                for cb in air.sequential(1, NCB):
                                    w_l1.get(w, indices=[tx, ty])
                                    x_l1.get(x, indices=[tx, ty])
                                    acc(x, w, yacc)
                                flush(yacc, y)
                                y_out.put(y, indices=[tx, ty])

    return launch.build(target="npu2")


def bf16_floor(t):
    """fp32 -> bf16 rounding toward -inf, the core's default mode, which FLM's
    kernels run in (they never set one)."""
    b = t.contiguous().view(torch.int32)
    hi = b & ~0xFFFF
    up = ((b & 0xFFFF) != 0) & (b < 0)  # negative with dropped bits: one ulp down
    return (hi + up.to(torch.int32) * 0x10000).view(torch.float32)


def reference(layer, expert, seed):
    """Expected output with FLM's rounding: each 32-column group's partial sum
    is rounded to bf16 (toward -inf) before the e8m0 scale is applied."""
    w = g.GptOssQ4nx()
    t = w._f.get_slice(f"model.layers.{layer}.ffn_gate_up_down_exps.weight")[expert]
    t = t.view(torch.uint8).contiguous()
    blocks = g._ungroup_rows(t, R)
    wf, bias = g.dequant_mxfp4(blocks)
    scale = g._e8m0(blocks[..., 0:128]).reshape(-1, NCB, 4, g.ROW_BLOCK)
    scale = scale.permute(0, 3, 1, 2).reshape(-1, NCB * 4)  # [rows, groups]
    x = torch.zeros(g.D_PAD)
    x[: g.D] = torch.randn(g.D, generator=torch.Generator().manual_seed(seed))
    x = x.to(torch.bfloat16).float()
    e2m1 = (wf / scale.repeat_interleave(32, dim=1)).reshape(wf.shape[0], -1, 32)
    part = bf16_floor((e2m1 * x.reshape(1, -1, 32)).sum(-1))
    y = (part * scale).sum(-1) + bias
    xb = x.to(torch.bfloat16).view(torch.int16).numpy().view(bfloat16)
    return t.numpy(), xb, bf16_floor(y).reshape(NRG, R, g.ROW_BLOCK).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    a = ap.parse_args()

    module = build_module()
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("mvm_mxfp4", moe=True)
    wbytes, x, y = reference(a.layer, a.expert, a.seed)
    runner = XRTRunner(output_format=a.output_format, instance_name="mvm_mxfp4")
    # bit-exact: the reference reproduces the kernel's rounding
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[wbytes, x],
            expected_outputs=[y.astype(bfloat16)],
            rtol=0,
            atol=0,
        )
    )
