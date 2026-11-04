# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS MoE decode block alone: the MoE half of the decode layer, on the
# shared MoE builders, as FastFlowLM lays it out
# (FLM_Xclbin/GPTOSS/gpt_npu_bin/iron/{flowArch_aie,router,proj_tile}.py):
#
#   column 6: router master (row 4) computes the router logits; the slave
#             (row 3) sorts them, starts the top 4 experts' weight transfers
#             itself -- W[idx[i], ...] is an L3 slice whose offset the core
#             chose -- takes the softmax, and combines the down rows; the GLU
#             core (row 2) activates the gate/up rounds.
#   columns 2..5: the GEMV array; expert rank i runs on column 2 + i.
#
# Checked against the same pipeline emulated on the CPU (and, for scale,
# against exact arithmetic: FLM's kernels round bf16 toward -inf throughout).
import argparse
import os
import sys
import tempfile
from pathlib import Path

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
import gpt_oss_20b_q4nx_weights as g  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
from shared.moe.array import MvmArray, Phase  # noqa: E402
from shared.moe.ffn import Moe, MoeShape, router_kernels  # noqa: E402

R = 4
NCB = g.D_PAD // g.COL_BLOCK  # 23 column blocks
NRG_GU, NRG_DN = 46, 23  # gate/up row groups (interleaved), then down
NRG = g.EXPERT_ROW_GROUPS
NB_ROUTER = g.D // 64  # 45 router blocks of 32x64
GLU_DEST, SLAVE_DEST = range(2)


def build_module():
    """Router, gate/up, GLU, down, combine -> MLP out, in one dispatch."""
    ex = lambda n, s=(), o="mvm.o": air.extern(  # noqa: E731
        n, link_with=o, scalars=list(s)
    )
    mx_flush, mx_step = ex("mxfp4_flush_hdr"), ex("mxfp4_step", [i32])
    glu = ex("swiglu_slice", (), "swiglu.o")
    arr = MvmArray(row_block=g.ROW_BLOCK, block_bytes=g.BLOCK_BYTES)
    shape = MoeShape(
        n_experts=g.N_EXPERTS,
        top_k=g.TOP_K,
        d_pad=g.D_PAD,
        router_blocks=NB_ROUTER,
        rg_gu=NRG_GU,
        rg_dn=NRG_DN,
        col_block=g.COL_BLOCK,
    )
    moe = Moe(shape, arr, router_kernels())

    W = air.tensor([g.N_EXPERTS, NRG, NCB, R, g.BLOCK_BYTES], ui8)
    X = air.tensor([g.D_PAD], bf16)
    WR = air.tensor([NB_ROUTER + 1, g.N_EXPERTS * 64], bf16)
    OUT = air.tensor([NRG_DN, g.COL_BLOCK], bf16)

    x_m = air.channel("XM")
    wr_m = air.channel("WRM")
    x_l2 = air.channel("XL2")
    arr.channels(n_dests=2)
    glu_out = air.channel("GluOut")
    out_ch = air.channel("Out")

    with air.launch(name="moe") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    x_m.put(X)
                    wr_m.put(WR)
                    x_l2.put(X)
                    for rg in air.sequential(0, NRG_DN):
                        out_ch.get(OUT[rg, :])

                    # assemble memtile: x, then each expert's GLU output
                    def l2(shape):
                        return air.alloc(shape, bf16, scope=seg.private(), column=1)

                    l2x, hid = l2([g.D_PAD]), l2([NRG_GU // 2, arr.round])
                    x_l2.get(l2x)
                    for sl in air.sequential(0, NRG_GU // 2):
                        glu_out.get(hid[sl, :])
                    arr.broadcast(
                        lambda i: [
                            l2x,
                            hid[:, i * arr.coly : i * arr.coly + g.COL_BLOCK],
                        ]
                    )
                    arr.gather(seg, NRG)
                    arr.feed(seg, NRG * NCB + 1)

                    moe.shared_buffers(seg)
                    moe.master(x_m.get, wr_m)
                    moe.slave(W, 0, SLAVE_DEST, out_ch)
                    moe.glu(glu, GLU_DEST, glu_out)

                    def phases():
                        yield Phase(g.D_PAD, NRG_GU * NCB // 2, NCB, mx_step, GLU_DEST)
                        # the down phase's odd block count, plus the one filler
                        # block the host sends after it
                        yield Phase(
                            g.D_PAD, (NRG_DN * NCB + 1) // 2, NCB, mx_step, SLAVE_DEST
                        )

                    arr.herd(g.D_PAD, mx_flush, phases)

            # one filler block per core after the experts, which keeps the down
            # phase's block count even for the cores' two weight buffers
            arr.put_weights(lambda i, h: [W[0, 0, 0, 0:2, :]])

    return launch.build(target="npu2")


def bf16_floor(t):
    b = t.contiguous().view(torch.int32)
    up = ((b & 0xFFFF) != 0) & (b < 0)
    return ((b & ~0xFFFF) + up.to(torch.int32) * 0x10000).view(torch.float32)


def expert_gemv_ref(t, x, rows):
    """Row groups rows[0]:rows[1] of expert bundle tensor t times x, with the
    MXFP4 kernel's rounding (bit-exact: see kernel_tests/mvm_mxfp4.py)."""
    blocks = g._ungroup_rows(t[rows[0] : rows[1]].contiguous(), R)
    wf, bias = g.dequant_mxfp4(blocks)
    scale = g._e8m0(blocks[..., 0:128]).reshape(-1, NCB, 4, g.ROW_BLOCK)
    scale = scale.permute(0, 3, 1, 2).reshape(-1, NCB * 4)
    e2m1 = (wf / scale.repeat_interleave(32, dim=1)).reshape(wf.shape[0], -1, 32)
    part = bf16_floor((e2m1 * x.reshape(1, -1, 32)).sum(-1))
    return bf16_floor((part * scale).sum(-1) + bias)


def moe_ref(w, layer, x, experts):
    """The block's output as the device computes it: the expert GEMVs with the
    kernel's rounding, FLM's GLU (its LUT is 0 below -3.875) and combine in
    bf16 rounded toward -inf. Router and softmax in exact arithmetic."""
    wr, br = w.router(layer)
    top_v, top_i = torch.topk(wr @ x[: g.D] + br, g.TOP_K)
    topw = bf16_floor(torch.softmax(top_v.to(torch.bfloat16).float(), 0))
    out = torch.zeros(g.D_PAD)
    for k, e in enumerate(top_i.tolist()):
        gu = expert_gemv_ref(experts[e], x, (0, NRG_GU)).reshape(-1, 2, R * g.ROW_BLOCK)
        gate, up = gu[:, 0].ravel(), gu[:, 1].ravel()
        h = g.swiglu(gate, up)
        h = bf16_floor(torch.where(gate < -3.875, torch.zeros_like(h), h))
        out += topw[k] * expert_gemv_ref(experts[e], h, (NRG_GU, NRG))
    return bf16_floor(out)[: g.D], top_i.tolist()


def inputs(layer, token):
    w = g.GptOssQ4nx()
    h = w.embed(torch.tensor([token]))[0]
    x = g.rmsnorm(h, w.norms(layer)[1]).to(torch.bfloat16).float()
    xp = torch.zeros(g.D_PAD)
    xp[: g.D] = x
    wr = w.raw(f"model.layers.{layer}.mlp.router.weight").reshape(NB_ROUTER, 32 * 64)
    wr = torch.cat((torch.zeros(1, 32 * 64, dtype=torch.bfloat16), wr))
    wr[0, :32] = w.router(layer)[1].to(torch.bfloat16)
    experts = w.raw(f"model.layers.{layer}.ffn_gate_up_down_exps.weight").view(
        torch.uint8
    )
    return w, xp, wr, experts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=12650)
    ap.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        return 0
    common.compile_kernel("mvm", model=True)
    common.compile_kernel("router", moe=True)
    common.compile_kernel("swiglu")
    w, x, wr, experts = inputs(a.layer, a.token)
    want, ids = moe_ref(w, a.layer, x, experts)
    exact = g.moe(w, a.layer, x[None, : g.D])[0][0]
    print(f"reference top-4 experts: {ids}")
    # fused_decode's settings: a persistent core loop, and daisy-chained locks
    # for the memtile gathers
    backend = XRTBackend(
        output_format=a.output_format,
        instance_name="moe",
        stack_size=4096,
        omit_while_true_loop=False,
        use_lock_race_condition_fix_v2=True,
    )
    art = backend.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        f = backend.load(art)
        res = f(
            experts.numpy(),
            x.numpy().astype(bfloat16),
            wr.view(torch.int16).numpy().view(bfloat16),
            np.zeros((NRG_DN, g.COL_BLOCK), bfloat16),
        )
    got = torch.from_numpy(res[3].view(bfloat16).astype(np.float32).ravel()[: g.D])

    def cos(u, v):
        return float(torch.nn.functional.cosine_similarity(u, v, dim=0))

    c = cos(got, want)
    print(
        f"MoE out vs emulated pipeline: cos {c:.6f}; vs exact: cos {cos(got, exact):.5f}"
    )
    ok = c > 0.9999
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
