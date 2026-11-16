# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A Q4_K projection of the bundle against 8 token rows through q4k_b8_* on one
# column of four cores: each core takes one 32-row block of a group of four,
# its blocks as stored, and the x chunk of each block alongside it.
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
LLMS = HERE.parents[2]
sys.path.insert(0, str(LLMS / "qwen3_coder_30b_a3b_q4k"))
sys.path.insert(0, str(HERE.parent))

import q4k_layout  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, ui8  # noqa: E402

R, RB, CB, BB, TOK = 4, 32, 256, 4736, 8
# the bfp16 x chunk and the unpacked block
SCRATCH = 32 * 72 + 4 * 32 * 72
PREFILL_KERNELS = LLMS / "shared" / "fused_prefill" / "kernels"


def compile_kernel():
    peano = os.environ["PEANO_INSTALL_DIR"]
    inc = Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"
    subprocess.run(
        [
            f"{peano}/bin/clang++",
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-DNDEBUG",
            "-D__AIE_API_AIE_ADF_HPP__",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            f"-I{inc}",
            f"-I{PREFILL_KERNELS}",
            "-c",
            str(HERE.parent / "kernels" / "q4k_b8.cc"),
            "-o",
            "q4k_b8.o",
        ],
        check=True,
    )


def build_module(nrg, ncb):
    k_ = lambda n: air.extern(n, link_with="q4k_b8.o")  # noqa: E731
    zero, block, flush = k_("q4k_b8_zero"), k_("q4k_b8_block"), k_("q4k_b8_flush")
    Wt = air.tensor([nrg, ncb, R, BB], ui8)
    X = air.tensor([ncb, CB * TOK], bf16)
    Y = air.tensor([nrg, R, TOK * RB], bf16)
    w_l2 = air.channel("WL2")
    w_l1 = air.channel("WL1", size=[1, R])
    x_l2 = air.channel("XL2")
    x_l1 = air.channel("XL1", size=[1, 1], broadcast_shape=[1, R])
    y_out = air.channel("YOut", size=[1, R])

    with air.launch(name="q4k_b8") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    x_l2.put(X)
                    l2x = air.alloc([ncb, CB * TOK], bf16, scope=seg.private())
                    x_l2.get(l2x)
                    l2w = air.alloc([R, BB], ui8, scope=seg.private())
                    for rg in air.sequential(0, nrg):
                        for cb in air.sequential(0, ncb):
                            w_l2.put(Wt[rg, cb, :, :])
                            w_l2.get(l2w)
                            for r in range(R):
                                w_l1.put(l2w[r, :], indices=[0, r])
                            x_l1.put(l2x[cb, :], indices=[0, 0])
                    for r in range(R):
                        y_out.get(Y[:, r, :], indices=[0, r])

                    with air.herd([range(1), range(R)], name="mm", shape=(1, R)) as h:

                        @h.body
                        def _(tx, ty):
                            p = h.private()
                            w = air.alloc([BB], ui8, scope=p)
                            x = air.alloc([CB * TOK], bf16, scope=p)
                            ws = air.alloc([SCRATCH], ui8, scope=p)
                            acc = air.alloc([TOK * RB], f32, scope=p)
                            y = air.alloc([TOK * RB], bf16, scope=p)
                            for rg in air.sequential(0, nrg):
                                zero(acc)
                                for cb in air.sequential(0, ncb):
                                    w_l1.get(w, indices=[tx, ty])
                                    x_l1.get(x, indices=[tx, ty])
                                    block(x, w, acc, ws)
                                flush(acc, y)
                                y_out.put(y, indices=[tx, ty])

    return launch.build(target="npu2")


def x_tiles(x):
    """x [TOK, K] -> [K / 256, 32 col tiles, TOK, 8]."""
    ncb = x.shape[1] // CB
    return np.ascontiguousarray(
        x.reshape(TOK, ncb, CB // 8, 8).transpose(1, 2, 0, 3)
    ).reshape(ncb, CB * TOK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--part", default="self_attn.k_proj.weight")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    w = W.Qwen3CoderQ4k()
    name = W.name(a.layer, a.part)
    nbi, ncb, block_bytes = w.shape(name)
    assert block_bytes == BB
    nrg = nbi // R
    module = build_module(nrg, ncb)
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    compile_kernel()
    raw = np.asarray(w.raw(name)).reshape(-1, ncb, BB)[: nrg * R]
    blocks = q4k_layout.tiled(raw.reshape(nrg, R, ncb, BB).transpose(0, 2, 1, 3))
    K = ncb * CB
    x = np.random.default_rng(a.seed).standard_normal((TOK, K)).astype(bfloat16)
    wf = w.dequant_rows(name, 0, nrg * R * RB, K).astype(np.float64)
    y = x.astype(np.float64) @ wf.T  # [TOK, rows]
    want = y.reshape(TOK, nrg, R, RB).transpose(1, 2, 0, 3).reshape(nrg, R, TOK * RB)
    from air.backend.xrt import XRTBackend

    be = XRTBackend(instance_name="q4k_b8", output_format="elf", stack_size=6144)
    art = be.compile(module)
    out = np.zeros((nrg, R, TOK * RB), bfloat16)
    run = be.load(art)
    got = run(blocks, x_tiles(x), out)[2].astype(np.float64)
    be.unload()
    # the same weights rounded to bf16, accumulated exactly
    yb = x.astype(np.float64) @ wf.astype(bfloat16).astype(np.float64).T
    ok = True
    for what, ref in (("exact", y), ("bf16 weights", yb)):
        r = ref.reshape(TOK, nrg, R, RB).transpose(1, 2, 0, 3).ravel()
        g = got.ravel()
        cos = g @ r / np.linalg.norm(g) / np.linalg.norm(r)
        l2 = np.linalg.norm(g - r) / np.linalg.norm(r)
        print(f"vs {what}: cos {cos:.6f} rel L2 {l2:.2e}")
        ok &= l2 < 1.5e-2
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
