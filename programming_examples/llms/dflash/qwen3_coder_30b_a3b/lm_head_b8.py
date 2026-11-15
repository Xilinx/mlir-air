# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The Q8_0 LM head for 8 token rows: the decode's program (lm_head.py, 8
# columns x 4 rows of cores, one 32-row block per core per round) with x as
# bfp16 tiles of the 8 normalised rows and the 8-row block kernel.
import argparse
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import filelock
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
LLMS = HERE.parents[1]
CODER = LLMS / "qwen3_coder_30b_a3b_q4k"
sys.path.insert(0, str(CODER))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(LLMS.parent)
]

import lm_head as LM  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

D, VOCAB = W.D, W.VOCAB
NCOL, NROW, RB, KB, BB, ROUNDS = LM.NCOL, LM.NROW, LM.RB, LM.KB, LM.BB, LM.ROUNDS
TOK = 8
XB = KB * 32 * 72  # x as bfp16 tiles
YB = TOK * RB  # a core's rows of a round


def build_module():
    k_ = lambda n, s=(): air.extern(  # noqa: E731
        n, link_with="q8_b8.o", scalars=list(s)
    )
    zero, block, flush = k_("q8_b8_zero"), k_("q8_b8_block", [i32]), k_("q8_b8_flush")

    XT = air.tensor([XB], ui8)
    # [column][shim channel][round][k-block][core of the pair][block]
    WL = air.tensor([NCOL, 2, ROUNDS, KB, 2, BB], ui8)
    Y = air.tensor([NCOL, ROUNDS, NROW * YB], bf16)

    ch = air.channel
    # x shares a shim channel with one weight stream, as in FLM: both packet
    pkt = dict(channel_type="npu_dma_packet")
    xb = ch("XB", size=[1, 1], broadcast_shape=[NCOL, NROW], **pkt)
    wl = ch("WL", size=[NCOL, 2], **pkt)
    wc = ch("WC", size=[NCOL, NROW])
    yc = ch("YC", size=[NCOL, NROW])
    yo = ch("YO", size=[NCOL])

    with air.launch(name="lm_head") as launch:

        @launch.body
        def _():
            xb.put(XT)
            for c in range(NCOL):
                for h in range(2):
                    wl.put(WL[c, h, :, :, :, :], indices=[c, h])
                yo.get(Y[c, :, :], indices=[c])
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    def l2(shape, col, dt=bf16):
                        return air.alloc(shape, dt, scope=seg.private(), column=col)

                    for c in range(NCOL):
                        for h in range(2):
                            for _r in air.sequential(0, ROUNDS * KB):
                                slot = l2([2, BB], c, ui8)
                                wl.get(slot, indices=[c, h])
                                for s in range(2):
                                    wc.put(slot[s, :], indices=[c, 2 * h + s])
                                air.dealloc(slot)
                        for _r in air.sequential(0, ROUNDS):
                            yb = l2([NROW * YB], c)
                            for i in range(NROW):
                                yc.get(yb[i * YB : (i + 1) * YB], indices=[c, i])
                            yo.put(yb, indices=[c])
                            air.dealloc(yb)

                    with air.herd(
                        [range(NCOL), range(NROW)],
                        name="lm",
                        shape=(NCOL, NROW),
                        at=(0, 2),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            p = h.private()
                            xw = air.alloc([XB], ui8, scope=p)
                            wb = [air.alloc([BB], ui8, scope=p) for _ in range(2)]
                            acc = air.alloc([4 * 64], f32, scope=p)
                            y = air.alloc([YB], bf16, scope=p)
                            xb.get(xw, indices=[tx, ty])
                            for _r in air.sequential(0, ROUNDS):
                                zero(acc)
                                # two weight buffers, filled alternately
                                for k2 in air.sequential(0, KB // 2):
                                    for half in range(2):
                                        wc.get(wb[half], indices=[tx, ty])
                                        block(xw, wb[half], acc, 2 * k2 + half)
                                flush(acc, y)
                                yc.put(y, indices=[tx, ty])

    return launch.build(target="npu2")


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
            "-Wno-parentheses",
            "-Wno-deprecated-declarations",
            f"-I{inc}",
            "-c",
            str(HERE / "kernels" / "q8_b8.cc"),
            "-o",
            "q8_b8.o",
        ],
        check=True,
    )


# x is queued ahead of the weights on the shim channel they share
OPTS = dict(stack_size=4096, allow_deadlock_prone_routing=True)


def unpack_logits(y):
    """[column][round][core][token][32] -> logits [8, VOCAB]."""
    y = np.asarray(y).reshape(NCOL, ROUNDS, NROW, TOK, RB)
    return y.transpose(3, 1, 0, 2, 4).reshape(TOK, -1)[:, :VOCAB]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    if a.print_module_only:
        print(build_module())
        return 0
    from air_examples.llms.shared.fused_prefill import hostops as H
    import model_b8 as M

    H.build(".", 4)
    H.load(".")
    compile_kernel()
    w = W.Qwen3CoderQ4k()
    rng = np.random.default_rng(0)
    xn = rng.standard_normal((TOK, D)).astype(bfloat16)
    args = [
        M.to_tiles(xn),
        LM.pack_weights(w),
        np.zeros((NCOL, ROUNDS, NROW * YB), bfloat16),
    ]
    be = XRTBackend(instance_name="lm_head", output_format="elf", **OPTS)
    art = be.compile(build_module())
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        res = run(*args)
        be.unload()
        if a.bench:
            tb = XRTBackend(
                instance_name="lm_head",
                output_format="elf",
                n_perf_iters=a.bench,
                n_warmup_iters=3,
                **OPTS,
            )
            trun = tb.load(art)
            trun(*args)
            print(f"[bench] {tb.last_latency_us / 1e3:.3f} ms a dispatch")
            tb.unload()
    got = unpack_logits(res[2]).astype(np.float32)
    want = xn.astype(np.float32) @ w.lm_head.T
    ok = True
    for t in range(TOK):
        g, r_ = got[t].astype(np.float64), want[t].astype(np.float64)
        cos = g @ r_ / np.linalg.norm(g) / np.linalg.norm(r_)
        l2 = np.linalg.norm(g - r_) / np.linalg.norm(r_)
        same = int(g.argmax()) == int(r_.argmax())
        print(f"row {t}: cos {cos:.6f} rel L2 {l2:.2e} argmax {'=' if same else '!='}")
        ok &= cos > 0.9999 and l2 < 0.015 and same
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
