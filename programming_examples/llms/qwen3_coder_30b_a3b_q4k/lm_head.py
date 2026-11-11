# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-Coder LM head: the final RMS norm and the Q8_0 vocabulary projection,
# Qwen3.6's program (after FastFlowLM's FLM_Xclbin/Qwen3_5/lm_head_npu_bin) at
# this vocabulary, a run of its own after the layers.
#
# 8 columns x 4 rows of cores. Every core takes x and the norm weight by
# broadcast, normalises x, then one 32-row block per round: round r, column c,
# row i is row block 32 r + 4 c + i. A column's weights arrive on its two shim
# channels into its memtile, each channel feeding two of its cores one k-block
# each per slot; the four cores' rows of a round meet in the memtile and go
# out on the column's shim.
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
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.015

D, VOCAB = W.D, W.VOCAB
NCOL, NROW, RB, KB = 8, 4, 32, W.D // 256
BB = W.Q8_BYTES  # a Q8_0 block: 32 rows x 256 columns
ROUNDS = -(-VOCAB // (NCOL * NROW * RB))
VOCAB_PAD = ROUNDS * NCOL * NROW * RB


def build_module():
    obj = "lm_head.o"
    step = air.extern("q8_step", link_with=obj, scalars=[i32])
    flush = air.extern("q8_flush", link_with=obj)
    rms = air.extern("lm_rms_norm", link_with=obj)

    X = air.tensor([D], bf16)
    NW = air.tensor([D], bf16)
    # [column][shim channel][round][k-block][core of the pair][block]
    WL = air.tensor([NCOL, 2, ROUNDS, KB, 2, BB], ui8)
    Y = air.tensor([NCOL, ROUNDS, NROW * RB], bf16)

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
            xb.put(X)
            xb.put(NW)
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
                            yb = l2([NROW * RB], c)
                            for i in range(NROW):
                                yc.get(yb[i * RB : (i + 1) * RB], indices=[c, i])
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
                            xw = air.alloc([2 * D], bf16, scope=p)
                            wb = [air.alloc([BB], ui8, scope=p) for _ in range(2)]
                            acc = air.alloc([2 * RB], f32, scope=p)
                            y = air.alloc([16 + RB], bf16, scope=p)
                            xb.get(xw[0:D], indices=[tx, ty])
                            xb.get(xw[D : 2 * D], indices=[tx, ty])
                            rms(xw)
                            for _r in air.sequential(0, ROUNDS):
                                # two weight buffers, filled alternately
                                for k2 in air.sequential(0, KB // 2):
                                    for half in range(2):
                                        wc.get(wb[half], indices=[tx, ty])
                                        step(xw, wb[half], acc, 2 * k2 + half)
                                flush(acc, y)
                                yc.put(y[16 : 16 + RB], indices=[tx, ty])

    return launch.build(target="npu2")


def pack_weights(w):
    """The Q8_0 row blocks -> [column][channel][round][k-block][core][block]."""
    raw = np.asarray(w.raw("lm_head.weight")).view(np.uint8).reshape(-1, KB, BB)
    pad = np.zeros((VOCAB_PAD // RB - raw.shape[0], KB, BB), np.uint8)
    rb = np.concatenate([raw, pad]).reshape(ROUNDS, NCOL, 2, 2, KB, BB)
    return np.ascontiguousarray(rb.transpose(1, 2, 0, 4, 3, 5))


def unpack_logits(y):
    """[column][round][row block of the column][32] -> logits[VOCAB]."""
    y = np.asarray(y).reshape(NCOL, ROUNDS, NROW * RB)
    return y.transpose(1, 0, 2).reshape(-1)[:VOCAB]


def compile_lm_head():
    common.compile_kernel("lm_head")
    # x is queued ahead of the weights on the shim channel they share
    be = XRTBackend(
        instance_name="lm_head", output_format="elf", allow_deadlock_prone_routing=True
    )
    return be.compile(build_module())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--token", type=int, default=9625)
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    if a.print_module_only:
        print(build_module())
        return 0
    import moe_block

    w = W.Qwen3CoderQ4k()
    art = compile_lm_head()
    x = w.embed([a.token])[0].astype(bfloat16)
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    args = [x, nw, pack_weights(w), np.zeros((NCOL, ROUNDS, NROW * RB), bfloat16)]
    if a.bench:
        return moe_block.bench(art, args, a.bench)
    be = XRTBackend(instance_name="lm_head", output_format="elf")
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        res = run(*args)
        be.unload()
    got = unpack_logits(res[3]).astype(np.float32)
    xf = x.astype(np.float32)
    xn = xf / np.sqrt(np.mean(xf * xf) + W.EPS) * nw.astype(np.float32)
    want = w.lm_head @ xn.astype(bfloat16).astype(np.float32)
    c, l2, e = common.compare(got, want)
    print(f"logits: cos {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
    print(
        "top5 dev", np.argsort(-got)[:5].tolist(), "ref", np.argsort(-want)[:5].tolist()
    )
    ok = c > 0.9999 and l2 < TOL and np.argmax(got) == np.argmax(want)
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
