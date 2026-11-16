# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8 LM head (Qwen3.6-35B-A3B's, qwen36_35b_a3b_q4nx/lm_head.py, at D = 5120): the final RMS norm and the Q8_0 vocabulary projection, as
# FastFlowLM runs it (FLM_Xclbin/Qwen3_5/lm_head_npu_bin), a run of its own
# after the layers.
#
# 8 columns x 4 rows of cores. Every core takes x and the norm weight by
# broadcast, normalises x, then one 32-row block per round: round r, column c,
# row i is row block 32 r + 4 c + i. A column's weights arrive on its two shim
# channels into its memtile, each channel feeding two of its cores one k-block
# each per slot; the four cores' rows of a round meet in the memtile and go
# out on the column's shim.
#
# rows=2 is the speculative decode's verify pass: both tokens' x, each weight
# block dequantized once for both (kernels/lm_head2.cc); a core's output is
# its row block for token 0, then for token 1. vocab < VOCAB takes the first
# rows only: the speculative decode's draft (BPE merge order puts the common
# tokens first).
import argparse
import os
import sys
import tempfile
import time
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
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32, ui8  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

# a check's bound on the relative L2 error
TOL = 0.015

D, VOCAB = W.D, W.VOCAB
NCOL, NROW, RB, KB = 8, 4, 32, W.D // 256
BB = 8704  # a Q8_0 block: 32 rows x 256 columns


def rounds(vocab):
    """Rounds of NCOL x NROW row blocks for the first `vocab` rows."""
    return -(-vocab // (NCOL * NROW * RB))


ROUNDS = rounds(VOCAB)


def build_module(rows=1, vocab=VOCAB):
    # kernels for one row (lm_head.cc) and two (lm_head2.cc)
    assert rows in (1, 2), f"{rows} rows: the LM head takes 1 or 2"
    obj = "lm_head.o" if rows == 1 else "lm_head2.o"
    sfx = "" if rows == 1 else "2"
    step = air.extern("q8_step" + sfx, link_with=obj, scalars=[i32])
    flush = air.extern("q8_flush" + sfx, link_with=obj)
    rms = air.extern("lm_rms_norm" + sfx, link_with=obj)
    ob = rows * RB  # a core's output per round
    nrd = rounds(vocab)

    X = air.tensor([rows * D], bf16)
    NW = air.tensor([D], bf16)
    # [column][shim channel][round][k-block][core of the pair][block]
    WL = air.tensor([NCOL, 2, nrd, KB, 2, BB], ui8)
    Y = air.tensor([NCOL, nrd, NROW * ob], bf16)

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
                            for _r in air.sequential(0, nrd * KB):
                                slot = l2([2, BB], c, ui8)
                                wl.get(slot, indices=[c, h])
                                for s in range(2):
                                    wc.put(slot[s, :], indices=[c, 2 * h + s])
                                air.dealloc(slot)
                        for _r in air.sequential(0, nrd):
                            yb = l2([NROW * ob], c)
                            for i in range(NROW):
                                yc.get(yb[i * ob : (i + 1) * ob], indices=[c, i])
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
                            xw = air.alloc([(rows + 1) * D], bf16, scope=p)
                            wb = [air.alloc([BB], ui8, scope=p) for _ in range(2)]
                            acc = air.alloc([2 * RB], f32, scope=p)
                            y = air.alloc([16 + ob], bf16, scope=p)
                            xb.get(xw[0 : rows * D], indices=[tx, ty])
                            xb.get(xw[rows * D : (rows + 1) * D], indices=[tx, ty])
                            rms(xw)
                            for _r in air.sequential(0, nrd):
                                # two weight buffers, filled alternately
                                for k2 in air.sequential(0, KB // 2):
                                    for half in range(2):
                                        wc.get(wb[half], indices=[tx, ty])
                                        step(xw, wb[half], acc, 2 * k2 + half)
                                flush(acc, y)
                                yc.put(y[16 : 16 + ob], indices=[tx, ty])

    return launch.build(target="npu2")


def pack_weights(w, vocab=VOCAB):
    """FLM's Q8_0 row blocks of the first `vocab` rows -> [column][channel]
    [round][k-block][core][block]."""
    nrd = rounds(vocab)
    raw = np.asarray(w.raw("lm_head.weight")).view(np.uint8).reshape(-1, KB, BB)
    raw = raw[: nrd * NCOL * NROW]
    pad = np.zeros((nrd * NCOL * NROW - raw.shape[0], KB, BB), np.uint8)
    rb = np.concatenate([raw, pad]).reshape(nrd, NCOL, 2, 2, KB, BB)
    return np.ascontiguousarray(rb.transpose(1, 2, 0, 4, 3, 5))


def unpack_logits(y, rows=1, vocab=VOCAB):
    """[column][round][core][row][32] -> logits[vocab], or [rows][vocab]."""
    y = np.asarray(y).reshape(NCOL, rounds(vocab), NROW, rows, RB)
    y = y.transpose(3, 1, 0, 2, 4).reshape(rows, -1)[:, :vocab]
    return y[0] if rows == 1 else y


def compile_lm_head(rows=1, vocab=VOCAB):
    common.compile_kernel("lm_head" if rows == 1 else "lm_head2")
    # x is queued ahead of the weights on the shim channel they share
    be = XRTBackend(
        instance_name="lm_head", output_format="elf", allow_deadlock_prone_routing=True
    )
    return be.compile(build_module(rows, vocab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--tokens", default="9338", help="one token, or two (rows=2)")
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    toks = [int(v) for v in a.tokens.split(",")]
    rows = len(toks)
    if rows not in (1, 2):
        ap.error("--tokens: one token, or two")
    if a.print_module_only:
        print(build_module(rows))
        return 0
    import mlp_block

    w = W.Qwen38Q4nx()
    art = compile_lm_head(rows)
    x = w.embed(toks).astype(bfloat16)
    nw = np.asarray(w.final_norm()).astype(bfloat16)
    y0 = np.zeros((NCOL, ROUNDS, NROW * rows * RB), bfloat16)
    args = [x.ravel(), nw, pack_weights(w), y0]
    if a.bench:
        return mlp_block.bench(art, args, a.bench, (1, 2))
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        be = XRTBackend(instance_name="lm_head", output_format="elf")
        run = be.load(art)
        res = run(*args)
        be.unload()
    got = unpack_logits(res[3], rows).astype(np.float32).reshape(rows, VOCAB)
    xf = x.astype(np.float32)
    xn = (
        xf
        / np.sqrt(np.mean(xf * xf, -1, keepdims=True) + W.EPS)
        * nw.astype(np.float32)
    )
    xnb = xn.astype(bfloat16).astype(np.float32)
    want = np.concatenate(
        [
            xnb @ w.lm_head_rows(r, min(r + 8192, VOCAB)).T
            for r in range(0, VOCAB, 8192)
        ],
        axis=1,
    )
    ok = True
    for t in range(rows):
        c, l2, e = common.compare(got[t], want[t])
        print(f"row {t} logits: corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}")
        print(
            "top5 dev",
            np.argsort(-got[t])[:5].tolist(),
            "ref",
            np.argsort(-want[t])[:5].tolist(),
        )
        ok &= c > 0.9999 and l2 < TOL and np.argmax(got[t]) == np.argmax(want[t])
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
