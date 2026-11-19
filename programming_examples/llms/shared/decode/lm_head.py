# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The final RMS norm and the Q8_0 vocabulary projection, as FastFlowLM's
# Qwen3.5 LM head runs them: a run of its own after the layers.
#
# 8 columns x 4 rows of cores. Every core takes x and the norm weight by
# broadcast, normalises x, then one 32-row block per round: round r, column c,
# row i is row block 32 r + 4 c + i. A column's weights arrive on its two shim
# channels into its memtile, each channel feeding two of its cores one k-block
# each per slot; the four cores' rows of a round meet in the memtile and go
# out on the column's shim.
#
# rows=2 takes two tokens' x and dequantizes each weight block once for both;
# a core's output is its row block for token 0, then for token 1. A vocab
# below the model's takes the first rows only.
import numpy as np

from air import api as air
from air.api import bf16, f32, i32, ui8
from air.backend.xrt import XRTBackend

NCOL, NROW, RB = 8, 4, 32
BB = 8704  # a Q8_0 block: 32 rows x 256 columns


def rounds(vocab):
    """Rounds of NCOL x NROW row blocks for the first `vocab` rows."""
    return -(-vocab // (NCOL * NROW * RB))


def build_module(d, vocab, rows=1):
    assert rows in (1, 2), f"{rows} rows: the LM head takes 1 or 2"
    obj = "lm_head.o" if rows == 1 else "lm_head2.o"
    sfx = "" if rows == 1 else "2"
    step = air.extern("q8_step" + sfx, link_with=obj, scalars=[i32])
    flush = air.extern("q8_flush" + sfx, link_with=obj)
    rms = air.extern("lm_rms_norm" + sfx, link_with=obj)
    kb = d // 256
    ob = rows * RB  # a core's output per round
    nrd = rounds(vocab)

    X = air.tensor([rows * d], bf16)
    NW = air.tensor([d], bf16)
    # [column][shim channel][round][k-block][core of the pair][block]
    WL = air.tensor([NCOL, 2, nrd, kb, 2, BB], ui8)
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
                            for _r in air.sequential(0, nrd * kb):
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
                            xw = air.alloc([(rows + 1) * d], bf16, scope=p)
                            wb = [air.alloc([BB], ui8, scope=p) for _ in range(2)]
                            acc = air.alloc([2 * RB], f32, scope=p)
                            y = air.alloc([16 + ob], bf16, scope=p)
                            xb.get(xw[0 : rows * d], indices=[tx, ty])
                            xb.get(xw[rows * d : (rows + 1) * d], indices=[tx, ty])
                            rms(xw)
                            for _r in air.sequential(0, nrd):
                                # two weight buffers, filled alternately
                                for k2 in air.sequential(0, kb // 2):
                                    for half in range(2):
                                        wc.get(wb[half], indices=[tx, ty])
                                        step(xw, wb[half], acc, 2 * k2 + half)
                                flush(acc, y)
                                yc.put(y[16 : 16 + ob], indices=[tx, ty])

    return launch.build(target="npu2")


def pack_weights(raw, d, vocab):
    """Q8_0 row blocks of the first `vocab` rows -> [column][channel][round]
    [k-block][core][block]."""
    kb, nrd = d // 256, rounds(vocab)
    raw = np.asarray(raw).view(np.uint8).reshape(-1, kb, BB)[: nrd * NCOL * NROW]
    pad = np.zeros((nrd * NCOL * NROW - raw.shape[0], kb, BB), np.uint8)
    rb = np.concatenate([raw, pad]).reshape(nrd, NCOL, 2, 2, kb, BB)
    return np.ascontiguousarray(rb.transpose(1, 2, 0, 4, 3, 5))


def unpack_logits(y, vocab, rows=1):
    """[column][round][core][row][32] -> logits[vocab], or [rows][vocab]."""
    y = np.asarray(y).reshape(NCOL, rounds(vocab), NROW, rows, RB)
    y = y.transpose(3, 1, 0, 2, 4).reshape(rows, -1)[:, :vocab]
    return y[0] if rows == 1 else y


def compile_lm_head(compile_kernel, d, vocab, rows=1):
    """`compile_kernel(name)` builds the model's kernel object `name`.o."""
    compile_kernel("lm_head" if rows == 1 else "lm_head2")
    # x is queued ahead of the weights on the shim channel they share
    be = XRTBackend(
        instance_name="lm_head", output_format="elf", allow_deadlock_prone_routing=True
    )
    return be.compile(build_module(d, vocab, rows))
