# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""int4 LM-head GEMV on its own device.

  y[V] = W[V, K] x[K], V = 262144, K = 1536, Q4NX weights (w = q * scale + min)

Each of the 32 cores owns V/32 contiguous rows, processed as 64-row tiles of
six 256-wide K packets (gemv_q4.cc). Weights differ per core, so each column's
shim stream interleaves its four cores' packets and the memtile splits them.
x reaches every core once per dispatch on the cores' second S2MM channel, and
the logits come back 1024 rows at a time through a column gather. The memtile
channels are count-free rings, so a dispatch is the shim transfers plus one
RTP.
"""

import numpy as np
from air import api as air
from air.api.types import bf16, f32, i8, i32

V, K = 262144, 1536
NC, NR = 8, 4
GN, GK = 64, 256
KC = K // GK
PKT = GN * GK // 2 + 2 * 2 * (GK // 32) * GN  # bytes per packet
ROWS = V // (NC * NR)  # per core
TILES = ROWS // GN
FLUSH = 16  # row tiles per output flush
OBJ = "gemv_q4.o"


def build():
    W = air.tensor([NC, TILES * KC * NR * PKT], i8)
    X = air.tensor([K], bf16)
    Y = air.tensor([NC, TILES // FLUSH * NR * FLUSH * GN], bf16)

    expand = air.extern("gv_expand", link_with=OBJ)
    zero = air.extern("gv_zero", link_with=OBJ)
    acc_k = air.extern("gv_acc", link_with=OBJ, scalars=[i32])
    flush = air.extern("gv_flush", link_with=OBJ, scalars=[i32])

    w_in = air.channel("WIn", size=[NC])
    x_in = air.channel("XIn", size=[1])
    w2l1 = air.channel("W2L1", size=[NC, NR])
    x2l1 = air.channel("X2L1", size=[1, 1], broadcast_shape=[NC, NR])
    o2l2 = air.channel("O2L2", size=[NC, NR])
    o_out = air.channel("OOut", size=[NC])

    with air.launch(
        repeat=1, name="lm_gemv", attrs=["air.preserve_shim_dma_order"]
    ) as lch:

        @lch.body
        def _(wave):
            x_in.put(X[0:K])
            for c in range(NC):
                w_in.put(W[c : c + 1, 0 : TILES * KC * NR * PKT], indices=[c])
            for c in range(NC):
                o_out.get(Y[c : c + 1, 0 : TILES * NR * GN], indices=[c])

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    go = air.rtp(wave * 0 + 1)
                    x_l2 = air.alloc([K], bf16, scope=seg.private(), column=0)
                    x_in.get(x_l2)
                    x2l1.put(x_l2, indices=[0, 0])
                    for c in range(NC):
                        w_l2 = [
                            air.alloc(
                                [NR * PKT],
                                i8,
                                scope=seg.private(),
                                column=c,
                                split=False,
                            )
                            for _ in range(2)
                        ]
                        for pp in range(2):
                            w_in.get(w_l2[pp], indices=[c])
                        for pp in range(2):
                            for r in range(NR):
                                w2l1.put(
                                    w_l2[pp][r * PKT : r * PKT + PKT], indices=[c, r]
                                )
                        o_l2 = air.alloc(
                            [NR, FLUSH * GN],
                            bf16,
                            scope=seg.private(),
                            column=c,
                            split=False,
                        )
                        for r in range(NR):
                            o2l2.get(o_l2[r : r + 1, 0 : FLUSH * GN], indices=[c, r])
                        o_out.put(o_l2, indices=[c])

                    with air.herd(
                        [range(NC), range(NR)],
                        name="gemv",
                        shape=(NC, NR),
                        link_with=OBJ,
                        params=[go],
                    ) as hd:

                        @hd.body
                        def _(tx, ty):
                            xb = air.alloc([K], bf16, scope=hd.private())
                            xe = air.alloc([K * 8], bf16, scope=hd.private())
                            wb = [
                                air.alloc([PKT], i8, scope=hd.private())
                                for _ in range(2)
                            ]
                            acc = air.alloc([GN * 8], f32, scope=hd.private())
                            ob = air.alloc([FLUSH * GN], bf16, scope=hd.private())
                            for _g in air.sequential(go):
                                x2l1.get(xb, indices=[tx, ty])
                                expand(xb, xe)
                                for _f in air.sequential(0, TILES // FLUSH):
                                    for j in air.sequential(0, FLUSH):
                                        zero(acc)
                                        for kc in range(KC):
                                            w2l1.get(wb[kc % 2], indices=[tx, ty])
                                            acc_k(wb[kc % 2], xe, acc, kc)
                                        flush(acc, ob, j)
                                    o2l2.put(ob, indices=[tx, ty])

    return lch


def pack(q, sc, mn):
    """Q4NX raw (q [V, K], scale / min [V, K/32]) -> W [NC, ...] uint8 in
    stream order: per column, for each row tile and K packet, its four cores'
    packets. Core (c, r) owns rows (c*NR + r) * ROWS + [0, ROWS)."""
    from ml_dtypes import bfloat16 as _bf

    # per packet: [GN/8][GK/8][8 k][8 n] nibbles, then scale, min [GK/32][GN]
    qt = q.reshape(NC, NR, TILES, GN // 8, 8, KC, GK // 8, 8)  # c r j nb nn kc kb kk
    qt = qt.transpose(0, 2, 5, 1, 3, 6, 7, 4).reshape(NC, TILES, KC, NR, -1)
    qb = (qt[..., 0::2] | (qt[..., 1::2] << 4)).astype(np.uint8)

    def grp(a):
        a = np.asarray(a, _bf).reshape(NC, NR, TILES, GN, KC, GK // 32)  # c r j n kc g
        return (
            a.transpose(0, 2, 4, 1, 5, 3).reshape(NC, TILES, KC, NR, -1).view(np.uint8)
        )

    return np.concatenate([qb, grp(sc), grp(mn)], axis=-1).reshape(NC, -1)


def unpack_y(y):
    """Y [NC, flush, NR, FLUSH*GN] bf16 -> logits [V] in row order."""
    y = y.reshape(NC, TILES // FLUSH, NR, FLUSH * GN).transpose(0, 2, 1, 3)
    return y.reshape(V)
