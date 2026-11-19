# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FastFlowLM's decode GEMV array, which every projection of a layer shares.

``cols`` columns of ``rows`` cores. Each column's shim streams its weight
blocks on two channels into the column memtile, which hands one block to each
core; the input vector is broadcast down the column. A core accumulates one
row block over the column blocks of a row group, then sends it as a packet
whose header names its consumer. The column memtiles gather a round (one row
group per column), the assemble memtile joins the columns, and one egress
delivers the round to the core its header names.

Weights are ``[row group, col block, row block, bytes]`` per column: row group
g of a projection goes to column g % cols, and shim channel h of a column
carries row blocks 2h and 2h + 1.
"""

from dataclasses import dataclass
from typing import Callable

from air import api as air
from air.api import bf16, f32, ui8

HDR = 2  # packet header words in front of a packet put's region


@dataclass
class Phase:
    """One projection on the array: ``npairs`` pairs of weight blocks, input
    width ``xw``, ``ncb`` column blocks per row group, ``step(x, w, acc, cb)``
    the kernel of one block, and the consumer ``dest`` of its rows. The input
    goes to the core's x buffer ``xbuf``; ``fresh=False`` reuses what an
    earlier phase left there."""

    xw: int
    npairs: object
    ncb: int
    step: Callable
    dest: int
    xbuf: int = 0
    fresh: bool = True


class MvmArray:
    def __init__(
        self,
        *,
        row_block,
        block_bytes,
        cols=4,
        rows=4,
        col0=2,
        row0=2,
        assemble_col=1,
        acc_words=None,
        columns=None,
    ):
        assert rows % 2 == 0, "a column's two shim channels split its rows"
        self.row_block, self.block_bytes = row_block, block_bytes
        self.cols, self.rows = cols, rows
        self.col0, self.row0, self.assemble_col = col0, row0, assemble_col
        # the physical column of each array column; a herd per contiguous run
        self.columns = list(columns or range(col0, col0 + cols))
        assert len(self.columns) == cols
        self.coly = rows * row_block  # one column's rows of a round
        self.round = cols * self.coly
        # a core's float32 accumulator between blocks: one row block unless
        # the block kernel carries more (Q4_K: its scale and min paths)
        self.acc_words = acc_words or row_block

    def channels(self, n_dests):
        """The array's channels; ``n_dests`` consumers share its egress."""
        ch = air.channel
        pkt = dict(channel_type="npu_dma_packet")
        c, r = self.cols, self.rows
        self.w_ex = ch("WEx", size=[c, 2])
        self.w_l1 = ch("WL1", size=[c, r])
        self.x_b = ch("XB", size=[c, 1], broadcast_shape=[c, r])
        self.out_a = ch("OutA", size=[c, r], **pkt)
        self.to_main = ch("ToMain", size=[c], **pkt)
        self.out_y = ch("OutY", size=[1, 1], broadcast_shape=[1, n_dests], **pkt)

    def put_weights(self, src):
        """Stream weights: ``src(col, half)`` lists the L3 slices shim channel
        ``half`` of column ``col`` sends, in order."""
        for i in range(self.cols):
            for h in range(2):
                for s in src(i, h):
                    self.w_ex.put(s, indices=[i, h])

    def broadcast(self, src):
        """Send each column the inputs ``src(col)`` lists, in order."""
        for i in range(self.cols):
            for b in src(i):
                self.x_b.put(b, indices=[i, 0])

    def _l2(self, seg, shape, col, dt=bf16):
        return air.alloc(shape, dt, scope=seg.private(), column=col)

    def feed(self, seg, nblk):
        """The column memtiles: ``nblk`` blocks per core, one L2 buffer each.
        The rings are count-free: every phase goes round the same buffers."""
        half = self.rows // 2
        for i in range(self.cols):
            lw = [
                self._l2(seg, [self.block_bytes], self.columns[i], ui8)
                for _ in range(self.rows)
            ]
            for b in air.sequential(0, nblk):
                for r in range(self.rows):
                    self.w_ex.get(lw[r], indices=[i, r // half])
                for r in range(self.rows):
                    self.w_l1.put(lw[r], indices=[i, r])

    def gather(self, seg, nrd):
        """``nrd`` rounds through the column memtiles and the assemble memtile
        to the egress. Only the first piece of each gather carries a header."""
        rb = self.row_block
        for rd in air.sequential(0, nrd):
            for i in range(self.cols):
                grp = self._l2(seg, [HDR + self.coly], self.columns[i])
                for r in range(self.rows):
                    off = 0 if r == 0 else HDR + r * rb
                    sz = HDR + rb if r == 0 else rb
                    self.out_a.get(grp[off : off + sz], indices=[i, r])
                self.to_main.put(grp, indices=[i])
                air.dealloc(grp)
            ml = self._l2(seg, [HDR + self.round], self.assemble_col)
            for i in range(self.cols):
                off = 0 if i == 0 else HDR + i * self.coly
                sz = HDR + self.coly if i == 0 else self.coly
                self.to_main.get(ml[off : off + sz], indices=[i])
            self.out_y.put(ml, indices=[0, 0])
            air.dealloc(ml)

    def herd(self, xw, flush, phases, params=(), name="mvm", layers=None):
        """The array's cores. ``phases()`` yields the Phases in order; it runs
        in the herd body, so a count may read a herd parameter.
        ``flush(acc, y)`` writes a row block at y + 16, after its header.
        ``xw`` is the width of the x buffer, or a list of widths, one buffer
        each. With ``layers``, the cores loop that many times over
        ``phases(j)``, j the loop index: one program for layers whose counts
        and consumers differ, as long as their transfers do not."""
        runs = []
        for i, c in enumerate(self.columns):
            if runs and c == runs[-1][1] + runs[-1][2]:
                runs[-1][2] += 1
            else:
                runs.append([i, c, 1])
        for base, c, n in runs:
            self._herd(xw, flush, phases, params, name, base, c, n, layers)

    def _herd(self, xw, flush, phases, params, name, base, col, n, layers):
        rb = self.row_block
        if base or n != self.cols:
            name = f"{name}{base}"
        with air.herd(
            [range(n), range(self.rows)],
            name=name,
            shape=(n, self.rows),
            at=(col, self.row0),
            params=list(params),
        ) as he:

            @he.body
            def _(tx, ty):
                tx = tx + base if base else tx
                p = he.private()
                # two weight buffers, filled alternately: the DMA brings one
                # block in while the kernel reads the other
                wab = [air.alloc([self.block_bytes], ui8, scope=p) for _ in range(2)]
                xs = [
                    air.alloc([w], bf16, scope=p)
                    for w in (xw if isinstance(xw, (list, tuple)) else [xw])
                ]
                yacc = air.alloc([self.acc_words], f32, scope=p)
                y = air.alloc([16 + rb], bf16, scope=p)

                if layers:
                    for j in air.sequential(0, layers):
                        self._phases(phases(j), xs, wab, yacc, y, flush, tx, ty)
                else:
                    self._phases(phases(), xs, wab, yacc, y, flush, tx, ty)

    def _phases(self, phases, xs, wab, yacc, y, flush, tx, ty):
        rb = self.row_block
        for ph in phases:
            # blocks two at a time; every phase has an even count, so each
            # starts on buffer 0
            x = xs[ph.xbuf]
            if ph.fresh:
                self.x_b.get(x[0 : ph.xw], indices=[tx, ty])
            for k in air.sequential(0, ph.npairs):
                for half, wb in enumerate(wab):
                    cb = (k * 2 + half) % ph.ncb
                    self.w_l1.get(wb, indices=[tx, ty])
                    ph.step(x, wb, yacc, cb)
                    with air.ops.branch(cb == ph.ncb - 1):
                        flush(yacc, y)
                        self.out_a.put(
                            y[16 - HDR : 16 + rb],
                            indices=[tx, ty],
                            dest=ph.dest,
                        )
