# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4 attention of the eight-token decoder (kernels/attn_b8.cc).

Each KV head has four shards; a shard computes two tokens' four query heads.
A shard has two inbound DMA channels, so Q reaches it on K's: the K stream is
four Q frames, one per shard, then the K blocks, and every shard keeps its own
frame. The K and V streams read whole KV rows (a run-time length needs a
contiguous shim transfer); a memory tile sends each head's half to that
head's shards as [DH/8][row][8].

Q frames are read from DDR, where the rope core's output lands in frame
order, [shard][token][q head % 4][KV head][DH]; tests/check_attention.py
checks this graph on its own.
"""

SHARDS = 4
Q_PER_KV = 4


class GemmaAttention:
    def __init__(
        self, f, dh=512, kv_heads=2, cols=(3, 4), ring_cols=(2, 4), out_cols=(3, 5)
    ):
        self.f, self.dh, self.kv = f, dh, kv_heads
        self.out_cols = out_cols
        self.cols, self.ring_cols = cols, ring_cols
        self.bf = f.BF16Type.get()
        self.fp = f.F32Type.get()
        self.i32 = f.IntegerType.get_signless(32)
        self.l1 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L1)
        self.l2 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L2)
        self.rw = kv_heads * dh  # one KV row
        self.frame = 8 * dh  # eight keys of one head, or one shard's Q
        n = self.frame
        self.start = self.function(
            "g8_start",
            [self.mem(n), self.mem(n), self.mem(n, self.fp)]
            + [self.mem(16, self.fp)] * 2,
        )
        self.block = self.function(
            "g8_block",
            [self.mem(n)] * 3
            + [self.mem(n, self.fp), self.mem(16, self.fp), self.mem(16, self.fp)]
            + [self.i32, self.i32, self.i32],
        )
        self.finish = self.function(
            "g8_finish", [self.mem(n, self.fp), self.mem(16, self.fp), self.mem(n)]
        )
        f.channel_decl("inK", size=[1])
        f.channel_decl("inV", size=[1])
        for h in range(kv_heads):
            for name in ("pk", "pv"):
                f.Channel(f"{name}{h}", size=[1, 1], broadcast_shape=[1, SHARDS])
            # circuits: a memory tile's packet BDs take packets in arrival
            # order, not by source, and four shards' outputs then stall
            f.channel_decl(f"po{h}", size=[1, SHARDS])
            f.channel_decl(f"oj{h}", size=[1])

    def mem(self, n, dtype=None, space=None):
        return self.f.MemRefType.get(
            [n], dtype or self.bf, memory_space=space or self.l1
        )

    def ix(self, n):
        return self.f.arith.ConstantOp.create_index(n)

    def ci(self, n):
        return self.f.arith.ConstantOp(self.f.IntegerAttr.get(self.i32, n), None).result

    def function(self, name, args):
        fn = self.f.FuncOp(name, (args, []), visibility="private")
        fn.attributes["link_with"] = self.f.StringAttr.get("attn_b8.o")
        return fn

    def blocks(self, length):
        """Eight-key blocks the last token sees: length + 7 keys."""
        f = self.f
        return f.arith.index_cast(
            f.IndexType.get(),
            f.arith.divui(f.arith.addi(length, self.ci(14)), self.ci(8)),
        )

    def gate(self, arm, body):
        f = self.f

        def decode():
            body()
            f.yield_([])

        f.index_switch(
            [],
            f.arith.index_cast(f.IndexType.get(), arm),
            [0],
            case_body_builder=lambda op, i, cv: f.yield_([]),
            default_body_builder=lambda op: decode(),
        )

    # ---- launch side: shim transfers ----

    def put_query(self, buf, offset):
        """The four Q frames, in frame order at `offset` of `buf`."""
        f = self.f
        f.ChannelPut(
            "inK",
            buf,
            offsets=[offset],
            sizes=[self.ix(SHARDS * 2 * self.frame)],
            strides=[self.ix(1)],
        )

    def first_block(self, length, window):
        """The first eight-key block any of the eight tokens sees: the first
        token (position length - 1) sees keys from length - window on."""
        f = self.f
        idx_t = f.IndexType.get()
        first = f.arith.subi(length, window)
        first = f.arith.select(
            f.arith.andi(
                f.arith.cmpi(f.arith.CmpIPredicate.ne, window, self.ci(0)),
                f.arith.cmpi(f.arith.CmpIPredicate.sgt, first, self.ci(0)),
            ),
            first,
            self.ci(0),
        )
        return f.arith.index_cast(idx_t, f.arith.divui(first, self.ci(8)))

    def put_kv(self, kv, k_offset, v_offset, length, window):
        """Whole KV rows of the blocks [first_block, blocks): K after the Q
        frames, and V. window: i32 as in emit."""
        f = self.f
        b0 = self.first_block(length, window)
        nb = f.arith.subi(self.blocks(length), b0)
        skip = f.arith.muli(b0, self.ix(8 * self.rw))
        for name, off in (("inK", k_offset), ("inV", v_offset)):
            f.ChannelPut(
                name,
                kv,
                offsets=[self.ix(0), f.arith.addi(off, skip)],
                sizes=[nb, self.ix(8 * self.rw)],
                strides=[self.ix(8 * self.rw), self.ix(1)],
            )

    # ---- segment side ----

    def emit(self, length, arm, window):
        """window: i32, the keys a token sees back from itself, 0 for all."""
        f, ix, ci = self.f, self.ix, self.ci

        def rings():
            for name, col in (("K", self.ring_cols[0]), ("V", self.ring_cols[1])):
                for _ in f.for_(ix(0), ix(2), ix(1)):
                    buf = f.AllocOp(self.mem(8 * self.rw, space=self.l2), [], [])
                    buf.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                        self.i32, col
                    )
                    f.ChannelGet(f"in{name}", buf)
                    for h in range(self.kv):
                        f.ChannelPut(
                            f"p{name.lower()}{h}",
                            buf,
                            indices=[ix(0), ix(0)],
                            offsets=[ix(0), ix(0), ix(h * self.dh)],
                            sizes=[ix(self.dh // 8), ix(8), ix(8)],
                            strides=[ix(8), ix(self.rw), ix(1)],
                        )
                    f.DeallocOp(buf)
                    f.yield_([])

        self.gate(arm, rings)

        for h in range(self.kv):

            def shard(tx, ty, sx, sy, L, A, W, h=h):
                def run():
                    n = self.frame
                    kb = f.AllocOp(self.mem(n), [], [])
                    vb = f.AllocOp(self.mem(n), [], [])
                    qs = f.AllocOp(self.mem(n), [], [])
                    y = f.AllocOp(self.mem(n, self.fp), [], [])
                    m = f.AllocOp(self.mem(16, self.fp), [], [])
                    ls = f.AllocOp(self.mem(16, self.fp), [], [])
                    vis = f.arith.addi(
                        L, f.arith.muli(f.arith.index_cast(self.i32, ty), ci(2))
                    )
                    # one get on K's channel for every frame: a runtime trip
                    # count cannot share the channel with another
                    b0 = self.first_block(L, W)
                    frames = f.arith.addi(f.arith.subi(self.blocks(L), b0), ix(SHARDS))
                    for fr in f.for_(ix(0), frames, ix(1)):
                        f.ChannelGet(f"pk{h}", kb, indices=[tx, ty])
                        is_q = f.IfOp(
                            f.arith.cmpi(f.arith.CmpIPredicate.ult, fr, ix(SHARDS)),
                            [],
                            has_else=True,
                        )
                        with f.InsertionPoint(is_q.thenRegion.blocks[0]):
                            mine = f.IfOp(
                                f.arith.cmpi(f.arith.CmpIPredicate.eq, fr, ty),
                                [],
                                has_else=False,
                            )
                            with f.InsertionPoint(mine.thenRegion.blocks[0]):
                                f.CallOp(self.start, [kb, qs, y, m, ls])
                                f.yield_([])
                            f.yield_([])
                        with f.InsertionPoint(is_q.elseRegion.blocks[0]):
                            f.ChannelGet(f"pv{h}", vb, indices=[tx, ty])
                            b = f.arith.index_cast(
                                self.i32, f.arith.addi(f.arith.subi(fr, ix(SHARDS)), b0)
                            )
                            f.CallOp(self.block, [qs, kb, vb, y, m, ls, b, vis, W])
                            f.yield_([])
                        f.yield_([])
                    f.CallOp(self.finish, [y, ls, kb])
                    f.ChannelPut(f"po{h}", kb, indices=[tx, ty])
                    for buf in (kb, vb, qs, y, m, ls):
                        f.DeallocOp(buf)

                self.gate(A, run)

            hd = f.herd(
                name=f"attn_h{h}", sizes=[1, SHARDS], operands=[length, arm, window]
            )(shard)
            hd.attributes["link_with"] = f.StringAttr.get("attn_b8.o")
            hd.attributes["x_loc"] = f.IntegerAttr.get(f.T.i64(), self.cols[h])
            hd.attributes["y_loc"] = f.IntegerAttr.get(f.T.i64(), 2)

    def gather(self, slab, qrow):
        """Every shard's output into slab, [token][half][q head][DH / 2]:
        each head's four shards into a memory tile of its own (a tile's six
        inbound channels do not take all eight), then both into slab."""
        f, ix = self.f, self.ix
        rows = 8 * Q_PER_KV * self.dh
        for h in range(self.kv):
            ob = f.AllocOp(self.mem(rows, space=self.l2), [], [])
            ob.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                self.i32, self.out_cols[h]
            )
            for s in range(SHARDS):
                f.ChannelGet(
                    f"po{h}",
                    ob,
                    indices=[ix(0), ix(s)],
                    offsets=[ix(s * self.frame)],
                    sizes=[ix(self.frame)],
                    strides=[ix(1)],
                )
            f.ChannelPut(f"oj{h}", ob)
            f.DeallocOp(ob)
        # half-major columns, [half][q head][dh / 2]: a sliding layer's o
        # projection reads only the first half of each head (its v is at the
        # front of the slot), so its weighted x blocks come first
        half = self.dh // 2
        for h in range(self.kv):
            f.ChannelGet(
                f"oj{h}",
                slab,
                offsets=[ix(0), ix(0), ix(0), ix(h * Q_PER_KV * half)],
                sizes=[ix(8), ix(Q_PER_KV), ix(2), ix(half)],
                strides=[ix(qrow), ix(half), ix(qrow // 2), ix(1)],
            )
