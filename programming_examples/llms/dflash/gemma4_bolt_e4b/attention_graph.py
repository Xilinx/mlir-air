# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4 attention of the eight-token decoder (kernels/attn_b8.cc).

Each KV head has one shard per two tokens; a shard computes two tokens' four
query heads.
A shard has two inbound DMA channels, so Q reaches it on K's: the K stream is
four Q frames, one per shard, then the K blocks, and every shard keeps its own
frame. The K and V streams read whole KV rows (a run-time length needs a
contiguous shim transfer); a memory tile sends each head's half to that
head's shards as [DH/8][row][8].

Q frames are read from DDR, where the rope core's output lands in frame
order, [shard][token][q head % 4][KV head][DH]; tests/check_attention.py
checks this graph on its own.
"""

TOKENS_PER_SHARD = 2
Q_PER_KV = 4


class GemmaAttention:
    """One sequence (SEQUENCES == 1): the batch's tokens share one cache, two
    tokens to a shard. Independent sequences: a shard, a K/V stream and a
    length per sequence; a shard's frame holds its one token's four query
    heads, and the second token's rows are ignored."""

    def __init__(
        self,
        f,
        dh=512,
        kv_heads=2,
        cols=(3, 4),
        ring_cols=((2, 4), (3, 3)),
        out_cols=(3, 5),
    ):
        self.f, self.dh, self.kv = f, dh, kv_heads
        self.batch = f.BATCH
        self.seqs = getattr(f, "SEQUENCES", 1)
        self.shards = self.seqs if self.seqs > 1 else f.BATCH // TOKENS_PER_SHARD
        # tokens a K/V stream serves, and Q frames at the head of each
        self.stream_tokens = 1 if self.seqs > 1 else f.BATCH
        self.stream_frames = 1 if self.seqs > 1 else self.shards
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
        for st in range(self.seqs):
            f.channel_decl(self.stream("inK", st), size=[1])
            f.channel_decl(self.stream("inV", st), size=[1])
        for h in range(kv_heads):
            for name in ("pk", "pv"):
                if self.seqs > 1:
                    f.channel_decl(f"{name}{h}", size=[1, self.seqs])
                else:
                    f.Channel(
                        f"{name}{h}", size=[1, 1], broadcast_shape=[1, self.shards]
                    )
            # circuits: a memory tile's packet BDs take packets in arrival
            # order, not by source, and four shards' outputs then stall
            f.channel_decl(f"po{h}", size=[1, self.shards])
            f.channel_decl(f"oj{h}", size=[1])

    def stream(self, name, seq):
        return name if self.seqs == 1 else f"{name}{seq}"

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
        """Eight-key blocks a stream's last token sees: length + tokens - 1
        keys."""
        f = self.f
        return f.arith.index_cast(
            f.IndexType.get(),
            f.arith.divui(
                f.arith.addi(length, self.ci(self.stream_tokens - 1 + 7)),
                self.ci(8),
            ),
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
        """The Q frames, in frame order at `offset` of `buf`. Independent
        sequences: sequence s's frame starts at its token's rows, so its
        second half is the next token's (or padding) and is ignored."""
        f = self.f
        if self.seqs == 1:
            f.ChannelPut(
                "inK",
                buf,
                offsets=[offset],
                sizes=[self.ix(self.shards * 2 * self.frame)],
                strides=[self.ix(1)],
            )
            return
        for st in range(self.seqs):
            f.ChannelPut(
                self.stream("inK", st),
                buf,
                offsets=[f.arith.addi(offset, self.ix(st * Q_PER_KV * self.rw))],
                sizes=[self.ix(2 * self.frame)],
                strides=[self.ix(1)],
            )

    def first_block(self, length, window):
        """The first eight-key block any token of the block sees: the first
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

    def put_kv(self, kv, k_offset, v_offset, length, window, seq=0):
        """Whole KV rows of the blocks [first_block, blocks): K after the Q
        frames, and V, on sequence `seq`'s streams. window: i32 as in emit."""
        f = self.f
        b0 = self.first_block(length, window)
        nb = f.arith.subi(self.blocks(length), b0)
        skip = f.arith.muli(b0, self.ix(8 * self.rw))
        for name, off in (("inK", k_offset), ("inV", v_offset)):
            f.ChannelPut(
                self.stream(name, seq),
                kv,
                offsets=[self.ix(0), f.arith.addi(off, skip)],
                sizes=[nb, self.ix(8 * self.rw)],
                strides=[self.ix(8 * self.rw), self.ix(1)],
            )

    # ---- segment side ----

    def emit(self, lengths, arm, window):
        """lengths: one i32 per sequence. window: i32, the keys a token sees
        back from itself, 0 for all."""
        f, ix, ci = self.f, self.ix, self.ci
        if not isinstance(lengths, (list, tuple)):
            lengths = [lengths]

        def rings():
            for st in range(self.seqs):
                cols = self.ring_cols[st]
                for name, col in (("K", cols[0]), ("V", cols[1])):
                    ring(st, name, col)

        def ring(st, name, col):
            for _ in f.for_(ix(0), ix(2), ix(1)):
                buf = f.AllocOp(self.mem(8 * self.rw, space=self.l2), [], [])
                buf.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                    self.i32, col
                )
                f.ChannelGet(self.stream(f"in{name}", st), buf)
                for h in range(self.kv):
                    f.ChannelPut(
                        f"p{name.lower()}{h}",
                        buf,
                        indices=[ix(0), ix(st)],
                        offsets=[ix(0), ix(0), ix(h * self.dh)],
                        sizes=[ix(self.dh // 8), ix(8), ix(8)],
                        strides=[ix(8), ix(self.rw), ix(1)],
                    )
                f.DeallocOp(buf)
                f.yield_([])

        self.gate(arm, rings)

        for h in range(self.kv):

            def shard(tx, ty, sx, sy, *ops, h=h):
                *Ls, A, W = ops

                def run():
                    n = self.frame
                    kb = f.AllocOp(self.mem(n), [], [])
                    vb = f.AllocOp(self.mem(n), [], [])
                    qs = f.AllocOp(self.mem(n), [], [])
                    y = f.AllocOp(self.mem(n, self.fp), [], [])
                    m = f.AllocOp(self.mem(16, self.fp), [], [])
                    ls = f.AllocOp(self.mem(16, self.fp), [], [])
                    if self.seqs > 1:
                        # this shard's sequence: its own length
                        L = Ls[0]
                        for st in range(1, self.seqs):
                            L = f.arith.select(
                                f.arith.cmpi(f.arith.CmpIPredicate.eq, ty, ix(st)),
                                Ls[st],
                                L,
                            )
                        vis = L
                    else:
                        (L,) = Ls
                        vis = f.arith.addi(
                            L, f.arith.muli(f.arith.index_cast(self.i32, ty), ci(2))
                        )
                    nq = self.stream_frames
                    # one get on K's channel for every frame: a runtime trip
                    # count cannot share the channel with another
                    b0 = self.first_block(L, W)
                    frames = f.arith.addi(f.arith.subi(self.blocks(L), b0), ix(nq))
                    for fr in f.for_(ix(0), frames, ix(1)):
                        f.ChannelGet(f"pk{h}", kb, indices=[tx, ty])
                        is_q = f.IfOp(
                            f.arith.cmpi(f.arith.CmpIPredicate.ult, fr, ix(nq)),
                            [],
                            has_else=True,
                        )
                        with f.InsertionPoint(is_q.thenRegion.blocks[0]):
                            # a stream of its own carries one frame, this shard's
                            mine = f.IfOp(
                                f.arith.cmpi(
                                    f.arith.CmpIPredicate.eq,
                                    fr,
                                    ix(0) if self.seqs > 1 else ty,
                                ),
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
                                self.i32, f.arith.addi(f.arith.subi(fr, ix(nq)), b0)
                            )
                            f.CallOp(self.block, [qs, kb, vb, y, m, ls, b, vis, W])
                            f.yield_([])
                        f.yield_([])
                    f.CallOp(self.finish, [y, ls, kb])
                    if self.seqs > 1:
                        # the token's four heads; the frame's other half is
                        # padding
                        f.ChannelPut(
                            f"po{h}",
                            kb,
                            indices=[tx, ty],
                            offsets=[ix(0)],
                            sizes=[ix(Q_PER_KV * self.dh)],
                            strides=[ix(1)],
                        )
                    else:
                        f.ChannelPut(f"po{h}", kb, indices=[tx, ty])
                    for buf in (kb, vb, qs, y, m, ls):
                        f.DeallocOp(buf)

                self.gate(A, run)

            hd = f.herd(
                name=f"attn_h{h}",
                sizes=[1, self.shards],
                operands=list(lengths) + [arm, window],
            )(shard)
            hd.attributes["link_with"] = f.StringAttr.get("attn_b8.o")
            hd.attributes["x_loc"] = f.IntegerAttr.get(f.T.i64(), self.cols[h])
            hd.attributes["y_loc"] = f.IntegerAttr.get(f.T.i64(), 2)

    def gather(self, slab, qrow):
        """Every shard's output into slab, [token][half][q head][DH / 2]:
        each head's shards into a memory tile of its own (a tile's six
        inbound channels do not take all eight), then both into slab."""
        f, ix = self.f, self.ix
        rows = self.batch * Q_PER_KV * self.dh
        for h in range(self.kv):
            ob = f.AllocOp(self.mem(rows, space=self.l2), [], [])
            ob.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                self.i32, self.out_cols[h]
            )
            # a shard's tokens, four heads each
            span = (self.batch // self.shards) * Q_PER_KV * self.dh
            for s in range(self.shards):
                f.ChannelGet(
                    f"po{h}",
                    ob,
                    indices=[ix(0), ix(s)],
                    offsets=[ix(s * span)],
                    sizes=[ix(span)],
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
                sizes=[ix(self.batch), ix(Q_PER_KV), ix(2), ix(half)],
                strides=[ix(qrow), ix(half), ix(qrow // 2), ix(1)],
            )
