# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Four-row owner mapping shared by the 8B B8 graphs.

Core coordinates: RMS (2,2)/(5,2), caches (2,3)/(5,5). RoPE moves to (2,5),
while the two attention relays keep (2,4)/(5,4). Together with the existing
projection, attention and GLU cores, this occupies all 32 compute tiles.

MemTile (2,1) demultiplexes residual updates to the two owners. (3,1) joins
normalized halves into the original packet stream; the FFN intermediate
cache moves from (3,1) to (5,1). Activation broadcast stays at (4,1). This split leaves each owner room for its gamma vectors and staging buffers.
"""


class ShardedOwners:
    def __init__(self, f, scale, norm, residual, replay):
        self.f = f
        self.scale, self.norm, self.residual, self.replay = (
            scale,
            norm,
            residual,
            replay,
        )
        self.bf, self.fp, self.i32 = (
            f.BF16Type.get(),
            f.F32Type.get(),
            f.IntegerType.get_signless(32),
        )
        self.l1 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L1)
        self.l2 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L2)
        assert (
            f.K == 4096 and f.BATCH == 8 and f.NORMALIZATION_CACHE and f.XMT_PCOL == 4
        )

    def mem(self, n, space=None, dtype=None):
        return self.f.MemRefType.get(
            [n], dtype or self.bf, memory_space=space or self.l1
        )

    def ix(self, n):
        return self.f.arith.ConstantOp.create_index(n)

    def ci(self, n):
        return self.f.arith.ConstantOp(self.f.IntegerAttr.get(self.i32, n), None).result

    def gate(self, arm, body):
        f = self.f
        branch = f.IfOp(
            f.arith.cmpi(f.arith.CmpIPredicate.ne, arm, self.ci(0)), [], has_else=False
        )
        branch.operation.attributes["air.arm_select"] = f.UnitAttr.get()
        with f.InsertionPoint(branch.thenRegion.blocks[0]):
            body()
            f.yield_([])

    def emit(self, arm):
        f = self.f
        ix = self.ix
        ci = self.ci

        # The existing projection packet order is converted to [token,512]
        # before the two independent four-row writes. One staging allocation
        # cannot alias either core's live normalization output.
        def updates():
            for _ in f.for_(ix(0), ix(f.I2P[1] + f.I2P[3]), ix(1)):
                buf = f.AllocOp(self.mem(8 * 512, self.l2), [], [])
                buf.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                    self.i32, 2
                )
                offsets, sizes, strides = f.outy_tokmajor(512)
                f.ChannelGet(
                    "outY",
                    buf,
                    indices=[ix(0), ix(f.RMS_DEST)],
                    offsets=[ix(v) for v in offsets],
                    sizes=[ix(v) for v in sizes],
                    strides=[ix(v) for v in strides],
                )
                for owner in range(2):
                    f.ChannelPut(
                        f"owner_update_{owner}",
                        buf,
                        offsets=[owner * 2048],
                        sizes=[2048],
                        strides=[1],
                    )
                f.DeallocOp(buf)
                f.yield_([])

        self.gate(arm, updates)

        def join(count):
            for _ in f.for_(ix(0), ix(count * 8), ix(1)):
                buf = f.AllocOp(self.mem(4096, self.l2), [], [])
                buf.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                    self.i32, 3
                )
                for owner in range(2):
                    f.ChannelGet(
                        f"owner_replay_{owner}",
                        buf,
                        offsets=[owner * 2048],
                        sizes=[2048],
                        strides=[1],
                    )
                f.ChannelPut("xnorm", buf, offsets=[0], sizes=[4096], strides=[1])
                f.DeallocOp(buf)
                f.yield_([])

        def voc():
            join(f.VOCAB_RNDS)
            f.yield_([])

        def dec():
            join(f.REFEED[0] + f.REFEED[2])
            f.yield_([])

        f.index_switch(
            [],
            f.arith.index_cast(f.IndexType.get(), arm),
            [0],
            case_body_builder=lambda op, i, cv: voc(),
            default_body_builder=lambda op: dec(),
        )

        for owner in range(2):

            @f.herd(name=f"rms_owner_{owner}", sizes=[1, 1], operands=[arm])
            def rms(tx, ty, sx, sy, A, owner=owner):
                xb = f.AllocOp(self.mem(4 * 4096), [], [])
                weights = [f.AllocOp(self.mem(4096), [], []) for _ in range(2)]
                out, update = [f.AllocOp(self.mem(2048), [], []) for _ in range(2)]
                scales = f.AllocOp(self.mem(4, dtype=self.fp), [], [])
                all_buffers = [xb, *weights, out, update, scales]
                for buf in all_buffers:
                    buf.operation.attributes["air.disable_alloc_sink"] = (
                        f.UnitAttr.get()
                    )
                for buf in [xb, *weights]:
                    f.ChannelGet(f"owner_source_{owner}", buf)

                def normalized(w):
                    for row in range(4):
                        f.CallOp(self.scale, [scales, xb, ci(row), A])
                    for chunk in f.for_(ix(0), ix(8), ix(1)):
                        f.CallOp(
                            self.norm,
                            [
                                out,
                                xb,
                                w,
                                scales,
                                ci(4),
                                f.arith.index_cast(self.i32, chunk),
                                ci(512),
                            ],
                        )
                        f.ChannelPut(
                            f"owner_fill_{owner}",
                            out,
                            offsets=[0],
                            sizes=[2048],
                            strides=[1],
                        )
                        f.yield_([])

                def accumulated():
                    for chunk in f.for_(ix(0), ix(8), ix(1)):
                        f.ChannelGet(f"owner_update_{owner}", update)
                        offset = f.arith.muli(
                            f.arith.index_cast(self.i32, chunk), ci(512)
                        )
                        for row in range(4):
                            f.CallOp(
                                self.residual, [xb, update, ci(row), offset, ci(512)]
                            )
                        f.yield_([])

                normalized(weights[0])

                def decode_tail():
                    accumulated()
                    normalized(weights[1])
                    accumulated()
                    f.ChannelPut(
                        f"owner_final_{owner}",
                        xb,
                        offsets=[0],
                        sizes=[16384],
                        strides=[1],
                    )

                self.gate(A, decode_tail)
                for buf in reversed(all_buffers):
                    f.DeallocOp(buf)

            rms.attributes["link_with"] = f.StringAttr.get("rms_residual.o")
            rms.attributes["x_loc"] = f.IntegerAttr.get(
                f.T.i64(), 2 if owner == 0 else 5
            )
            rms.attributes["y_loc"] = f.IntegerAttr.get(f.T.i64(), 2)

            @f.herd(name=f"norm_cache_{owner}", sizes=[1, 1], operands=[arm])
            def cache(tx, ty, sx, sy, A, owner=owner):
                slab, out = f.AllocOp(self.mem(16384), [], []), f.AllocOp(
                    self.mem(2048), [], []
                )
                for buf in (slab, out):
                    buf.operation.attributes["air.disable_alloc_sink"] = (
                        f.UnitAttr.get()
                    )

                def emit(count):
                    f.ChannelGet(f"owner_fill_{owner}", slab)
                    for _ in f.for_(ix(0), count, ix(1)):
                        for chunk in f.for_(ix(0), ix(8), ix(1)):
                            f.CallOp(
                                self.replay,
                                [out, slab, f.arith.index_cast(self.i32, chunk)],
                            )
                            f.ChannelPut(
                                f"owner_replay_{owner}",
                                out,
                                offsets=[0],
                                sizes=[2048],
                                strides=[1],
                            )
                            f.yield_([])
                        f.yield_([])

                decoding = f.arith.cmpi(f.arith.CmpIPredicate.ne, A, ci(0))
                emit(f.arith.select(decoding, ix(f.REFEED[0]), ix(f.VOCAB_RNDS)))
                self.gate(A, lambda: emit(ix(f.REFEED[2])))
                f.DeallocOp(out)
                f.DeallocOp(slab)

            cache.attributes["link_with"] = f.StringAttr.get("norm_cache.o")
            cache.attributes["x_loc"] = f.IntegerAttr.get(
                f.T.i64(), 2 if owner == 0 else 5
            )
            cache.attributes["y_loc"] = f.IntegerAttr.get(
                f.T.i64(), 3 if owner == 0 else 5
            )
