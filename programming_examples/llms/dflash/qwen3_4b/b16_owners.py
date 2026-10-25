# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Two eight-row residual owners and normalized caches for Qwen3-4B B16.

Owner storage: 40 KiB residual, 10 KiB gamma, two 2 KiB staging buffers,
a 1-KiB source buffer, 32-byte factors and a 6080-byte stack. Fill chunks
are 128 elements per row. Two caches join 256-wide projection packets with
separate receive/output storage to preserve DMA ownership.
"""


class B16Owners:
    def __init__(self, f):
        self.f = f
        assert f.K == 2560 and f.BATCH == 16 and f.NORMALIZATION_CACHE
        assert f.XCHUNK == 256 and f.XMT_PCOL == 4
        self.bf, self.fp, self.i32 = (
            f.BF16Type.get(),
            f.F32Type.get(),
            f.IntegerType.get_signless(32),
        )
        self.l1 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L1)
        self.l2 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L2)

        def fn(name, args, obj):
            op = f.FuncOp(name, (args, []), visibility="private")
            op.attributes["link_with"] = f.StringAttr.get(obj)
            return op

        self.scale = fn(
            "rms_scale_row_aie",
            [self.mem(8, dtype=self.fp), self.mem(8 * f.K), self.i32, self.i32],
            "rms_residual.o",
        )
        self.norm = fn(
            "rms_chunk_aie",
            [
                self.mem(1024),
                self.mem(8 * f.K),
                self.mem(f.K),
                self.mem(8, dtype=self.fp),
                self.i32,
                self.i32,
                self.i32,
            ],
            "rms_residual.o",
        )
        self.residual = fn(
            "residual_acc_row_aie",
            [self.mem(8 * f.K), self.mem(1024), self.i32, self.i32, self.i32],
            "rms_residual.o",
        )
        self.replay = fn(
            "b16_normalized_cache_chunk",
            [self.mem(2048), self.mem(8 * f.K), self.i32],
            "norm_cache.o",
        )
        self.join = fn(
            "b16_normalized_cache_join",
            [self.mem(4096), self.mem(2048), self.mem(8 * f.K), self.i32],
            "norm_cache.o",
        )
        self.load = fn(
            "owner_load_chunk",
            [
                self.mem(512),
                self.mem(8 * f.K),
                self.mem(f.K),
                self.mem(f.K),
                self.i32,
                self.i32,
            ],
            "rms_residual.o",
        )
        f.channel_decl("owner_source_host", size=[1])
        broadcast = f.Channel("owner_source_all", size=[1, 1], broadcast_shape=[2, 1])
        broadcast.operation.attributes["air.shared_resident_ring"] = f.UnitAttr.get()
        for owner in range(2):
            for name in (
                "owner_fill",
                "owner_replay",
                "owner_update",
                "owner_final",
            ):
                f.channel_decl(f"{name}_{owner}", size=[1])

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
            f.arith.cmpi(f.arith.CmpIPredicate.ne, arm, self.ci(0)),
            [],
            has_else=False,
        )
        branch.operation.attributes["air.arm_select"] = f.UnitAttr.get()
        with f.InsertionPoint(branch.thenRegion.blocks[0]):
            body()
            f.yield_([])

    def emit(self, arm):
        f = self.f
        ix = self.ix
        ci = self.ci

        # One shim endpoint, then circuit broadcast. This avoids backpressured
        # host-to-owner packet streams sharing arbiters with control packets.
        for _ in f.for_(ix(0), ix(2), ix(1)):
            source = f.AllocOp(self.mem(512, self.l2), [], [])
            source.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                self.i32, 3
            )
            f.ChannelGet("owner_source_host", source)
            f.ChannelPut("owner_source_all", source, indices=[ix(0), ix(0)])
            f.DeallocOp(source)
            f.yield_([])

        def updates():
            for _ in f.for_(ix(0), ix(f.I2P[1] + f.I2P[f.DOWN_PHASE]), ix(1)):
                buf = f.AllocOp(self.mem(16 * 512, self.l2), [], [])
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
                # One DMA task per owner streams four narrow packets. Keeping
                # the four chunks in one descriptor preserves the shared
                # buffer's two-reader release chain in current AIR lowering.
                for owner in range(2):
                    f.ChannelPut(
                        f"owner_update_{owner}",
                        buf,
                        offsets=[0, 0, owner * 4096],
                        sizes=[4, 8, 128],
                        strides=[128, 512, 1],
                    )
                f.DeallocOp(buf)
                f.yield_([])

        self.gate(arm, updates)

        for owner in range(2):

            @f.herd(name=f"rms_owner_{owner}", sizes=[1, 1], operands=[arm])
            def rms(tx, ty, sx, sy, A, owner=owner):
                xb = f.AllocOp(self.mem(8 * f.K), [], [])
                weights = [f.AllocOp(self.mem(f.K), [], []) for _ in range(2)]
                out, update = [f.AllocOp(self.mem(1024), [], []) for _ in range(2)]
                scales = f.AllocOp(self.mem(8, dtype=self.fp), [], [])
                source = f.AllocOp(self.mem(512), [], [])
                buffers = [xb, *weights, out, update, scales, source]
                for buf in buffers:
                    buf.operation.attributes["air.disable_alloc_sink"] = (
                        f.UnitAttr.get()
                    )
                for chunk in f.for_(ix(0), ix(90), ix(1)):
                    f.ChannelGet("owner_source_all", source, indices=[ix(owner), ix(0)])
                    f.CallOp(
                        self.load,
                        [
                            source,
                            xb,
                            *weights,
                            f.arith.index_cast(self.i32, chunk),
                            ci(owner),
                        ],
                    )
                    f.yield_([])

                def normalized(w):
                    for row in range(8):
                        f.CallOp(self.scale, [scales, xb, ci(row), A])
                    for chunk in f.for_(ix(0), ix(20), ix(1)):
                        f.CallOp(
                            self.norm,
                            [
                                out,
                                xb,
                                w,
                                scales,
                                ci(8),
                                f.arith.index_cast(self.i32, chunk),
                                ci(128),
                            ],
                        )
                        f.ChannelPut(f"owner_fill_{owner}", out)
                        f.yield_([])

                def accumulated():
                    for chunk in f.for_(ix(0), ix(20), ix(1)):
                        f.ChannelGet(f"owner_update_{owner}", update)
                        offset = f.arith.muli(
                            f.arith.index_cast(self.i32, chunk), ci(128)
                        )
                        for row in range(8):
                            f.CallOp(
                                self.residual, [xb, update, ci(row), offset, ci(128)]
                            )
                        f.yield_([])

                normalized(weights[0])

                def tail():
                    accumulated()
                    normalized(weights[1])
                    accumulated()
                    f.ChannelPut(f"owner_final_{owner}", xb)

                self.gate(A, tail)
                for buf in reversed(buffers):
                    f.DeallocOp(buf)

            rms.attributes["link_with"] = f.StringAttr.get("rms_residual.o")
            rms.attributes["x_loc"] = f.IntegerAttr.get(
                f.T.i64(), 2 if owner == 0 else 5
            )
            rms.attributes["y_loc"] = f.IntegerAttr.get(f.T.i64(), 2)

            @f.herd(name=f"norm_cache_{owner}", sizes=[1, 1], operands=[arm])
            def cache(tx, ty, sx, sy, A, owner=owner):
                slab, out = f.AllocOp(self.mem(8 * f.K), [], []), f.AllocOp(
                    self.mem(2048 if owner == 0 else 4096), [], []
                )
                peer = f.AllocOp(self.mem(2048), [], []) if owner else None
                for buf in (slab, out, *([peer] if peer is not None else [])):
                    buf.operation.attributes["air.disable_alloc_sink"] = (
                        f.UnitAttr.get()
                    )

                def emit(count):
                    f.ChannelGet(f"owner_fill_{owner}", slab)
                    for _ in f.for_(ix(0), count, ix(1)):
                        for chunk in f.for_(ix(0), ix(10), ix(1)):
                            if owner == 0:
                                f.CallOp(
                                    self.replay,
                                    [out, slab, f.arith.index_cast(self.i32, chunk)],
                                )
                                f.ChannelPut("owner_replay_0", out)
                            else:
                                # Receive the first eight rows, then fill our
                                # eight rows in the same output packet. This
                                # fits in L1 and removes the L2 join entirely.
                                f.ChannelGet(
                                    "owner_replay_0",
                                    peer,
                                    offsets=[0],
                                    sizes=[2048],
                                    strides=[1],
                                )
                                f.CallOp(
                                    self.join,
                                    [
                                        out,
                                        peer,
                                        slab,
                                        f.arith.index_cast(self.i32, chunk),
                                    ],
                                )
                                f.ChannelPut("xnorm", out)
                            f.yield_([])
                        f.yield_([])

                decoding = f.arith.cmpi(f.arith.CmpIPredicate.ne, A, ci(0))
                emit(f.arith.select(decoding, ix(f.REFEED[0]), ix(f.VOCAB_RNDS)))
                self.gate(A, lambda: emit(ix(f.REFEED[f.GATEUP_PHASE])))
                if peer is not None:
                    f.DeallocOp(peer)
                f.DeallocOp(out)
                f.DeallocOp(slab)

            cache.attributes["link_with"] = f.StringAttr.get("norm_cache.o")
            cache.attributes["x_loc"] = f.IntegerAttr.get(
                f.T.i64(), 2 if owner == 0 else 5
            )
            cache.attributes["y_loc"] = f.IntegerAttr.get(
                f.T.i64(), 3 if owner == 0 else 5
            )
