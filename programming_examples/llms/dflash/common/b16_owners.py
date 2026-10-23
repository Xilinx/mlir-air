# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""4096-wide B16: four resident four-row owners, DMA normalized cache in DDR.

Residual/gamma/output buffers and arithmetic reuse the four-row B8 owner.
B16 adds a 512-word source buffer and, on the paired output writers, a
separate 512-word cascade receive buffer.
The former two cache cores become residual owners. Y scratch caches all
16 normalized rows; no normalization replay computation is introduced.
"""


class B16Owners:
    def __init__(self, f, scale, norm, residual, replay, load, store, copy_peer):
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
            f.K == 4096 and f.BATCH == 16 and f.NORMALIZATION_CACHE and f.XMT_PCOL == 4
        )
        self.load = load
        self.store = store
        self.copy_peer = copy_peer

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
        # Route the host stream through one L2 DMA endpoint, then broadcast
        # chunks on chip. The shim placer need not pack a four-way fanout.
        for _ in f.for_(ix(0), ix(2), ix(1)):
            source = f.AllocOp(self.mem(2048, self.l2), [], [])
            source.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                self.i32, 3
            )
            f.ChannelGet("owner_source_host", source)
            f.ChannelPut("owner_source_all", source, indices=[ix(0), ix(0)])
            f.DeallocOp(source)
            f.yield_([])

        # The existing projection packet order is converted to [token,512]
        # before the two independent four-row writes. One staging allocation
        # cannot alias either core's live normalization output.
        def updates():
            for _ in f.for_(ix(0), ix(f.I2P[1] + f.I2P[3]), ix(1)):
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
                # All owners drain the four row groups in the same order;
                # each retains only its own group. One broadcast endpoint
                # replaces four independent cross-array update routes.
                f.ChannelPut("owner_update_all", buf, indices=[ix(0), ix(0)])
                f.DeallocOp(buf)
                f.yield_([])

        self.gate(arm, updates)

        for owner in range(4):

            @f.herd(name=f"rms_owner_{owner}", sizes=[1, 1], operands=[arm])
            def rms(tx, ty, sx, sy, A, owner=owner):
                xb = f.AllocOp(self.mem(4 * 4096), [], [])
                weights = [f.AllocOp(self.mem(4096), [], []) for _ in range(2)]
                out, update = [f.AllocOp(self.mem(2048), [], []) for _ in range(2)]
                source = f.AllocOp(self.mem(512), [], [])
                peer = f.AllocOp(self.mem(512), [], []) if owner < 2 else None
                scales = f.AllocOp(self.mem(4, dtype=self.fp), [], [])
                all_buffers = [xb, *weights, out, update, source, scales]
                if peer is not None:
                    all_buffers.append(peer)
                for buf in all_buffers:
                    buf.operation.attributes["air.disable_alloc_sink"] = (
                        f.UnitAttr.get()
                    )
                for chunk in f.for_(ix(0), ix(144), ix(1)):
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

                def send():
                    if owner >= 2:
                        for part in f.for_(ix(0), ix(4), ix(1)):
                            f.ChannelPut(
                                "owner_pair",
                                out,
                                indices=[ix(owner - 2)],
                                offsets=[f.arith.muli(part, ix(512))],
                                sizes=[512],
                                strides=[1],
                            )
                            f.yield_([])
                    else:
                        f.ChannelPut(f"owner_fill_{owner}", out)
                        # The cascade receiver never writes the pending DMA
                        # output. Ordinary copy calls carry its write dependency.
                        for part in f.for_(ix(0), ix(4), ix(1)):
                            f.ChannelGet("owner_pair", peer, indices=[ix(owner)])
                            f.CallOp(
                                self.copy_peer,
                                [out, peer, f.arith.index_cast(self.i32, part)],
                            )
                            f.yield_([])
                        f.ChannelPut(f"owner_fill_{owner}", out)

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
                        send()
                        f.yield_([])

                def accumulated():
                    for chunk in f.for_(ix(0), ix(8), ix(1)):
                        offset = f.arith.muli(
                            f.arith.index_cast(self.i32, chunk), ci(512)
                        )
                        for group in range(4):
                            f.ChannelGet(
                                "owner_update_all", update, indices=[ix(owner), ix(0)]
                            )
                            if group == owner:
                                for row in range(4):
                                    f.CallOp(
                                        self.residual,
                                        [xb, update, ci(row), offset, ci(512)],
                                    )
                        f.yield_([])

                normalized(weights[0])

                def decode_tail():
                    accumulated()
                    normalized(weights[1])
                    accumulated()
                    for chunk in f.for_(ix(0), ix(8), ix(1)):
                        f.CallOp(
                            self.store, [out, xb, f.arith.index_cast(self.i32, chunk)]
                        )
                        send()
                        f.yield_([])

                self.gate(A, decode_tail)
                for buf in reversed(all_buffers):
                    f.DeallocOp(buf)

            rms.attributes["link_with"] = f.StringAttr.get("rms_residual.o")
            rms.attributes["x_loc"] = f.IntegerAttr.get(
                f.T.i64(), ((2, 2), (5, 2), (2, 3), (5, 3))[owner][0]
            )
            rms.attributes["y_loc"] = f.IntegerAttr.get(
                f.T.i64(), ((2, 2), (5, 2), (2, 3), (5, 3))[owner][1]
            )
