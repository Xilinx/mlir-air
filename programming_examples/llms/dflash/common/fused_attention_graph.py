# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""B8 attention: eight four-query/head shards and two Q/K relays.

This file is the selected fused-shards attention graph used by target and drafter builds.
Keep each query's key-block order and pair grouping unchanged. The full decoder
supplies one KV stream per head-column after its append/readback dependency.
"""


class ParallelAttentionGraph:
    def __init__(self, f):
        self.f = f
        self.bf, self.fp, self.i32 = (
            f.BF16Type.get(),
            f.F32Type.get(),
            f.IntegerType.get_signless(32),
        )
        self.l1 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L1)
        self.l2 = f.IntegerAttr.get(f.T.i32(), f.MemorySpace.L2)
        self.qk = self.function(
            "dflash_fused_qk",
            [
                self.mem(8192),
                self.mem(4096),
                self.mem(64),
                self.mem(32, dtype=self.fp),
                self.mem(768),
                self.i32,
                self.i32,
                self.i32,
                self.i32,
            ],
        )
        self.pv = self.function(
            "dflash_fused_pv",
            [
                self.mem(768),
                self.mem(4096),
                self.mem(4096, dtype=self.fp),
                self.mem(64, dtype=self.fp),
                self.i32,
                self.i32,
                self.i32,
            ],
        )
        self.fin = self.function(
            "attn_kv_fin_tile",
            [
                self.mem(4096, dtype=self.fp),
                self.mem(64, dtype=self.fp),
                self.mem(8192),
                self.i32,
            ],
        )
        self.copy = self.function("dflash_copy_query", [self.mem(8192), self.mem(4096)])
        self.relayq = self.function(
            "dflash_relay_query", [self.mem(8192), self.mem(8192)]
        )
        self.relayk = self.function(
            "dflash_relay_key", [self.mem(8192), self.mem(8192)]
        )
        self.joinsend = self.function(
            "dflash_join_send", [self.mem(768), self.mem(8192), self.i32]
        )
        self.joinrecv = self.function(
            "dflash_join_receive", [self.mem(8192), self.mem(768), self.i32]
        )
        f.channel_decl("parallel_attention_output", size=[4])
        f.channel_decl(
            "parallel_attention_cascade", size=[4], channel_type="npu_cascade"
        )
        for group in range(2):
            for name in ("pqcol", "pkcol", "pqkout"):
                f.channel_decl(f"{name}{group}", size=[1])
        for cu in range(4):
            for name in ("pk", "pv"):
                ch = f.Channel(f"{name}{cu}", size=[1, 1], broadcast_shape=[1, 2])
                ch.operation.attributes["air.shared_resident_ring"] = f.UnitAttr.get()

    def mem(self, n, space=None, dtype=None):
        return self.f.MemRefType.get(
            [n], dtype or self.bf, memory_space=space or self.l1
        )

    def ix(self, n):
        return self.f.arith.ConstantOp.create_index(n)

    def ci(self, n):
        return self.f.arith.ConstantOp(self.f.IntegerAttr.get(self.i32, n), None).result

    def function(self, name, args):
        fn = self.f.FuncOp(name, (args, []), visibility="private")
        fn.attributes["link_with"] = self.f.StringAttr.get("attn_fused.ll")
        fn.attributes["link_with_mode"] = self.f.StringAttr.get("merge")
        return fn

    def query(self, slab):
        f = self.f
        for group in range(2):
            f.ChannelPut(
                f"pqcol{group}",
                slab,
                offsets=[0, 0, group * 16, 0],
                sizes=[f.BATCH, 16, 16, 8],
                strides=[4096, 8, 128, 1],
            )

    def gate(self, arm, body):
        f = self.f

        def idle():
            f.yield_([])

        def decode():
            body()
            f.yield_([])

        f.index_switch(
            [],
            f.arith.index_cast(f.IndexType.get(), arm),
            [0],
            case_body_builder=lambda op, i, cv: idle(),
            default_body_builder=lambda op: decode(),
        )

    def rounds(self, length):
        f = self.f
        return f.arith.index_cast(
            f.IndexType.get(),
            f.arith.divui(f.arith.addi(length, self.ci(f.BATCH + 14)), self.ci(16)),
        )

    def emit(self, length, arm):
        f = self.f
        ix = self.ix
        ci = self.ci
        for group in range(2):

            def memory(group=group):
                # L2 lowers to persistent lock-driven DMA rings. Express two
                # slots without a runtime L2 loop bound; only the shim producers
                # and core consumers count live blocks. Audit that every emitted
                # L2 BD loops indefinitely before executing this template.
                for _ in f.for_(ix(0), ix(2), ix(1)):
                    kb = f.AllocOp(self.mem(8192, self.l2), [], [])
                    vb = f.AllocOp(self.mem(8192, self.l2), [], [])
                    # Distribute the K/V relay buffers across memory tiles
                    # to fit the six MM2S channels per memory tile.
                    kb.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                        self.i32, (2) if group == 0 else 4
                    )
                    vb.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                        self.i32, 3 + group
                    )
                    for kind, buf in (("K", kb), ("V", vb)):
                        f.ChannelGet(
                            (f"inKV_{kind}"),
                            buf,
                            indices=[ix(group)],
                        )
                    f.ChannelPut(
                        f"pkcol{group}",
                        kb,
                        offsets=[0, 0, 0, 0],
                        sizes=[2, 32, 16, 8],
                        strides=[256, 8, 512, 1],
                    )
                    for local in range(2):
                        pv = f.ChannelPut(
                            f"pv{2*group+local}",
                            vb,
                            indices=[ix(0), ix(0)],
                            offsets=[0, 0, 0, local * 256],
                            sizes=[2, 32, 8, 8],
                            strides=[4096, 8, 512, 1],
                        )
                    f.DeallocOp(kb)
                    f.DeallocOp(vb)
                    f.yield_([])
                for _ in f.for_(ix(0), ix(2), ix(1)):
                    stage = f.AllocOp(self.mem(8192, self.l2), [], [])
                    stage.operation.attributes["air.memtile_col"] = f.IntegerAttr.get(
                        self.i32, 3 + group
                    )
                    f.ChannelGet(f"pqkout{group}", stage)
                    for local in range(2):
                        put = f.ChannelPut(
                            f"pk{2*group+local}",
                            stage,
                            indices=[ix(0), ix(0)],
                            offsets=[local * 4096],
                            sizes=[4096],
                            strides=[1],
                        )
                    f.DeallocOp(stage)
                    f.yield_([])

            self.gate(arm, memory)

            def relay(tx, ty, sx, sy, L, A, group=group):
                def run():
                    qi, ki, stage = [
                        f.AllocOp(self.mem(8192), [], []) for _ in range(3)
                    ]
                    for buf in (qi, ki, stage):
                        buf.operation.attributes["air.disable_alloc_sink"] = (
                            f.UnitAttr.get()
                        )
                    frames = f.arith.addi(self.rounds(L), ix(2))
                    for frame in f.for_(
                        ix(0), f.arith.muli(frames, ix(f.BATCH // 8)), ix(1)
                    ):
                        local_frame = f.arith.remui(frame, frames)
                        branch = f.IfOp(
                            f.arith.cmpi(f.arith.CmpIPredicate.ult, local_frame, ix(2)),
                            [],
                            has_else=True,
                        )
                        with f.InsertionPoint(branch.thenRegion.blocks[0]):
                            f.ChannelGet(f"pqcol{group}", qi)
                            f.CallOp(self.relayq, [stage, qi])
                            f.yield_([])
                        with f.InsertionPoint(branch.elseRegion.blocks[0]):
                            f.ChannelGet(f"pkcol{group}", ki)
                            f.CallOp(self.relayk, [stage, ki])
                            f.yield_([])
                        f.ChannelPut(f"pqkout{group}", stage)
                        f.yield_([])
                    for buf in (stage, ki, qi):
                        f.DeallocOp(buf)

                self.gate(A, run)

            rh = f.herd(name=f"qk_relay{group}", sizes=[1, 1], operands=[length, arm])(
                relay
            )
            rh.attributes["x_loc"] = f.IntegerAttr.get(
                f.T.i64(), 2 if group == 0 else 5
            )
            rh.attributes["y_loc"] = f.IntegerAttr.get(f.T.i64(), 4)

        for cu in range(4):

            def shard(tx, ty, sx, sy, L, A, cu=cu):
                def run():
                    # Each wave owns its query/output slab until output DMA
                    # releases it; the next wave must not overwrite pending output.
                    for wave_id in f.for_(ix(0), ix(f.BATCH // 8), ix(1)):
                        types = [self.mem(n) for n in (4096, 4096, 64)] + [
                            self.mem(32, dtype=self.fp),
                            self.mem(768),
                            self.mem(4096, dtype=self.fp),
                            self.mem(64, dtype=self.fp),
                            self.mem(8192),
                        ]
                        k, v, m, c, s, y, l, o = [f.AllocOp(t, [], []) for t in types]
                        q = o  # Query storage is dead when the joined output is formed.
                        for buf in (q, k, v, m, c, s, y, l, o):
                            buf.operation.attributes["air.disable_alloc_sink"] = (
                                f.UnitAttr.get()
                            )
                        first = f.arith.muli(f.arith.index_cast(self.i32, ty), ci(4))
                        frames = f.arith.addi(self.rounds(L), ix(2))
                        for local_frame in f.for_(ix(0), frames, ix(1)):
                            wave = f.arith.index_cast(self.i32, wave_id)
                            delta = (
                                f.arith.addi(first, f.arith.muli(wave, ci(8)))
                                if f.CAUSAL
                                else ci(f.BATCH - 1)
                            )
                            mask_length = f.arith.addi(L, delta)

                            def consume_value(block):
                                f.ChannelGet(f"pv{cu}", v, indices=[ix(0), ty])
                                f.CallOp(
                                    self.pv,
                                    [
                                        s,
                                        v,
                                        y,
                                        l,
                                        block,
                                        mask_length,
                                        ci(int(f.CAUSAL)),
                                    ],
                                )

                            f.ChannelGet(f"pk{cu}", k, indices=[ix(0), ty])
                            branch = f.IfOp(
                                f.arith.cmpi(
                                    f.arith.CmpIPredicate.ult, local_frame, ix(2)
                                ),
                                [],
                                has_else=True,
                            )
                            with f.InsertionPoint(branch.thenRegion.blocks[0]):
                                keep = f.IfOp(
                                    f.arith.cmpi(
                                        f.arith.CmpIPredicate.eq, local_frame, ty
                                    ),
                                    [],
                                    has_else=False,
                                )
                                with f.InsertionPoint(keep.thenRegion.blocks[0]):
                                    f.CallOp(self.copy, [q, k])
                                    f.yield_([])
                                f.yield_([])
                            with f.InsertionPoint(branch.elseRegion.blocks[0]):
                                block = f.arith.index_cast(
                                    self.i32, f.arith.subi(local_frame, ix(2))
                                )
                                f.CallOp(
                                    self.qk,
                                    [
                                        q,
                                        k,
                                        m,
                                        c,
                                        s,
                                        block,
                                        mask_length,
                                        ci(0),
                                        ci(int(f.CAUSAL)),
                                    ],
                                )
                                if not f.EARLY_KEY_RELEASE:
                                    consume_value(block)
                                f.yield_([])
                            if f.EARLY_KEY_RELEASE:
                                # Finish K's last-use region before acquiring V.
                                consume = f.IfOp(
                                    f.arith.cmpi(
                                        f.arith.CmpIPredicate.uge, local_frame, ix(2)
                                    ),
                                    [],
                                    has_else=False,
                                )
                                with f.InsertionPoint(consume.thenRegion.blocks[0]):
                                    block = f.arith.index_cast(
                                        self.i32, f.arith.subi(local_frame, ix(2))
                                    )
                                    consume_value(block)
                                    f.yield_([])
                            f.yield_([])
                        f.CallOp(self.fin, [y, l, o, first])
                        join = f.IfOp(
                            f.arith.cmpi(f.arith.CmpIPredicate.eq, ty, ix(1)),
                            [],
                            has_else=True,
                        )
                        with f.InsertionPoint(join.thenRegion.blocks[0]):
                            for part in f.for_(ix(0), ix(8), ix(1)):
                                f.CallOp(
                                    self.joinsend,
                                    [s, o, f.arith.index_cast(self.i32, part)],
                                )
                                f.ChannelPut(
                                    "parallel_attention_cascade", s, indices=[ix(cu)]
                                )
                                f.yield_([])
                            f.yield_([])
                        with f.InsertionPoint(join.elseRegion.blocks[0]):
                            for part in f.for_(ix(0), ix(8), ix(1)):
                                f.ChannelGet(
                                    "parallel_attention_cascade", s, indices=[ix(cu)]
                                )
                                f.CallOp(
                                    self.joinrecv,
                                    [o, s, f.arith.index_cast(self.i32, part)],
                                )
                                f.yield_([])
                            f.ChannelPut(
                                "parallel_attention_output", o, indices=[ix(cu)]
                            )
                            f.yield_([])
                        for buf in (o, l, y, s, c, m, v, k):
                            f.DeallocOp(buf)
                        f.yield_([])

                self.gate(A, run)

            core = f.herd(
                name=f"attention_shards{cu}", sizes=[1, 2], operands=[length, arm]
            )(shard)
            core.attributes["x_loc"] = f.IntegerAttr.get(f.T.i64(), 3 + cu // 2)
            core.attributes["y_loc"] = f.IntegerAttr.get(f.T.i64(), 2 + 2 * (cu % 2))

    def gather(self, slab):
        f = self.f
        for cu in range(4):
            f.ChannelGet(
                "parallel_attention_output",
                slab,
                indices=[self.ix(cu)],
                offsets=[self.ix(0), self.ix(0), self.ix(0), self.ix(cu * 1024)],
                sizes=[self.ix(f.BATCH), self.ix(16), self.ix(8), self.ix(8)],
                strides=[self.ix(4096), self.ix(8), self.ix(128), self.ix(1)],
            )
