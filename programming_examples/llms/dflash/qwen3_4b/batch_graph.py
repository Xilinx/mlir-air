# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-4B B8 graph: cached normalization and fused attention shards.

The layout argument supplies shared model geometry and AIR operations.
The persistent projection weight path comes from fused_decode. DFlash adds
batched projections, per-row positions/visibility, layer-boundary export and
parallel attention, while keeping projection weights quantized in DDR.
"""


def build_module(layout):

    @layout.module_builder
    def build():
        bf16 = layout.BF16Type.get()
        f32 = layout.F32Type.get()
        i32 = layout.IntegerType.get_signless(32)
        idx_t = layout.IndexType.get()
        l1 = layout.IntegerAttr.get(layout.T.i32(), layout.MemorySpace.L1)
        l2 = layout.IntegerAttr.get(layout.T.i32(), layout.MemorySpace.L2)
        x_l3 = layout.MemRefType.get([layout.X_SLOTS * layout.BATCH * layout.K], bf16)
        _w_blocks = (
            layout.UNI_DEC * layout.W_TOTAL_BLOCKS
            + layout.UNI_LM * layout.VOCAB_W_BLOCKS
        )
        w_l3 = layout.MemRefType.get([_w_blocks * layout.BLOCK_BF16], bf16)
        rms_l3 = layout.MemRefType.get(
            [
                layout.UNI_DEC * layout.RMS_LAYER
                + layout.UNI_DEC * layout.ROPE_W_LEN * layout.BATCH
                + layout.K
            ],
            bf16,
        )
        _y_elems = (
            (layout.HOST_ROUNDS + layout.LAYER_RNDS) * layout.PAYLOAD * layout.BATCH
            + layout.UNI_LM * layout.VOCAB_SIZE_PADDED * layout.BATCH
        )
        y_l3 = layout.MemRefType.get([_y_elems], bf16)
        # This graph uses a single KV buffer and does not implement the decode
        # engine's KV split, so it cannot go past 4 GiB (see kv_layout).
        if layout.KV_BO_SPLIT:
            raise SystemExit(
                f"KV capacity {layout.ATTN_MAXL} puts the cache past the 4 GiB one "
                f"buffer is addressable over, and the batch graph has no KV split"
            )
        kvc_l3 = layout.MemRefType.get(
            [layout.UNI_DEC * layout.ATTN_MAXL * layout.KVSZ_TOK], bf16
        )
        wblk_l1 = layout.MemRefType.get([layout.BLOCK_BF16], bf16, memory_space=l1)
        xblk_mm_l1 = layout.MemRefType.get(
            [layout.BATCH * layout.COL_BLOCK], bf16, memory_space=l1
        )
        yacc_mm_l1 = layout.MemRefType.get(
            [layout.BATCH * layout.ROW_BLOCK], f32, memory_space=l1
        )
        wscr_mm_l1 = layout.MemRefType.get(
            [layout.ROW_BLOCK * layout.COL_BLOCK], bf16, memory_space=l1
        )
        _YPAIR = 16 + layout.PAIR_ROWS * layout.ROW_BLOCK * layout.PACKET_BATCH
        ypair_mm_l1 = layout.MemRefType.get(
            [_YPAIR + -_YPAIR % layout.L1_VEC_BF16], bf16, memory_space=l1
        )
        rms_l1 = layout.MemRefType.get([layout.K], bf16, memory_space=l1)
        rmsb_l1 = layout.MemRefType.get(
            [layout.BATCH * layout.K], bf16, memory_space=l1
        )
        rstg_l1 = layout.MemRefType.get(
            [layout.BATCH * layout.STG_W], bf16, memory_space=l1
        )
        rscl_l1 = layout.MemRefType.get([layout.BATCH], layout.T.f32(), memory_space=l1)
        glu_x_l1 = layout.MemRefType.get(
            [layout.BATCH * layout.GLU_SLICE], bf16, memory_space=l1
        )
        glu_hid_l1 = layout.MemRefType.get(
            [layout.BATCH * layout.GLU_HID], bf16, memory_space=l1
        )
        qkv_l1 = layout.MemRefType.get([layout.M], bf16, memory_space=l1)
        ropeq_l1 = layout.MemRefType.get([layout.DQ_PADDED], bf16, memory_space=l1)
        ropekv_l1 = layout.MemRefType.get([layout.DK], bf16, memory_space=l1)
        ropelut_l1 = layout.MemRefType.get([layout.ROPE_W_LEN], bf16, memory_space=l1)
        qmt_l2 = layout.MemRefType.get(
            [layout.BATCH * layout.DQ_PADDED], bf16, memory_space=l2
        )
        omt_l2 = layout.MemRefType.get(
            [layout.BATCH * layout.DQ], bf16, memory_space=l2
        )
        qkvmt_l2 = layout.MemRefType.get(
            [layout.BATCH * layout.M], bf16, memory_space=l2
        )
        xmt_l2 = layout.MemRefType.get(
            [layout.BATCH * layout.XCHUNK], bf16, memory_space=l2
        )
        wfan_l2 = layout.MemRefType.get(
            [layout.NCY // 2 * layout.BLOCK_BF16], bf16, memory_space=l2
        )
        grp_l2 = layout.MemRefType.get([layout.GRP_ROWS_B], bf16, memory_space=l2)
        main_l2 = layout.MemRefType.get([layout.MAIN_ROWS_B], bf16, memory_space=l2)
        relay_l2 = layout.MemRefType.get([layout.PAYLOAD_B], bf16, memory_space=l2)
        down_l2 = layout.MemRefType.get(
            [layout.BATCH * layout.GLU_OUT], bf16, memory_space=l2
        )
        RELAY_COLS = [3, 5, 4][: layout.NDEST]
        mm_zero = layout.FuncOp(
            "proj_qmm_mm_zero", ([yacc_mm_l1, i32], []), visibility="private"
        )
        mm_zero.attributes["link_with"] = layout.StringAttr.get("proj_qmm.o")
        mm_acc = layout.FuncOp(
            "proj_qmm_mm_acc",
            ([xblk_mm_l1, wblk_l1, yacc_mm_l1, wscr_mm_l1], []),
            visibility="private",
        )
        mm_acc.attributes["link_with"] = layout.StringAttr.get("proj_qmm.o")
        mm_flush = layout.FuncOp(
            "proj_qmm_mm_flush_token",
            ([yacc_mm_l1, ypair_mm_l1, i32, i32], []),
            visibility="private",
        )
        mm_flush.attributes["link_with"] = layout.StringAttr.get("proj_qmm.o")
        rms_scale_row_aie = layout.FuncOp(
            "rms_scale_row_aie",
            ([rscl_l1, rmsb_l1, i32, i32], []),
            visibility="private",
        )
        rms_scale_row_aie.attributes["link_with"] = layout.StringAttr.get(
            "rms_residual.o"
        )
        rms_chunk_aie = layout.FuncOp(
            "rms_chunk_aie",
            ([rstg_l1, rmsb_l1, rms_l1, rscl_l1, i32, i32, i32], []),
            visibility="private",
        )
        rms_chunk_aie.attributes["link_with"] = layout.StringAttr.get("rms_residual.o")
        residual_acc_row_aie = layout.FuncOp(
            "residual_acc_row_aie",
            ([rmsb_l1, rstg_l1, i32, i32, i32], []),
            visibility="private",
        )
        residual_acc_row_aie.attributes["link_with"] = layout.StringAttr.get(
            "rms_residual.o"
        )
        glu_row_aie = layout.FuncOp(
            "glu_row_aie", ([glu_hid_l1, glu_x_l1, i32, i32], []), visibility="private"
        )
        glu_row_aie.attributes["link_with"] = layout.StringAttr.get("glu.o")
        glu_copy_aie = layout.FuncOp(
            "glu_copy_aie",
            ([glu_hid_l1, glu_x_l1, i32, i32, i32], []),
            visibility="private",
        )
        glu_copy_aie.attributes["link_with"] = layout.StringAttr.get("glu.o")
        rope_compute = layout.FuncOp(
            "rope_compute",
            ([ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropelut_l1, i32], []),
            visibility="private",
        )
        rope_compute.attributes["link_with"] = layout.StringAttr.get("rope.o")
        parallel_attention = layout.ParallelAttentionGraph(layout)
        assert layout.BATCH == 8 and layout.XCHUNK == 512
        cache_copy = layout.FuncOp(
            "normalized_cache_chunk",
            ([rstg_l1, rmsb_l1, i32], []),
            visibility="private",
        )
        cache_copy.attributes["link_with"] = layout.StringAttr.get("norm_cache.o")
        layout.channel_decl("rmsX", size=[1], channel_type="npu_dma_packet")
        layout.channel_decl("rmsW", size=[1])
        layout.channel_decl("rmsW2", size=[1])
        # xnorm_fill carries five 8-KiB chunks into the cache core's 40-KiB
        # slab over a circuit stream. xnorm replays cached chunks as packets
        # through the activation MemTile to the projection cores.
        layout.channel_decl("xnorm", size=[1], channel_type="npu_dma_packet")
        _fill = layout.channel_decl("xnorm_fill", size=[1])
        _fill.operation.attributes["air.tile_dma_channel"] = layout.IntegerAttr.get(
            i32, 0
        )
        _XCHUNK = layout.XCHUNK

        def _xnorm_put(buf, width, **kw):
            assert (
                width % _XCHUNK == 0
            ), f"@xnorm producer width {width} is not a multiple of {_XCHUNK}; the X memtile stages whole chunks"
            return layout.ChannelPut(
                "xnorm",
                buf,
                offsets=[idx(0), idx(0), idx(0)],
                sizes=[idx(width // _XCHUNK), idx(layout.BATCH), idx(_XCHUNK)],
                strides=[idx(_XCHUNK), idx(width), idx(1)],
                **kw,
            )

        _inX = layout.Channel(
            "inX", size=[1, 1], broadcast_shape=[layout.NCX, layout.NCY]
        )
        _inX.operation.attributes["air.shared_resident_ring"] = layout.UnitAttr.get()
        layout.channel_decl("ropeLUT", size=[1])
        layout.channel_decl("toRope", size=[1])
        layout.channel_decl("ropeQ", size=[1])
        layout.channel_decl("appendK", size=[1], channel_type="npu_dma_packet")
        layout.channel_decl("appendV", size=[1], channel_type="npu_dma_packet")
        for kind in ("K", "V"):
            layout.channel_decl(f"inKV_{kind}", size=[len(layout.ATTN_COL_GROUPS)])
        for _wc in range(layout.NCX):
            for _nm in (layout._wname(0, _wc), layout._wname(1, _wc)):
                layout.channel_decl(_nm, size=[1])
        _wL2 = layout.channel_decl("wL2ToL1", size=[layout.NCX, layout.NCY])
        _wL2.operation.attributes["air.shared_resident_ring"] = layout.UnitAttr.get()
        _gather_type = {"channel_type": "npu_dma_packet"}
        layout.channel_decl("outA", size=[layout.NCX, layout.PAIRS_PC], **_gather_type)
        _main_type = {"channel_type": "npu_dma_packet"}
        layout.channel_decl("toMain", size=[layout.N_GRP], **_main_type)
        _outY = layout.Channel("outY", size=[1, 1], broadcast_shape=[1, layout.NDEST])
        _outY.operation.attributes["channel_type"] = layout.StringAttr.get(
            "npu_dma_packet"
        )
        _outY.operation.attributes["air.tile_dma_channel"] = layout.IntegerAttr.get(
            layout.T.i32(), 1
        )
        _layerOut = layout.channel_decl("layerOut", size=[1])
        _layerOut.operation.attributes["air.tile_dma_channel"] = layout.IntegerAttr.get(
            i32, 1
        )
        layout.channel_decl("gluOut", size=[1])
        layout.channel_decl("vocOut", size=[1])

        def idx(v):
            return layout.arith.ConstantOp.create_index(v)

        def grp_leads(g):
            _cpg = layout.NCX // layout.N_GRP
            out = []
            for lc in range(_cpg):
                for pp in range(layout.PAIRS_PC):
                    out.append((g * _cpg + lc, pp))
            return out

        _fn_args = [x_l3, w_l3, rms_l3, y_l3, kvc_l3, i32]

        @layout.FuncOp.from_py_func(*_fn_args)
        def q4nx_decode(*_fa):

            def launch_body(*_la):
                X, W, RMS, Y = (_la[4], _la[5], _la[6], _la[7])
                KVC = _la[8]
                L_rt = _la[4 + len(_fa) - 1]

                def _live_rounds(length):
                    # L is the anchor's one-based position. All queries share
                    # ceil((L + BATCH - 1)/16) transferred blocks; each still
                    # applies its own causal or bidirectional mask.
                    total = layout.arith.addi(
                        length,
                        layout.arith.ConstantOp(
                            layout.IntegerAttr.get(i32, layout.BATCH + 14), None
                        ).result,
                    )
                    rounds = layout.arith.divui(
                        total,
                        layout.arith.ConstantOp(
                            layout.IntegerAttr.get(i32, 16), None
                        ).result,
                    )
                    return layout.arith.index_cast(idx_t, rounds)

                a_iv = _la[-1] if len(_la) > 4 + len(_fa) else None

                def _lb(slab):
                    return 0 if a_iv is None else layout.arith.muli(a_iv, idx(slab))

                def _lo(base, extra):
                    if a_iv is None:
                        return base + extra
                    return layout.arith.addi(base, idx(extra)) if extra else base

                def _loi(base, extra):
                    if a_iv is None:
                        return idx(base + extra)
                    return layout.arith.addi(base, idx(extra)) if extra else base

                def _loi_slot(base, extra):
                    position = layout.arith.subi(
                        L_rt,
                        layout.arith.ConstantOp(
                            layout.IntegerAttr.get(i32, 1), None
                        ).result,
                    )
                    slot = layout.arith.muli(
                        position,
                        layout.arith.ConstantOp(
                            layout.IntegerAttr.get(i32, layout.REGION_W), None
                        ).result,
                    )
                    offset = layout.arith.addi(
                        base if a_iv is not None else idx(base),
                        layout.arith.index_cast(idx_t, slot),
                    )
                    return layout.arith.addi(offset, idx(extra)) if extra else offset

                _X_SLOT = layout.BATCH * layout.K

                def _x_in():
                    return _lb(_X_SLOT)

                def _x_out():
                    return _lo(_lb(_X_SLOT), _X_SLOT)

                _x_final = layout.UNI_DEC * _X_SLOT
                blk = layout.BLOCK_BF16
                wstep = layout.NCY * blk

                def _feed_wcol(_cx, _cbase, nsteps, Wb):
                    nch = 2
                    cstep = wstep // nch
                    span = nsteps * cstep
                    half = nsteps // 2 * cstep
                    for ci in range(nch):
                        base = (
                            _cbase if ci == 0 else layout.arith.addi(_cbase, idx(span))
                        )
                        ch = layout._wname(ci, _cx)
                        ix = [idx(0)]
                        layout.ChannelPut(
                            ch,
                            Wb,
                            indices=ix,
                            offsets=[base],
                            sizes=[half],
                            strides=[1],
                        )
                        layout.ChannelPut(
                            ch,
                            Wb,
                            indices=ix,
                            offsets=[layout.arith.addi(base, idx(half))],
                            sizes=[span - half],
                            strides=[1],
                        )

                def _feed_wcols(_wcol0, _colspan, nsteps):

                    def _fan(Wb):
                        for _cxi in range(layout.NCX):
                            _cbase = (
                                _wcol0 + _cxi * _colspan
                                if isinstance(_wcol0, int)
                                else layout.arith.addi(_wcol0, idx(_cxi * _colspan))
                            )
                            if isinstance(_cbase, int):
                                _cbase = idx(_cbase)
                            _feed_wcol(_cxi, _cbase, nsteps, Wb)
                        return

                    _fan(W)

                for _layer in range(layout.NLAYERS if a_iv is None else 1):
                    _wbase = _lb(layout.W_LAYER)
                    _rbase = _lb(layout.RMS_LAYER)
                    _kbase = _lb(layout.KV_LAYER)
                    _ybase = _lb(layout.Y_LAYER)
                    _u1 = layout.arith.ConstantOp(
                        layout.IntegerAttr.get(i32, 1), None
                    ).result
                    _u0 = layout.arith.ConstantOp(
                        layout.IntegerAttr.get(i32, 0), None
                    ).result
                    if a_iv is None:
                        _uarm = _u1
                    else:
                        _ucmp = layout.arith.cmpi(
                            layout.arith.CmpIPredicate.slt, a_iv, idx(layout.UNI_DEC)
                        )
                        _uarm = layout.arith.select(_ucmp, _u1, _u0)
                    _uarm_i = layout.arith.index_cast(idx_t, _uarm)
                    _final_norm_off = (
                        layout.UNI_DEC * layout.RMS_LAYER
                        + layout.UNI_DEC * layout.ROPE_W_LEN * layout.BATCH
                    )

                    def _uni_voc():
                        _vwb = layout.arith.addi(
                            idx(layout.UNI_DEC * layout.W_LAYER),
                            layout.arith.muli(
                                layout.arith.subi(a_iv, idx(layout.UNI_DEC)),
                                idx(layout.VOCAB_W_BLOCKS * layout.BLOCK_BF16),
                            ),
                        )
                        _vyb = layout.arith.addi(
                            idx(
                                (layout.HOST_ROUNDS + layout.LAYER_RNDS)
                                * layout.PAYLOAD
                                * layout.BATCH
                            ),
                            layout.arith.muli(
                                layout.arith.subi(a_iv, idx(layout.UNI_DEC)),
                                idx(layout.VOCAB_SIZE_PADDED * layout.BATCH),
                            ),
                        )
                        layout.ChannelPut(
                            "rmsX",
                            X,
                            offsets=[_x_final],
                            sizes=[layout.BATCH * layout.K],
                            strides=[1],
                        )
                        layout.ChannelPut(
                            "rmsW",
                            RMS,
                            offsets=[_final_norm_off],
                            sizes=[layout.K],
                            strides=[1],
                        )
                        layout.ChannelPut(
                            "rmsW2", RMS, offsets=[0], sizes=[layout.K], strides=[1]
                        )
                        assert layout.VOCAB_PER_COL % layout.NCY == 0
                        _colspan = layout.VOCAB_PER_COL * blk
                        _feed_wcols(_vwb, _colspan, layout.VOCAB_PER_COL // layout.NCY)
                        layout.ChannelGet(
                            "vocOut",
                            Y,
                            indices=[idx(0)],
                            offsets=[_vyb],
                            sizes=[
                                layout.VOCAB_RNDS // layout.VOC_SLICE_RNDS,
                                layout.BATCH,
                                layout.GLU_SLICE,
                            ],
                            strides=[layout.GLU_SLICE, layout.VOCAB_SIZE_PADDED, 1],
                        )
                        layout.yield_([])

                    def _uni_dec():
                        layout.ChannelPut(
                            "rmsX",
                            X,
                            offsets=[_x_in()],
                            sizes=[layout.BATCH * layout.K],
                            strides=[1],
                        )
                        layout.ChannelPut(
                            "rmsW", RMS, offsets=[_rbase], sizes=[layout.K], strides=[1]
                        )
                        layout.ChannelPut(
                            "rmsW2",
                            RMS,
                            offsets=[_lo(_rbase, layout.K)],
                            sizes=[layout.K],
                            strides=[1],
                        )
                        _lut_off = layout.UNI_DEC * layout.RMS_LAYER
                        _rope_off = _lo(_lb(layout.BATCH * layout.ROPE_W_LEN), _lut_off)
                        # One RoPE LUT row per token, in token order.
                        _tl_range = range(layout.BATCH)
                        for _tl in _tl_range:
                            _lo_t = _tl * layout.ROPE_W_LEN
                            if not _lo_t:
                                _off_t = _rope_off
                            elif isinstance(_rope_off, int):
                                _off_t = _rope_off + _lo_t
                            else:
                                _off_t = layout.arith.addi(_rope_off, idx(_lo_t))
                            layout.ChannelPut(
                                "ropeLUT",
                                RMS,
                                offsets=[_off_t],
                                sizes=[layout.ROPE_W_LEN * (1)],
                                strides=[1],
                            )

                        def _emit_append(_kbase=_kbase):
                            _bt = [idx(layout.BATCH)]
                            _bs = [idx(layout.REGION_W)]
                            _bo = [idx(0), idx(0)]
                            layout.ChannelGet(
                                "appendK",
                                KVC,
                                indices=[idx(0)],
                                offsets=_bo + [_loi_slot(_kbase, 0)],
                                sizes=_bt + [idx(layout.NGRP), idx(layout.REGION_W)],
                                strides=_bs + [idx(layout.REGION_STRIDE), idx(1)],
                            )
                            layout.ChannelGet(
                                "appendV",
                                KVC,
                                indices=[idx(0)],
                                offsets=_bo + [_loi_slot(_kbase, layout._vreg_off(0))],
                                sizes=_bt + [idx(layout.NGRP), idx(layout.REGION_W)],
                                strides=_bs + [idx(layout.REGION_STRIDE), idx(1)],
                            )
                            return

                        def _emit_readback(_kbase=_kbase):
                            _emit_readback_one(_kbase)

                        def _emit_readback_one(_kbase=_kbase):
                            _NRB = 1
                            _nb = layout.RB_ROUNDS
                            _cbk = (_nb + _NRB - 1) // _NRB
                            _ci = 0
                            while _ci < _nb:
                                _cb = min(_cbk, _nb - _ci)
                                _coff = _ci * 16 * layout.REGION_W
                                _cbv = lambda: _live_rounds(L_rt)
                                # AIR lowers a runtime-length linear transfer
                                # or a runtime repeat, but not both in one
                                # transfer. Emit the eight reads one by one and
                                # keep one contiguous live span.
                                _rb = [idx(1)]
                                _r0 = [idx(0)]
                                for _query in range(layout.KV_QUERY_GROUPS):
                                    for gi in range(layout.NGRP):
                                        layout.ChannelPut(
                                            "inKV_K",
                                            KVC,
                                            indices=[idx(gi)],
                                            offsets=_r0
                                            + [
                                                _loi(
                                                    _kbase, layout._kreg_off(gi) + _coff
                                                )
                                            ],
                                            sizes=(
                                                _rb
                                                + [
                                                    _cbv(),
                                                    idx(16),
                                                    idx(layout.REGION_W),
                                                ]
                                            ),
                                            strides=(
                                                _r0
                                                + [
                                                    idx(16 * layout.REGION_W),
                                                    idx(layout.REGION_W),
                                                    idx(1),
                                                ]
                                            ),
                                        )
                                        layout.ChannelPut(
                                            "inKV_V",
                                            KVC,
                                            indices=[idx(gi)],
                                            offsets=_r0
                                            + [
                                                _loi(
                                                    _kbase, layout._vreg_off(gi) + _coff
                                                )
                                            ],
                                            sizes=(
                                                _rb
                                                + [
                                                    _cbv(),
                                                    idx(16),
                                                    idx(layout.REGION_W),
                                                ]
                                            ),
                                            strides=(
                                                _r0
                                                + [
                                                    idx(16 * layout.REGION_W),
                                                    idx(layout.REGION_W),
                                                    idx(1),
                                                ]
                                            ),
                                        )
                                _ci += _cb
                            return

                        woff = 0
                        for p in range(layout.NPH):
                            per_col = layout.PER_COL_PH[p]
                            assert per_col % layout.NCY == 0
                            _colspan = per_col * blk
                            _wcol0 = _lo(_wbase, woff)
                            _feed_wcols(_wcol0, _colspan, per_col // layout.NCY)
                            woff += layout.NCX * per_col * blk
                            if layout.MULTIBLK and p == 0:
                                _emit_append()
                                _emit_readback()
                        roff = 0
                        for p in layout.HOST_DRAIN:
                            if p == 0:
                                pass
                            else:
                                for rr in range(layout.ROUNDS_PER_DEST[p]):
                                    layout.ChannelGet(
                                        "toShim",
                                        Y,
                                        indices=[idx(p)],
                                        offsets=[
                                            _lo(_ybase, (roff + rr) * layout.PAYLOAD)
                                        ],
                                        sizes=[layout.PAYLOAD],
                                        strides=[1],
                                    )
                            roff += layout.ROUNDS_PER_DEST[p]
                        _out_bo = X
                        _out_base = _x_out()
                        layout.ChannelGet(
                            "layerOut",
                            _out_bo,
                            indices=[idx(0)],
                            offsets=[_out_base],
                            sizes=[layout.BATCH * layout.LAYER_RNDS * layout.PAYLOAD],
                            strides=[1],
                        )
                        layout.yield_([])

                    layout.index_switch(
                        [],
                        _uarm_i,
                        [0],
                        case_body_builder=lambda op, i, cv: _uni_voc(),
                        default_body_builder=lambda op: _uni_dec(),
                    )
                _seg_opers = ([a_iv] if a_iv is not None else []) + ([L_rt])

                @layout.segment(name="seg", operands=_seg_opers)
                def seg(*_sa):
                    _seg_iv = _sa[0] if a_iv is not None else None
                    _seg_L = _sa[-1]

                    def _seg_blocks():
                        r = _seg_rounds()
                        return layout.arith.muli(
                            r, idx(layout.BATCH // layout.ATTENTION_QUERY_TILE)
                        )

                    def _seg_rounds():
                        return _live_rounds(_seg_L)

                    if _seg_iv is not None:
                        _seg_cmp = layout.arith.cmpi(
                            layout.arith.CmpIPredicate.slt, _seg_iv, idx(layout.UNI_DEC)
                        )
                        _seg_arm = layout.arith.select(
                            _seg_cmp,
                            layout.arith.ConstantOp(
                                layout.IntegerAttr.get(i32, 1), None
                            ).result,
                            layout.arith.ConstantOp(
                                layout.IntegerAttr.get(i32, 0), None
                            ).result,
                        )
                        _seg_arm_i = layout.arith.index_cast(idx_t, _seg_arm)
                    else:
                        _seg_cmp = None
                        _seg_arm_i = None
                        _seg_arm = layout.arith.ConstantOp(
                            layout.IntegerAttr.get(i32, 1), None
                        ).result

                    def _feed_inX(src, total_chunks):
                        for _rc in layout.for_(idx(0), idx(total_chunks), idx(1)):
                            xb = layout.AllocOp(xmt_l2, [], [])
                            xb.operation.attributes["air.memtile_col"] = (
                                layout.IntegerAttr.get(layout.T.i32(), layout.XMT_PCOL)
                            )
                            layout.ChannelGet(
                                src,
                                xb,
                                offsets=[0],
                                sizes=[layout.BATCH * layout.XCHUNK],
                                strides=[1],
                            )
                            for _jj in layout.for_(
                                idx(0), idx(layout.XCHUNK_MUL), idx(1)
                            ):
                                _sz, _st = layout._XFEED_BD
                                _o0 = layout.arith.muli(
                                    _jj, idx(layout.COL_BLOCK // _st[0])
                                )
                                layout.ChannelPut(
                                    "inX",
                                    xb,
                                    offsets=[_o0] + [idx(0)] * (len(_sz) - 1),
                                    sizes=[idx(v) for v in _sz],
                                    strides=[idx(v) for v in _st],
                                )
                                layout.yield_([])
                            layout.DeallocOp(xb)
                            layout.yield_([])

                    # Projection inputs use K, DQ, K and GLU_OUT respectively.
                    # Derive the total from their packed block counts.
                    assert layout.N_XBLK % layout.XCHUNK_MUL == 0
                    _xc_dec = layout.N_XBLK // layout.XCHUNK_MUL
                    _xc_voc = layout.VOCAB_RNDS * (layout.K // layout.XCHUNK)
                    if _seg_arm_i is not None:

                        def _xs_voc():
                            _feed_inX("xnorm", _xc_voc)
                            layout.yield_([])

                        def _xs_dec():
                            _feed_inX("xnorm", _xc_dec)
                            layout.yield_([])

                        layout.index_switch(
                            [],
                            _seg_arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _xs_voc(),
                            default_body_builder=lambda op: _xs_dec(),
                        )
                    else:
                        _feed_inX("xnorm", _xc_dec)
                    _fan_groups = [
                        (0, 0, layout.NCY // 2),
                        (1, layout.NCY // 2, layout.NCY),
                    ]
                    for cx in range(layout.NCX):
                        for _ci, _cy0, _cy1 in _fan_groups:
                            _wch = layout._wname(_ci, cx)
                            for _ in layout.for_(
                                idx(0), idx(layout.W_FAN_STEPS), idx(1)
                            ):
                                wf = layout.AllocOp(wfan_l2, [], [])
                                wf.operation.attributes["air.memtile_col"] = (
                                    layout.IntegerAttr.get(
                                        layout.T.i32(), layout.PCOL[cx]
                                    )
                                )
                                layout.ChannelGet(_wch, wf, indices=[idx(0)])
                                for cy in range(_cy0, _cy1):
                                    layout.ChannelPut(
                                        "wL2ToL1",
                                        wf,
                                        indices=[idx(cx), idx(cy)],
                                        offsets=[(cy - _cy0) * layout.BLOCK_BF16],
                                        sizes=[layout.BLOCK_BF16],
                                        strides=[1],
                                    )
                                layout.DeallocOp(wf)
                                layout.yield_([])

                    def _egress(_nrc):
                        for _r in layout.for_(
                            idx(0), idx(_nrc * (layout.BATCH)), idx(1)
                        ):
                            for g in range(layout.N_GRP):
                                grp = layout.AllocOp(grp_l2, [], [])
                                grp.operation.attributes["air.memtile_col"] = (
                                    layout.IntegerAttr.get(
                                        layout.T.i32(), layout.GRP_PCOL[g]
                                    )
                                )
                                for k, (cx, pp) in enumerate(grp_leads(g)):
                                    _pay = layout.PAIR_PAY * layout.PACKET_BATCH
                                    off = 0 if k == 0 else layout.HDR + k * _pay
                                    sz = layout.HDR + _pay if k == 0 else _pay
                                    layout.ChannelGet(
                                        "outA",
                                        grp,
                                        indices=[idx(cx), idx(pp)],
                                        offsets=[off],
                                        sizes=[sz],
                                        strides=[1],
                                    )
                                _skip = layout.HDR if layout.CIRCUIT_MAIN and g else 0
                                layout.ChannelPut(
                                    "toMain",
                                    grp,
                                    indices=[idx(g)],
                                    offsets=[_skip],
                                    sizes=[layout.GRP_ROWS_B - _skip],
                                    strides=[1],
                                )
                                layout.DeallocOp(grp)
                            ml = layout.AllocOp(main_l2, [], [])
                            ml.operation.attributes["air.memtile_col"] = (
                                layout.IntegerAttr.get(layout.T.i32(), layout.MAIN_PCOL)
                            )
                            for g in range(layout.N_GRP):
                                _gpay = (
                                    layout.LEADS_PER_GRP
                                    * layout.PAIR_PAY
                                    * layout.PACKET_BATCH
                                )
                                off = (
                                    0 if g == 0 else layout.GRP_ROWS_B + (g - 1) * _gpay
                                )
                                sz = layout.GRP_ROWS_B if g == 0 else _gpay
                                layout.ChannelGet(
                                    "toMain",
                                    ml,
                                    indices=[idx(g)],
                                    offsets=[off],
                                    sizes=[sz],
                                    strides=[1],
                                )
                            layout.ChannelPut(
                                "outY",
                                ml,
                                indices=[idx(0), idx(0)],
                                offsets=[0],
                                sizes=[layout.MAIN_ROWS_B],
                                strides=[1],
                            )
                            layout.DeallocOp(ml)
                            layout.yield_([])

                    if _seg_arm_i is not None:

                        def _egr_voc():
                            _egress(layout.VOCAB_RNDS)
                            layout.yield_([])

                        def _egr_dec():
                            _egress(layout.N_ROUNDS)
                            layout.yield_([])

                        layout.index_switch(
                            [],
                            _seg_arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _egr_voc(),
                            default_body_builder=lambda op: _egr_dec(),
                        )
                    else:
                        _egress(layout.N_ROUNDS)

                    def _qkv_transpose():
                        qmt = layout.AllocOp(qkvmt_l2, [], [])
                        qmt.operation.attributes["air.memtile_col"] = (
                            layout.IntegerAttr.get(layout.T.i32(), RELAY_COLS[0])
                        )
                        _qo, _qs, _qt = layout.outy_tokmajor(
                            layout.M, rounds=layout.ROUNDS_PER_DEST[0]
                        )
                        layout.ChannelGet(
                            "outY",
                            qmt,
                            indices=[idx(0), idx(0)],
                            offsets=[idx(v) for v in _qo],
                            sizes=[idx(v) for v in _qs],
                            strides=[idx(v) for v in _qt],
                        )
                        _pt_all = [(0, layout.BATCH * layout.M)]
                        for _off, _len in _pt_all:
                            _pt = layout.ChannelPut(
                                "toRope",
                                qmt,
                                offsets=[idx(_off)],
                                sizes=[idx(_len)],
                                strides=[idx(1)],
                            )
                        layout.DeallocOp(qmt)

                    if _seg_arm_i is not None:

                        def _qt_dec():
                            _qkv_transpose()
                            layout.yield_([])

                        layout.index_switch(
                            [],
                            _seg_arm_i,
                            layout.ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: layout.yield_([]),
                            default_body_builder=lambda op: _qt_dec(),
                        )
                    else:
                        _qkv_transpose()
                    for p in layout.HOST_DRAIN:
                        if p == 0:
                            continue
                        for _rp in layout.for_(
                            idx(0), idx(layout.ROUNDS_PER_DEST[p]), idx(1)
                        ):
                            rb = layout.AllocOp(relay_l2, [], [])
                            rb.operation.attributes["air.memtile_col"] = (
                                layout.IntegerAttr.get(layout.T.i32(), RELAY_COLS[p])
                            )
                            layout.ChannelGet(
                                "outY",
                                rb,
                                indices=[idx(0), idx(p)],
                                offsets=[0],
                                sizes=[layout.PAYLOAD],
                                strides=[1],
                            )
                            layout.ChannelPut(
                                "toShim",
                                rb,
                                indices=[idx(p)],
                                offsets=[0],
                                sizes=[layout.PAYLOAD],
                                strides=[1],
                            )
                            layout.DeallocOp(rb)
                            layout.yield_([])
                    _arm_rope = _seg_arm

                    @layout.herd(name="rope", sizes=[1, 1], operands=[_arm_rope])
                    def rope_h(tx, ty, _sx, _sy, _arm):

                        def _dec_one():
                            a_qkv = layout.AllocOp(qkv_l1, [], [])
                            layout.ChannelGet(
                                "toRope",
                                a_qkv,
                                offsets=[idx(0)],
                                sizes=[idx(layout.M)],
                                strides=[idx(1)],
                            )
                            a_lut = layout.AllocOp(ropelut_l1, [], [])
                            layout.ChannelGet("ropeLUT", a_lut, indices=[idx(0)])
                            a_q = layout.AllocOp(ropeq_l1, [], [])
                            a_k = layout.AllocOp(ropekv_l1, [], [])
                            a_v = layout.AllocOp(ropekv_l1, [], [])
                            layout.CallOp(
                                rope_compute, [a_q, a_k, a_v, a_qkv, a_lut, _arm]
                            )
                            layout.ChannelPut(
                                "ropeQ",
                                a_q,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(layout.DQ_PADDED)],
                                strides=[idx(1)],
                            )
                            layout.ChannelPut(
                                "appendK",
                                a_k,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(layout.DK_TOT_A)],
                                strides=[idx(1)],
                            )
                            layout.ChannelPut(
                                "appendV",
                                a_v,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(layout.DK_TOT_A)],
                                strides=[idx(1)],
                            )
                            layout.DeallocOp(a_qkv)
                            layout.DeallocOp(a_lut)
                            layout.DeallocOp(a_q)
                            layout.DeallocOp(a_k)
                            layout.DeallocOp(a_v)

                        def _dec():
                            for _t in layout.for_(idx(0), idx(layout.BATCH), idx(1)):
                                _dec_one()
                                layout.yield_([])
                            layout.yield_([])

                        def _voc():
                            layout.yield_([])

                        layout.index_switch(
                            [],
                            layout.arith.index_cast(idx_t, _arm),
                            layout.ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: _voc(),
                            default_body_builder=lambda op: _dec(),
                        )

                    rope_h.attributes["link_with"] = layout.StringAttr.get("rope.o")
                    rope_h.attributes["x_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), layout.RMS_PCOL
                    )
                    rope_h.attributes["y_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), 3
                    )

                    def _qmtb_dec():
                        qmtb = layout.AllocOp(qmt_l2, [], [])
                        qmtb.operation.attributes["air.memtile_col"] = (
                            layout.IntegerAttr.get(layout.T.i32(), 5)
                        )
                        layout.ChannelGet(
                            "ropeQ",
                            qmtb,
                            indices=[idx(0)],
                            offsets=[idx(0)],
                            sizes=[idx(layout.BATCH * layout.DQ_PADDED)],
                            strides=[idx(1)],
                        )
                        parallel_attention.query(qmtb)
                        layout.DeallocOp(qmtb)

                    if _seg_arm_i is not None:

                        def _q_voc():
                            layout.yield_([])

                        def _q_dec():
                            _qmtb_dec()
                            layout.yield_([])

                        layout.index_switch(
                            [],
                            _seg_arm_i,
                            layout.ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: _q_voc(),
                            default_body_builder=lambda op: _q_dec(),
                        )
                    else:
                        _qmtb_dec()

                    _Lc = _seg_L
                    parallel_attention.emit(_Lc, _seg_arm)

                    def _omtb_dec():
                        omtb = layout.AllocOp(omt_l2, [], [])
                        omtb.operation.attributes["air.memtile_col"] = (
                            layout.IntegerAttr.get(layout.T.i32(), 5)
                        )

                        parallel_attention.gather(omtb)
                        layout.refeed(
                            layout.OPROJ_REFEED,
                            lambda: _xnorm_put(
                                omtb,
                                layout.N_ATTN_CU * layout.DQ_PER_CU,
                                indices=[idx(0)],
                            ),
                        )
                        layout.DeallocOp(omtb)

                    if _seg_arm_i is not None:

                        def _o_voc():
                            layout.yield_([])

                        def _o_dec():
                            _omtb_dec()
                            layout.yield_([])

                        layout.index_switch(
                            [],
                            _seg_arm_i,
                            layout.ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: _o_voc(),
                            default_body_builder=lambda op: _o_dec(),
                        )
                    else:
                        _omtb_dec()
                    _arm_glu = _seg_arm

                    @layout.herd(name="glu", sizes=[1, 1], operands=[_arm_glu])
                    def glu_h(tx, ty, _sx, _sy, _arm):
                        _voc_arm = layout.BATCH > 1 and layout.GLU_DEST >= 0

                        def _get_slice(gx):
                            assert layout.GLU_SLICE % layout.PAYLOAD == 0
                            _go, _gs, _gt = layout.outy_tokmajor(
                                layout.GLU_SLICE,
                                rounds=layout.GLU_SLICE // layout.PAYLOAD,
                            )
                            layout.ChannelGet(
                                "outY",
                                gx,
                                indices=[idx(0), idx(layout.GLU_DEST)],
                                offsets=[idx(v) for v in _go],
                                sizes=[idx(v) for v in _gs],
                                strides=[idx(v) for v in _gt],
                            )

                        def _glu_slice(gx, mk_gh):
                            _get_slice(gx)
                            gh = mk_gh()
                            for _t in range(layout.BATCH):
                                layout.CallOp(
                                    glu_row_aie,
                                    [
                                        gh,
                                        gx,
                                        layout.arith.ConstantOp(
                                            layout.IntegerAttr.get(i32, _t), None
                                        ).result,
                                        _arm,
                                    ],
                                )
                            layout.ChannelPut(
                                "gluOut",
                                gh,
                                offsets=[idx(0)],
                                sizes=[idx(layout.BATCH * layout.GLU_HID)],
                                strides=[idx(1)],
                            )
                            return gh

                        def _voc_slice(gx, mk_gh):
                            _get_slice(gx)
                            gh = mk_gh()
                            _half = layout.BATCH * layout.GLU_HID
                            for _h in range(2):
                                layout.CallOp(
                                    glu_copy_aie,
                                    [
                                        gh,
                                        gx,
                                        layout.arith.ConstantOp(
                                            layout.IntegerAttr.get(i32, _h * _half),
                                            None,
                                        ).result,
                                        layout.arith.ConstantOp(
                                            layout.IntegerAttr.get(i32, _half), None
                                        ).result,
                                        _arm,
                                    ],
                                )
                                layout.ChannelPut(
                                    "vocOut",
                                    gh,
                                    offsets=[idx(0)],
                                    sizes=[idx(_half)],
                                    strides=[idx(1)],
                                )
                            return gh

                        assert (
                            layout.NGLU % 2 == 0
                        ), f"odd NGLU={layout.NGLU} desyncs the GLU BD ring (GLU_PKTS={layout.GLU_PKTS})"

                        def _dec():

                            def _slice():
                                gx = layout.AllocOp(glu_x_l1, [], [])
                                gh = _glu_slice(
                                    gx, lambda: layout.AllocOp(glu_hid_l1, [], [])
                                )
                                layout.DeallocOp(gx)
                                layout.DeallocOp(gh)

                            for _s in layout.for_(
                                idx(0), idx(layout.NGLU // 2), idx(1)
                            ):
                                _slice()
                                _slice()
                                layout.yield_([])

                        if not _voc_arm:

                            def _dec_case():
                                _dec()
                                layout.yield_([])

                            layout.index_switch(
                                [],
                                layout.arith.index_cast(idx_t, _arm),
                                layout.ARM_IDLE_CASES,
                                case_body_builder=lambda op, i, cv: layout.yield_([]),
                                default_body_builder=lambda op: _dec_case(),
                            )
                        else:
                            _gx = [layout.AllocOp(glu_x_l1, [], []) for _ in range(2)]
                            _gh = [layout.AllocOp(glu_hid_l1, [], []) for _ in range(2)]

                            def _glu_dec_body():
                                for _s in layout.for_(
                                    idx(0), idx(layout.NGLU // 2), idx(1)
                                ):
                                    _glu_slice(_gx[0], lambda: _gh[0])
                                    _glu_slice(_gx[1], lambda: _gh[1])
                                    layout.yield_([])

                            def _glu_voc_body():
                                for _s in layout.for_(
                                    idx(0), idx(layout.VOC_ROTATIONS), idx(1)
                                ):
                                    _voc_slice(_gx[0], lambda: _gh[0])
                                    _voc_slice(_gx[1], lambda: _gh[1])
                                    layout.yield_([])

                            _outer = None
                            _ctx = (
                                layout.InsertionPoint(_outer.thenRegion.blocks[0])
                                if _outer is not None
                                else layout.contextlib.nullcontext()
                            )
                            with _ctx:
                                _ifa = layout.IfOp(
                                    layout.arith.cmpi(
                                        layout.arith.CmpIPredicate.ne,
                                        _arm,
                                        layout.arith.ConstantOp(
                                            layout.IntegerAttr.get(i32, 0), None
                                        ).result,
                                    ),
                                    [],
                                    has_else=True,
                                )
                                _ifa.operation.attributes["air.arm_select"] = (
                                    layout.UnitAttr.get()
                                )
                                with layout.InsertionPoint(_ifa.thenRegion.blocks[0]):
                                    _glu_dec_body()
                                    layout.yield_([])
                                with layout.InsertionPoint(_ifa.elseRegion.blocks[0]):
                                    _glu_voc_body()
                                    layout.yield_([])
                                if _outer is not None:
                                    layout.yield_([])
                            for _b in _gx + _gh:
                                layout.DeallocOp(_b)

                    glu_h.attributes["link_with"] = layout.StringAttr.get("glu.o")
                    glu_h.attributes["x_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), layout.GLU_PCOL
                    )
                    glu_h.attributes["y_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), 3
                    )
                    db = layout.AllocOp(down_l2, [], [])
                    db.operation.attributes["air.memtile_col"] = layout.IntegerAttr.get(
                        layout.T.i32(), layout.DOWN_PCOL
                    )
                    for _s in layout.for_(idx(0), idx(layout.NGLU), idx(1)):
                        soff = layout.arith.muli(_s, idx(layout.GLU_HID))
                        layout.ChannelGet(
                            "gluOut",
                            db,
                            offsets=[idx(0), soff],
                            sizes=[idx(layout.BATCH), idx(layout.GLU_HID)],
                            strides=[idx(layout.GLU_OUT), idx(1)],
                        )
                        layout.yield_([])
                    layout.refeed(
                        layout.DOWN_REFEED, lambda: _xnorm_put(db, layout.GLU_OUT)
                    )
                    layout.DeallocOp(db)

                    def _psw(ph, vals, ty):
                        if len(vals) == 1:
                            return vals[0]
                        return layout.index_switch(
                            [ty],
                            ph,
                            list(range(len(vals) - 1)),
                            case_body_builder=lambda op, i, cv: layout.yield_(
                                [vals[i]]
                            ),
                            default_body_builder=lambda op: layout.yield_([vals[-1]]),
                        )

                    def _wscr_alloc():
                        scratch = layout.AllocOp(wscr_mm_l1, [], [])
                        scratch.operation.attributes["air.disable_alloc_sink"] = (
                            layout.UnitAttr.get()
                        )
                        return scratch

                    def _core_blk(base_cx):

                        def body(tx, ty, _sx, _sy, _arm):
                            gcx = layout.arith.addi(idx(base_cx), tx)
                            gcy = ty
                            i2c = [idx(v) for v in layout.I2P]
                            j2c = [idx(v) for v in layout.J2P]
                            pktc = [idx(d) for d in layout.DEST]
                            c2 = idx(2)
                            _ws = _wscr_alloc()

                            def _mm(J2v):
                                J2x2 = layout.arith.muli(J2v, c2)
                                a_acc = layout.AllocOp(yacc_mm_l1, [], [])
                                a_ws = _ws
                                layout.CallOp(mm_zero, [a_acc, _arm])
                                a_x_resident = layout.AllocOp(xblk_mm_l1, [], [])
                                a_x_resident.operation.attributes[
                                    "air.disable_alloc_sink"
                                ] = layout.UnitAttr.get()
                                for _j in layout.for_(idx(0), J2x2, idx(1)):
                                    a_x = a_x_resident
                                    layout.ChannelGet("inX", a_x, indices=[gcx, gcy])
                                    a_w = layout.AllocOp(wblk_l1, [], [])
                                    layout.ChannelGet(
                                        "wL2ToL1", a_w, indices=[gcx, gcy]
                                    )
                                    layout.CallOp(mm_acc, [a_x, a_w, a_acc, a_ws])
                                    layout.DeallocOp(a_w)
                                    layout.yield_([])
                                layout.DeallocOp(a_x_resident)
                                return a_acc

                            _tokstr = layout.arith.ConstantOp(
                                layout.IntegerAttr.get(i32, layout.PAIR_ROWS), None
                            ).result

                            def _emit(a_acc, destv):
                                for token in layout.for_(
                                    idx(0), idx(layout.BATCH), idx(1)
                                ):
                                    yb = layout.AllocOp(ypair_mm_l1, [], [])
                                    layout.CallOp(
                                        mm_flush,
                                        [
                                            a_acc,
                                            yb,
                                            layout.arith.index_cast(i32, token),
                                            _tokstr,
                                        ],
                                    )
                                    layout.ChannelPut(
                                        "outA",
                                        yb,
                                        indices=[gcx, ty],
                                        offsets=[idx(14)],
                                        sizes=[idx(layout.HDR + layout.PAIR_PAY)],
                                        strides=[idx(1)],
                                        dest=destv,
                                    )
                                    layout.DeallocOp(yb)
                                    layout.yield_([])
                                layout.DeallocOp(a_acc)
                                return

                            _arm_i = layout.arith.index_cast(idx_t, _arm)
                            _id4 = idx(layout.VOCAB_DEST)

                            def _sel(voc_val, dec_thunk, ty_, ex=None):
                                return layout.index_switch(
                                    [ty_],
                                    _arm_i,
                                    [0],
                                    case_body_builder=lambda op, i, cv: layout.yield_(
                                        [voc_val]
                                    ),
                                    default_body_builder=lambda op: layout.yield_(
                                        [dec_thunk()]
                                    ),
                                )

                            nph_v = _sel(
                                idx(1), lambda: idx(layout.NPH), idx_t, [1] * 0
                            )
                            for ph in layout.for_(idx(0), nph_v, idx(1)):
                                I2v = _sel(
                                    idx(layout.VOCAB_I2),
                                    lambda: _psw(ph, i2c, idx_t),
                                    idx_t,
                                    [],
                                )
                                J2v = _sel(
                                    idx(layout.VOCAB_J2),
                                    lambda: _psw(ph, j2c, idx_t),
                                    idx_t,
                                    [],
                                )
                                pktv = _sel(
                                    _id4, lambda: _psw(ph, pktc, idx_t), idx_t, []
                                )
                                for _v1 in layout.for_(idx(0), I2v, idx(1)):
                                    for _e in range(layout.PAIR_ROWS):
                                        _emit(_mm(J2v), pktv)
                                    layout.yield_([])
                                layout.yield_([])

                        return body

                    _arm_proj = _seg_arm
                    for blk in range(layout.NCX // 2):
                        base_cx = blk * 2
                        _ops = [_arm_proj]
                        blk_h = layout.herd(
                            name=f"proj_blk{blk}", sizes=[2, 4], operands=_ops
                        )(_core_blk(base_cx))
                        blk_h.attributes["link_with"] = layout.StringAttr.get(
                            "proj_qmm.o"
                        )
                        blk_h.attributes["x_loc"] = layout.IntegerAttr.get(
                            layout.T.i64(), layout.PCOL[base_cx]
                        )
                        blk_h.attributes["y_loc"] = layout.IntegerAttr.get(
                            layout.T.i64(), 2
                        )
                    OPROJ_RNDS = layout.PAIR_ROWS * layout.I2P[1]
                    DOWN_RNDS = layout.PAIR_ROWS * layout.I2P[layout.DOWN_PHASE]
                    _arm_rms = _seg_arm

                    def _rms_body(tx, ty, _sx, _sy, _arm):
                        _rms_batched(_arm)
                        return

                    def _rms_batched_norm(xb, stg, scl, wbuf, nrefeed, _arm):
                        _nchunk = layout.K // layout.XCHUNK

                        def _i32(v):
                            return layout.arith.ConstantOp(
                                layout.IntegerAttr.get(i32, v), None
                            ).result

                        for t in range(layout.BATCH):
                            layout.CallOp(rms_scale_row_aie, [scl, xb, _i32(t), _arm])
                        _n = nrefeed if isinstance(nrefeed, int) else None
                        _bound = idx(_n) if _n is not None else nrefeed
                        for _r in layout.for_(idx(0), _bound, idx(1)):
                            for _c in layout.for_(idx(0), idx(_nchunk), idx(1)):
                                layout.CallOp(
                                    rms_chunk_aie,
                                    [
                                        stg,
                                        xb,
                                        wbuf,
                                        scl,
                                        _i32(layout.BATCH),
                                        layout.arith.index_cast(i32, _c),
                                        _i32(layout.XCHUNK),
                                    ],
                                )
                                layout.ChannelPut(
                                    "xnorm_fill",
                                    stg,
                                    offsets=[idx(0)],
                                    sizes=[idx(layout.BATCH * layout.XCHUNK)],
                                    strides=[idx(1)],
                                )
                                layout.yield_([])
                            layout.yield_([])

                    def _rms_batched(_arm):
                        assert (
                            layout.RMS_DEST >= 0
                        ), "DECODE_BATCH>1 needs the FULL4 residual path (RMS_DEST >= 0); the debug configs feed no o-proj"
                        assert layout.K % layout.XCHUNK == 0

                        def _i32(v):
                            return layout.arith.ConstantOp(
                                layout.IntegerAttr.get(i32, v), None
                            ).result

                        def _cnt(voc, dec, ex=None):
                            return layout.index_switch(
                                [idx_t],
                                layout.arith.index_cast(idx_t, _arm),
                                [0],
                                case_body_builder=lambda op, i, cv: layout.yield_(
                                    [idx(voc)]
                                ),
                                default_body_builder=lambda op: layout.yield_(
                                    [idx(dec)]
                                ),
                            )

                        def _if_decode(body, decode_only=False):
                            _pred = (
                                layout.arith.cmpi(
                                    layout.arith.CmpIPredicate.eq, _arm, _i32(1)
                                )
                                if decode_only
                                else layout.arith.cmpi(
                                    layout.arith.CmpIPredicate.ne, _arm, _i32(0)
                                )
                            )
                            _if = layout.IfOp(_pred, [], has_else=False)
                            _if.operation.attributes["air.arm_select"] = (
                                layout.UnitAttr.get()
                            )
                            with layout.InsertionPoint(_if.thenRegion.blocks[0]):
                                body()
                                layout.yield_([])

                        xb = layout.AllocOp(rmsb_l1, [], [])
                        layout.ChannelGet("rmsX", xb, indices=[idx(0)])
                        stg = layout.AllocOp(rstg_l1, [], [])
                        scl = layout.AllocOp(rscl_l1, [], [])
                        w = layout.AllocOp(rms_l1, [], [])
                        w2 = layout.AllocOp(rms_l1, [], [])

                        def _emit_norm(wbuf, nrefeed):
                            _rms_batched_norm(xb, stg, scl, wbuf, nrefeed, _arm)

                        def _accumulate(nrnds, stage, guard=None):
                            offsets, sizes, strides = layout.outy_tokmajor(
                                layout.PAYLOAD
                            )
                            for row in layout.for_(idx(0), idx(nrnds), idx(1)):
                                layout.ChannelGet(
                                    "outY",
                                    stg,
                                    indices=[idx(0), idx(layout.RMS_DEST)],
                                    offsets=[idx(v) for v in offsets],
                                    sizes=[idx(v) for v in sizes],
                                    strides=[idx(v) for v in strides],
                                )
                                off = layout.arith.muli(
                                    layout.arith.index_cast(i32, row),
                                    _i32(layout.PAYLOAD),
                                )
                                for t in range(layout.BATCH):
                                    layout.CallOp(
                                        residual_acc_row_aie,
                                        [xb, stg, _i32(t), off, _i32(layout.PAYLOAD)],
                                    )
                                layout.yield_([])

                        def _layer_out():
                            layout.ChannelPut(
                                "layerOut",
                                xb,
                                offsets=[idx(0)],
                                sizes=[idx(layout.BATCH * layout.K)],
                                strides=[idx(1)],
                            )

                        layout.ChannelGet("rmsW", w, indices=[idx(0)])
                        _emit_norm(w, 1)
                        layout.ChannelGet("rmsW2", w2, indices=[idx(0)])
                        _if_decode(lambda: _accumulate(OPROJ_RNDS, 1, guard=None))
                        _emit_norm(w2, _cnt(0, 1))
                        _if_decode(
                            lambda: _accumulate(DOWN_RNDS, 2), decode_only=bool(0)
                        )
                        _if_decode(_layer_out)
                        for _b in (w2, w, scl, stg, xb):
                            layout.DeallocOp(_b)

                    rms_h = layout.herd(name="rms", sizes=[1, 1], operands=[_arm_rms])(
                        _rms_body
                    )
                    rms_h.attributes["link_with"] = layout.StringAttr.get(
                        "rms_residual.o"
                    )
                    rms_h.attributes["x_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), layout.RMS_PCOL
                    )
                    rms_h.attributes["y_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), 2
                    )

                    @layout.herd(name="norm_cache", sizes=[1, 1], operands=[_arm_rms])
                    def cache_h(tx, ty, sx, sy, arm):
                        slab = layout.AllocOp(rmsb_l1, [], [])
                        out = layout.AllocOp(rstg_l1, [], [])
                        for buf in (slab, out):
                            buf.operation.attributes["air.disable_alloc_sink"] = (
                                layout.UnitAttr.get()
                            )

                        def emit(count):
                            layout.ChannelGet("xnorm_fill", slab)
                            for _r in layout.for_(idx(0), count, idx(1)):
                                for c in layout.for_(
                                    idx(0), idx(layout.K // layout.XCHUNK), idx(1)
                                ):
                                    layout.CallOp(
                                        cache_copy,
                                        [out, slab, layout.arith.index_cast(i32, c)],
                                    )
                                    layout.ChannelPut(
                                        "xnorm",
                                        out,
                                        offsets=[idx(0)],
                                        sizes=[idx(layout.BATCH * layout.XCHUNK)],
                                        strides=[idx(1)],
                                    )
                                    layout.yield_([])
                                layout.yield_([])

                        is_decode = layout.arith.cmpi(
                            layout.arith.CmpIPredicate.ne,
                            arm,
                            layout.arith.ConstantOp(
                                layout.IntegerAttr.get(i32, 0), None
                            ).result,
                        )
                        first_count = layout.arith.select(
                            is_decode, idx(layout.REFEED[0]), idx(layout.VOCAB_RNDS)
                        )
                        emit(first_count)
                        branch = layout.IfOp(is_decode, [], has_else=False)
                        branch.operation.attributes["air.arm_select"] = (
                            layout.UnitAttr.get()
                        )
                        with layout.InsertionPoint(branch.thenRegion.blocks[0]):
                            emit(idx(layout.REFEED[layout.GATEUP_PHASE]))
                            layout.yield_([])
                        layout.DeallocOp(out)
                        layout.DeallocOp(slab)

                    cache_h.attributes["link_with"] = layout.StringAttr.get(
                        "norm_cache.o"
                    )
                    cache_h.attributes["x_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), layout.RMS_PCOL
                    )
                    cache_h.attributes["y_loc"] = layout.IntegerAttr.get(
                        layout.T.i64(), 3
                    )

            def _emit_wave_loop(lo, hi):
                if lo >= hi:
                    return
                for _iv in layout.for_(idx(lo), idx(hi), idx(1)):
                    layout.launch(
                        sizes=[1, 1],
                        operands=list(_fa) + [_iv],
                        attributes={
                            "air.preserve_shim_dma_order": layout.UnitAttr.get()
                        },
                    )(launch_body)
                    layout.yield_([])

            _emit_wave_loop(layout.UNI_WAVE_LO, layout.UNI_WAVE_HI)

    return build()
