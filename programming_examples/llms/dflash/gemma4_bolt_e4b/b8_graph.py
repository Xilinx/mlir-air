# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B eight-token decoder graph.

dflash/qwen3_4b/batch_graph.py's structure (weight stream, batched
projections, X memory tile, egress gather, normalized-activation cache), with
what Gemma4 adds per layer:

- Post norms on the attention, MLP and per-layer-embedding outputs. Those
  outputs go to the cache core, whose slab is free once the projections have
  re-read it; it normalizes them and sends the chunks to the residual core.
- The per-layer embedding as two more projection phases: its gate (on the
  residual, unnormalized), GELU times the token's per-layer input on the GLU
  core, and its projection back to the model width.
- No output scale on device: the residual is kept divided by the product
  of the earlier layers' scales, which weights_b8.py folds into the post
  norms' weights and the per-layer embedding gate. Norms do not see it; the
  host multiplies the features it reads back.
- KV-shared layers read the slab of the layer they share with; their own
  appends land in a scratch slab.
- Gemma4 attention (attention_graph.py), its Q passing through DDR in the
  order the shards take it.
- Each layer class streams only its own weights (b8_config.class_geometry):
  the projection cores take none for the rounds and x blocks past them.

One wave per layer, then the LM-head waves. Every wave reads the layer's
input from X slot iv and writes slot iv + 1, so the host can read the draft
model's features from any layer boundary.
"""


def build_module(layout, attention):
    L = layout

    @L.module_builder
    def build():
        bf16 = L.BF16Type.get()
        f32 = L.F32Type.get()
        i32 = L.IntegerType.get_signless(32)
        idx_t = L.IndexType.get()
        l1 = L.IntegerAttr.get(L.T.i32(), L.MemorySpace.L1)
        l2 = L.IntegerAttr.get(L.T.i32(), L.MemorySpace.L2)
        B, K = L.BATCH, L.K
        el = L.ABI["elements"]
        x_l3 = L.MemRefType.get([el["x"]], bf16)
        w_l3 = L.MemRefType.get([el["w"]], bf16)
        rms_l3 = L.MemRefType.get([el["rms"]], bf16)
        y_l3 = L.MemRefType.get([el["y"]], bf16)
        kv_l3 = L.MemRefType.get([el["kv"]], bf16)
        pli_l3 = L.MemRefType.get([el["pli"]], bf16)

        def mem(n, dt=None, space=None):
            return L.MemRefType.get([n], dt or bf16, memory_space=space or l1)

        wblk_l1 = mem(L.BLOCK_BF16)
        xblk_mm_l1 = mem(B * L.COL_BLOCK)
        yacc_mm_l1 = mem(B * L.ROW_BLOCK, f32)
        wscr_mm_l1 = mem(L.ROW_BLOCK * L.COL_BLOCK)
        _YPAIR = 16 + L.ROW_BLOCK
        ypair_mm_l1 = mem(_YPAIR + -_YPAIR % L.L1_VEC_BF16)
        rms_l1 = mem(K)
        rmsb_l1 = mem(B * K)
        rstg_l1 = mem(B * L.STG_W)
        rscl_l1 = mem(B, f32)
        glu_x_l1 = mem(B * L.GLU_SLICE)
        glu_hid_l1 = mem(B * L.GLU_HID)
        qkv_l1 = mem(L.M)
        ropeq_l1 = mem(L.DQ_PADDED)
        ropekv_l1 = mem(L.DK)
        ropelut_l1 = mem(L.ROPE_W_LEN)
        omt_l2 = mem(B * L.DQ, space=l2)
        qkvmt_l2 = mem(B * L.M, space=l2)
        xmt_l2 = mem(B * L.XCHUNK, space=l2)
        wfan_l2 = mem(L.NCY // 2 * L.BLOCK_BF16, space=l2)
        grp_l2 = mem(L.GRP_ROWS_B, space=l2)
        main_l2 = mem(L.MAIN_ROWS_B, space=l2)
        down_l2 = mem(B * L.GLU_OUT, space=l2)

        def fn(name, args, obj):
            f = L.FuncOp(name, (args, []), visibility="private")
            f.attributes["link_with"] = L.StringAttr.get(obj)
            return f

        mm_zero = fn("proj_qmm_mm_zero", [yacc_mm_l1, i32], "proj_qmm.o")
        mm_acc = fn(
            "proj_qmm_mm_acc",
            [xblk_mm_l1, wblk_l1, yacc_mm_l1, wscr_mm_l1],
            "proj_qmm.o",
        )
        # independent sequences: the mask of rows that have a token
        mm_acc_m = None
        if L.SEQUENCES > 1:
            mm_acc_m = fn(
                "proj_qmm_mm_acc_m",
                [xblk_mm_l1, wblk_l1, yacc_mm_l1, wscr_mm_l1, i32],
                "proj_qmm.o",
            )
        mm_flush = fn(
            "proj_qmm_mm_flush_token", [yacc_mm_l1, ypair_mm_l1, i32, i32], "proj_qmm.o"
        )
        r_scale = fn("rms_scale_row_aie", [rscl_l1, rmsb_l1, i32, i32], "rms_b8.o")
        r_chunk = fn(
            "rms_chunk_aie",
            [rstg_l1, rmsb_l1, rms_l1, rscl_l1, i32, i32, i32],
            "rms_b8.o",
        )
        r_acc = fn(
            "residual_acc_row_aie", [rmsb_l1, rstg_l1, i32, i32, i32], "rms_b8.o"
        )
        r_acc_w = fn(
            "residual_acc_w_row_aie",
            [rmsb_l1, rstg_l1, rms_l1, i32, i32, i32],
            "rms_b8.o",
        )
        r_raw = fn("raw_chunk_aie", [rstg_l1, rmsb_l1, i32, i32, i32], "rms_b8.o")
        c_copy = fn("normalized_cache_chunk", [rstg_l1, rmsb_l1, i32], "cache_b8.o")
        c_scale = fn("cache_scale_rows_aie", [rscl_l1, rmsb_l1, i32], "cache_b8.o")
        c_chunk = fn(
            "cache_chunk_aie", [rstg_l1, rmsb_l1, rscl_l1, i32, i32], "cache_b8.o"
        )
        glu_row = fn("glu_row_aie", [glu_hid_l1, glu_x_l1, i32, i32], "glu.o")
        glu_copy = fn("glu_copy_aie", [glu_hid_l1, glu_x_l1, i32, i32, i32], "glu.o")
        ple_act = fn("ple_act_aie", [glu_hid_l1, glu_x_l1, glu_hid_l1, i32], "glu.o")
        rope_fn = fn(
            "rope_b8",
            [ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropelut_l1, i32],
            "rope_b8.o",
        )
        att = attention(L)

        # ---- channels ----
        # the residual core's inputs: the layer's input, then its norm
        # weights (the order is the residual core's)
        L.channel_decl("rmsW", size=[1])
        # The cache core's post-norm weights and the GLU core's per-layer
        # inputs reach them from a memory tile on circuits. Straight from the
        # shim they would be packets (AIR upgrades the shim's L1 channels under
        # pressure), and a core's packet channel takes packets in arrival
        # order: these arrive early and would land in the buffer the core is
        # receiving projection outputs into.
        # shares the cache core's packet channel with the projection outputs
        L.channel_decl("pliM", size=[1])
        L.channel_decl("subn", size=[1])
        L.channel_decl("xnorm", size=[1], channel_type="npu_dma_packet")
        # a circuit of its own on the residual core's MM2S 0: a channel that
        # also carried layerOut would lower to one-shot tasks (its transfers
        # differ between decode and LM-head waves) and stall the second wave
        _fill = L.channel_decl("xnorm_fill", size=[1])
        _fill.operation.attributes["air.tile_dma_channel"] = L.IntegerAttr.get(i32, 0)
        _XCHUNK = L.XCHUNK

        def _xnorm_put(buf, width, row_stride=None, **kw):
            assert width % _XCHUNK == 0, width
            return L.ChannelPut(
                "xnorm",
                buf,
                offsets=[idx(0), idx(0), idx(0)],
                sizes=[idx(width // _XCHUNK), idx(B), idx(_XCHUNK)],
                strides=[idx(_XCHUNK), idx(row_stride or width), idx(1)],
                **kw,
            )

        _inX = L.Channel("inX", size=[1, 1], broadcast_shape=[L.NCX, L.NCY])
        _inX.operation.attributes["air.shared_resident_ring"] = L.UnitAttr.get()
        L.channel_decl("ropeLUT", size=[1])
        L.channel_decl("toRope", size=[1])
        L.channel_decl("ropeQ", size=[1])
        L.channel_decl("appendK", size=[1], channel_type="npu_dma_packet")
        L.channel_decl("appendV", size=[1], channel_type="npu_dma_packet")
        for _wc in range(L.NCX):
            for _nm in (L._wname(0, _wc), L._wname(1, _wc)):
                L.channel_decl(_nm, size=[1])
        _wL2 = L.channel_decl("wL2ToL1", size=[L.NCX, L.NCY])
        _wL2.operation.attributes["air.shared_resident_ring"] = L.UnitAttr.get()
        L.channel_decl("outA", size=[L.NCX, L.PAIRS_PC], channel_type="npu_dma_packet")
        L.channel_decl("toMain", size=[L.N_GRP], channel_type="npu_dma_packet")
        _outY = L.Channel("outY", size=[1, 1], broadcast_shape=[1, L.NDEST])
        _outY.operation.attributes["channel_type"] = L.StringAttr.get("npu_dma_packet")
        _outY.operation.attributes["air.tile_dma_channel"] = L.IntegerAttr.get(
            L.T.i32(), 1
        )
        _layerOut = L.channel_decl("layerOut", size=[1])
        _layerOut.operation.attributes["air.tile_dma_channel"] = L.IntegerAttr.get(
            i32, 1
        )
        L.channel_decl("gluOut", size=[1])
        L.channel_decl("vocOut", size=[1])
        L.channel_decl("pliIn", size=[1])

        def idx(v):
            return L.arith.ConstantOp.create_index(v)

        def c32(v):
            return L.arith.ConstantOp(L.IntegerAttr.get(i32, v), None).result

        def _class_arm(layer):
            """b8_config.arm_of_layer of a wave index, as i32."""
            _sl = L.arith.cmpi(
                L.arith.CmpIPredicate.ne,
                L.arith.remui(layer, idx(L.FULL_PERIOD)),
                idx(L.FULL_PERIOD - 1),
            )
            _sh = L.arith.cmpi(L.arith.CmpIPredicate.uge, layer, idx(L.KV_OWNERS))
            return L.arith.addi(
                L.arith.addi(c32(1), L.arith.select(_sl, c32(L.SLIDING_BIT), c32(0))),
                L.arith.select(_sh, c32(L.SHARED_BIT), c32(0)),
            )

        def _arm_table(arm, vals):
            """vals[arm] for a decode arm (an i32), as an index."""
            arms = sorted(vals)
            v = idx(vals[arms[0]])
            for a in arms[1:]:
                v = L.arith.select(
                    L.arith.cmpi(L.arith.CmpIPredicate.eq, arm, c32(a)), idx(vals[a]), v
                )
            return v

        def grp_leads(g):
            _cpg = L.NCX // L.N_GRP
            return [
                (g * _cpg + lc, pp) for lc in range(_cpg) for pp in range(L.PAIRS_PC)
            ]

        # one length per sequence: a sequence's tokens sit at positions
        # [length - 1, length - 1 + its tokens)
        _NSEQ = L.SEQUENCES
        _KVSEQ = L.ABI["geometry"]["kv_sequence"]

        # Independent sequences also take each one's append row and a mask of
        # the sequences that have a token this step. An idle sequence attends
        # one key and appends to a row the host keeps out of its context, and
        # the projections skip its row.
        _NSCAL = _NSEQ if _NSEQ == 1 else 2 * _NSEQ + 1

        @L.FuncOp.from_py_func(
            x_l3, w_l3, rms_l3, y_l3, kv_l3, pli_l3, *([i32] * _NSCAL)
        )
        def gemma4_b8_decode(*_fa):
            def launch_body(*_la):
                X, W, RMS, Y, KVC, PLI = _la[4:10]
                L_rts = list(_la[10 : 10 + _NSEQ])
                L_rt = L_rts[0]
                A_rts = list(_la[10 + _NSEQ : 10 + 2 * _NSEQ]) if _NSEQ > 1 else []
                M_rt = [_la[10 + 2 * _NSEQ]] if _NSEQ > 1 else []
                iv = _la[10 + _NSCAL]

                def _lb(slab):
                    return L.arith.muli(iv, idx(slab))

                def _add(base, extra):
                    return L.arith.addi(base, idx(extra)) if extra else base

                def _slot(base, extra=0, row=None):
                    """base + (L - 1) rows of one KV row: this call's first slot,
                    or the given row."""
                    pos = L.arith.subi(L_rt, c32(1)) if row is None else row
                    off = L.arith.muli(pos, c32(attention_row()))
                    return _add(
                        L.arith.addi(base, L.arith.index_cast(idx_t, off)), extra
                    )

                def attention_row():
                    return L.KV_HEADS_G * L.DH_G

                _X_SLOT = B * K
                blk = L.BLOCK_BF16
                wstep = L.NCY * blk

                def _feed_wcol(_cx, _cbase, nsteps):
                    nch = 2
                    cstep = wstep // nch
                    span = nsteps * cstep
                    half = nsteps // 2 * cstep
                    for ci in range(nch):
                        base = _cbase if ci == 0 else L.arith.addi(_cbase, idx(span))
                        ch = L._wname(ci, _cx)
                        L.ChannelPut(
                            ch,
                            W,
                            indices=[idx(0)],
                            offsets=[base],
                            sizes=[half],
                            strides=[1],
                        )
                        L.ChannelPut(
                            ch,
                            W,
                            indices=[idx(0)],
                            offsets=[L.arith.addi(base, idx(half))],
                            sizes=[span - half],
                            strides=[1],
                        )

                def _feed_wcols(_wcol0, _colspan, nsteps):
                    for _cxi in range(L.NCX):
                        _feed_wcol(_cxi, _add(_wcol0, _cxi * _colspan), nsteps)

                _udec = L.arith.cmpi(L.arith.CmpIPredicate.slt, iv, idx(L.UNI_DEC))
                _uarm_i = L.arith.index_cast(
                    idx_t, L.arith.select(_udec, c32(1), c32(0))
                )
                _rope_base = L.UNI_DEC * L.RMS_LAYER
                _final_norm_off = _rope_base + L.UNI_DEC * B * L.ROPE_W_LEN

                def _uni_voc():
                    _vwb = L.arith.addi(
                        idx(L.W_OFFS[L.UNI_DEC]),
                        L.arith.muli(
                            L.arith.subi(iv, idx(L.UNI_DEC)),
                            idx(L.VOCAB_W_BLOCKS * blk),
                        ),
                    )
                    _vyb = L.arith.addi(
                        idx(L.ABI["offsets"]["logits"]),
                        L.arith.muli(
                            L.arith.subi(iv, idx(L.UNI_DEC)),
                            idx(L.VOCAB_SIZE_PADDED * B),
                        ),
                    )
                    L.ChannelPut(
                        "rmsW",
                        X,
                        offsets=[L.UNI_DEC * _X_SLOT],
                        sizes=[_X_SLOT],
                        strides=[1],
                    )
                    L.ChannelPut(
                        "rmsW", RMS, offsets=[_final_norm_off], sizes=[K], strides=[1]
                    )
                    # four dummy weights: the residual core's input chain is
                    # the same in every wave
                    L.ChannelPut("rmsW", RMS, offsets=[0], sizes=[4 * K], strides=[1])
                    assert L.VOCAB_PER_COL % L.NCY == 0
                    _feed_wcols(_vwb, L.VOCAB_PER_COL * blk, L.VOCAB_PER_COL // L.NCY)
                    L.ChannelGet(
                        "vocOut",
                        Y,
                        indices=[idx(0)],
                        offsets=[_vyb],
                        sizes=[L.VOCAB_RNDS // L.VOC_SLICE_RNDS, B, L.GLU_SLICE],
                        strides=[L.GLU_SLICE, L.VOCAB_SIZE_PADDED, 1],
                    )
                    L.yield_([])

                def _per_layer(vals):
                    """vals[iv] (the launch loop unrolls, so this folds)."""
                    v = idx(vals[0])
                    for _l in range(1, len(vals)):
                        v = L.arith.select(
                            L.arith.cmpi(L.arith.CmpIPredicate.eq, iv, idx(_l)),
                            idx(vals[_l]),
                            v,
                        )
                    return v

                def _by_arm(kinds, feeds, wbase):
                    """The phase's whole column fan in one switch case per
                    class (a switch per column would split the fan)."""
                    _a = _per_layer([L.arm_of_layer(_l) for _l in range(L.UNI_DEC)])

                    def _case(a):
                        _o, _pc = feeds[a]
                        _feed_wcols(_add(wbase, _o), _pc * blk, _pc // L.NCY)
                        L.yield_([])

                    L.index_switch(
                        [],
                        _a,
                        kinds[:-1],
                        case_body_builder=lambda op, i, cv: _case(kinds[i]),
                        default_body_builder=lambda op: _case(kinds[-1]),
                    )

                def _uni_dec():
                    _wbase = _per_layer(L.W_OFFS[: L.UNI_DEC])
                    _arms = [L.arm_of_layer(_l) for _l in range(L.UNI_DEC)]
                    _kinds = sorted(set(_arms))
                    _rbase = _lb(L.RMS_LAYER)
                    L.ChannelPut(
                        "rmsW", X, offsets=[_lb(_X_SLOT)], sizes=[_X_SLOT], strides=[1]
                    )
                    # the layer's five norm weights, in the order the residual
                    # core takes them
                    L.ChannelPut(
                        "rmsW", RMS, offsets=[_rbase], sizes=[5 * K], strides=[1]
                    )
                    L.ChannelPut(
                        "ropeLUT",
                        RMS,
                        offsets=[_add(_lb(B * L.ROPE_W_LEN), _rope_base)],
                        sizes=[B * L.ROPE_W_LEN],
                        strides=[1],
                    )
                    L.ChannelPut(
                        "pliM",
                        PLI,
                        offsets=[_lb(B * L.PLE_PAD)],
                        sizes=[B * L.PLE_PAD],
                        strides=[1],
                    )
                    # KV slabs: appends go to the layer's own slab, or to the
                    # scratch slab past the owners; reads come from the slab of
                    # the layer whose cache this one uses.
                    _owner = L.arith.cmpi(
                        L.arith.CmpIPredicate.ult, iv, idx(L.KV_OWNERS)
                    )
                    _app = L.arith.select(_owner, iv, idx(L.KV_SLABS - 1))
                    _full = L.arith.cmpi(
                        L.arith.CmpIPredicate.eq,
                        L.arith.remui(iv, idx(L.FULL_PERIOD)),
                        idx(L.FULL_PERIOD - 1),
                    )
                    _shared = L.arith.select(
                        _full, idx(L.KV_OWNERS - 1), idx(L.KV_OWNERS - 2)
                    )
                    _rd = L.arith.select(_owner, iv, _shared)
                    _vreg = L.CAPACITY * attention_row()
                    _kapp = L.arith.muli(_app, idx(L.KV_LAYER))
                    _krd = L.arith.muli(_rd, idx(L.KV_LAYER))
                    _woff = {a: 0 for a in _kinds}
                    for p in range(L.NPH):
                        # this phase in each class present: (offset, blocks per column)
                        _f = {a: (_woff[a], L.CLASS_PER_COL[a][p]) for a in _kinds}
                        for a in _kinds:
                            _woff[a] += L.NCX * L.CLASS_PER_COL[a][p] * blk
                        if len(set(_f.values())) == 1:
                            _o, _pc = _f[_kinds[0]]
                            _feed_wcols(_add(_wbase, _o), _pc * blk, _pc // L.NCY)
                        elif len({pc for _, pc in _f.values()}) == 1:
                            _pc = _f[_kinds[0]][1]
                            _o = _per_layer([_f[a][0] for a in _arms])
                            _feed_wcols(
                                L.arith.addi(_wbase, _o), _pc * blk, _pc // L.NCY
                            )
                        else:
                            _by_arm(_kinds, _f, _wbase)
                        if p == 0:
                            # the tokens' appends, B rows of one sequence or
                            # one row each to its own sequence's cache
                            _rows = B // _NSEQ
                            for kind, extra in (("appendK", 0), ("appendV", _vreg)):
                                for _s in range(_NSEQ):
                                    L.ChannelGet(
                                        kind,
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=[
                                            _slot(
                                                _add(_kapp, _s * _KVSEQ),
                                                extra,
                                                A_rts[_s] if A_rts else None,
                                            )
                                        ],
                                        sizes=[_rows * attention_row()],
                                        strides=[1],
                                    )
                            qoff = L.ABI["offsets"]["q_scratch"]
                            # rope's q, written in the shards' frame order:
                            # [token][KV head][q head % 4] in, frame order out
                            L.ChannelGet(
                                "ropeQ",
                                Y,
                                offsets=[idx(0), idx(0), idx(0), idx(qoff)],
                                sizes=[B, L.KV_HEADS_G, 4, L.DH_G],
                                strides=[
                                    L.KV_HEADS_G * 4 * L.DH_G,
                                    L.DH_G,
                                    L.KV_HEADS_G * L.DH_G,
                                    1,
                                ],
                            )
                            att.put_query(Y, idx(qoff))
                            for _s in range(_NSEQ):
                                _kb = _add(_krd, _s * _KVSEQ)
                                att.put_kv(
                                    KVC,
                                    _kb,
                                    _add(_kb, _vreg),
                                    L_rts[_s],
                                    L.arith.select(
                                        _full, c32(0), c32(L.SLIDING_WINDOW_G)
                                    ),
                                    seq=_s,
                                )
                    L.ChannelGet(
                        "layerOut",
                        X,
                        indices=[idx(0)],
                        offsets=[_add(_lb(_X_SLOT), _X_SLOT)],
                        sizes=[_X_SLOT],
                        strides=[1],
                    )
                    L.yield_([])

                L.index_switch(
                    [],
                    _uarm_i,
                    [0],
                    case_body_builder=lambda op, i, cv: _uni_voc(),
                    default_body_builder=lambda op: _uni_dec(),
                )

                @L.segment(name="seg", operands=[iv] + L_rts + M_rt)
                def seg(_seg_iv, *_seg_ops):
                    _seg_Ls = _seg_ops[:_NSEQ]
                    _seg_M = list(_seg_ops[_NSEQ:])
                    _cmp = L.arith.cmpi(
                        L.arith.CmpIPredicate.slt, _seg_iv, idx(L.UNI_DEC)
                    )
                    _arm = L.arith.select(_cmp, _class_arm(_seg_iv), c32(0))
                    _arm_i = L.arith.index_cast(idx_t, _arm)

                    def _switch(voc, dec):
                        def _v():
                            voc()
                            L.yield_([])

                        def _d():
                            dec()
                            L.yield_([])

                        L.index_switch(
                            [],
                            _arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _v(),
                            default_body_builder=lambda op: _d(),
                        )

                    # ---- X memory tile: every projection input ----
                    def _feed_inX(total_chunks):
                        for _rc in L.for_(idx(0), idx(total_chunks), idx(1)):
                            xb = L.AllocOp(xmt_l2, [], [])
                            xb.operation.attributes["air.memtile_col"] = (
                                L.IntegerAttr.get(L.T.i32(), L.XMT_PCOL)
                            )
                            L.ChannelGet(
                                "xnorm",
                                xb,
                                offsets=[0],
                                sizes=[B * L.XCHUNK],
                                strides=[1],
                            )
                            for _jj in L.for_(idx(0), idx(L.XCHUNK_MUL), idx(1)):
                                _sz, _st = L._XFEED_BD
                                _o0 = L.arith.muli(_jj, idx(L.COL_BLOCK // _st[0]))
                                L.ChannelPut(
                                    "inX",
                                    xb,
                                    offsets=[_o0] + [idx(0)] * (len(_sz) - 1),
                                    sizes=[idx(v) for v in _sz],
                                    strides=[idx(v) for v in _st],
                                )
                                L.yield_([])
                            L.DeallocOp(xb)
                            L.yield_([])

                    assert L.N_XBLK % L.XCHUNK_MUL == 0
                    _switch(
                        lambda: _feed_inX(L.VOCAB_RNDS * (K // L.XCHUNK)),
                        lambda: _feed_inX(L.N_XBLK // L.XCHUNK_MUL),
                    )

                    # ---- shim to core relays (see the channel declarations) ----
                    def _relays():
                        pb = L.AllocOp(mem(B * L.PLE_PAD, space=l2), [], [])
                        pb.operation.attributes["air.memtile_col"] = L.IntegerAttr.get(
                            L.T.i32(), 5
                        )
                        L.ChannelGet("pliM", pb)
                        L.ChannelPut("pliIn", pb)
                        L.DeallocOp(pb)

                    _switch(lambda: None, _relays)

                    # ---- weight fan ----
                    for cx in range(L.NCX):
                        for _ci, _cy0, _cy1 in (
                            (0, 0, L.NCY // 2),
                            (1, L.NCY // 2, L.NCY),
                        ):
                            _wch = L._wname(_ci, cx)
                            for _ in L.for_(idx(0), idx(L.W_FAN_STEPS), idx(1)):
                                wf = L.AllocOp(wfan_l2, [], [])
                                wf.operation.attributes["air.memtile_col"] = (
                                    L.IntegerAttr.get(L.T.i32(), L.PCOL[cx])
                                )
                                L.ChannelGet(_wch, wf, indices=[idx(0)])
                                for cy in range(_cy0, _cy1):
                                    L.ChannelPut(
                                        "wL2ToL1",
                                        wf,
                                        indices=[idx(cx), idx(cy)],
                                        offsets=[(cy - _cy0) * L.BLOCK_BF16],
                                        sizes=[L.BLOCK_BF16],
                                        strides=[1],
                                    )
                                L.DeallocOp(wf)
                                L.yield_([])

                    # ---- egress: projection outputs to their consumers ----
                    def _egress(_nrc):
                        for _r in L.for_(idx(0), idx(_nrc * B), idx(1)):
                            for g in range(L.N_GRP):
                                grp = L.AllocOp(grp_l2, [], [])
                                grp.operation.attributes["air.memtile_col"] = (
                                    L.IntegerAttr.get(L.T.i32(), L.GRP_PCOL[g])
                                )
                                for k, (cx, pp) in enumerate(grp_leads(g)):
                                    off = 0 if k == 0 else L.HDR + k * L.PAIR_PAY
                                    sz = L.HDR + L.PAIR_PAY if k == 0 else L.PAIR_PAY
                                    L.ChannelGet(
                                        "outA",
                                        grp,
                                        indices=[idx(cx), idx(pp)],
                                        offsets=[off],
                                        sizes=[sz],
                                        strides=[1],
                                    )
                                L.ChannelPut(
                                    "toMain",
                                    grp,
                                    indices=[idx(g)],
                                    offsets=[0],
                                    sizes=[L.GRP_ROWS_B],
                                    strides=[1],
                                )
                                L.DeallocOp(grp)
                            ml = L.AllocOp(main_l2, [], [])
                            ml.operation.attributes["air.memtile_col"] = (
                                L.IntegerAttr.get(L.T.i32(), L.MAIN_PCOL)
                            )
                            for g in range(L.N_GRP):
                                _gpay = L.LEADS_PER_GRP * L.PAIR_PAY
                                off = 0 if g == 0 else L.GRP_ROWS_B + (g - 1) * _gpay
                                sz = L.GRP_ROWS_B if g == 0 else _gpay
                                L.ChannelGet(
                                    "toMain",
                                    ml,
                                    indices=[idx(g)],
                                    offsets=[off],
                                    sizes=[sz],
                                    strides=[1],
                                )
                            L.ChannelPut(
                                "outY",
                                ml,
                                indices=[idx(0), idx(0)],
                                offsets=[0],
                                sizes=[L.MAIN_ROWS_B],
                                strides=[1],
                            )
                            L.DeallocOp(ml)
                            L.yield_([])

                    _switch(lambda: _egress(L.VOCAB_RNDS), lambda: _egress(L.N_ROUNDS))

                    # ---- QKV to the rope core, token by token ----
                    def _qkv_transpose():
                        qmt = L.AllocOp(qkvmt_l2, [], [])
                        # on the memory tile of the main gather that sends
                        # it; the attention outputs' tile takes circuits only
                        qmt.operation.attributes["air.memtile_col"] = L.IntegerAttr.get(
                            L.T.i32(), L.MAIN_PCOL
                        )
                        _qo, _qs, _qt = L.outy_tokmajor(
                            L.M, rounds=L.ROUNDS_PER_DEST[0]
                        )
                        L.ChannelGet(
                            "outY",
                            qmt,
                            indices=[idx(0), idx(0)],
                            offsets=[idx(v) for v in _qo],
                            sizes=[idx(v) for v in _qs],
                            strides=[idx(v) for v in _qt],
                        )
                        L.ChannelPut(
                            "toRope",
                            qmt,
                            offsets=[idx(0)],
                            sizes=[idx(B * L.M)],
                            strides=[idx(1)],
                        )
                        L.DeallocOp(qmt)

                    _switch(lambda: None, _qkv_transpose)

                    @L.herd(name="rope", sizes=[1, 1], operands=[_arm])
                    def rope_h(tx, ty, _sx, _sy, arm):
                        def _one():
                            a_qkv = L.AllocOp(qkv_l1, [], [])
                            L.ChannelGet(
                                "toRope",
                                a_qkv,
                                offsets=[idx(0)],
                                sizes=[idx(L.M)],
                                strides=[idx(1)],
                            )
                            a_lut = L.AllocOp(ropelut_l1, [], [])
                            L.ChannelGet("ropeLUT", a_lut, indices=[idx(0)])
                            a_q = L.AllocOp(ropeq_l1, [], [])
                            a_k = L.AllocOp(ropekv_l1, [], [])
                            a_v = L.AllocOp(ropekv_l1, [], [])
                            L.CallOp(rope_fn, [a_q, a_k, a_v, a_qkv, a_lut, arm])
                            L.ChannelPut("ropeQ", a_q, indices=[idx(0)])
                            L.ChannelPut("appendK", a_k, indices=[idx(0)])
                            L.ChannelPut("appendV", a_v, indices=[idx(0)])
                            for b in (a_qkv, a_lut, a_q, a_k, a_v):
                                L.DeallocOp(b)

                        def _dec():
                            for _t in L.for_(idx(0), idx(B), idx(1)):
                                _one()
                                L.yield_([])
                            L.yield_([])

                        L.index_switch(
                            [],
                            L.arith.index_cast(idx_t, arm),
                            [0],
                            case_body_builder=lambda op, i, cv: L.yield_([]),
                            default_body_builder=lambda op: _dec(),
                        )

                    rope_h.attributes["link_with"] = L.StringAttr.get("rope_b8.o")
                    rope_h.attributes["x_loc"] = L.IntegerAttr.get(L.T.i64(), 5)
                    rope_h.attributes["y_loc"] = L.IntegerAttr.get(L.T.i64(), 2)

                    # sliding layers see the last SLIDING_WINDOW keys
                    _full = L.arith.cmpi(
                        L.arith.CmpIPredicate.eq,
                        L.arith.remui(_seg_iv, idx(L.FULL_PERIOD)),
                        idx(L.FULL_PERIOD - 1),
                    )
                    _window = L.arith.select(_full, c32(0), c32(L.SLIDING_WINDOW_G))
                    att.emit(_seg_Ls, _arm, _window)

                    # ---- attention output to the o projection ----
                    def _omtb():
                        omtb = L.AllocOp(omt_l2, [], [])
                        # with the X memory tile: its re-reads stay in the tile
                        omtb.operation.attributes["air.memtile_col"] = (
                            L.IntegerAttr.get(L.T.i32(), L.XMT_PCOL)
                        )
                        att.gather(omtb, L.DQ)
                        L.refeed(
                            L.REFEED[1],
                            lambda: _xnorm_put(omtb, L.DQ, indices=[idx(0)]),
                        )
                        L.DeallocOp(omtb)

                    _switch(lambda: None, _omtb)

                    # ---- GLU core: the MLP's GLU, the per-layer embedding's
                    # gate, and the LM head's drain ----
                    @L.herd(name="glu", sizes=[1, 1], operands=[_arm])
                    def glu_h(tx, ty, _sx, _sy, arm):
                        def _get_slice(gx):
                            # in arrival order, [round][token][PAYLOAD]
                            L.ChannelGet(
                                "outY",
                                gx,
                                indices=[idx(0), idx(L.GLU_DEST)],
                                offsets=[idx(0)],
                                sizes=[idx(B * L.GLU_SLICE)],
                                strides=[idx(1)],
                            )

                        def _rows(gh, gx):
                            for _t in range(B):
                                L.CallOp(glu_row, [gh, gx, c32(_t), arm])
                            L.ChannelPut(
                                "gluOut",
                                gh,
                                offsets=[idx(0)],
                                sizes=[idx(B * L.GLU_HID)],
                                strides=[idx(1)],
                            )

                        def _voc_slice(gx, gh):
                            _get_slice(gx)
                            _half = B * L.GLU_HID
                            for _h in range(2):
                                L.CallOp(
                                    glu_copy, [gh, gx, c32(_h * _half), c32(_half), arm]
                                )
                                L.ChannelPut(
                                    "vocOut",
                                    gh,
                                    offsets=[idx(0)],
                                    sizes=[idx(_half)],
                                    strides=[idx(1)],
                                )

                        # One buffer each way: every slice, the per-layer
                        # embedding's gate and the LM head's drain then use the
                        # same transfer, so each channel is one circular BD
                        # in decode and LM-head waves alike.
                        gx = L.AllocOp(glu_x_l1, [], [])
                        gh = L.AllocOp(glu_hid_l1, [], [])
                        # the LM head's drain has its own: two output channels
                        # on one buffer would both wait on its lock, and the
                        # wrong one can take it
                        gv = L.AllocOp(glu_hid_l1, [], [])
                        # the per-layer input: a buffer that only DMA writes
                        gp = L.AllocOp(glu_hid_l1, [], [])

                        def _dec_body():
                            for _s in L.for_(idx(0), idx(L.NGLU), idx(1)):
                                _get_slice(gx)
                                _rows(gh, gx)
                                L.yield_([])
                            # per-layer embedding: the gate arrives as a slice
                            # whose first round holds it; GELU(gate) * pli
                            L.ChannelGet("pliIn", gp)
                            _get_slice(gx)
                            L.CallOp(ple_act, [gh, gx, gp, c32(B * L.PLE_PAD)])
                            # once per term of the projection's input, [g | g]
                            for _ in range(L.PLE_TERMS_G):
                                L.ChannelPut(
                                    "gluOut",
                                    gh,
                                    offsets=[idx(0)],
                                    sizes=[idx(B * L.GLU_HID)],
                                    strides=[idx(1)],
                                )

                        def _voc_body():
                            for _s in L.for_(idx(0), idx(2 * L.VOC_ROTATIONS), idx(1)):
                                _voc_slice(gx, gv)
                                L.yield_([])

                        _ifa = L.IfOp(
                            L.arith.cmpi(L.arith.CmpIPredicate.ne, arm, c32(0)),
                            [],
                            has_else=True,
                        )
                        with L.InsertionPoint(_ifa.thenRegion.blocks[0]):
                            _dec_body()
                            L.yield_([])
                        with L.InsertionPoint(_ifa.elseRegion.blocks[0]):
                            _voc_body()
                            L.yield_([])
                        L.DeallocOp(gx)
                        L.DeallocOp(gh)
                        L.DeallocOp(gv)
                        L.DeallocOp(gp)

                    glu_h.attributes["link_with"] = L.StringAttr.get("glu.o")
                    glu_h.attributes["x_loc"] = L.IntegerAttr.get(L.T.i64(), L.GLU_PCOL)
                    glu_h.attributes["y_loc"] = L.IntegerAttr.get(L.T.i64(), 3)

                    # ---- GLU outputs to the down and per-layer projections ----
                    db = L.AllocOp(down_l2, [], [])
                    db.operation.attributes["air.memtile_col"] = L.IntegerAttr.get(
                        L.T.i32(), L.DOWN_PCOL
                    )
                    # every slice in one transfer (one BD rather than one
                    # per slice): [slice][token][GLU_HID] into [token][GLU_OUT]
                    L.ChannelGet(
                        "gluOut",
                        db,
                        offsets=[idx(0), idx(0), idx(0)],
                        sizes=[idx(L.NGLU), idx(B), idx(L.GLU_HID)],
                        strides=[idx(L.GLU_HID), idx(L.GLU_OUT), idx(1)],
                    )
                    L.refeed(L.REFEED[3], lambda: _xnorm_put(db, L.GLU_OUT))
                    L.DeallocOp(db)
                    # a buffer of its own: db's re-reads have their own
                    # count
                    pg = L.AllocOp(mem(L.PLE_TERMS_G * B * L.PLE_PAD, space=l2), [], [])
                    pg.operation.attributes["air.memtile_col"] = L.IntegerAttr.get(
                        L.T.i32(), L.DOWN_PCOL
                    )
                    # [term][token][PLE_PAD]: each re-read sends both
                    # terms, a chunk each
                    L.ChannelGet("gluOut", pg)
                    L.refeed(
                        L.PLE_G_REFEEDS // L.PLE_TERMS_G,
                        lambda: L.ChannelPut(
                            "xnorm",
                            pg,
                            offsets=[idx(0)],
                            sizes=[idx(L.PLE_TERMS_G * B * L.PLE_PAD)],
                            strides=[idx(1)],
                        ),
                    )
                    L.DeallocOp(pg)

                    # ---- projections ----
                    def _psw(ph, vals):
                        if len(vals) == 1:
                            return vals[0]
                        return L.index_switch(
                            [idx_t],
                            ph,
                            list(range(len(vals) - 1)),
                            case_body_builder=lambda op, i, cv: L.yield_([vals[i]]),
                            default_body_builder=lambda op: L.yield_([vals[-1]]),
                        )

                    def _core_blk(base_cx):
                        def body(tx, ty, _sx, _sy, arm, *mask):
                            gcx = L.arith.addi(idx(base_cx), tx)
                            gcy = ty
                            i2c = [idx(v) for v in L.I2P]
                            j2c = [idx(v) for v in L.J2P]
                            pktc = [idx(d) for d in L.DEST]
                            scratch = L.AllocOp(wscr_mm_l1, [], [])
                            scratch.operation.attributes["air.disable_alloc_sink"] = (
                                L.UnitAttr.get()
                            )

                            def _mm(J2v, nw):
                                """One round: 2 J2v x blocks, of which the
                                first nw (even) carry weights; none at all if
                                nw is 0. Blocks go by pairs, each weight get in
                                a 0/1-trip loop of the pair's bound, so the x
                                and weight buffers keep alternating."""
                                a_acc = L.AllocOp(yacc_mm_l1, [], [])
                                L.CallOp(mm_zero, [a_acc, arm])
                                # x for the round only if it has weights
                                _span = L.arith.select(
                                    L.arith.cmpi(L.arith.CmpIPredicate.ne, nw, idx(0)),
                                    L.arith.muli(J2v, idx(2)),
                                    idx(0),
                                )
                                for _jp in L.for_(idx(0), _span, idx(2)):
                                    _hw = L.arith.select(
                                        L.arith.cmpi(
                                            L.arith.CmpIPredicate.ult, _jp, nw
                                        ),
                                        idx(1),
                                        idx(0),
                                    )
                                    for _o in range(2):
                                        a_x = L.AllocOp(xblk_mm_l1, [], [])
                                        L.ChannelGet("inX", a_x, indices=[gcx, gcy])
                                        for _w in L.for_(idx(0), _hw, idx(1)):
                                            a_w = L.AllocOp(wblk_l1, [], [])
                                            L.ChannelGet(
                                                "wL2ToL1", a_w, indices=[gcx, gcy]
                                            )
                                            if mask:
                                                L.CallOp(
                                                    mm_acc_m,
                                                    [a_x, a_w, a_acc, scratch, mask[0]],
                                                )
                                            else:
                                                L.CallOp(
                                                    mm_acc, [a_x, a_w, a_acc, scratch]
                                                )
                                            L.DeallocOp(a_w)
                                            L.yield_([])
                                        L.DeallocOp(a_x)
                                    L.yield_([])
                                return a_acc

                            def _emit(a_acc, destv):
                                for token in L.for_(idx(0), idx(B), idx(1)):
                                    yb = L.AllocOp(ypair_mm_l1, [], [])
                                    L.CallOp(
                                        mm_flush,
                                        [
                                            a_acc,
                                            yb,
                                            L.arith.index_cast(i32, token),
                                            c32(1),
                                        ],
                                    )
                                    L.ChannelPut(
                                        "outA",
                                        yb,
                                        indices=[gcx, ty],
                                        offsets=[idx(14)],
                                        sizes=[idx(L.HDR + L.PAIR_PAY)],
                                        strides=[idx(1)],
                                        dest=destv,
                                    )
                                    L.DeallocOp(yb)
                                    L.yield_([])
                                L.DeallocOp(a_acc)

                            _ai = L.arith.index_cast(idx_t, arm)

                            def _sel(voc_val, dec_thunk):
                                return L.index_switch(
                                    [idx_t],
                                    _ai,
                                    [0],
                                    case_body_builder=lambda op, i, cv: L.yield_(
                                        [voc_val]
                                    ),
                                    default_body_builder=lambda op: L.yield_(
                                        [dec_thunk()]
                                    ),
                                )

                            # the class's real qkv rounds and o x-block pairs
                            def _by_class(k, p):
                                v = idx(L.CLASS_GEOM[1][k][p])
                                for a in L.CLASS_GEOM:
                                    if a != 1:
                                        _is = L.arith.cmpi(
                                            L.arith.CmpIPredicate.eq, _ai, idx(a)
                                        )
                                        v = L.arith.select(
                                            _is, idx(L.CLASS_GEOM[a][k][p]), v
                                        )
                                return v

                            _i2q = _by_class(0, 0)
                            _j2o = _by_class(1, 1)
                            nph_v = _sel(idx(1), lambda: idx(L.NPH))
                            for ph in L.for_(idx(0), nph_v, idx(1)):
                                I2v = _sel(idx(L.VOCAB_I2), lambda: _psw(ph, i2c))
                                J2v = _sel(idx(L.VOCAB_J2), lambda: _psw(ph, j2c))
                                pktv = _sel(idx(L.VOCAB_DEST), lambda: _psw(ph, pktc))
                                _is0 = L.arith.cmpi(
                                    L.arith.CmpIPredicate.eq, ph, idx(0)
                                )
                                _is1 = L.arith.cmpi(
                                    L.arith.CmpIPredicate.eq, ph, idx(1)
                                )
                                _dec = L.arith.cmpi(
                                    L.arith.CmpIPredicate.ne, _ai, idx(0)
                                )
                                I2w = L.arith.select(
                                    L.arith.andi(_dec, _is0), _i2q, I2v
                                )
                                J2w = L.arith.select(
                                    L.arith.andi(_dec, _is1), _j2o, J2v
                                )
                                for _v1 in L.for_(idx(0), I2v, idx(1)):
                                    _nw = L.arith.select(
                                        L.arith.cmpi(
                                            L.arith.CmpIPredicate.ult, _v1, I2w
                                        ),
                                        L.arith.muli(J2w, idx(2)),
                                        idx(0),
                                    )
                                    _emit(_mm(J2v, _nw), pktv)
                                    L.yield_([])
                                L.yield_([])

                        return body

                    for blk_i in range(L.NCX // 2):
                        base_cx = blk_i * 2
                        blk_h = L.herd(
                            name=f"proj_blk{blk_i}",
                            sizes=[2, 4],
                            operands=[_arm] + _seg_M,
                        )(_core_blk(base_cx))
                        blk_h.attributes["link_with"] = L.StringAttr.get("proj_qmm.o")
                        blk_h.attributes["x_loc"] = L.IntegerAttr.get(
                            L.T.i64(), L.PCOL[base_cx]
                        )
                        blk_h.attributes["y_loc"] = L.IntegerAttr.get(L.T.i64(), 2)

                    _nchunk = K // L.XCHUNK

                    # ---- residual core ----
                    @L.herd(name="rms", sizes=[1, 1], operands=[_arm])
                    def rms_h(tx, ty, _sx, _sy, arm):
                        xb = L.AllocOp(rmsb_l1, [], [])
                        stg = L.AllocOp(rstg_l1, [], [])
                        scl = L.AllocOp(rscl_l1, [], [])
                        w = L.AllocOp(rms_l1, [], [])
                        w2 = L.AllocOp(rms_l1, [], [])
                        L.ChannelGet("rmsW", xb, indices=[idx(0)])

                        def _fill(chunk_fn):
                            for _c in L.for_(idx(0), idx(_nchunk), idx(1)):
                                chunk_fn(L.arith.index_cast(i32, _c))
                                L.ChannelPut(
                                    "xnorm_fill",
                                    stg,
                                    offsets=[idx(0)],
                                    sizes=[idx(B * L.XCHUNK)],
                                    strides=[idx(1)],
                                )
                                L.yield_([])

                        def _norm(wbuf):
                            for t in range(B):
                                L.CallOp(r_scale, [scl, xb, c32(t), arm])
                            _fill(
                                lambda c: L.CallOp(
                                    r_chunk,
                                    [stg, xb, wbuf, scl, c32(B), c, c32(L.XCHUNK)],
                                )
                            )

                        def _add_sub(wpost):
                            """A sublayer's normalized output, chunk by chunk from
                            the cache core, times its post-norm weight."""
                            for _c in L.for_(idx(0), idx(_nchunk), idx(1)):
                                L.ChannelGet("subn", stg, indices=[idx(0)])
                                off = L.arith.muli(
                                    L.arith.index_cast(i32, _c), c32(L.XCHUNK)
                                )
                                for t in range(B):
                                    L.CallOp(
                                        r_acc_w,
                                        [xb, stg, wpost, c32(t), off, c32(L.XCHUNK)],
                                    )
                                L.yield_([])

                        def _dec(body):
                            _i = L.IfOp(
                                L.arith.cmpi(L.arith.CmpIPredicate.ne, arm, c32(0)),
                                [],
                                has_else=False,
                            )
                            with L.InsertionPoint(_i.thenRegion.blocks[0]):
                                body()
                                L.yield_([])

                        # The input chain is the same in every wave: x, then
                        # five weights into w, w2, w, w2, w (input, pre-FFN,
                        # post-attention, post-MLP, post-embedding; LM-head
                        # waves send the final norm and four dummies). Each
                        # weight lands once the one before it in its buffer
                        # has been used.
                        L.ChannelGet("rmsW", w, indices=[idx(0)])
                        L.ChannelGet("rmsW", w2, indices=[idx(0)])
                        _norm(w)
                        L.ChannelGet("rmsW", w, indices=[idx(0)])  # post-attention

                        def _attn_mlp():
                            _add_sub(w)
                            _norm(w2)

                        _dec(_attn_mlp)
                        L.ChannelGet("rmsW", w2, indices=[idx(0)])  # post-MLP

                        def _mlp_ple():
                            _add_sub(w2)
                            _fill(
                                lambda c: L.CallOp(
                                    r_raw, [stg, xb, c32(B), c, c32(L.XCHUNK)]
                                )
                            )

                        _dec(_mlp_ple)
                        L.ChannelGet("rmsW", w, indices=[idx(0)])  # post-embedding

                        def _out():
                            _add_sub(w)
                            L.ChannelPut(
                                "layerOut",
                                xb,
                                offsets=[idx(0)],
                                sizes=[idx(B * K)],
                                strides=[idx(1)],
                            )

                        _dec(_out)
                        for b in (w2, w, scl, stg, xb):
                            L.DeallocOp(b)

                    rms_h.attributes["link_with"] = L.StringAttr.get("rms_b8.o")
                    rms_h.attributes["x_loc"] = L.IntegerAttr.get(L.T.i64(), L.RMS_PCOL)
                    rms_h.attributes["y_loc"] = L.IntegerAttr.get(L.T.i64(), 2)

                    # ---- cache core ----
                    @L.herd(name="norm_cache", sizes=[1, 1], operands=[_arm])
                    def cache_h(tx, ty, sx, sy, arm):
                        slab = L.AllocOp(rmsb_l1, [], [])
                        out = L.AllocOp(rstg_l1, [], [])
                        scl = L.AllocOp(rscl_l1, [], [])
                        for buf in (slab, out):
                            buf.operation.attributes["air.disable_alloc_sink"] = (
                                L.UnitAttr.get()
                            )

                        def _replay(count):
                            L.ChannelGet("xnorm_fill", slab)
                            for _r in L.for_(idx(0), count, idx(1)):
                                for c in L.for_(idx(0), idx(_nchunk), idx(1)):
                                    L.CallOp(
                                        c_copy, [out, slab, L.arith.index_cast(i32, c)]
                                    )
                                    L.ChannelPut(
                                        "xnorm",
                                        out,
                                        offsets=[idx(0)],
                                        sizes=[idx(B * L.XCHUNK)],
                                        strides=[idx(1)],
                                    )
                                    L.yield_([])
                                L.yield_([])

                        def _post_norm():
                            # the sublayer's output as it arrives, [round][token]
                            # [PAYLOAD], normalized without its weight
                            assert L.PAYLOAD == L.XCHUNK
                            L.ChannelGet(
                                "outY",
                                slab,
                                indices=[idx(0), idx(L.RMS_DEST)],
                                offsets=[idx(0)],
                                sizes=[idx(B * K)],
                                strides=[idx(1)],
                            )
                            L.CallOp(c_scale, [scl, slab, c32(B)])
                            for c in L.for_(idx(0), idx(_nchunk), idx(1)):
                                L.CallOp(
                                    c_chunk,
                                    [
                                        out,
                                        slab,
                                        scl,
                                        c32(B),
                                        L.arith.index_cast(i32, c),
                                    ],
                                )
                                L.ChannelPut("subn", out, indices=[idx(0)])
                                L.yield_([])

                        is_dec = L.arith.cmpi(L.arith.CmpIPredicate.ne, arm, c32(0))
                        # a qkv round past the class's has no weights, and
                        # the projection cores take no x for it
                        _replay(
                            L.arith.select(
                                is_dec,
                                _arm_table(
                                    arm,
                                    {a: g[0][0] for a, g in L.CLASS_GEOM.items()},
                                ),
                                idx(L.VOCAB_RNDS),
                            )
                        )
                        branch = L.IfOp(is_dec, [], has_else=False)
                        with L.InsertionPoint(branch.thenRegion.blocks[0]):
                            _post_norm()  # attention
                            _replay(idx(L.REFEED[2]))
                            _post_norm()  # MLP
                            _replay(idx(L.PLE_X_REPLAYS))
                            _post_norm()  # per-layer embedding
                            L.yield_([])
                        for b in (scl, out, slab):
                            L.DeallocOp(b)

                    cache_h.attributes["link_with"] = L.StringAttr.get("cache_b8.o")
                    # above the residual core: the two exchange every chunk
                    cache_h.attributes["x_loc"] = L.IntegerAttr.get(L.T.i64(), 2)
                    cache_h.attributes["y_loc"] = L.IntegerAttr.get(L.T.i64(), 3)

            for _iv in L.for_(idx(L.UNI_WAVE_LO), idx(L.UNI_WAVE_HI), idx(1)):
                L.launch(
                    sizes=[1, 1],
                    operands=list(_fa) + [_iv],
                    attributes={"air.preserve_shim_dma_order": L.UnitAttr.get()},
                )(launch_body)
                L.yield_([])

    return build()
