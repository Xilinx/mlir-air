# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused prefill device: every op of a 128-token prefill chunk on one
configured device, laid out like FastFlowLM's fused_prefill.

  columns 1-6  GEMM, int4 (Q4NX) or bf16 weights
  other cols   attention, one herd group per column (Config.attn)

The herds loop forever and take their work from RTPs, and the memtile
channels are count-free rings, so one PDI serves every op and shape and the
ops differ only in their insts. build() emits the insts of one op;
insts.strip_columns then drops the other groups' columns, so those cores stay
parked.

An op's rounds loop inside the cores and are the outer axis of its shim
patterns, so each op is a single launch.
"""

from typing import NamedTuple

from air import api as air
from air.api.types import bf16, f32, i8, i32

M = 128  # rows per chunk
MM = 8  # aie2p mmul block

# GEMM: herd 6 x 4, core tile TM x TN, K step TK; a round is NR output columns
GCOL0, GNC = 1, 6
TM, TN, TK = 32, 64, 128
HR = M // TM
NR = TN * GNC
QB = TN * TK // 2 + 2 * 2 * (TK // 32) * TN  # bytes of one B packet
MAX_N = 32 * NR  # widest GEMM op: more rounds no longer fit one shim BD


def n_split(n):
    """[n0, n1) column ranges of a GEMM n wide, each at most MAX_N."""
    return [(n0, min(n, n0 + MAX_N)) for n0 in range(0, n, MAX_N)]


MM_OBJ, MM_SUF = "mm_m32n64.o", "_m32n64"
DQ_OBJ = "dq4.o"

NQT = 4  # cores per attention column


class Attn(NamedTuple):
    """One attention column. lkp is both the K block and the q tile of a
    core. kv_heads > 1 is GQA: the op's heads are kv_heads groups of
    heads / kv_heads, and each KV head's records sit max_blocks blocks apart."""

    col: int
    dh: int
    lkp: int
    dvt: int
    window: int = 0  # sliding-window mask when nonzero, else causal
    kv_heads: int = 1
    max_blocks: int = 0
    # the window mask with its width an RTP, so ops differing only in the
    # window (Config.windows) share one core program
    window_rtp: bool = False
    # "bfp16": attn_bfp16.cc, the host packing q, K and V as bfp16 and one
    # call per K block (dh <= 128, dvt == dh); "npu2": attn_npu2.cc
    kern: str = "npu2"


# a window no prompt reaches, which makes the window mask causal
NO_WINDOW = 1 << 20


class Config(NamedTuple):
    attn: dict  # group name -> Attn
    act: str = "gelu"  # the GEMM's activation drain, f32_to_bf16_<act>_mn
    # attention op name -> the groups one dispatch drives, splitting the
    # op's heads evenly; by default each group is its own op
    ops: dict = None
    windows: dict = None  # attention op -> window in tokens, for window_rtp groups


def attn_ops(cfg):
    return cfg.ops or {a: (a,) for a in cfg.attn}


def cols(cfg):
    """op name -> the columns its dispatches keep."""
    out = {"g": set(range(GCOL0, GCOL0 + GNC))}
    out.update({op: {cfg.attn[a].col for a in gs} for op, gs in attn_ops(cfg).items()})
    return out


def attn_rows(g):
    """q rows of one attention round."""
    return NQT * g.lkp


def attn_qr(g):
    """rounds per head for one chunk."""
    return M // attn_rows(g)


def kv_rec(g):
    """bf16 elements of one K block's record: K tile, then the V tiles (as
    bfp16, 9 bytes per 8 values, for attn_bfp16)."""
    n = 2 * g.lkp * g.dh
    return n * 9 // 16 if g.kern == "bfp16" else n


def q_tile(g):
    """bf16 elements of one core's q tile as attn_bfp16 takes it (bfp16)."""
    return g.lkp * g.dh * 9 // 16


def _gemm(wave, A, B, C, k, act, wq, rounds, kern, puts):
    ksteps = k // TK
    zero, mmul, cast, cast_act, dq, cp = kern
    npk = 1 if wq else 4  # B packets per K step: int4, or bf16 in 32-K slices
    a_in = air.channel("gAIn", size=[HR])
    b_in = air.channel("gBIn", size=[GNC])
    a2l1 = air.channel("gA2L1", size=[1, HR], broadcast_shape=[GNC, HR])
    b2l1 = air.channel("gB2L1", size=[GNC, 1], broadcast_shape=[GNC, HR])
    c2l2 = air.channel("gC2L2", size=[GNC, HR])
    c_out = air.channel("gCOut", size=[GNC])

    def shim():
        L = ksteps * QB * npk
        for ty in range(HR):
            a_in.put(
                A[ty : ty + 1, 0:ksteps, 0:TM, 0:TK]
                .reshape(ksteps * TM * TK)
                .broadcast_to(rounds, ksteps * TM * TK),
                indices=[ty],
            )
        for tx in range(GNC):
            b_in.put(B.reshape(rounds, GNC, L)[0:rounds, tx, 0:L], indices=[tx])
        for tx in range(GNC):
            c_out.get(
                C.reshape(M, rounds, GNC, TN)[0:M, 0:rounds, tx, 0:TN].transpose(
                    1, 0, 2, 3
                ),
                indices=[tx],
            )

    puts.append(shim)

    def body(seg):
        kq = air.rtp(wave * 0 + (ksteps // 2) * wq)
        kb = air.rtp(wave * 0 + (ksteps // 2) * (1 - wq))
        nr = air.rtp(wave * 0 + rounds)
        actp = air.rtp(wave * 0 + act)
        plain = air.rtp(wave * 0 + (1 - act))

        def blk(v, r, c):
            return v.reshape(r // MM, MM, c // MM, MM).transpose(2, 0, 1, 3)

        a_l2 = [
            [
                air.alloc([TM, TK], bf16, scope=seg.private(), column=GCOL0 + ty)
                for _ in range(2)
            ]
            for ty in range(HR)
        ]
        for ty in range(HR):
            for pp in range(2):
                a_in.get(a_l2[ty][pp], indices=[ty])
            for pp in range(2):
                a2l1.put(blk(a_l2[ty][pp], TM, TK), indices=[0, ty])
        b_l2 = [
            [
                air.alloc([QB], i8, scope=seg.private(), column=GCOL0 + tx)
                for _ in range(2)
            ]
            for tx in range(GNC)
        ]
        c_l2 = [
            air.alloc(
                [HR, TM, TN], bf16, scope=seg.private(), column=GCOL0 + tx, split=False
            )
            for tx in range(GNC)
        ]
        for tx in range(GNC):
            for pp in range(2):
                b_in.get(b_l2[tx][pp], indices=[tx])
            for pp in range(2):
                b2l1.put(b_l2[tx][pp], indices=[tx, 0])

        with air.herd(
            [range(GNC), range(HR)],
            name="gemm",
            shape=(GNC, HR),
            link_with=MM_OBJ,
            at=(GCOL0, 2),
            params=[kq, kb, nr, actp, plain],
        ) as hd:

            @hd.body
            def _(tx, ty):
                a_l1 = [
                    air.alloc([TK // MM, TM // MM, MM, MM], bf16, scope=hd.private())
                    for _ in range(2)
                ]
                b_l1 = [air.alloc([QB], i8, scope=hd.private()) for _ in range(2)]
                bdq = air.alloc([TN // MM, TK // MM, MM, MM], bf16, scope=hd.private())
                acc = air.alloc([TN // MM, TM // MM, MM, MM], f32, scope=hd.private())
                out = air.alloc([TN // MM, TM // MM, MM, MM], bf16, scope=hd.private())
                for _r in air.sequential(nr):
                    zero(acc)
                    # int4 packets are dequantized; bf16 packets (four per K
                    # step) are copied. Two loops, one of them zero-trip, so
                    # the gets alternate the two buffers in program order,
                    # which is the BD chain air-to-aie builds (#2040).
                    for _k in air.sequential(kq):
                        for pp in range(2):
                            a2l1.get(a_l1[pp], indices=[tx, ty])
                            b2l1.get(b_l1[pp], indices=[tx, ty])
                            dq(b_l1[pp], bdq)
                            mmul(a_l1[pp], bdq, acc)
                    for _k in air.sequential(kb):
                        for pp in range(2):
                            a2l1.get(a_l1[pp], indices=[tx, ty])
                            for j in range(4):
                                b2l1.get(b_l1[j % 2], indices=[tx, ty])
                                cp(b_l1[j % 2], bdq, j)
                            mmul(a_l1[pp], bdq, acc)
                    # the activation in the drain when act is set
                    for _a in air.sequential(actp):
                        cast_act(acc, out)
                    for _p in air.sequential(plain):
                        cast(acc, out)
                    c2l2.put(out.transpose(1, 2, 0, 3), indices=[tx, ty])

        for tx in range(GNC):
            for ty in range(HR):
                c2l2.get(c_l2[tx][ty, 0:TM, 0:TN], indices=[tx, ty])
            c_out.put(c_l2[tx], indices=[tx])

    return body


def _kv_put(kvin, KV, g, nkv, rounds):
    n = nkv * kv_rec(g)
    if g.kv_heads == 1:
        kvin.put(KV[0:n].broadcast_to(rounds, n))
    else:
        # rounds walk the heads kv-head-major: each KV head's records
        # repeat for its group's rounds
        h = g.kv_heads
        kvin.put(
            KV.reshape(h, 1, g.max_blocks * kv_rec(g))[0:h, 0:1, 0:n].broadcast_to(
                h, rounds // h, n
            )
        )


def _attention(name, g, wave, Q, KV, O, nkv, q0, k0, win, rounds, puts, externs):
    """Attention, one head per round, the column's cores splitting the
    round's q rows. The cores read nkv K blocks starting at block k0; the
    chunk's first q block is q0. Both are RTPs, in units of lkp rows."""
    col, dh, lkp, dvt = g.col, g.dh, g.lkp, g.dvt
    tq, R, QR = lkp, attn_rows(g), attn_qr(g)
    dvc = dh // dvt
    suf = f"_d{dh}"
    obj = f"attn{suf}.o"

    def ext(f, **kw):
        # groups sharing a head_dim share the kernel symbols
        if f + suf not in externs:
            externs[f + suf] = air.extern(f + suf, link_with=obj, **kw)
        return externs[f + suf]

    zero_g, zero_gp, zero_sp = (
        ext("zero_fill_g_bf16"),
        ext("zero_fill_gp_bf16"),
        ext("zero_fill_sp_bf16"),
    )
    ninf_up = ext("neg_inf_fill_up_bf16")
    mm_ab, mm_gb = ext("matmul_a_b_bf16"), ext("matmul_g_b_bf16")
    softmax, mul_r_gp = ext("fused_softmax"), ext("mul_r_gp")
    accum, vcopy = ext("accum_sp_r_s"), ext("vector_copy_32elems", scalars=[i32])
    div = ext("div_gp_sp")
    if g.window_rtp:
        wmask = ext("apply_window_mask", scalars=[i32, i32, i32])
        mask = lambda s, qb, kb, w: wmask(s, qb, kb, w)  # noqa: E731
    elif g.window:
        wmask = ext("apply_window_mask", scalars=[i32, i32, i32])
        mask = lambda s, qb, kb, w: wmask(s, qb, kb, g.window // lkp)  # noqa: E731
    else:
        causal = ext("apply_causal_mask", scalars=[i32, i32])
        mask = lambda s, qb, kb, w: causal(s, qb, kb)  # noqa: E731

    qin = air.channel(f"{name}QIn", size=[1])
    q2l1 = air.channel(f"{name}Q2L1", size=[NQT])
    kvin = air.channel(f"{name}KVIn", size=[1])
    kv2l1 = air.channel(f"{name}KV2L1", size=[1, 1], broadcast_shape=[1, NQT])
    gp2l2 = air.channel(f"{name}Gp2L2", size=[NQT])
    gpout = air.channel(f"{name}GpOut", size=[1])

    def shim():
        qin.put(Q[0 : rounds * R, 0:dh])
        _kv_put(kvin, KV, g, nkv, rounds)
        gpout.get(O[0 : rounds * R, 0:dh])

    puts.append(shim)

    def body(seg):
        nk = air.rtp(wave * 0 + nkv)
        nr = air.rtp(wave * 0 + rounds)
        qb0 = air.rtp(wave * 0 + q0)
        kb0 = air.rtp(wave * 0 + k0)
        params = [nk, nr, qb0, kb0]
        wb = None
        if g.window_rtp:
            wb = air.rtp(wave * 0 + win // lkp)
            params.append(wb)
        q_l2 = air.alloc([R, dh], bf16, scope=seg.private(), column=col, split=False)
        k_l2 = air.alloc([lkp, dh], bf16, scope=seg.private(), column=col)
        v_l2 = [
            air.alloc([lkp, dvt], bf16, scope=seg.private(), column=col)
            for _ in range(dvc)
        ]
        gp_l2 = air.alloc([R, dh], bf16, scope=seg.private(), column=col, split=False)
        qin.get(q_l2)
        for r in range(NQT):
            q2l1.put(
                q_l2[r * tq : r * tq + tq, 0:dh]
                .reshape(tq // MM, MM, dh // MM, MM)
                .transpose(2, 0, 1, 3),
                indices=[r],
            )
        kvin.get(k_l2)
        kv2l1.put(
            k_l2.reshape(lkp // MM, MM, dh // MM, MM).transpose(2, 0, 1, 3),
            indices=[0, 0],
        )
        for z in range(dvc):
            kvin.get(v_l2[z])
            kv2l1.put(
                v_l2[z].reshape(lkp // MM, MM, dvt // MM, MM).transpose(2, 0, 1, 3),
                indices=[0, 0],
            )

        with air.herd(
            [range(1), range(NQT)],
            name=f"{name}attn",
            shape=(1, NQT),
            at=(col, 2),
            link_with=obj,
            params=params,
        ) as h:

            @h.body
            def _(tx, ty):
                q_saved = air.alloc([tq, dh], bf16, scope=h.private())
                qk = air.alloc([lkp, dh], bf16, scope=h.private())
                v_l1 = air.alloc([lkp, dvt], bf16, scope=h.private())
                s = air.alloc([tq, lkp], bf16, scope=h.private())
                gp = [air.alloc([tq, dvt], bf16, scope=h.private()) for _ in range(dvc)]
                up = air.alloc([tq, 1], bf16, scope=h.private())
                sp = air.alloc([tq, 1], bf16, scope=h.private())
                s_tmp = air.alloc([tq, 1], bf16, scope=h.private())
                r_tmp = air.alloc([tq, 1], bf16, scope=h.private())
                # q block from the round index: a per-core counter updated
                # in this loop crashes air-fuse-alloc-dealloc (#2041)
                for r in air.sequential(nr):
                    for z in range(dvc):
                        zero_gp(gp[z])
                    zero_sp(sp)
                    ninf_up(up)
                    qb = qb0 + (r % QR) * NQT + ty
                    q2l1.get(q_saved, indices=[ty])
                    for blk in air.sequential(nk):
                        zero_g(s.reshape(tq * lkp))
                        kv2l1.get(qk, indices=[tx, ty])
                        mm_ab(q_saved, qk, s.reshape(tq * lkp))
                        mask(s, qb, kb0 + blk, wb)
                        softmax(s.reshape(tq * lkp), up, s_tmp, r_tmp)
                        for z in range(dvc):
                            kv2l1.get(v_l1, indices=[tx, ty])
                            mul_r_gp(r_tmp, gp[z])
                            mm_gb(s.reshape(tq * lkp), v_l1, gp[z])
                        accum(sp, r_tmp, s_tmp)
                        vcopy(0, s_tmp, sp)
                    for z in range(dvc):
                        div(sp, gp[z])
                    for z in range(dvc):
                        gp2l2.put(
                            gp[z]
                            .reshape(dvt // MM, tq // MM, MM, MM)
                            .transpose(1, 2, 0, 3),
                            indices=[ty],
                        )

        for r in range(NQT):
            for z in range(dvc):
                gp2l2.get(
                    gp_l2[r * tq : r * tq + tq, z * dvt : z * dvt + dvt], indices=[r]
                )
        gpout.put(gp_l2)

    return body


def _attention_bfp16(name, g, wave, Q, KV, O, nkv, q0, k0, win, rounds, puts, externs):
    """_attention on attn_bfp16.cc: q, K and V arrive as bfp16 (the host packs
    them), K and V of a block land together and one call runs the block."""
    col, dh, lkp = g.col, g.dh, g.lkp
    tq, R, QR = lkp, attn_rows(g), attn_qr(g)
    assert g.dvt == dh, g
    suf = f"_d{dh}"
    obj = f"attn_bfp16{suf}.o"

    def ext(f, **kw):
        if f + suf not in externs:
            externs[f + suf] = air.extern(f + suf, link_with=obj, **kw)
        return externs[f + suf]

    init, fin = ext("attn_init"), ext("attn_fin")
    blk_fn = ext("attn_blk", scalars=[i32, i32, i32])

    qin = air.channel(f"{name}QIn", size=[1])
    q2l1 = air.channel(f"{name}Q2L1", size=[NQT])
    kvin = air.channel(f"{name}KVIn", size=[1])
    kv2l1 = air.channel(f"{name}KV2L1", size=[1, 1], broadcast_shape=[1, NQT])
    gp2l2 = air.channel(f"{name}Gp2L2", size=[NQT])
    gpout = air.channel(f"{name}GpOut", size=[1])

    def shim():
        qin.put(Q[0 : rounds * NQT * q_tile(g)])
        _kv_put(kvin, KV, g, nkv, rounds)
        gpout.get(O[0 : rounds * R, 0:dh])

    puts.append(shim)

    def body(seg):
        assert nkv % 2 == 0, nkv
        nk2 = air.rtp(wave * 0 + nkv // 2)
        nr = air.rtp(wave * 0 + rounds)
        qb0 = air.rtp(wave * 0 + q0)
        kb0 = air.rtp(wave * 0 + k0)
        wb = air.rtp(wave * 0 + win // lkp)
        qt, kt = q_tile(g), kv_rec(g) // 2
        q_l2 = air.alloc([NQT * qt], bf16, scope=seg.private(), column=col, split=False)
        k_l2 = air.alloc([kt], bf16, scope=seg.private(), column=col)
        v_l2 = air.alloc([kt], bf16, scope=seg.private(), column=col)
        gp_l2 = air.alloc([R, dh], bf16, scope=seg.private(), column=col, split=False)
        qin.get(q_l2)
        for r in range(NQT):
            q2l1.put(q_l2[r * qt : r * qt + qt], indices=[r])
        for t in (k_l2, v_l2):
            kvin.get(t)
            kv2l1.put(t, indices=[0, 0])

        with air.herd(
            [range(1), range(NQT)],
            name=f"{name}attn",
            shape=(1, NQT),
            at=(col, 2),
            link_with=obj,
            params=[nk2, nr, qb0, kb0, wb],
        ) as h:

            @h.body
            def _(tx, ty):
                q_bfp = air.alloc([qt], bf16, scope=h.private())
                # two K/V buffers, so a block's DMA overlaps the previous
                # block's compute
                k_l1 = [air.alloc([kt], bf16, scope=h.private()) for _ in "01"]
                v_l1 = [air.alloc([kt], bf16, scope=h.private()) for _ in "01"]
                gp = air.alloc([tq, dh], bf16, scope=h.private())
                ml = air.alloc([tq // 2 + tq * 8], f32, scope=h.private())
                for r in air.sequential(nr):
                    init(gp, ml)
                    qb = qb0 + (r % QR) * NQT + ty
                    q2l1.get(q_bfp, indices=[ty])
                    for blk in air.sequential(nk2):
                        for pp in range(2):
                            kv2l1.get(k_l1[pp], indices=[tx, ty])
                            kv2l1.get(v_l1[pp], indices=[tx, ty])
                            kb = kb0 + blk * 2 + pp
                            blk_fn(q_bfp, k_l1[pp], v_l1[pp], gp, ml, qb, kb, wb)
                    fin(gp, ml)
                    gp2l2.put(
                        gp.reshape(dh // MM, tq // MM, MM, MM).transpose(1, 2, 0, 3),
                        indices=[ty],
                    )

        for r in range(NQT):
            gp2l2.get(gp_l2[r * tq : r * tq + tq, 0:dh], indices=[r])
        gpout.put(gp_l2)

    return body


def _kv_tensor(g, nkv):
    if g.kv_heads == 1:
        return air.tensor([nkv * kv_rec(g)], bf16)
    return air.tensor([g.kv_heads * g.max_blocks * kv_rec(g)], bf16)


def build(cfg, op, *shape):
    """The module for one op; the other groups get minimal shapes.

    op 'g': (k, n, act, wq), C[128, n] = A[128, k] @ W, with the activation
            drain if act and int4 weight packets if wq (bf16 otherwise)
    op <attention op>: (heads, nkv, q0, k0), attention for one chunk
    """
    g = dict(k=2 * TK, n=NR, act=0, wq=1, rounds=1)
    at = {
        a: dict(
            nkv=2 if ga.kern == "bfp16" else 1,
            q0=0,
            k0=0,
            win=NO_WINDOW,
            rounds=ga.kv_heads,
        )
        for a, ga in cfg.attn.items()
    }
    if op == "g":
        k, n, act, wq = shape
        assert k % (2 * TK) == 0 and n % NR == 0, (k, n)
        g.update(k=k, n=n, act=act, wq=wq, rounds=n // NR)
    else:
        heads, nkv, q0, k0 = shape
        groups = attn_ops(cfg)[op]
        hg = heads // len(groups)
        for a in groups:
            ga = cfg.attn[a]
            assert hg * len(groups) == heads and hg % ga.kv_heads == 0, (heads, ga)
            win = (cfg.windows or {}).get(op) or NO_WINDOW
            at[a].update(nkv=nkv, q0=q0, k0=k0, win=win, rounds=hg * attn_qr(ga))

    kern = tuple(
        air.extern(f"{f}{MM_SUF}", link_with=MM_OBJ)
        for f in (
            "zero_f32_mn",
            "op_has_no_registered_library_name",
            "f32_to_bf16_mn",
            f"f32_to_bf16_{cfg.act}_mn",
        )
    ) + (
        air.extern("dq4_bf16", link_with=DQ_OBJ),
        air.extern("bf16_pkt", link_with=DQ_OBJ, scalars=[i32]),
    )
    gt = (
        air.tensor([HR, g["k"] // TK, TM, TK], bf16),
        air.tensor([g["n"] // TN, g["k"] // TK * (1 if g["wq"] else 4), QB], i8),
        air.tensor([M, g["n"]], bf16),
    )
    att = {
        a: (
            (
                air.tensor([at[a]["rounds"] * NQT * q_tile(ga)], bf16)
                if ga.kern == "bfp16"
                else air.tensor([at[a]["rounds"] * attn_rows(ga), ga.dh], bf16)
            ),
            _kv_tensor(ga, at[a]["nkv"]),
            air.tensor([at[a]["rounds"] * attn_rows(ga), ga.dh], bf16),
        )
        for a, ga in cfg.attn.items()
    }

    with air.launch(
        repeat=1, name="fused_prefill", attrs=["air.preserve_shim_dma_order"]
    ) as lch:

        @lch.body
        def _(wave):
            puts, externs = [], {}
            bodies = [
                _gemm(wave, *gt, g["k"], g["act"], g["wq"], g["rounds"], kern, puts)
            ]
            for a, ga in cfg.attn.items():
                p = at[a]
                fn = _attention_bfp16 if ga.kern == "bfp16" else _attention
                bodies.append(
                    fn(
                        a,
                        ga,
                        wave,
                        *att[a],
                        p["nkv"],
                        p["q0"],
                        p["k0"],
                        p["win"],
                        p["rounds"],
                        puts,
                        externs,
                    )
                )
            for p in puts:
                p()
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    for b in bodies:
                        b(seg)

    return lch
