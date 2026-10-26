# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused prefill device: every op of a 128-token prefill chunk on one
configured device, laid out like FastFlowLM's fused_prefill.

  columns 1-6  GEMM, int4 (Q4NX) or bf16 weights
  other cols   attention (Config.attn); groups on one column share its cores

The herds loop forever and take their work from RTPs, and the memtile
channels are count-free rings, so one PDI serves every op and shape and the
ops differ only in their insts. build(own=True) emits one op's func, whose
segment holds only that op's herds; the funcs of all ops share segment @seg,
which aircc builds as one device with a runtime sequence per op. A dispatch
drives only its op's columns, so the other cores stay parked.

An op's rounds loop inside the cores and are the outer axis of its shim
patterns, so each op is a single launch.
"""

from typing import NamedTuple

from air import api as air
from air.api.types import bf16, f32, i8, i32

M = 128  # rows per chunk
KV_CAP = 1 << 16  # KV tokens an op with a run-time KV length can address
MM = 8  # aie2p mmul block

# GEMM: herd 6 x 4, core tile TM x TN, K step TK; a round is NR output columns
GCOL0, GNC = 1, 6
TM, TN, TK = 32, 64, 128
HR = M // TM
NR = TN * GNC
QB = TN * TK // 2 + 2 * 2 * (TK // 32) * TN  # bytes of one B packet
# bytes of the bfp16 A and B tiles of a K step (gemm_bfp16.cc)
A_BFP, B_BFP = TM * TK * 9 // 8, TN * TK * 9 // 8
MAX_N = 32 * NR  # widest GEMM op: more rounds no longer fit one shim BD


def k_pad(k):
    """A GEMM's depth on the device: whole groups of 4 K steps, which the
    cores take per loop trip (_gemm)."""
    return -(-k // (4 * TK)) * 4 * TK


def n_split(n):
    """[n0, n1) column ranges of a GEMM n wide, each at most MAX_N."""
    return [(n0, min(n, n0 + MAX_N)) for n0 in range(0, n, MAX_N)]


MM_OBJ, MM_SUF = "mm_m32n64.o", "_m32n64"
GEMM_OBJ = "gemm_bfp16.o"

NQT = 4  # cores per attention column


class Attn(NamedTuple):
    """One attention group, on column col, on attn_bfp16.cc: the host packs q,
    K and V as bfp16 and one call runs a K block. Groups on one column share
    its cores, so their per-core tiles must be the same size. lkp is both the
    K block and the q tile of a core. kv_heads > 1 is GQA: the op's heads are
    kv_heads groups of heads / kv_heads, and each KV head's records sit
    max_blocks blocks apart."""

    col: int
    dh: int
    lkp: int
    kv_heads: int = 1
    max_blocks: int = 0
    # take the running max off the scores before they are rounded to bf16,
    # for models whose q/k biases put scores far above the differences
    # between them (attn_bfp16.cc, ATTN_SPLIT_SCORES)
    split_scores: bool = False


# a window no prompt reaches, which makes the window mask causal
NO_WINDOW = 1 << 20


class Config(NamedTuple):
    attn: dict  # group name -> Attn
    act: str = "gelu"  # the GEMM's activation drain, f32_to_bf16_<act>_mn
    # attention op name -> the groups one dispatch drives, splitting the
    # op's heads evenly; by default each group is its own op
    ops: dict = None
    windows: dict = None  # attention op -> sliding window in tokens


def attn_ops(cfg):
    return cfg.ops or {a: (a,) for a in cfg.attn}


def attn_rows(g):
    """q rows of one attention round."""
    return NQT * g.lkp


def attn_qr(g):
    """rounds per head for one chunk."""
    return M // attn_rows(g)


def kv_rec(g):
    """bf16 elements of one K block's record: the K tile, then the V tile,
    both bfp16 (9 bytes per 8 values)."""
    return 2 * g.lkp * g.dh * 9 // 16


def sum_lanes(dh):
    """Floats the cores keep of each q row's running sum: the eight lanes the
    row sum fills, or one where L1 has no room for eight."""
    return 8 if dh <= 128 else 1


def q_tile(g):
    """bf16 elements of one core's q tile, bfp16."""
    return g.lkp * g.dh * 9 // 16


def _gemm(wave, A, B, C, k, act, wq, pair, rounds, kern, puts, rtps):
    ksteps = k // TK
    P = 1 + pair
    zero, mmul, mmul_pkt, cast, cast_act, dq = kern
    npk = 1 if wq else 2  # B packets per K step: int4, or bfp16 in K halves
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
                A[ty : ty + 1, 0:ksteps, 0:P, 0:A_BFP]
                .reshape(ksteps * P * A_BFP)
                .broadcast_to(rounds, ksteps * P * A_BFP),
                indices=[ty],
            )
        # One put per round and column, so a packet's start goes in the
        # buffer offset: as a BD stride, GNC packets of a long K overflow
        # the shim BD's stride field.
        Bp = B.reshape(rounds * GNC, L)
        for r in range(rounds):
            for tx in range(GNC):
                i = r * GNC + tx
                b_in.put(Bp[i : i + 1, 0:L], indices=[tx])
        for tx in range(GNC):
            c_out.get(
                C.reshape(P, M, rounds, GNC, TN)[
                    0:P, 0:M, 0:rounds, tx, 0:TN
                ].transpose(2, 0, 1, 3, 4),
                indices=[tx],
            )

    puts.append(shim)
    # computed in the launch, so the segment body does not depend on the shape
    kq = air.rtp(wave * 0 + (ksteps // 4) * wq * (1 - pair))
    kp = air.rtp(wave * 0 + (ksteps // 2) * wq * pair)
    kb = air.rtp(wave * 0 + (ksteps // 4) * (1 - wq) * (1 - pair))
    kbp = air.rtp(wave * 0 + (ksteps // 2) * (1 - wq) * pair)
    nr = air.rtp(wave * 0 + rounds)
    actp = air.rtp(wave * 0 + act)
    plain = air.rtp(wave * 0 + (1 - act))
    actp2 = air.rtp(wave * 0 + act * pair)
    plain2 = air.rtp(wave * 0 + (1 - act) * pair)
    rtps += [kq, kp, kb, kbp, nr, actp, plain, actp2, plain2]

    def body(seg):
        a_l2 = [
            [
                air.alloc([A_BFP], i8, scope=seg.private(), column=GCOL0 + ty)
                for _ in range(2)
            ]
            for ty in range(HR)
        ]
        for ty in range(HR):
            for pp in range(2):
                a_in.get(a_l2[ty][pp], indices=[ty])
            for pp in range(2):
                a2l1.put(a_l2[ty][pp], indices=[0, ty])
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
            params=[kq, kp, kb, kbp, nr, actp, plain, actp2, plain2],
        ) as hd:

            @hd.body
            def _(tx, ty):
                a_l1 = [air.alloc([A_BFP], i8, scope=hd.private()) for _ in range(4)]
                b_l1 = [air.alloc([QB], i8, scope=hd.private()) for _ in range(2)]
                bdq = air.alloc([B_BFP], i8, scope=hd.private())
                acc = air.alloc([TN // MM, TM // MM, MM, MM], f32, scope=hd.private())
                acc2 = air.alloc([TN // MM, TM // MM, MM, MM], f32, scope=hd.private())
                out = air.alloc([TN // MM, TM // MM, MM, MM], bf16, scope=hd.private())
                for _r in air.sequential(nr):
                    zero(acc)
                    zero(acc2)
                    # int4 packets are dequantized; bf16 weights come as bfp16
                    # in two packets per K step. With pair, each K step takes
                    # two chunks' A tiles against one B. Of the four loops all
                    # but one are zero-trip, and each trip gets the four A
                    # buffers in the same order, which is the BD chain
                    # air-to-aie builds (#2040).
                    for _k in air.sequential(kq):
                        for i in range(4):
                            a2l1.get(a_l1[i], indices=[tx, ty])
                            b2l1.get(b_l1[i % 2], indices=[tx, ty])
                            dq(b_l1[i % 2], bdq)
                            mmul(a_l1[i], bdq, acc)
                    for _k in air.sequential(kp):
                        for pp in range(2):
                            b2l1.get(b_l1[pp], indices=[tx, ty])
                            dq(b_l1[pp], bdq)
                            a2l1.get(a_l1[2 * pp], indices=[tx, ty])
                            mmul(a_l1[2 * pp], bdq, acc)
                            a2l1.get(a_l1[2 * pp + 1], indices=[tx, ty])
                            mmul(a_l1[2 * pp + 1], bdq, acc2)
                    for _k in air.sequential(kb):
                        for i in range(4):
                            a2l1.get(a_l1[i], indices=[tx, ty])
                            for h in range(2):
                                b2l1.get(b_l1[h], indices=[tx, ty])
                                mmul_pkt(a_l1[i], b_l1[h], acc, h)
                    for _k in air.sequential(kbp):
                        for pp in range(2):
                            a2l1.get(a_l1[2 * pp], indices=[tx, ty])
                            a2l1.get(a_l1[2 * pp + 1], indices=[tx, ty])
                            for h in range(2):
                                b2l1.get(b_l1[h], indices=[tx, ty])
                                mmul_pkt(a_l1[2 * pp], b_l1[h], acc, h)
                                mmul_pkt(a_l1[2 * pp + 1], b_l1[h], acc2, h)
                    # the activation in the drain when act is set
                    for _a in air.sequential(actp):
                        cast_act(acc, out)
                    for _p in air.sequential(plain):
                        cast(acc, out)
                    c2l2.put(out.transpose(1, 2, 0, 3), indices=[tx, ty])
                    for _a in air.sequential(actp2):
                        cast_act(acc2, out)
                        c2l2.put(out.transpose(1, 2, 0, 3), indices=[tx, ty])
                    for _p in air.sequential(plain2):
                        cast(acc2, out)
                        c2l2.put(out.transpose(1, 2, 0, 3), indices=[tx, ty])

        for tx in range(GNC):
            for ty in range(HR):
                c2l2.get(c_l2[tx][ty, 0:TM, 0:TN], indices=[tx, ty])
            c_out.put(c_l2[tx], indices=[tx])

    return body


def _rtp(wave, v):
    """An RTP from a build-time int, or from a run-time scalar argument."""
    return air.rtp(wave * 0 + v if isinstance(v, int) else v.value)


def _kv_put(kvin, KV, g, nkv, rounds):
    """nkv: an int, or a run-time scalar."""
    if not isinstance(nkv, int):
        return _kv_put_runtime(kvin, KV, g, nkv, rounds)
    n = nkv * kv_rec(g)
    if g.kv_heads == 1:
        kvin.put(KV[0:n].broadcast_to(rounds, n))
    else:
        # rounds walk the heads kv-head-major: each KV head's records
        # repeat for its group's rounds. Each head is its own put, so its
        # start goes in the buffer offset: as a BD stride, max_blocks
        # records overflow the shim BD's stride field at long max_len.
        h, span = g.kv_heads, g.max_blocks * kv_rec(g)
        for i in range(h):
            kvin.put(KV[i * span : i * span + n].broadcast_to(rounds // h, n))


def _kv_put_runtime(kvin, KV, g, nkv, rounds):
    """KV put where nkv is a run-time scalar: one shim transfer of nkv records
    per KV head, repeated `rounds` times. Built with ChannelPut directly since
    air.api slices only take static sizes."""
    from air.dialects.air import ChannelPut
    from air.dialects.arith import ConstantOp, IndexCastOp, MulIOp
    from air.ir import IndexType, IntegerAttr

    idx = IndexType.get()
    c = lambda v: ConstantOp(idx, IntegerAttr.get(idx, v)).result  # noqa: E731
    n = MulIOp(IndexCastOp(idx, nkv.value).result, c(kv_rec(g))).result
    h = g.kv_heads
    kvin._declare()
    ChannelPut(
        kvin.name,
        KV.value,
        offsets=[c(0), c(0), c(0)],
        sizes=[c(h), c(rounds // h), n],
        strides=[c(g.max_blocks * kv_rec(g)), c(0), c(1)],
        indices=kvin._indices(None, "put"),
    )


def _attention(name, gs, wave, puts, externs, rtps):
    """Attention on one column, one head per round, the column's cores
    splitting the round's q rows. gs: per group on the column (g, Q, KV, O,
    nkv, q0, k0, win, rounds); the cores run each group's rounds in turn, so
    groups of different ops share the column. The cores read nkv K blocks
    starting at block k0, two at a time; the chunk's first q block is q0 and
    the window win // lkp blocks. All are RTPs, nkv, q0 and k0 taken from the
    groups that have rounds. q, K and V arrive as bfp16 and the output goes
    back in the cores' 8x8 tile order, so the column's data movement is the
    same for every group on it."""
    g0 = gs[0][0]
    col = g0.col
    qt, kt, ot = q_tile(g0), kv_rec(g0) // 2, g0.lkp * g0.dh
    for g, *_ in gs:
        assert (q_tile(g), kv_rec(g) // 2, g.lkp * g.dh) == (qt, kt, ot), gs
    tqm = max(g.lkp for g, *_ in gs)
    lanes = max(sum_lanes(g.dh) for g, *_ in gs)
    lead = next((a for a in gs if a[-1]), gs[0])
    nkv, q0, k0 = lead[4:7]

    qin = air.channel(f"{name}QIn", size=[1])
    q2l1 = air.channel(f"{name}Q2L1", size=[NQT])
    kvin = air.channel(f"{name}KVIn", size=[1])
    kv2l1 = air.channel(f"{name}KV2L1", size=[1, 1], broadcast_shape=[1, NQT])
    gp2l2 = air.channel(f"{name}Gp2L2", size=[NQT])
    gpout = air.channel(f"{name}GpOut", size=[1])

    def shim(g, Q, KV, O, nkv, rounds):
        qin.put(Q[0 : rounds * NQT * qt])
        _kv_put(kvin, KV, g, nkv, rounds)
        gpout.get(O[0 : rounds * NQT * ot])

    for g, Q, KV, O, n, _, _, _, rounds in gs:
        if rounds:
            puts.append(lambda a=(g, Q, KV, O, n, rounds): shim(*a))
    if isinstance(nkv, int):
        assert nkv % 2 == 0, nkv
        nk2 = air.rtp(wave * 0 + nkv // 2)
    else:  # the host passes an even nkv
        from air.dialects.arith import ConstantOp, ShRUIOp
        from air.ir import IntegerAttr, IntegerType

        one = ConstantOp(
            IntegerType.get_signless(32),
            IntegerAttr.get(IntegerType.get_signless(32), 1),
        ).result
        nk2 = air.rtp(ShRUIOp(nkv.value, one).result)
    qb0 = _rtp(wave, q0)
    kb0 = _rtp(wave, k0)
    params = [nk2, qb0, kb0]
    loops = []
    for g, *_, win, rounds in gs:
        nr = air.rtp(wave * 0 + rounds)
        wb = air.rtp(wave * 0 + win // g.lkp)
        params += [nr, wb]
        suf = f"_d{g.dh}"
        obj = f"attn_bfp16{suf}.o"

        def ext(f, suf=suf, obj=obj, **kw):
            # groups sharing a head_dim share the kernel symbols
            if f + suf not in externs:
                externs[f + suf] = air.extern(f + suf, link_with=obj, **kw)
            return externs[f + suf]

        fns = (
            ext("attn_init"),
            ext("attn_blk", scalars=[i32, i32, i32]),
            ext("attn_fin"),
        )
        loops.append((g, nr, wb, fns))
    rtps += params

    def body(seg):
        q_l2 = air.alloc([NQT * qt], bf16, scope=seg.private(), column=col, split=False)
        k_l2 = air.alloc([kt], bf16, scope=seg.private(), column=col)
        v_l2 = air.alloc([kt], bf16, scope=seg.private(), column=col)
        gp_l2 = air.alloc(
            [NQT * ot], bf16, scope=seg.private(), column=col, split=False
        )
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
            params=params,
        ) as h:

            @h.body
            def _(tx, ty):
                q_bfp = air.alloc([qt], bf16, scope=h.private())
                # two K/V buffers, so a block's DMA overlaps the previous
                # block's compute
                k_l1 = [air.alloc([kt], bf16, scope=h.private()) for _ in "01"]
                v_l1 = [air.alloc([kt], bf16, scope=h.private()) for _ in "01"]
                gp = air.alloc([ot], bf16, scope=h.private())
                # the running max, padded to 64 bytes, then the running sums
                ml = air.alloc(
                    [-(-tqm // 32) * 16 + tqm * lanes], f32, scope=h.private()
                )
                for g, nr, wb, (init, blk_fn, fin) in loops:
                    QR = attn_qr(g)
                    # q block from the round index: a per-core counter updated
                    # in this loop crashes air-fuse-alloc-dealloc (#2041)
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
                        gp2l2.put(gp, indices=[ty])

        for r in range(NQT):
            gp2l2.get(gp_l2[r * ot : (r + 1) * ot], indices=[r])
        gpout.put(gp_l2)

    return body


def _kv_tensor(g, nkv):
    if g.kv_heads == 1:
        return air.tensor([nkv * kv_rec(g)], bf16)
    return air.tensor([g.kv_heads * g.max_blocks * kv_rec(g)], bf16)


def build(cfg, op, *shape, name="fused_prefill", own=False):
    """The module for one op; the other groups get minimal shapes.

    op 'g': (k, n, act, wq, pair), C[128 p, n] = A[128 p, k] @ W for p chunks
            (2 with pair, else 1), with the activation drain if act and int4
            weight packets if wq (bf16 otherwise)
    op <attention op>: (heads, nkv, q0, k0), attention for one chunk

    With own, the segment holds only the op's own groups: the device is then
    the union of every op's segment @seg (see air-to-aie), and the op's func
    drives only its groups' columns.
    """
    drive = ({"g"} if op == "g" else set(attn_ops(cfg)[op])) if own else None
    # with own, an attention op takes (nkv, q0, k0) at run time
    runtime = own and op != "g"
    g = dict(k=4 * TK, n=NR, act=0, wq=1, pair=0, rounds=1)
    at = {
        a: dict(
            nkv=2,
            q0=0,
            k0=0,
            win=NO_WINDOW,
            rounds=0,
        )
        for a, ga in cfg.attn.items()
    }
    if op == "g":
        k, n, act, wq, pair = shape
        assert n % NR == 0, n
        g.update(k=k_pad(k), n=n, act=act, wq=wq, pair=pair, rounds=n // NR)
    else:
        heads, nkv, q0, k0 = shape
        groups = attn_ops(cfg)[op]
        hg = heads // len(groups)
        for a in groups:
            ga = cfg.attn[a]
            assert hg * len(groups) == heads and hg % ga.kv_heads == 0, (heads, ga)
            win = (cfg.windows or {}).get(op) or NO_WINDOW
            at[a].update(nkv=nkv, q0=q0, k0=k0, win=win, rounds=hg * attn_qr(ga))

    mm = [
        air.extern(f"{f}{MM_SUF}", link_with=MM_OBJ)
        for f in ("zero_f32_mn", "f32_to_bf16_mn", f"f32_to_bf16_{cfg.act}_mn")
    ]
    kern = (
        mm[0],
        air.extern("mm_bfp16", link_with=GEMM_OBJ),
        air.extern("mm_bfp16_pkt", link_with=GEMM_OBJ, scalars=[i32]),
        mm[1],
        mm[2],
        air.extern("dq4_bfp16", link_with=GEMM_OBJ),
    )
    gt = (
        air.tensor([HR, g["k"] // TK, 1 + g["pair"], A_BFP], i8),
        air.tensor([g["n"] // TN, g["k"] // TK * (1 if g["wq"] else 2), QB], i8),
        air.tensor([(1 + g["pair"]) * M, g["n"]], bf16),
    )
    att = {}
    for a, ga in cfg.attn.items():
        nr = max(1, at[a]["rounds"])
        o = nr * attn_rows(ga) * ga.dh
        q = nr * NQT * q_tile(ga)
        att[a] = (
            air.tensor([q], bf16),
            _kv_tensor(ga, KV_CAP // ga.lkp if runtime else at[a]["nkv"]),
            air.tensor([o], bf16),
        )
    cols = {}
    for a, ga in cfg.attn.items():
        cols.setdefault(ga.col, []).append(a)
    if runtime:
        scalars = {n: air.tensor([], i32, name=n) for n in ("nkv", "q0", "k0")}
        for a in drive:
            at[a].update(scalars)

    with air.launch(repeat=1, name=name, attrs=["air.preserve_shim_dma_order"]) as lch:

        @lch.body
        def _(wave):
            puts, externs, rtps = [], {}, []
            bodies = []
            if drive is None or "g" in drive:
                bodies.append(
                    _gemm(
                        wave,
                        *gt,
                        g["k"],
                        g["act"],
                        g["wq"],
                        g["pair"],
                        g["rounds"],
                        kern,
                        puts,
                        rtps,
                    )
                )
            for names in cols.values():
                if drive is not None and not drive & set(names):
                    continue
                gs = [
                    (cfg.attn[a], *att[a])
                    + tuple(at[a][k] for k in ("nkv", "q0", "k0", "win", "rounds"))
                    for a in names
                ]
                bodies.append(_attention("".join(names), gs, wave, puts, externs, rtps))
            for p in puts:
                p()
            with air.segment(name="seg", params=rtps) as seg:

                @seg.body
                def _():
                    for b in bodies:
                        b(seg)

    return lch
