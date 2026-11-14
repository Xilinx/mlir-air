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
from air.api import ops
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


# a Q4_K block: 32 output columns by 256 k, as the bundle stores it and as the
# device reads it (hostops.q4k_prefill)
Q4K_COLS, Q4K_BYTES, Q4K_PF_BYTES = 256, 4736, 5120

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
    # every query sees every key before the op's run-time key count, which
    # the op takes as a fourth scalar (attn_bfp16.cc, ATTN_NONCAUSAL)
    noncausal: bool = False
    # each head's softmax has one more logit, with no value: the host writes
    # it after each core's q tile (SINK, attn_bfp16.cc ATTN_SINKS)
    sinks: bool = False


# a window no prompt reaches, which makes the window mask causal
NO_WINDOW = 1 << 20


class Config(NamedTuple):
    attn: dict  # group name -> Attn
    act: str = "gelu"  # the GEMM's activation drain, f32_to_bf16_<act>_mn
    # attention op name -> the groups one dispatch drives, splitting the
    # op's heads evenly; by default each group is its own op
    ops: dict = None
    windows: dict = None  # attention op -> sliding window in tokens
    # GEMMs whose weights are MXFP4 blocks as the gpt-oss bundle stores them
    # (wq=2, gemm_bfp16.cc dqmx_bfp16), read where they lie
    mxfp4: bool = False
    # every GEMM's weights are FastFlowLM's Q4_K blocks (wq=3, gemm_bfp16.cc
    # dq4k_bfp16) in the prefill's layout of them (hostops.q4k_prefill); a B
    # packet is then one block, so the device has no int4 or bf16 GEMMs
    q4k: bool = False
    # MoE expert GEMMs as one op over a run of blocks (Experts); see _gemm
    experts: "Experts" = None


class Experts(NamedTuple):
    """An "x" op runs one GEMM shape on up to kmax blocks of an expert's rows,
    a block per launch wave. A block is a pair of chunks or one chunk, each
    from its own buffer of cap blocks; its weights are those of expert e of a
    layer, stride rows of B past expert e - 1 (a B row is one column block of
    four Q4_K blocks, or a pair of MXFP4 ones)."""

    kmax: int
    cap: int
    stride: int
    experts: int


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


def attn_suffix(g):
    """The kernel symbols' and object's suffix of group g."""
    return f"_d{g.dh}" + ("n" if g.noncausal else "") + ("k" if g.sinks else "")


def sum_lanes(dh):
    """Floats the cores keep of each q row's running sum: the eight lanes the
    row sum fills, or one where L1 has no room for eight."""
    return 8 if dh <= 128 else 1


# bf16 elements after a core's q tile holding its head's sink: the logit as
# a score (bf16), then its weight against that score (f32), padded to 64 bytes
SINK = 32


def q_tile(g):
    """bf16 elements of one core's q tile, bfp16, and its sink if any."""
    return g.lkp * g.dh * 9 // 16 + (SINK if g.sinks else 0)


def _gemm(
    wave, A, B, C, k, act, wq, pair, rounds, kern, puts, rtps, mx, q4, ncb, blk=None
):
    """wq: 0 bf16 weights, 1 int4, 2 MXFP4 (a device built with mx), 3 Q4_K
    (a device built with q4). An MXFP4 weight is row groups of ncb blocks
    pairs: two 32-row blocks per K step make a packet, which is how the bundle
    lays out a 64-column panel's steps. A Q4_K weight is row groups of ncb
    column blocks of four 32-row blocks: a panel's two blocks of a column
    block are two packets, which cover two K steps.

    blk runs it on a run of expert blocks (Experts), one per launch wave:
    waves below blk["n1"] are pairs, A and C blocks i1 + wave of blk["A"][1]
    and blk["C"][1], the rest single chunks, blocks i0 + wave - n1 of
    blk["A"][0] and blk["C"][0]; the wave's B is blk["e"] rows into B. The
    pair RTPs are picked per wave, so the cores run both kinds."""
    ksteps = k // TK
    P = 1 + pair
    qb = Q4K_PF_BYTES if q4 else QB
    zero, mmul, mmul_pkt, cast, cast_act, dq, dqmx, dq4k = kern
    npk = 2 if wq == 0 else 1  # B packets per K step: bfp16 in K halves, else 1
    a_in = air.channel("gAIn", size=[HR])
    b_in = air.channel("gBIn", size=[GNC])
    a2l1 = air.channel("gA2L1", size=[1, HR], broadcast_shape=[GNC, HR])
    b2l1 = air.channel("gB2L1", size=[GNC, 1], broadcast_shape=[GNC, HR])
    c2l2 = air.channel("gC2L2", size=[GNC, HR])
    c_out = air.channel("gCOut", size=[GNC])

    def a_whole(ty, P):
        return (
            A[ty : ty + 1, 0:ksteps, 0:P, 0:A_BFP]
            .reshape(ksteps * P * A_BFP)
            .broadcast_to(rounds, ksteps * P * A_BFP)
        )

    def c_whole(tx, P):
        return C.reshape(P, M, rounds, GNC, TN)[0:P, 0:M, 0:rounds, tx, 0:TN].transpose(
            2, 0, 1, 3, 4
        )

    def shim():
        if blk is None:
            feeds(a_whole, c_whole, P, 0)
            return
        # block i of its kind's A and C buffers
        n1, i1, i0 = blk["n1"], blk["i1"], blk["i0"]
        blocks = {
            q: (
                lambda ty, P, Ab=blk["A"][q], ib=ib: Ab[
                    ib, ty : ty + 1, 0:ksteps, 0:P, 0:A_BFP
                ].broadcast_to(rounds, 1, 1, ksteps, P, A_BFP),
                lambda tx, P, Cb=blk["C"][q], ib=ib: Cb[
                    ib, 0:P, 0:M, 0:rounds, tx, 0:TN
                ].transpose(0, 3, 1, 2, 4, 5),
            )
            for q, ib in ((1, i1 + wave), (0, i0 + wave - n1))
        }
        with ops.branch(blk["pair"]) as arm:
            feeds(*blocks[1], 2, blk["e"])
        with arm.otherwise():
            feeds(*blocks[0], 1, blk["e"])

    def feeds(a_of, c_of, P, b0):
        """A, B and C of one block: a_of(ty, P) and c_of(tx, P) are its A put
        and C get, b0 its weights' first row of B."""
        L = ksteps * qb * npk
        for ty in range(HR):
            a_in.put(a_of(ty, P), indices=[ty])
        # One put per round and column, so a packet's start goes in the
        # buffer offset: as a BD stride, GNC packets of a long K overflow
        # the shim BD's stride field.
        if wq == 2:
            # panel p is half p % 2 of row group p // 2, its K steps a block
            # pair apart; the steps past ncb read the next group's, against
            # A's zero columns
            for r in range(rounds):
                for tx in range(GNC):
                    p = r * GNC + tx
                    g0 = b0 + p // 2 * ncb
                    h = p % 2
                    b_in.put(B[g0 : g0 + ksteps, h : h + 1, 0:QB], indices=[tx])
        elif wq == 3:
            # panel p is blocks 2 (p % 2) and the next of row group p // 2
            for r in range(rounds):
                for tx in range(GNC):
                    p = r * GNC + tx
                    g0 = b0 + p // 2 * ncb
                    s = 2 * (p % 2)
                    b_in.put(B[g0 : g0 + ksteps // 2, s : s + 2, 0:qb], indices=[tx])
        else:
            assert blk is None, "expert blocks take Q4_K or MXFP4 weights"
            Bp = B.reshape(rounds * GNC, L)
            for r in range(rounds):
                for tx in range(GNC):
                    i = r * GNC + tx
                    b_in.put(Bp[i : i + 1, 0:L], indices=[tx])
        for tx in range(GNC):
            c_out.get(c_of(tx, P), indices=[tx])

    puts.append(shim)
    # computed in the launch, so the segment body does not depend on the shape
    w4, wb, wm = (int(wq == v) for v in (1, 0, 2))

    def by_pair(f):
        """The RTP f(pair); with blk, f(1) on a pair's wave, f(0) on a single
        chunk's."""
        if blk is None:
            return air.rtp(wave * 0 + f(pair))
        from air.dialects import arith
        from air.ir import IntegerType

        i32_t = IntegerType.get_signless(32)
        v1, v0 = (arith.ConstantOp(i32_t, f(q)).result for q in (1, 0))
        return air.rtp(arith.select(blk["pair"].materialize(), v1, v0))

    kq = by_pair(lambda q: (ksteps // 4) * w4 * (1 - q))
    kp = by_pair(lambda q: (ksteps // 2) * w4 * q)
    kb = by_pair(lambda q: (ksteps // 4) * wb * (1 - q))
    kbp = by_pair(lambda q: (ksteps // 2) * wb * q)
    nr = air.rtp(wave * 0 + rounds)
    actp = air.rtp(wave * 0 + act)
    plain = air.rtp(wave * 0 + (1 - act))
    actp2 = by_pair(lambda q: act * q)
    plain2 = by_pair(lambda q: (1 - act) * q)
    rtps += [kq, kp, kb, kbp, nr, actp, plain, actp2, plain2]
    loops = [kq, kp, kb, kbp]
    if mx:
        kmx = by_pair(lambda q: (ksteps // 4) * wm * (1 - q))
        kmxp = by_pair(lambda q: (ksteps // 2) * wm * q)
        rtps += [kmx, kmxp]
        loops += [kmx, kmxp]
    if q4:
        w3 = int(wq == 3)
        k4k = by_pair(lambda q: (ksteps // 4) * w3 * (1 - q))
        k4kp = by_pair(lambda q: (ksteps // 2) * w3 * q)
        rtps += [k4k, k4kp]
        loops += [k4k, k4kp]

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
                air.alloc([qb], i8, scope=seg.private(), column=GCOL0 + tx)
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
            # in the segment's order
            params=[kq, kp, kb, kbp, nr, actp, plain, actp2, plain2, *loops[4:]],
        ) as hd:

            @hd.body
            def _(tx, ty):
                a_l1 = [air.alloc([A_BFP], i8, scope=hd.private()) for _ in range(4)]
                b_l1 = [air.alloc([qb], i8, scope=hd.private()) for _ in range(2)]
                bdq = air.alloc([B_BFP], i8, scope=hd.private())
                acc = air.alloc([TN // MM, TM // MM, MM, MM], f32, scope=hd.private())
                acc2 = air.alloc([TN // MM, TM // MM, MM, MM], f32, scope=hd.private())
                out = air.alloc([TN // MM, TM // MM, MM, MM], bf16, scope=hd.private())
                for _r in air.sequential(nr):
                    zero(acc)
                    zero(acc2)
                    # int4 and MXFP4 packets are dequantized; bf16 weights
                    # come as bfp16 in two packets per K step. With pair, each
                    # K step takes two chunks' A tiles against one B. Of the
                    # loops all but one are zero-trip, and each trip gets the
                    # four A buffers in the same order, which is the BD chain
                    # air-to-aie builds (#2040).
                    for kn, dqf in ((kq, dq), (kmx, dqmx)) if mx else ((kq, dq),):
                        for _k in air.sequential(kn):
                            for i in range(4):
                                a2l1.get(a_l1[i], indices=[tx, ty])
                                b2l1.get(b_l1[i % 2], indices=[tx, ty])
                                dqf(b_l1[i % 2], bdq)
                                mmul(a_l1[i], bdq, acc)
                    for kn, dqf in ((kp, dq), (kmxp, dqmx)) if mx else ((kp, dq),):
                        for _k in air.sequential(kn):
                            for pp in range(2):
                                b2l1.get(b_l1[pp], indices=[tx, ty])
                                dqf(b_l1[pp], bdq)
                                a2l1.get(a_l1[2 * pp], indices=[tx, ty])
                                mmul(a_l1[2 * pp], bdq, acc)
                                a2l1.get(a_l1[2 * pp + 1], indices=[tx, ty])
                                mmul(a_l1[2 * pp + 1], bdq, acc2)
                    if q4:
                        # a block pair is two K steps: both stay while the
                        # cores take its K halves in turn
                        for _k in air.sequential(k4k):
                            for s in range(2):
                                for pp in range(2):
                                    b2l1.get(b_l1[pp], indices=[tx, ty])
                                for h in range(2):
                                    dq4k(b_l1[0], b_l1[1], bdq, h)
                                    a2l1.get(a_l1[2 * s + h], indices=[tx, ty])
                                    mmul(a_l1[2 * s + h], bdq, acc)
                        for _k in air.sequential(k4kp):
                            for pp in range(2):
                                b2l1.get(b_l1[pp], indices=[tx, ty])
                            for h in range(2):
                                dq4k(b_l1[0], b_l1[1], bdq, h)
                                a2l1.get(a_l1[2 * h], indices=[tx, ty])
                                mmul(a_l1[2 * h], bdq, acc)
                                a2l1.get(a_l1[2 * h + 1], indices=[tx, ty])
                                mmul(a_l1[2 * h + 1], bdq, acc2)
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
    span = g.max_blocks * kv_rec(g)
    kvin._declare()
    if rounds // h > 1:
        ChannelPut(
            kvin.name,
            KV.value,
            offsets=[c(0), c(0), c(0)],
            sizes=[c(h), c(rounds // h), n],
            strides=[c(span), c(0), c(1)],
            indices=kvin._indices(None, "put"),
        )
        return
    # Without GQA each head is read once. A run-time size needs a contiguous
    # transfer or a repeat around it (airrt-to-npu), so a put per head.
    for i in range(h):
        ChannelPut(
            kvin.name,
            KV.value,
            offsets=[c(i * span)],
            sizes=[n],
            strides=[c(1)],
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
        # noncausal: the key count in tokens, else the window in blocks
        wb = _rtp(wave, win) if g.noncausal else air.rtp(wave * 0 + win // g.lkp)
        params += [nr, wb]
        suf = attn_suffix(g)
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
                        if g.sinks:
                            # the sink comes with q and starts the softmax
                            q2l1.get(q_bfp, indices=[ty])
                            init(gp, ml, q_bfp)
                            qb = qb0 + (r % QR) * NQT + ty
                        else:
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
            weight packets if wq is 1, MXFP4 row groups of k / 128 block
            pairs if 2 (cfg.mxfp4), bf16 if 0
    op 'x': (k, n, act, wq), the 'g' GEMM on a run of up to
            cfg.experts.kmax expert blocks, its block count and choice at
            run time: scalars nb (blocks), n1 (of them pairs, first), i1
            and i0 (the first pair's and single chunk's block in their
            buffers) and e0 ... (each block's expert); wq is 2 or 3
    op <attention op>: (heads, nkv, q0, k0), attention for one chunk; a
            noncausal op's keys are its nkv blocks, and with own it takes
            the key count as a fourth run-time scalar nk

    With own, the segment holds only the op's own groups: the device is then
    the union of every op's segment @seg (see air-to-aie), and the op's func
    drives only its groups' columns.
    """
    xb = op == "x"
    if xb:
        op = "g"
        shape = (*shape, 0)
    drive = ({"g"} if op == "g" else set(attn_ops(cfg)[op])) if own else None
    # with own, an attention op takes (nkv, q0, k0) at run time
    runtime = own and op != "g"
    g = dict(k=4 * TK, n=NR, act=0, wq=3 if cfg.q4k else 1, pair=0, rounds=1, ncb=0)
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
        assert wq != 2 or (cfg.mxfp4 and k % TK == 0), (k, wq)
        assert (wq == 3) == cfg.q4k and (wq != 3 or k % Q4K_COLS == 0), (k, wq)
        ncb = k // (Q4K_COLS if wq == 3 else TK)
        g.update(k=k_pad(k), n=n, act=act, wq=wq, pair=pair, rounds=n // NR, ncb=ncb)
    else:
        heads, nkv, q0, k0 = shape
        groups = attn_ops(cfg)[op]
        hg = heads // len(groups)
        for a in groups:
            ga = cfg.attn[a]
            assert hg * len(groups) == heads and hg % ga.kv_heads == 0, (heads, ga)
            win = (cfg.windows or {}).get(op) or NO_WINDOW
            if ga.noncausal:
                win = nkv * ga.lkp
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
        air.extern("dqmx_bfp16", link_with=GEMM_OBJ) if cfg.mxfp4 else None,
        (
            air.extern("dq4k_bfp16", link_with=GEMM_OBJ, scalars=[i32])
            if cfg.q4k
            else None
        ),
    )
    if g["wq"] == 2:
        assert QB == 2 * 2560, QB  # a packet is two MXFP4 blocks of 32 x 128
        # block pairs [group * ncb + step][2 halves], up to the last group's
        # K steps, which read past its ncb
        bshape = [(g["n"] // (2 * TN) - 1) * g["ncb"] + g["k"] // TK, 2, QB]
    elif g["wq"] == 3:
        # [group * ncb + column block][4 blocks], up to the last panel's
        # column blocks, which read past its group when k was padded
        bshape = [(g["n"] // (2 * TN) - 1) * g["ncb"] + g["k"] // (2 * TK), 4]
        bshape.append(Q4K_PF_BYTES)
    else:
        bshape = [g["n"] // TN, g["k"] // TK * (2 if g["wq"] == 0 else 1), QB]
    # declared in the kernel's argument order
    if xb:
        # pair and single blocks [cap] each, B a layer's experts
        ex = cfg.experts
        assert ex and g["wq"] in (2, 3), (ex, g["wq"])
        bshape[0] += (ex.experts - 1) * ex.stride

        def a_blocks(q):
            return air.tensor([ex.cap, HR, g["k"] // TK, 1 + q, A_BFP], i8)

        def c_blocks(q):
            return air.tensor([ex.cap, 1 + q, M, g["rounds"], GNC, TN], bf16)

        # in argument order: the pairs' A, B, the pairs' C, then the singles'
        gt = (a_blocks(1), air.tensor(bshape, i8), c_blocks(1))
        ab, cb = [a_blocks(0), gt[0]], [c_blocks(0), gt[2]]
    else:
        gt = (
            air.tensor([HR, g["k"] // TK, 1 + g["pair"], A_BFP], i8),
            air.tensor(bshape, i8),
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
        names = ("nkv", "q0", "k0")
        if any(cfg.attn[a].noncausal for a in drive):
            names += ("nk",)
        scalars = {n: air.tensor([], i32, name=n) for n in names}
        for a in drive:
            at[a].update({k: scalars[k] for k in ("nkv", "q0", "k0")})
            if cfg.attn[a].noncausal:
                at[a]["win"] = scalars["nk"]

    if xb:
        sc = {
            v: air.tensor([], i32, name=v)
            for v in ("nb", "n1", "i1", "i0", *(f"e{i}" for i in range(ex.kmax)))
        }
    with air.launch(
        repeat=sc["nb"] if xb else 1,
        name=name,
        attrs=["air.preserve_shim_dma_order"],
    ) as lch:

        @lch.body
        def _(wave):
            puts, externs, rtps = [], {}, []
            bodies = []
            blk = None
            if xb:
                from air.api._index import coerce_index

                es = [sc[f"e{i}"] for i in range(ex.kmax)]
                blk = dict(
                    A=ab,
                    C=cb,
                    n1=coerce_index(sc["n1"]),
                    i1=coerce_index(sc["i1"]),
                    i0=coerce_index(sc["i0"]),
                    pair=wave < coerce_index(sc["n1"]),
                    e=ops.switch(
                        wave,
                        [lambda t=t: coerce_index(t) * ex.stride for t in es],
                    ).as_index(),
                )
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
                        cfg.mxfp4,
                        cfg.q4k,
                        g["ncb"],
                        blk,
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
