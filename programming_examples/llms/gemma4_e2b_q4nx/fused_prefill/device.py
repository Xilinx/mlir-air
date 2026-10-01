# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused prefill device: every op of a 128-token prefill chunk on one
configured device, laid out like FastFlowLM's fused_prefill.

  columns 1-6  GEMM, int4 (Q4NX) or bf16 weights
  column 0     full attention, head_dim 512
  column 7     sliding-window attention, head_dim 256

The herds loop forever and take their work from RTPs, and the memtile
channels are count-free rings, so one PDI serves every op and shape and the
ops differ only in their insts. build() emits the insts of one op;
insts.strip_columns then drops the other groups' columns, so those cores stay
parked.

An op's rounds loop inside the cores and are the outer axis of its shim
patterns, so each op is a single launch.
"""

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
MM_OBJ, MM_SUF = "mm_m32n64.o", "_m32n64"
DQ_OBJ = "dq4.o"

# attention: name -> (column, head_dim, lkp == tile_q, dv_tile)
ATTN = {"f": (0, 512, 16, 256), "s": (7, 256, 32, 128)}
NQT = 4  # cores per attention column
WINDOW = 512

COLS = {"g": set(range(GCOL0, GCOL0 + GNC)), "f": {ATTN["f"][0]}, "s": {ATTN["s"][0]}}


def attn_rows(a):
    """q rows of one attention round."""
    return NQT * ATTN[a][2]


def attn_qr(a):
    """rounds per head for one chunk."""
    return M // attn_rows(a)


def kv_rec(a):
    """elements of one K block's record: K tile, then the V tiles."""
    _, dh, lkp, _ = ATTN[a]
    return 2 * lkp * dh


def _gemm(wave, A, B, C, k, act, wq, rounds, kern, puts):
    ksteps = k // TK
    zero, mmul, cast, cast_gelu, dq, cp = kern
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
                    # GELU(tanh) in the drain when act is set
                    for _a in air.sequential(actp):
                        cast_gelu(acc, out)
                    for _p in air.sequential(plain):
                        cast(acc, out)
                    c2l2.put(out.transpose(1, 2, 0, 3), indices=[tx, ty])

        for tx in range(GNC):
            for ty in range(HR):
                c2l2.get(c_l2[tx][ty, 0:TM, 0:TN], indices=[tx, ty])
            c_out.put(c_l2[tx], indices=[tx])

    return body


def _attention(name, wave, Q, KV, O, nkv, q0, k0, rounds, puts):
    """MQA attention, one head per round, the column's cores splitting the
    round's q rows. The cores read nkv K blocks starting at block k0; the
    chunk's first q block is q0. Both are RTPs, in units of lkp rows."""
    col, dh, lkp, dvt = ATTN[name]
    tq, R, QR = lkp, attn_rows(name), attn_qr(name)
    dvc = dh // dvt
    suf = f"_d{dh}"
    obj = f"attn{suf}.o"
    ext = lambda f, **kw: air.extern(f + suf, link_with=obj, **kw)  # noqa: E731
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
    if name == "s":
        wmask = ext("apply_window_mask", scalars=[i32, i32, i32])
        mask = lambda g, qb, kb: wmask(g, qb, kb, WINDOW // lkp)  # noqa: E731
    else:
        mask = ext("apply_causal_mask", scalars=[i32, i32])

    qin = air.channel(f"{name}QIn", size=[1])
    q2l1 = air.channel(f"{name}Q2L1", size=[NQT])
    kvin = air.channel(f"{name}KVIn", size=[1])
    kv2l1 = air.channel(f"{name}KV2L1", size=[1, 1], broadcast_shape=[1, NQT])
    gp2l2 = air.channel(f"{name}Gp2L2", size=[NQT])
    gpout = air.channel(f"{name}GpOut", size=[1])

    def shim():
        qin.put(Q[0 : rounds * R, 0:dh])
        kvin.put(KV[0 : nkv * kv_rec(name)].broadcast_to(rounds, nkv * kv_rec(name)))
        gpout.get(O[0 : rounds * R, 0:dh])

    puts.append(shim)

    def body(seg):
        nk = air.rtp(wave * 0 + nkv)
        nr = air.rtp(wave * 0 + rounds)
        qb0 = air.rtp(wave * 0 + q0)
        kb0 = air.rtp(wave * 0 + k0)
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
            params=[nk, nr, qb0, kb0],
        ) as h:

            @h.body
            def _(tx, ty):
                q_saved = air.alloc([tq, dh], bf16, scope=h.private())
                qk = air.alloc([lkp, dh], bf16, scope=h.private())
                v_l1 = air.alloc([lkp, dvt], bf16, scope=h.private())
                g = air.alloc([tq, lkp], bf16, scope=h.private())
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
                        zero_g(g.reshape(tq * lkp))
                        kv2l1.get(qk, indices=[tx, ty])
                        mm_ab(q_saved, qk, g.reshape(tq * lkp))
                        mask(g, qb, kb0 + blk)
                        softmax(g.reshape(tq * lkp), up, s_tmp, r_tmp)
                        for z in range(dvc):
                            kv2l1.get(v_l1, indices=[tx, ty])
                            mul_r_gp(r_tmp, gp[z])
                            mm_gb(g.reshape(tq * lkp), v_l1, gp[z])
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


def build(op, *shape):
    """The module for one op; the other groups get minimal shapes.

    op 'g': (k, n, act, wq), C[128, n] = A[128, k] @ W, with a GELU drain if
            act and int4 weight packets if wq (bf16 otherwise)
    op 'f' / 's': (heads, nkv, q0, k0), attention for one chunk
    """
    g = dict(k=2 * TK, n=NR, act=0, wq=1, rounds=1)
    at = {a: dict(nkv=1, q0=0, k0=0, rounds=1) for a in ATTN}
    if op == "g":
        k, n, act, wq = shape
        assert k % (2 * TK) == 0 and n % NR == 0, (k, n)
        g.update(k=k, n=n, act=act, wq=wq, rounds=n // NR)
    else:
        heads, nkv, q0, k0 = shape
        at[op].update(nkv=nkv, q0=q0, k0=k0, rounds=heads * attn_qr(op))

    kern = tuple(
        air.extern(f"{f}{MM_SUF}", link_with=MM_OBJ)
        for f in (
            "zero_f32_mn",
            "op_has_no_registered_library_name",
            "f32_to_bf16_mn",
            "f32_to_bf16_gelu_mn",
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
            air.tensor([at[a]["rounds"] * attn_rows(a), ATTN[a][1]], bf16),
            air.tensor([at[a]["nkv"] * kv_rec(a)], bf16),
            air.tensor([at[a]["rounds"] * attn_rows(a), ATTN[a][1]], bf16),
        )
        for a in ATTN
    }

    with air.launch(
        repeat=1, name="fused_prefill", attrs=["air.preserve_shim_dma_order"]
    ) as lch:

        @lch.body
        def _(wave):
            puts = []
            bodies = [
                _gemm(wave, *gt, g["k"], g["act"], g["wq"], g["rounds"], kern, puts)
            ]
            for a in ATTN:
                p = at[a]
                bodies.append(
                    _attention(
                        a, wave, *att[a], p["nkv"], p["q0"], p["k0"], p["rounds"], puts
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
