# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-Coder-30B-A3B for blocks of 8 token rows: the geometry the verify
# pass is built for, the layouts the host gives it, and float64 references.
import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
LLMS = HERE.parents[1]
CODER = LLMS / "qwen3_coder_30b_a3b_q4k"
sys.path.insert(0, str(CODER))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(LLMS.parent)
]

import attn_layer as AL  # noqa: E402
import moe_block as MB  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402

R, RB, BB, D = AL.R, AL.RB, AL.BB, AL.D
DH, NQ, NKV, DK = AL.DH, AL.NQ, AL.NKV, AL.DK
I, E, K = MB.I, MB.E, MB.K
NCB_X, NCB_O, NCB_H = AL.NCB_X, AL.NCB_O, MB.NCB_H
G_DENSE, RG_KV, RG_Q, RG_O = AL.G_DENSE, AL.RG_KV, AL.RG_Q, AL.RG_O
RG_GU, RG_DN = MB.RG_GU, MB.RG_DN
G_EXP = RG_GU * NCB_X + RG_DN * NCB_H
TOK = 8
NCOL = 4
COLUMNS, ASM = [0, 1, 6, 7], 2
# attention units: a qk core over a pv core per KV head
CU_COLS = [3, 4]
CU_LOC = [(col, row) for col in CU_COLS for row in (2, 4)]
UPM = NKV // len(CU_COLS)
HPM = UPM * DH
CHUNK = 32 * 72  # 256 columns of 8 rows as bfp16 tiles
XB = NCB_X * CHUNK
OB = NCB_O * CHUNK  # o_proj's x
OT = OB // NKV  # a unit's share of it
EGR = TOK * RB  # an egress message: a row block of 8 rows
RW_LEN = 2 * TOK * DH // 2 + 2 * DH
SSZ = 160
YT = 17 * 64
QH_DQ = 8 * DH  # a unit's query heads, one token
HDR = 256
HMSG = HDR + NCB_H * CHUNK  # an H message: header, then 3 chunks
SLICES = I // 128
MAXU = 64  # experts a plan holds
PLAN_W = 80 + 8 * MAXU
PLAN_B = PLAN_W * 4
TOK_W = 80  # a router token: the plan's head, through the experts' ids


def to_tiles(x):
    """x [8, K] (bf16 values) -> bfp16 tiles [K / 8][8 rows][9 bytes]."""
    from air_examples.llms.shared.fused_prefill import hostops as H

    t = np.ascontiguousarray(x.astype(bfloat16).reshape(TOK, -1, 8).transpose(1, 0, 2))
    return H.bfp_rows(t).reshape(-1)


def to_res(h):
    """[8, D] -> tiles [dim tile][token][8]."""
    return np.ascontiguousarray(
        np.asarray(h, np.float32).reshape(TOK, D // 8, 8).transpose(1, 0, 2)
    ).reshape(-1)


def rope_table(pos, qn, kn):
    """RW at positions pos .. pos + 7: cos, sin, then the q and k norms."""
    inv = 1.0 / (W.THETA ** (np.arange(0, DH, 2) / DH))
    f = (pos + np.arange(TOK))[:, None] * inv[None]
    rw = np.concatenate([np.cos(f).ravel(), np.sin(f).ravel(), qn, kn])
    return rw.astype(np.float32).astype(bfloat16)


def dense_weights(w, layer):
    """The layer's attention projections as the array streams them: K and V,
    q, then o_proj with its blocks by x half, row block, column block."""
    raw = lambda n: np.asarray(w.raw(W.name(layer, n)))  # noqa: E731
    sa = "self_attn."
    kvb = np.concatenate([raw(sa + "k_proj.weight"), raw(sa + "v_proj.weight")])
    parts = [
        AL.by_column(kvb, NCB_X),
        AL.by_column(raw(sa + "q_proj.weight"), NCB_X),
        AL.by_column(raw(sa + "o_proj.weight"), NCB_O),
    ]
    wd = np.concatenate([t.reshape(NCOL, -1, R, BB) for t in parts], 1)
    kvq = (RG_KV + RG_Q) * NCB_X
    o = wd[:, kvq:].reshape(NCOL, RG_O, 2, NCB_X, R, BB).transpose(0, 2, 1, 3, 4, 5)
    return np.ascontiguousarray(
        np.concatenate([wd[:, :kvq], o.reshape(NCOL, -1, R, BB)], axis=1)
    )


def expert_weights(w, layer):
    wgu, wdn, _ = MB.weights(w, layer)
    return np.concatenate([m.reshape(E, -1, R, BB) for m in (wgu, wdn)], axis=1)


def router_blocks(wr):
    """The router [E, D] as the planner takes it: per 16 columns
    [2 column tiles][16 expert tiles][8 columns][8 experts]."""
    t = np.asarray(wr, bfloat16).reshape(E // 8, 8, D // 16, 2, 8)
    return np.ascontiguousarray(t.transpose(2, 3, 0, 4, 1)).reshape(-1)


def rmsnorm(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + W.EPS) * g


def route(x, wr):
    """Each row's top 8 of the router's softmax, renormalised: (experts in
    first-use order, scores [experts, 8 rows])."""
    import torch

    lg = (torch.from_numpy(x) @ wr.T).numpy()
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    top = np.argsort(-p, -1, kind="stable")[:, :K]
    ex = list(dict.fromkeys(top.ravel().tolist()))
    sc = np.zeros((len(ex), TOK), np.float32)
    col = {e: i for i, e in enumerate(ex)}
    for r in range(TOK):
        s = p[r, top[r]]
        for e, v in zip(top[r], s / s.sum()):
            sc[col[int(e)], r] = v
    return ex, sc


def attention_reference(w, layer, x, cache, pos):
    """o_proj(attention) of x [8, D] at pos .. pos + 7 after cache (K, V
    [pos, DK], post norm and RoPE), float64."""
    qw, kw, vw, ow, qn, kn = (np.asarray(a, np.float64) for a in w.attn(layer))
    xf = x.astype(np.float64)

    def norm(a, g):
        return a / np.sqrt((a * a).mean(-1, keepdims=True) + W.EPS) * g

    inv = 1.0 / (W.THETA ** (np.arange(0, DH, 2) / DH))
    f = (pos + np.arange(TOK))[:, None, None] * inv[None, None]
    cos, sin = np.cos(f), np.sin(f)

    def rope(a):
        a1, a2 = a[..., : DH // 2], a[..., DH // 2 :]
        return np.concatenate([a1 * cos - a2 * sin, a2 * cos + a1 * sin], -1)

    q = rope(norm((xf @ qw.T).reshape(TOK, NQ, DH), qn))
    k = rope(norm((xf @ kw.T).reshape(TOK, NKV, DH), kn))
    v = (xf @ vw.T).reshape(TOK, NKV, DH)
    Kc = np.concatenate([cache[0][:pos].reshape(pos, NKV, DH), k])
    Vc = np.concatenate([cache[1][:pos].reshape(pos, NKV, DH), v])
    rep = NQ // NKV
    s = np.einsum("qhd,khd->hqk", q, np.repeat(Kc, rep, 1)) * DH**-0.5
    mask = np.arange(pos + TOK)[None] > (pos + np.arange(TOK))[:, None]
    s = np.where(mask[None], -np.inf, s)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", p, np.repeat(Vc, rep, 1)).reshape(TOK, -1)
    return o @ ow.T


def moe_reference(w, layer, x, ex, sc):
    """sum over the experts of score x expert(x), float64."""
    xf = x.astype(np.float64)
    y = np.zeros((TOK, D))
    for e, s in zip(ex, sc):
        g, u, dn = (np.asarray(a, np.float64) for a in w.expert(layer, int(e)))
        hh = (xf @ g.T) / (1 + np.exp(-(xf @ g.T))) * (xf @ u.T)
        y += s[:, None] * (hh @ dn.T)
    return y


def compare(got, want):
    g, r_ = np.asarray(got, np.float64).ravel(), np.asarray(want, np.float64).ravel()
    cos = g @ r_ / np.linalg.norm(g) / np.linalg.norm(r_)
    return cos, np.linalg.norm(g - r_) / np.linalg.norm(r_)
