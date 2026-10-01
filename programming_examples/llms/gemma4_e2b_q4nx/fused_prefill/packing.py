# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side layouts of the fused prefill device and the LM-head GEMV.

Weights are repacked once at load. The packed layouts keep Q4NX's values
(4-bit q, bf16 scale and min per 32-wide K group, w = q * scale + min); there
is no requantization.
"""

import numpy as np
from ml_dtypes import bfloat16

import device as D
import gemma4_e2b_q4nx_weights as gw


def q4nx_raw(model, name, rows, K, r0=0):
    """Rows [r0, r0+rows) of a Codec B matrix as (q [rows, K] uint8,
    scale [rows, K/32], min [rows, K/32]): the block walk of
    Q4nxModel.dequant, without the multiply."""
    nbi, nbj = rows // gw.ROW_BLOCK, K // gw.COL_BLOCK
    nb = nbi * nbj
    s, _ = model._hdr[name]["data_offsets"]
    i16 = np.fromfile(
        model.path,
        dtype=np.int16,
        count=nb * gw.BLOCK_BF16,
        offset=model._data_off + s + (r0 // gw.ROW_BLOCK) * nbj * gw.BLOCK_BF16 * 2,
    ).reshape(nb, gw.BLOCK_BF16)
    sc = gw._bf(i16[:, 0:256].copy()).reshape(nb, gw.N_GROUPS, gw.ROW_BLOCK)
    mn = gw._bf(i16[:, 256:512].copy()).reshape(nb, gw.N_GROUPS, gw.ROW_BLOCK)
    qb = (
        i16[:, 512 : gw.BLOCK_BF16]
        .copy()
        .view(np.uint8)
        .reshape(nb, gw._G, gw.COL_BLOCK, gw.PARALLEL // 2)
    )
    lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, gw.ROW_BLOCK // 2, gw.COL_BLOCK)
    hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, gw.ROW_BLOCK // 2, gw.COL_BLOCK)
    q = np.zeros((nb, gw.ROW_BLOCK, gw.COL_BLOCK), np.uint8)
    q[:, gw._EVEN, :] = lo
    q[:, gw._ODD, :] = hi
    q = (
        q.reshape(nbi, nbj, gw.ROW_BLOCK, gw.COL_BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(rows, K)
    )

    def grp(a):
        a = a.transpose(0, 2, 1).reshape(nbi, nbj, gw.ROW_BLOCK, gw.N_GROUPS)
        return a.transpose(0, 2, 1, 3).reshape(rows, K // 32)

    return q, grp(sc), grp(mn)


def layer_q4(model, L):
    """A layer's Q4NX projections as raw (q, scale, min), [out, in]."""
    dh = gw.head_dim(L)
    dq, dkv = gw.N_Q_HEADS * dh, gw.N_KV_HEADS * dh
    inter = model.mlp_inter(L)
    p = f"model.layers.{L}."
    d = gw.D
    w = dict(
        q=q4nx_raw(model, p + "self_attn.q_proj.weight", dq, d),
        o=q4nx_raw(model, p + "self_attn.o_proj.weight", d, dq),
        up=q4nx_raw(model, p + "mlp.up_proj.weight", inter, d),
        gate=q4nx_raw(model, p + "mlp.gate_proj.weight", inter, d),
        down=q4nx_raw(model, p + "mlp.down_proj.weight", d, inter),
    )
    if gw.owns_kv(L):
        w["k"] = q4nx_raw(model, p + "self_attn.k_proj.weight", dkv, d)
        w["v"] = q4nx_raw(model, p + "self_attn.v_proj.weight", dkv, d)
    return w


def n_pad(n):
    return -(-n // D.NR) * D.NR


def pack_q4(q, scale, mn):
    """[out, in] raw Q4NX -> GEMM B packets [N/TN, K/TK, QB] uint8 (dq4.cc),
    output columns padded to whole rounds with zero weights."""
    n, k = q.shape
    npd = n_pad(n)
    pad = lambda a: np.pad(a.T, ((0, 0), (0, npd - n)))  # noqa: E731
    q, scale, mn = pad(q), pad(scale), pad(mn)
    TK, TN, MM = D.TK, D.TN, D.MM
    qt = q.reshape(k // TK, TK // MM, MM, npd // TN, TN // MM, MM)  # ks kb kk nt nb nn
    qt = qt.transpose(3, 0, 4, 1, 2, 5).reshape(npd // TN, k // TK, -1)
    qb = (qt[..., 0::2] | (qt[..., 1::2] << 4)).astype(np.uint8)

    def grp(a):
        a = (
            np.asarray(a, bfloat16)
            .reshape(k // TK, TK // 32, npd // TN, TN)
            .transpose(2, 0, 1, 3)
        )
        return a.reshape(npd // TN, k // TK, -1).view(np.uint8)

    return np.ascontiguousarray(np.concatenate([qb, grp(scale), grp(mn)], axis=2))


def pack_bf16(w):
    """[out, in] float -> GEMM B packets [N/TN, K/TK * 4, QB] uint8: per K
    step four packets of TN x 32 bf16 in B-tile order (bf16_pkt in dq4.cc)."""
    n, k = w.shape
    npd = n_pad(n)
    b = np.zeros((k, npd), bfloat16)
    b[:, :n] = np.asarray(w, np.float32).T
    TK, TN, MM = D.TK, D.TN, D.MM
    t = b.reshape(k // TK, 4, 4, MM, npd // TN, TN // MM, MM)  # ks j kbl kk nt nb nn
    t = (
        t.transpose(4, 0, 1, 5, 2, 3, 6)
        .reshape(npd // TN, k // TK * 4, -1)
        .view(np.uint8)
    )
    out = np.zeros((npd // TN, k // TK * 4, D.QB), np.uint8)
    out[..., : t.shape[-1]] = t
    return out


def kv_records(a, k, v):
    """k, v [T, dh] -> attention KV records [ceil(T/lkp), rec] bf16 (K tile,
    then the V tiles of each block), the tail block zero-padded."""
    _, dh, lkp, dvt = D.ATTN[a]
    nkv = -(-k.shape[0] // lkp)
    kp = np.zeros((nkv * lkp, dh), bfloat16)
    vp = np.zeros((nkv * lkp, dh), bfloat16)
    kp[: k.shape[0]], vp[: v.shape[0]] = k, v
    recs = [kp.reshape(nkv, lkp * dh)]
    recs += [
        vp.reshape(nkv, lkp, dh // dvt, dvt)[:, :, z].reshape(nkv, -1)
        for z in range(dh // dvt)
    ]
    return np.concatenate(recs, axis=1)
