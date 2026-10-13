# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side layouts of the fused prefill device and the LM-head GEMV.

Weights are repacked once at load. The packed layouts keep Q4NX's values
(4-bit q, bf16 scale and min per 32-wide K group, w = q * scale + min); there
is no requantization.
"""

import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from ....fused_decode.proj_qmm_pack import (
    BLOCK_BF16,
    COL_BLOCK,
    N_GROUPS,
    PARALLEL,
    ROW_BLOCK,
)

from . import device as D

_G = ROW_BLOCK // PARALLEL
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


class Bundle:
    """A model.q4nx file (safetensors), read by tensor name."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n

    def _read(self, name, dtype, count=-1, at=0):
        s, e = self.hdr[name]["data_offsets"]
        if count < 0:
            count = (e - s) // np.dtype(dtype).itemsize - at
        return np.fromfile(
            self.path,
            dtype=dtype,
            count=count,
            offset=self.off + s + at * np.dtype(dtype).itemsize,
        )

    def bf16(self, name):
        """A bf16 tensor as float32, in its declared shape."""
        a = self._read(name, bfloat16).astype(np.float32)
        return a.reshape(self.hdr[name]["shape"])

    def rows(self, name, ids):
        """Rows ids of a 2-D bf16 tensor, float32."""
        w = self.hdr[name]["shape"][1]
        with open(self.path, "rb") as f:
            base = self.off + self.hdr[name]["data_offsets"][0]
            out = np.empty((len(ids), w), bfloat16)
            for i, t in enumerate(ids):
                f.seek(base + int(t) * w * 2)
                out[i] = np.frombuffer(f.read(w * 2), bfloat16)
        return out.astype(np.float32)

    def q4nx(self, name, rows, K):
        """A Q4NX (Codec B) matrix [rows, K] as raw (q, scale, min)."""
        nb = (rows // ROW_BLOCK) * (K // COL_BLOCK)
        i16 = self._read(name, np.int16, nb * BLOCK_BF16)
        return q4nx_raw(i16.reshape(nb, BLOCK_BF16), rows, K)


def q4nx_raw(i16, rows, K):
    """Block-major Codec B int16 blocks of a [rows, K] matrix as (q [rows, K]
    uint8, scale [rows, K/32], min [rows, K/32]): the block walk of a Q4NX
    dequant, without the multiply."""
    nbi, nbj = rows // ROW_BLOCK, K // COL_BLOCK
    nb = nbi * nbj
    bf = lambda a: a.view(bfloat16).astype(np.float32)  # noqa: E731
    sc = bf(i16[:, 0:256].copy()).reshape(nb, N_GROUPS, ROW_BLOCK)
    mn = bf(i16[:, 256:512].copy()).reshape(nb, N_GROUPS, ROW_BLOCK)
    qb = (
        i16[:, 512:BLOCK_BF16]
        .copy()
        .view(np.uint8)
        .reshape(nb, _G, COL_BLOCK, PARALLEL // 2)
    )
    lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
    hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
    q = np.zeros((nb, ROW_BLOCK, COL_BLOCK), np.uint8)
    q[:, _EVEN, :] = lo
    q[:, _ODD, :] = hi
    q = q.reshape(nbi, nbj, ROW_BLOCK, COL_BLOCK).transpose(0, 2, 1, 3).reshape(rows, K)

    def grp(a):
        a = a.transpose(0, 2, 1).reshape(nbi, nbj, ROW_BLOCK, N_GROUPS)
        return a.transpose(0, 2, 1, 3).reshape(rows, K // 32)

    return q, grp(sc), grp(mn)


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


def kv_records(g, k, v):
    """k, v [T, dh] -> attention group g's KV records [ceil(T/lkp), rec] bf16
    (K tile, then the V tiles of each block), the tail block zero-padded."""
    dh, lkp, dvt = g.dh, g.lkp, g.dvt
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
