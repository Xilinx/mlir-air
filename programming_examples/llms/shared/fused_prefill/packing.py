# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side layouts of the fused prefill device and the LM-head GEMV.

Weights are repacked once at load. The packed layouts keep Q4NX's values
(4-bit q, bf16 scale and min per 32-wide K group, w = q * scale + min). A
model whose weights are not a Q4NX bundle is put on that grid exactly (Q4_0)
or by the quantizer its fused decode uses (a full-precision checkpoint), so
prefill and decode compute with the same weights.
"""

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from ....fused_decode.proj_qmm_pack import (
    BLOCK_BF16,
    COL_BLOCK,
    ROW_BLOCK,
)

from ..infra import host_weights
from . import device as D
from . import hostops as H


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
        """A Q4NX (Codec B) matrix [rows, K] as raw (q, scale, min). A bundle
        may store one of the two dimensions zero-padded; the [rows, K] corner
        is returned."""
        nbi, nbj = rows // ROW_BLOCK, K // COL_BLOCK
        nb = self.hdr[name]["shape"][0]
        if nb != nbi * nbj:
            if nb % nbi == 0 and nb // nbi > nbj:
                nbj = nb // nbi
            elif nb % nbj == 0 and nb // nbj > nbi:
                nbi = nb // nbj
            else:
                raise ValueError(f"{name}: {nb} blocks do not hold [{rows}, {K}]")
        i16 = self._read(name, np.int16, nb * BLOCK_BF16)
        q, scale, mn = q4nx_raw(
            i16.reshape(nb, BLOCK_BF16), nbi * ROW_BLOCK, nbj * COL_BLOCK
        )
        g = K // 32
        return q[:rows, :K], scale[:rows, :g], mn[:rows, :g]


class Q40Bundle(Bundle):
    """A bundle whose int4 matrices are Q4_0: signed nibbles, w = q * scale,
    the min field unused. As Q4NX that is q + 8 with min = -8 scale, exact in
    bf16."""

    def q4nx(self, name, rows, K):
        q, scale, _ = super().q4nx(name, rows, K)
        return q ^ 8, scale, -8 * scale


class HFBundle:
    """A full-precision HF checkpoint behind Bundle's interface. Each matrix is
    rounded onto the Q4NX grid when read, by the fused decode's quantizer, with
    scale and min in bf16 as the device holds them."""

    def __init__(self, d):
        # q4_0_codec imports its fused_decode siblings top-level
        fd = str(Path(__file__).resolve().parents[3] / "fused_decode")
        if fd not in sys.path:
            sys.path.insert(0, fd)
        from q4_0_codec import HFModel

        self.hf = HFModel(str(d))

    def bf16(self, name):
        return self.hf.bf16(name)

    def rows(self, name, ids):
        return self.hf.rows(name, ids)

    def q4nx(self, name, rows, K):
        q, scale, mn = host_weights.requant_q4k(self.hf.bf16(name)[:rows, :K], 32)
        bf = lambda a: a.astype(bfloat16).astype(np.float32)  # noqa: E731
        return q, bf(scale), bf(mn)


def open_weights(codec, path):
    """The weights at `path` (resolve()'s result) for a model of `codec`."""
    if codec == "hf":
        return HFBundle(Path(path).parent)
    if codec == "q4_0":
        return Q40Bundle(path)
    return Bundle(path)


def q4nx_raw(i16, rows, K):
    """Block-major Codec B int16 blocks of a [rows, K] matrix as (q [rows, K]
    uint8, scale [rows, K/32], min [rows, K/32]): the block walk of a Q4NX
    dequant, without the multiply."""
    return host_weights.unpack_q4nx(i16, rows // ROW_BLOCK, K // COL_BLOCK)


def n_pad(n):
    return -(-n // D.NR) * D.NR


def pack_q4(q, scale, mn, gemm_format="bfp16"):
    """[out, in] raw Q4NX -> GEMM B packets [N/TN, K/TK, QB] uint8, output
    columns padded to whole rounds with zero weights. The nibbles of a B tile
    run along k for "bfp16" (gemm_bfp16.cc dq4_bfp16, rows n), along n for
    "bf16" (B tiles [MM k][MM n])."""
    n, k = q.shape
    npd = n_pad(n)
    pad = lambda a: np.pad(a.T, ((0, 0), (0, npd - n)))  # noqa: E731
    q, scale, mn = pad(q), pad(scale), pad(mn)
    TK, TN, MM = D.TK, D.TN, D.MM
    qt = q.reshape(k // TK, TK // MM, MM, npd // TN, TN // MM, MM)  # ks kb kk nt nb nn
    if gemm_format == "bf16":
        qt = qt.transpose(3, 0, 4, 1, 2, 5)
    else:
        qt = qt.transpose(3, 0, 4, 1, 5, 2)
    qt = qt.reshape(npd // TN, k // TK, -1)
    qb = (qt[..., 0::2] | (qt[..., 1::2] << 4)).astype(np.uint8)

    def grp(a):
        a = (
            np.asarray(a, bfloat16)
            .reshape(k // TK, TK // 32, npd // TN, TN)
            .transpose(2, 0, 1, 3)
        )
        return a.reshape(npd // TN, k // TK, -1).view(np.uint8)

    return np.ascontiguousarray(np.concatenate([qb, grp(scale), grp(mn)], axis=2))


def pack_bf16(w, gemm_format="bfp16"):
    """[out, in] float -> GEMM B packets uint8. "bfp16": [N/TN, K/TK * 2, QB],
    per K step two packets, one per K half, of bfp16 blocks [TN/16][TK/16][2]
    (gemm_bfp16.cc mm_bfp16_pkt). "bf16": [N/TN, K/TK * 4, QB], per K step four
    packets of TN x 32 bf16 as B tiles [TN/MM][32/MM][MM k][MM n]."""
    n, k = w.shape
    npd = n_pad(n)
    b = np.zeros((k, npd), bfloat16)
    b[:, :n] = np.asarray(w, np.float32).T
    TK, TN, MM = D.TK, D.TN, D.MM
    if gemm_format == "bf16":
        t = b.reshape(k // TK, 4, 32 // MM, MM, npd // TN, TN // MM, MM)
        t = t.transpose(4, 0, 1, 5, 2, 3, 6)  # nt ks j nb kb kk nn
        g = t.reshape(npd // TN, k // TK * 4, -1).view(np.uint8)
    else:
        t = b.reshape(k // TK, 2, TK // 16, MM, npd // TN, TN // 16, 2, MM)
        t = t.transpose(4, 0, 1, 5, 2, 6, 7, 3)  # nt ks h np kb i nn kk
        g = H.bfp_rows(t).reshape(npd // TN, k // TK * 2, -1)
    out = np.zeros(g.shape[:2] + (D.QB,), np.uint8)
    out[..., : g.shape[-1]] = g
    return out


def a_layout(gemm_format, k, chunks=1):
    """dtype and shape [HR, K steps, chunk, tile] of the GEMM's A for depth k
    on `chunks` chunks, a K step's tiles of the chunks side by side, K padded
    to the device's (device.k_pad). "bfp16" is this builder's GEMM
    (gemm_bfp16.cc); "bf16" is A as bf16 tiles [TM, TK], for a device whose
    GEMMs are built elsewhere and read bf16."""
    steps = D.k_pad(k) // D.TK
    if gemm_format == "bf16":
        return bfloat16, (D.HR, steps, chunks, D.TM, D.TK)
    assert gemm_format == "bfp16", gemm_format
    return np.uint8, (D.HR, steps, chunks, D.A_BFP)


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
