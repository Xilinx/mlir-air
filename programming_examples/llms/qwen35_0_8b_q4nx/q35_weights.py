# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM's Qwen3.5-0.8B `model.q4nx` (FastFlowLM/Qwen3.5-0.8B-NPU2, branch
flm_q4k_high_precision @ 1d16e5eaa2508889fb88eb1bfab1921a30a1a466, the file
FastFlowLM 1.0.11 loads), read by tensor name.

Every quantized tensor is stored in blocks of 32 rows x 256 columns, in one
of two formats:

  Q4K   (Q4_K, 8-bit sub-scales) 4736 bytes: sc uint8 [8][32], m uint8 [8][32]
        (per group of 32 columns and row), nibbles [256 columns][32 rows / 2]
        (low nibble = even row), d bf16 [32], dmin bf16 [32] (per row; dmin <= 0);
        w = d * sc * q + dmin * m, q in [0, 15].
  Q8NX  (Q8_0)  8704 bytes: scale [8][32] bf16, then int8 [256 columns][32
        rows]; w = scale * q, q in [-128, 127].

Q4K holds every projection but the beta/alpha projections and lm_head (Q8).
The bundle has no embedding: the embedding is tied to lm_head
(tie_word_embeddings), so embed() returns lm_head's Q8 values rounded to bf16.

The decode (q35_fdg) takes the Q4K blocks as stored and Q8 as exact
affine-Q4 row pairs (q8_pair_rows). The prefill's GEMM device reads the affine
grid (q, bf16 scale, bf16 min) per group: q4() gives a Q4K tensor there as
(q, bf16(d * sc), bf16(dmin * m)), the products rounded to bf16 (8 significant
bits; the device's BFP16 compute rounds the weights further), and a Q8 matrix
as two Q4 matrices stacked along K (q8_as_q4: with u = q + 128 = 16 h + l,
w = (16 scale) h - 128 scale + scale l, each term on the Q4 grid with bf16
scale and min exactly). The sum of the two products is the Q8 value.
"""

import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = "FastFlowLM/Qwen3.5-0.8B-NPU2"
REVISION = "1d16e5eaa2508889fb88eb1bfab1921a30a1a466"  # branch flm_q4k_high_precision
SHA256 = "787428ffe87be5e3e5a8fcaf241844b69896ca0c74a41f2432c436739f7ba4e4"
RB, CB, G = 32, 256, 32
Q8_BYTES, Q4K_BYTES = 8704, 4736
D, I, NV = 1024, 3584, 248320  # hidden, MLP intermediate, vocabulary


def bundle_path(source=None):
    """The model.q4nx to load: `source` (else $Q4NX_MODEL_SOURCE) when it is a
    file or a directory holding model.q4nx, else the Hub download of REPO at
    REVISION (cached under HF_HOME). Its sha256 must be SHA256."""
    import os

    source = source or os.environ.get("Q4NX_MODEL_SOURCE")
    if source and os.path.isdir(source):
        source = os.path.join(source, "model.q4nx")
    if source and os.path.isfile(source):
        path = source
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, "model.q4nx", revision=REVISION)
    got = sha256(path)
    if got != SHA256:
        raise ValueError(
            f"{path}: sha256 {got}, expected {SHA256} ({REPO} @ {REVISION})"
        )
    return path


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Q4nx:
    def __init__(self, path=None):
        self.path = Path(bundle_path(path))
        with open(self.path, "rb") as f:
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n
        self.mm = np.memmap(self.path, np.uint8, "r")

    def raw(self, name):
        s, e = self.hdr[name]["data_offsets"]
        return self.mm[self.off + s : self.off + e]

    def shape(self, name):
        return self.hdr[name]["shape"]

    def bf16(self, name):
        """A bf16 / f32 tensor as float32 in its declared shape."""
        dt = {"BF16": bfloat16, "F32": np.float32}[self.hdr[name]["dtype"]]
        return (
            np.asarray(self.raw(name))
            .view(dt)
            .astype(np.float32)
            .reshape(self.shape(name))
        )

    def kind(self, name):
        return {Q8_BYTES: "q8", Q4K_BYTES: "q4k"}[self.shape(name)[2]]

    def blocks(self, name, rows, K):
        nbi, nbj, nb = self.shape(name)
        assert nbi * RB >= rows and nbj * CB >= K, (name, self.shape(name), rows, K)
        return np.asarray(self.raw(name)).reshape(nbi, nbj, nb)

    def q4k(self, name, rows, K):
        """(q uint8 [rows, K], sc, m uint8 [rows, K / 32], d, dmin float32
        [rows, K / 256]): w = d * sc * q + dmin * m."""
        b = self.blocks(name, rows, K)
        nbi, nbj = b.shape[:2]

        def grp(a):  # [nbi, nbj, 8 groups, 32 rows] -> [rows, groups]
            a = a.reshape(nbi, nbj, CB // G, RB)
            return a.transpose(0, 3, 1, 2).reshape(nbi * RB, nbj * CB // G)

        def row(a):  # [nbi, nbj, 32 rows] bf16 -> [rows, nbj]
            a = a.copy().view(bfloat16).astype(np.float32).reshape(nbi, nbj, RB)
            return a.transpose(0, 2, 1).reshape(nbi * RB, nbj)

        sc, m = grp(b[..., :256]), grp(b[..., 256:512])
        qb = b[..., 512:4608].reshape(nbi, nbj, CB, RB // 2)
        q = np.stack([qb & 15, qb >> 4], -1)  # [i, j, col, 16, 2]: row = 2r + b
        q = q.transpose(0, 3, 4, 1, 2).reshape(nbi * RB, nbj * CB)
        d, dm = row(b[..., 4608:4672]), row(b[..., 4672:4736])
        return (
            q[:rows, :K],
            sc[:rows, : K // G],
            m[:rows, : K // G],
            d[:rows, : K // CB],
            dm[:rows, : K // CB],
        )

    def q4(self, name, rows, K):
        """(q uint8 [rows, K], scale, min float32 [rows, K / 32]) of a Q4K
        tensor: bf16(d * sc), bf16(dmin * m)."""
        q, sc, m, d, dm = self.q4k(name, rows, K)
        r8 = lambda a: np.repeat(a, CB // G, 1)  # noqa: E731
        return (
            q,
            (r8(d) * sc).astype(bfloat16).astype(np.float32),
            (r8(dm) * m).astype(bfloat16).astype(np.float32),
        )

    def q8(self, name, rows, K):
        """(q int8 [rows, K], scale float32 [rows, K / 32])."""
        b = self.blocks(name, rows, K)
        nbi, nbj = b.shape[:2]
        sc = (
            b[..., :512]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(nbi, nbj, CB // G, RB)
        )
        sc = sc.transpose(0, 3, 1, 2).reshape(nbi * RB, nbj * CB // G)
        q = b[..., 512:].copy().view(np.int8).reshape(nbi, nbj, CB, RB)  # column-major
        q = q.transpose(0, 3, 1, 2).reshape(nbi * RB, nbj * CB)
        return q[:rows, :K], sc[:rows, : K // G]

    def dequant(self, name, rows, K):
        """The logical weight values, float32 [rows, K]."""
        if self.kind(name) == "q4k":  # exact in float64, then float32
            q, sc, m, d, dm = self.q4k(name, rows, K)
            r = lambda a, n: np.repeat(a.astype(np.float64), n, 1)  # noqa: E731
            return (r(d, CB) * r(sc, G) * q + r(dm, CB) * r(m, G)).astype(np.float32)
        q, sc = self.q8(name, rows, K)
        return q * np.repeat(sc, G, 1)

    def as_q4(self, name, rows, K):
        """The matrix on the GEMM's Q4 grid: (q, scale, min) of depth K for a
        Q4 tensor, of depth 2K (q8_as_q4) for a Q8 tensor."""
        if self.kind(name) == "q4k":
            return self.q4(name, rows, K)
        return q8_as_q4(*self.q8(name, rows, K))

    def embed(self):
        """The embedding table, bf16 [NV, D]: the bundle's own, or (this
        bundle: none, tied) lm_head's Q8 values rounded to bf16."""
        if "model.embed_tokens.weight" in self.hdr:
            return (
                np.asarray(self.raw("model.embed_tokens.weight"))
                .view(bfloat16)
                .reshape(NV, D)
            )
        out = np.empty((NV, D), bfloat16)
        for r0 in range(0, NV, 8192):  # in slices (memory)
            r1 = min(NV, r0 + 8192)
            out[r0:r1] = self.dequant_rows("lm_head.weight", r0, r1, D).astype(bfloat16)
        return out

    def dequant_rows(self, name, r0, r1, K):
        """dequant() of rows r0..r1 (multiples of 32 but the last)."""
        nbi, nbj, nb = self.shape(name)
        s, _ = self.hdr[name]["data_offsets"]
        b0, b1 = r0 // RB, -(-r1 // RB)
        sub = self.mm[self.off + s + b0 * nbj * nb : self.off + s + b1 * nbj * nb]
        t = Q4nx.__new__(Q4nx)
        t.hdr = {
            name: {
                "shape": [b1 - b0, nbj, nb],
                "data_offsets": [0, sub.size],
                "dtype": "I8",
            }
        }
        t.off, t.mm = 0, sub
        return t.dequant(name, r1 - b0 * RB, K)[r0 - b0 * RB :]


def q8_as_q4(q, sc):
    """Q8 (q, scale) -> two Q4 matrices stacked along K, against [x | x]."""
    u = q.astype(np.int16) + 128
    h, l = (u >> 4).astype(np.uint8), (u & 15).astype(np.uint8)
    sc = sc.astype(bfloat16).astype(np.float32)
    return (
        np.concatenate([h, l], 1),
        np.concatenate([16 * sc, sc], 1),
        np.concatenate([-128 * sc, np.zeros_like(sc)], 1),
    )


def q8_pair_rows(q, sc):
    """Q8 (q, scale) [n, K] -> 2n Q4 rows at depth K: row 2i = (q8_as_q4's
    high half of row i), row 2i + 1 = its low half; rows 2i and 2i + 1 sum to
    row i (the decode's Q8 rows: lm_head, beta / alpha)."""
    hq, hs, hm = q8_as_q4(q, sc)
    K = q.shape[1]
    G_ = K // 32
    out = []
    for a, b in (
        (hq[:, :K], hq[:, K:]),
        (hs[:, :G_], hs[:, G_:]),
        (hm[:, :G_], hm[:, G_:]),
    ):
        out.append(np.stack([a, b], 1).reshape(2 * a.shape[0], -1))
    return tuple(out)
