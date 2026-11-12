# ./python/test/api/llms_fused_prefill_gemm_format.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root %t | FileCheck %s

"""The fused prefill's host side for a device whose GEMMs read bf16 operands
(manifest "gemm_format": "bf16") and take one chunk per dispatch
("gemm_pairs": false).

The A tilers must write bf16 tiles straight into a bf16 A, and those tiles must
be exactly what the bfp16 path converts, so the two formats carry the same
values. pack_q4 and pack_bf16 must put every weight where a bf16 B tile reads
it.
"""

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(SRC / "programming_examples")
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import hostops as H  # noqa: E402
from air_examples.llms.shared.fused_prefill import packing as P  # noqa: E402

FP = SRC / "programming_examples/llms/shared/fused_prefill"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["gcc", "-O2", "-fopenmp", "-shared", "-fPIC"]
    + [str(FP / "kernels/hostops.c"), "-o", str(OUT / "libhostops.so"), "-lm"],
    check=True,
)
H.load(str(OUT))
rng = np.random.default_rng(0)
# a K the device pads, and rows that end inside a row block
K, T = 3 * D.TK, D.TM + 13


def a_bufs(chunks):
    """Zeroed A buffers of both formats for `chunks` chunks."""
    out = {}
    for fmt in ("bf16", "bfp16"):
        dt, shape = P.a_layout(fmt, K, chunks)
        out[fmt] = np.zeros(shape, dt)
    return out


def view(buf, slot):
    """Chunk `slot`'s A, as Engine.a_chunk hands it to the tilers."""
    return buf[:, : K // D.TK, slot]


def tiled(x):
    """x [t, k] -> bf16 tiles [HR, k/TK, TM, TK], rows past t zero."""
    a = np.zeros((D.M, x.shape[1]), bfloat16)
    a[: x.shape[0]] = x
    hr, ks = D.M // D.TM, x.shape[1] // D.TK
    return a.reshape(hr, D.TM, ks, D.TK).transpose(0, 2, 1, 3)


def converted(a16, t, k0=0, k1=None):
    """The bfp16 bytes the bfp16 path makes from bf16 tiles a16."""
    out = view(a_bufs(1)["bfp16"], 0)
    H._to_bfp(np.ascontiguousarray(a16).view(np.uint16), t, out, k0, k1)
    return out


def check_rest(bufs, slot):
    """Only chunk `slot`'s K steps below K were written."""
    for buf in bufs.values():
        rest = np.delete(buf, slot, axis=2) if buf.shape[2] > 1 else buf[:, :0]
        assert not rest.view(np.uint8).any()
        assert not buf[:, K // D.TK :].view(np.uint8).any()


def same(a, b):
    return np.array_equal(a.view(np.uint16), b.view(np.uint16))


# K is padded to whole groups of K steps; a pair's tiles sit side by side.
# CHECK: a_layout ok
assert D.k_pad(K) > K
for chunks in (1, 2):
    bufs = a_bufs(chunks)
    assert bufs["bf16"].nbytes == chunks * D.M * D.k_pad(K) * 2
    assert bufs["bfp16"].nbytes == chunks * D.HR * D.k_pad(K) // D.TK * D.A_BFP
print("a_layout ok")

# Each tiler, on a lone chunk and on either chunk of a pair.
# CHECK: tile_a ok
x = rng.standard_normal((T, K)).astype(np.float32)
for chunks, slot in ((1, 0), (2, 0), (2, 1)):
    bufs = a_bufs(chunks)
    for buf in bufs.values():
        H.tile_a(x, view(buf, slot), D.TM, D.TK)
    a16 = view(bufs["bf16"], slot)
    assert same(a16, tiled(x.astype(bfloat16)))
    assert np.array_equal(view(bufs["bfp16"], slot), converted(a16, T))
    check_rest(bufs, slot)
print("tile_a ok")

# CHECK: glu_tile ok
g = rng.standard_normal((T, K + 8)).astype(bfloat16)
u = rng.standard_normal((T, K + 8)).astype(bfloat16)
for chunks, slot in ((1, 0), (2, 1)):
    bufs = a_bufs(chunks)
    for buf in bufs.values():
        H.glu_tile(g, u, T, K, view(buf, slot), D.TM, D.TK)
    a16 = view(bufs["bf16"], slot)
    assert np.array_equal(view(bufs["bfp16"], slot), converted(a16, T))
    check_rest(bufs, slot)
print("glu_tile ok")


def row_bytes(a, r):
    """Row r's bfp16 groups in a, a chunk's bfp16 A (tiles_bfp's layout)."""
    ty, b, g = r // D.TM, r % D.TM // 8, r % 8
    nkb = D.TK // 8
    return np.concatenate(
        [
            a[ty, s, off : off + 9]
            for s in range(K // D.TK)
            for kb in range(nkb)
            for off in [72 * (b % 2) + 144 * (b // 2 * nkb + kb) + 9 * g]
        ]
    )


def same_rows(got, want, t):
    """Rows below t as in want; the rest of t's last 8 rows zero."""
    for r in range(t):
        assert np.array_equal(row_bytes(got, r), row_bytes(want, r)), r
    for r in range(t, -(-t // 8) * 8):
        assert not row_bytes(got, r).any(), r


# tile_a_rows reads rows by index: tile_a of the gathered rows, on either
# chunk of a pair, and again with fewer rows into the same A.
# CHECK: tile_a_rows ok
X = rng.standard_normal((3 * T, K)).astype(np.float32)
rows = rng.permutation(3 * T)[:T]
for chunks, slot in ((1, 0), (2, 0), (2, 1)):
    bufs = {"bfp16": a_bufs(chunks)["bfp16"]}
    got = view(bufs["bfp16"], slot)
    for t in (T, T - 9):
        H.tile_a_rows(X, rows[:t], got, D.TM, D.TK)
        same_rows(got, converted(tiled(X[rows[:t]].astype(bfloat16)), t), t)
    check_rest(bufs, slot)
print("tile_a_rows ok")

# glu_tile_a is the expert GLU, bf16(u * g * sigmoid(g)) with hostops'
# exp, then the bfp16 pass, in one; u and g alternate every TK columns.
# CHECK: glu_tile_a ok
f32 = np.float32


def vexp(x):
    """hostops.c vexp, op for op in float32."""
    x = np.clip(x, f32(-87), f32(88))
    n = np.rint(x * f32(1.44269504))
    r = x - n * f32(0.693145752)
    r = r - n * f32(1.42860677e-6)
    p = np.full_like(x, f32(1) / f32(720))
    for c in (120, 24, 6):
        p = p * r + f32(1) / f32(c)
    for c in (f32(0.5), f32(1), f32(1)):
        p = p * r + c
    return p * ((n.astype(np.int32) + 127) << 23).view(f32)


gu = (rng.standard_normal((T, 2 * K)) * 3).astype(bfloat16)
cols = gu.astype(f32).reshape(T, K // D.TK, 2, D.TK)
u, g = cols[:, :, 0].reshape(T, K), cols[:, :, 1].reshape(T, K)
want = (u * g * (f32(1) / (f32(1) + vexp(-g)))).astype(bfloat16)
for chunks, slot in ((1, 0), (2, 1)):
    bufs = {"bfp16": a_bufs(chunks)["bfp16"]}
    got = view(bufs["bfp16"], slot)
    for t in (T, T - 9):
        H.glu_tile_a(gu, t, K, got, D.TM, D.TK)
        same_rows(got, converted(tiled(want[:t]), t), t)
    check_rest(bufs, slot)
print("glu_tile_a ok")

# round_bf16 is the bf16 round trip, in place.
# CHECK: round_bf16 ok
r = np.concatenate(
    [
        rng.standard_normal(4096).astype(np.float32) * 1e3,
        np.float32([1e-40, -1e-40, 3.4e38, 1.00390625, 1.01171875, 0.0, -0.0]),
    ]
)
assert np.array_equal(
    H.round_bf16(r.copy()).view(np.uint32),
    r.astype(bfloat16).astype(np.float32).view(np.uint32),
)
print("round_bf16 ok")

# o_tile fills only its heads' columns; the rest of A keeps what it had.
# CHECK: o_tile ok
nh, dh, h0 = 2, 64, 1
o = rng.standard_normal((nh, D.M, dh)).astype(bfloat16)
before = rng.standard_normal((T, K)).astype(np.float32)
want = before.astype(bfloat16)
want[:, h0 * dh : (h0 + nh) * dh] = o[:, :T].transpose(1, 0, 2).reshape(T, -1)
# o as the attention cores write it: per tq rows, a tq x dh tile
# [dh / 8][tq / 8][8][8]
tq = 16
o_cores = (
    o.reshape(nh, D.M // tq, tq // 8, 8, dh // 8, 8)
    .transpose(0, 1, 4, 2, 3, 5)
    .reshape(nh, D.M, dh)
    .copy()
)
for chunks, slot in ((1, 0), (2, 1)):
    bufs = a_bufs(chunks)
    a16 = view(bufs["bf16"], slot)
    H.tile_a(before, a16, D.TM, D.TK)
    H.o_tile(o_cores, T, h0, K, a16, D.TM, D.TK, tq)
    assert same(a16, tiled(want))
    a8 = view(bufs["bfp16"], slot)
    H.o_tile(o_cores, T, h0, K, a8, D.TM, D.TK, tq)
    assert np.array_equal(a8, converted(a16, T, h0 * dh, (h0 + nh) * dh))
    check_rest(bufs, slot)
print("o_tile ok")

# Per K step four packets; packet j holds K rows [32 j, 32 j + 32) of the step
# as B tiles [TN/MM][32/MM][MM k][MM n], then zeros. Columns past n are zero.
# CHECK: pack_bf16 ok
n = 2 * D.TN - 24
npd = P.n_pad(n)
mm = D.MM
w = rng.standard_normal((n, K)).astype(np.float32)
pk = P.pack_bf16(w, "bf16")
assert pk.shape == (npd // D.TN, K // D.TK * 4, D.QB)
nbytes = D.TN * 32 * 2
assert not pk[..., nbytes:].any()
b = pk[..., :nbytes].copy().view(bfloat16)
b = b.reshape(npd // D.TN, K // D.TK, 4, D.TN // mm, 32 // mm, mm, mm)
b = b.transpose(1, 2, 4, 5, 0, 3, 6).reshape(K, npd)  # (ks j kb kk) x (nt nb nn)
want = np.zeros((K, npd), bfloat16)
want[:, :n] = w.T
assert np.array_equal(b.view(np.uint16), want.view(np.uint16))
print("pack_bf16 ok")

# One packet per K step: the nibbles as B tiles [TN/MM][TK/MM][MM k][MM n],
# two per byte along n, low first; then scale and min [TK/32][TN] as bf16.
# CHECK: pack_q4 ok
q = rng.integers(0, 16, (n, K)).astype(np.uint8)
sc = rng.standard_normal((n, K // 32)).astype(np.float32)
mn = rng.standard_normal((n, K // 32)).astype(np.float32)
pk = P.pack_q4(q, sc, mn, "bf16")
assert pk.shape == (npd // D.TN, K // D.TK, D.QB)
qb = D.TN * D.TK // 2
nib = np.stack([pk[..., :qb] & 0xF, pk[..., :qb] >> 4], -1)
nib = nib.reshape(npd // D.TN, K // D.TK, D.TN // mm, D.TK // mm, mm, mm)
nib = nib.transpose(1, 3, 4, 0, 2, 5).reshape(K, npd)  # (ks kb kk) x (nt nb nn)
want = np.zeros((K, npd), np.uint8)
want[:, :n] = q.T
assert np.array_equal(nib, want)
g = pk[..., qb:].copy().view(bfloat16).reshape(npd // D.TN, K // D.TK, 2, -1, D.TN)
for i, v in enumerate((sc, mn)):
    got = g[:, :, i].transpose(1, 2, 0, 3).reshape(K // 32, npd)
    want = np.zeros((K // 32, npd), bfloat16)
    want[:, :n] = v.T
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))
print("pack_q4 ok")


# A device without pair ops runs every chunk alone; with them, chunks pair
# up from three chunks on, and an odd last chunk runs alone.
# CHECK: chunk pairs ok
sync = SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=0, XCL_BO_SYNC_BO_FROM_DEVICE=1)
sys.modules.setdefault("pyxrt", SimpleNamespace(xclBOSyncDirection=sync))
from air_examples.llms.shared.fused_prefill.engine import Engine  # noqa: E402

eng = object.__new__(Engine)
for pairs, n, want in (
    (True, 5, [(0, 0), (0, 1), (1, 0), (1, 1), None]),
    (True, 2, [None, None]),
    (False, 5, [None] * 5),
):
    eng.gemm_pairs = pairs
    eng.chunk_rows(n * D.M)
    assert [eng._pair(c) for c in range(n)] == want, (pairs, n)
print("chunk pairs ok")
