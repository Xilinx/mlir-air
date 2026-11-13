# ./python/test/api/llms_fused_prefill_batched.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root %t | FileCheck %s

"""Several of the fused prefill's ops in one dispatch, and the host helpers
that feed them in one parallel region.

txn.concat must repeat a TXN stream with each copy's DDR patches moved by its
own offsets, past 4 GB too, and leave every other word and a valid header.
tile_a_rows_n, glu_tile_a_n and scatter_add_rows_n must write what the single
calls do, byte for byte; scatter_add_rows_n's jobs share rows, so its sums
must also come in the single calls' order.
"""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(SRC / "programming_examples")
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import hostops as H  # noqa: E402
from air_examples.llms.shared.fused_prefill import packing as P  # noqa: E402
from air_examples.llms.shared.fused_prefill import txn  # noqa: E402

FP = SRC / "programming_examples/llms/shared/fused_prefill"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["gcc", "-O2", "-fopenmp", "-shared", "-fPIC", "-DNT=4"]
    + [str(FP / "kernels/hostops.c"), "-o", str(OUT / "libhostops.so"), "-lm"],
    check=True,
)
H.load(str(OUT))
rng = np.random.default_rng(0)


# A TXN stream as aie_runtime's builders write it (TxnEncoding.h).
def write32(addr, v):
    return [txn.WRITE, 0, addr, 0, v, 24]


def blockwrite(addr, data):
    return [txn.BLOCKWRITE, 0, addr, 4 * (4 + len(data))] + list(data)


def maskwrite(addr, v, m):
    return [txn.MASKWRITE, 0, addr, 0, v, m, 28]


def patch(addr, arg, plus):
    return [
        txn.DDR_PATCH,
        48,
        0,
        0,
        0,
        0,
        addr,
        0,
        arg,
        0,
        plus & 0xFFFFFFFF,
        plus >> 32,
    ]


def sync(col, chan):
    return [txn.TCT, 16, col << 16, chan << 24]


def stream(ops):
    body = [w for op in ops for w in op]
    return np.asarray(
        [0x02040001, 0x0108, len(ops), 4 * (4 + len(body))] + body, np.uint32
    )


# CHECK: txn ok
CORE = (1 << 25) | (2 << 20)  # column 1, row 2: a core tile
ops = [
    write32(CORE | 0x3E80, 7),  # an RTP
    write32(CORE | 0x1F000, 1),  # its lock
    write32(0x1D204, 5),  # a shim register
    blockwrite(0x1D000, [1, 2, 3, 4, 5, 6, 7, 8]),
    patch(0x1D004, 0, 0),
    patch(0x1D024, 1, 9472),
    maskwrite(0x1D200, 3840, 7936),
    patch(0x1D044, 2, 128),
    patch(0x1D064, 6, 0x80000000 + 64),  # an arg past the firmware's five
    sync(0, 1),
]
w = stream(ops)
assert [o for _, o, _ in txn.ops(w)] == [0, 0, 0, 1, 0x81, 0x81, 3, 0x81, 0x81, 0x80]
c = txn.Copies(w)
for f in (c.first, c.rest):
    assert list(f.arg) == [0, 1, 2, 6] and list(f.plus) == [0, 9472, 128, 0x80000040]
# a copy after the first leaves out the RTP write, and only that
assert c.first.nops == len(ops) and c.rest.nops == len(ops) - 1
assert np.array_equal(c.first.body, w[4:])
assert np.array_equal(c.rest.body, np.concatenate([w[4 + 6 :]]))
offs = np.array(
    [[0, 0, 0, 0, 0, 0, 0], [4096, 5 << 30, 8192, 0, 0, 0, 3], [1, 2, 3, 0, 0, 0, 4]],
    np.int64,
)
small = txn.Copies(stream(ops[3:5] + [sync(1, 0)]))
runs = [(c, offs), (small, np.array([[1 << 33, 0, 0]], np.int64))]
out = np.zeros(txn.size(runs) + 5, np.uint32)
cat = txn.concat(runs, out)
# the header counts every copy's ops and words; nothing else moves
assert cat.size == txn.size(runs) == 4 + w.size - 4 + 2 * c.rest.body.size + 7 * 4
assert cat[2] == len(ops) + 2 * (len(ops) - 1) + 3 and cat[3] == 4 * cat.size
assert cat[0] == w[0] and cat[1] == w[1]
got = txn.ops(cat)
assert len(got) == cat[2]
plus = [
    (int(cat[p + 8]), int(cat[p + 10]) | int(cat[p + 11]) << 32)
    for p, o, _ in got
    if o == 0x81
]
assert plus == [
    (0, 0), (1, 9472), (2, 128), (6, 0x80000040),
    (0, 4096), (1, 9472 + (5 << 30)), (2, 128 + 8192), (6, 0x80000043),
    (0, 1), (1, 9474), (2, 131), (6, 0x80000044),
    (0, 1 << 33),
], plus  # fmt: skip
mask = np.ones(cat.size, bool)
mask[2:4] = False  # the counts, checked above
for p, o, _ in got:
    if o == 0x81:
        mask[p + 10 : p + 12] = False
ref = np.concatenate([w[:4], c.first.body, c.rest.body, c.rest.body, small.first.body])
assert np.array_equal(cat[mask], ref[mask])
try:
    txn.ops(np.concatenate([w, [9, 0]]).astype(np.uint32))
    raise AssertionError("trailing words accepted")
except ValueError:
    pass
print("txn ok")

# Blocks of an expert's rows, as the MoE gives them: one chunk, or a pair
# (two chunks) of up to 2 * M rows, rows that end inside a row block.
K = 3 * D.TK
X = rng.standard_normal((4 * D.M, K)).astype(np.float32)
blocks = [(1, 2 * D.M - 5), (0, D.M), (1, D.M + 1), (0, 37), (0, 1)]


def a_views(q):
    dt, shape = P.a_layout("bfp16", K, 1 + q)
    buf = np.zeros(shape, dt)
    return buf, [buf[:, : K // D.TK, hh] for hh in range(1 + q)]


# CHECK: tile_a_rows_n ok
want, got, jobs = [], [], []
for q, t in blocks:
    rows = rng.choice(X.shape[0], t, replace=False)
    wb, wv = a_views(q)
    gb, gv = a_views(q)
    for hh in range(1 + q):
        r = rows[hh * D.M : (hh + 1) * D.M]
        H.tile_a_rows(X, r, wv[hh], D.TM, D.TK)
        jobs.append((r, gv[hh]))
    want.append(wb)
    got.append(gb)
H.tile_a_rows_n(X, jobs, D.TM, D.TK)
assert all(np.array_equal(a, b) for a, b in zip(got, want))
print("tile_a_rows_n ok")

# CHECK: glu_tile_a_n ok
gu = (rng.standard_normal((2 * D.M, 2 * K)) * 3).astype(bfloat16)
want, got, jobs = [], [], []
for q, t in blocks:
    wb, wv = a_views(q)
    gb, gv = a_views(q)
    for hh in range(1 + q):
        r0, r1 = hh * D.M, min(t, (hh + 1) * D.M)
        H.glu_tile_a(gu[r0:r1], r1 - r0, K, wv[hh], D.TM, D.TK)
        jobs.append((gu[r0:r1], r1 - r0, gv[hh]))
    want.append(wb)
    got.append(gb)
H.glu_tile_a_n(jobs, K, D.TM, D.TK)
assert all(np.array_equal(a, b) for a, b in zip(got, want))
print("glu_tile_a_n ok")

# CHECK: scatter_add_rows_n ok
N, n = 300, 200
out0 = rng.standard_normal((N, 256)).astype(np.float32)
want, got = out0.copy(), out0.copy()
b = rng.standard_normal(n).astype(np.float32)
jobs = []
for t in (N, 129, 40, 1):  # the first job's rows are every other job's too
    y = (rng.standard_normal((t, 256)) * 100).astype(bfloat16)
    rows = rng.choice(N, t, replace=False)
    wt = rng.random(t).astype(np.float32)
    H.scatter_add_rows(want, y, rows, wt, b, n)
    jobs.append((y, rows, wt))
H.scatter_add_rows_n(got, jobs, b, n)
assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
print("scatter_add_rows_n ok")
