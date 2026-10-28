# ./python/test/api/llms_fused_prefill_attn_format.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root %t | FileCheck %s

"""The fused prefill's host side for a device whose attention reads bf16 q and
KV records and writes a bf16 o, all in 8x8 blocks (manifest "attn_format":
"bf16").

q is q_pack's bf16 [heads, M, dh] as blocks [M/8, dh/8] per head, each block
row-major. A KV record is the K tile's blocks [lkp/8, dh/8], then V's in
"attn_v_chunks" column chunks, each chunk's blocks [lkp/8, dvt/8]; record r of
KV head h is at element (h * nblk + r) * 2 * lkp * dh. o comes back in q's
order and goes into the o-projection's A in either GEMM format. Without the
key the engine keeps the bfp16 path.
"""

import json
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


class NoDevice(Exception):
    pass


def no_device(i):
    raise NoDevice()


class SubBO:
    """xrt.bo(parent, size, offset), as attn_start makes it."""

    def __init__(self, parent, size, offset):
        self.parent, self.size, self.offset = parent, size, offset


class BO:
    """A host BO: its bytes, and the ranges synced to the device."""

    def __init__(self, nbytes):
        self.buf = bytearray(int(nbytes))
        self.synced = []

    def map(self):
        return self.buf

    def size(self):
        return len(self.buf)

    def sync(self, direction, nbytes=None, offset=0):
        if direction == 0:
            self.synced.append((nbytes, offset))


sync = SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=0, XCL_BO_SYNC_BO_FROM_DEVICE=1)
sys.modules["pyxrt"] = SimpleNamespace(
    xclBOSyncDirection=sync, bo=SubBO, device=no_device
)
from air_examples.llms.shared.fused_prefill.engine import Engine  # noqa: E402

# two KV heads with records max_blocks apart; a window, so the second chunk's
# attention starts past record 0
LKP, DH, KVH, MAX_LEN = 32, 64, 2, 256
g = D.Attn(col=7, dh=DH, lkp=LKP, kv_heads=KVH, max_blocks=-(-MAX_LEN // LKP) + 1)
cfg = D.Config(attn={"a": g}, windows={"a": 2 * LKP})


builds = iter(range(1 << 20))


def engine(man):
    """An engine on build dir manifest man, up to opening the device."""
    # a fresh directory each time: the engine loads the host library from it
    d = OUT / f"build_{next(builds)}"
    d.mkdir(exist_ok=True)
    man = dict(man, attn={"a": "attn_a"}, heads=4)
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "libhostops.so").write_bytes((OUT / "libhostops.so").read_bytes())
    eng = object.__new__(Engine)
    try:
        Engine.__init__(eng, d, cfg, MAX_LEN)
    except NoDevice:
        pass
    eng.fp = SimpleNamespace(bo=lambda nbytes, arg: BO(max(int(nbytes), 4096)))
    eng.dummy = [BO(4096) for _ in range(eng.nargs)]
    return eng


def same(a, b):
    return np.array_equal(a.view(np.uint16), b.view(np.uint16))


def blocks(x):
    """[..., rows, cols] -> its 8x8 blocks [..., rows/8, cols/8, 8, 8]."""
    *b, r, c = x.shape
    return x.reshape(*b, r // 8, 8, c // 8, 8).swapaxes(-3, -2)


VC = 2  # V's column chunks
bf = engine({"attn_format": "bf16", "attn_v_chunks": VC})
# CHECK: v chunks checked
# Each chunk must hold whole 8-column blocks of V.
for bad in (0, 3, 16, "2"):
    try:
        engine({"attn_format": "bf16", "attn_v_chunks": bad})
    except ValueError:
        continue
    raise AssertionError(bad)
print("v chunks checked")
# CHECK: default bfp16 ok
bfp = engine({})
assert (bf.attn_format, bfp.attn_format) == ("bf16", "bfp16")
assert bfp.kv_rec(g) == D.kv_rec(g)
assert bf.kv_rec(g) == 2 * LKP * DH
print("default bfp16 ok")

# Two appends: a whole chunk, then a partial one. Every record of each KV
# head, the spare block past max_len included, against K/V cut into blocks.
# CHECK: kv records ok
T1, T2 = D.M, 45
k = rng.standard_normal((T1 + T2, KVH, DH)).astype(np.float32)
v = rng.standard_normal((T1 + T2, KVH, DH)).astype(np.float32)
nblk = g.max_blocks
rec = 2 * LKP * DH
want = np.zeros((KVH, nblk, rec), bfloat16)
dvt = DH // VC
for h in range(KVH):
    for b in range(-(-(T1 + T2) // LKP)):
        kb = np.zeros((LKP, DH), bfloat16)
        vb = np.zeros((LKP, DH), bfloat16)
        n = min(LKP, T1 + T2 - b * LKP)
        kb[:n] = k[b * LKP : b * LKP + n, h]
        vb[:n] = v[b * LKP : b * LKP + n, h]
        chunks = [blocks(vb[:, z * dvt : (z + 1) * dvt]).ravel() for z in range(VC)]
        want[h, b] = np.concatenate([blocks(kb).ravel(), *chunks])
kv = bf.kv_bo("a")
bo = kv["parts"][0]["bo"]
assert bo.size() == KVH * nblk * rec * 2
bf.kv_append(kv, 0, k[:T1], v[:T1])
bf.kv_append(kv, T1, k[T1:], v[T1:])
got = np.frombuffer(bo.map(), bfloat16).reshape(want.shape)
assert same(got, want)
# record r of head h at (h * nblk + r) * rec; only the new blocks synced
nb1, nb2 = T1 // LKP, -(-T2 // LKP)
assert bo.synced == [(nb1 * rec * 2, h * nblk * rec * 2) for h in range(KVH)] + [
    (nb2 * rec * 2, (h * nblk + nb1) * rec * 2) for h in range(KVH)
]
print("kv records ok")

# The bfp16 records of the same tokens, unchanged.
# CHECK: kv bfp16 ok
kvp = bfp.kv_bo("a")
bfp.kv_append(kvp, 0, k[:T1], v[:T1])
gotp = np.frombuffer(kvp["parts"][0]["bo"].map(), np.uint8)
gotp = gotp.reshape(KVH, nblk, D.kv_rec(g) * 2)
want_p = np.zeros_like(gotp)
for h in range(KVH):
    kb = k[:T1, h].astype(bfloat16)
    vb = v[:T1, h].astype(bfloat16)
    H.kv_rec_bfp(kb, vb, 0, T1, DH, LKP, want_p[h])
assert np.array_equal(gotp, want_p)
print("kv bfp16 ok")

# The second chunk's attention reads the records from block k0 on: its KV
# argument is a sub-BO of the records from k0 * record bytes.
# CHECK: kv sub-BO ok
bf.attn_setup()
bf.chunk_attn = {}
args = {}
run = SimpleNamespace(
    set_arg=lambda i, x: args.__setitem__(i, x), r=SimpleNamespace(start=lambda: 0)
)
part = dict(q=BO(4096), o=BO(4096))
bf.chunk_attn[("a", 1)] = dict(
    parts=[part], ib=BO(4096), iv=np.zeros(4, np.uint32), run=run, pt=None
)
seqs = []
bf._insts = lambda name, *a: seqs.append(a) or np.zeros(4, np.uint32)
bf.attn_start(kv, 1, T1, T2)
k0 = T1 // LKP - 2
sub = args[3 + bf.slot["a"] + 1]
assert (sub.parent, sub.offset, sub.size) == (bo, k0 * rec * 2, bo.size() - sub.offset)
nend = -(-(T1 + T2) // LKP)
assert seqs == [(nend - k0 + (nend - k0) % 2, T1 // LKP, k0)], seqs
print("kv sub-BO ok")

# q: q_pack's output, written into the q BO in 8x8 blocks.
# CHECK: q bf16 ok
hg, T = 3, 77
scale = float(np.sqrt(DH) * 0.5)
q = rng.standard_normal((T, hg, DH)).astype(np.float32)
want = np.zeros((hg, D.M, DH), bfloat16)
H.q_pack(q, scale, want)
ref = np.zeros((hg, D.M, DH), bfloat16)
ref[:, :T] = (q * scale).astype(bfloat16).transpose(1, 0, 2)
assert same(want, ref)
bos = list(bf.dummy)
p = bf._attn_part(g, hg, bf.slot["a"], bos)
assert p["q"].size() == hg * D.M * DH * 2
p["qm"][:] = 1.0  # a previous chunk's q
q16 = bf.q_stage(p)
H.q_pack(q, scale, q16)
bf.q_write(p, q16)
got = np.frombuffer(p["q"].map(), bfloat16)[: want.size]
assert same(got, blocks(want).ravel())
# the bfp16 engine packs the same values
pp = bfp._attn_part(g, hg, bfp.slot["a"], list(bfp.dummy))
q16 = bfp.q_stage(pp)
assert q16 is not pp["qm"]
H.q_pack(q, scale, q16)
bfp.q_write(pp, q16)
qb = np.zeros(hg * D.M * DH * 9 // 8, np.uint8)
H.q_bfp(want, LKP, qb)
assert np.array_equal(np.frombuffer(pp["q"].map(), np.uint8)[: qb.size], qb)
print("q bf16 ok")

# o: heads h0.. of an o in 8x8 blocks into the o-projection's A, in either
# GEMM format; the rest of A keeps what it had. A head of 96 straddles K
# steps.
# CHECK: o blocks ok
K = 3 * D.TK


def tiled(x):
    """x [t, k] -> bf16 tiles [HR, k/TK, TM, TK], rows past t zero."""
    a = np.zeros((D.M, x.shape[1]), bfloat16)
    a[: x.shape[0]] = x
    hr, ks = D.M // D.TM, x.shape[1] // D.TK
    return a.reshape(hr, D.TM, ks, D.TK).transpose(0, 2, 1, 3)


def a_buf(fmt):
    dt, shape = P.a_layout(fmt, K, 1)
    return np.zeros(shape, dt)[:, : K // D.TK, 0]


for dh in (64, 96):
    nh, h0, T = 2, 1, D.TM + 13
    o = rng.standard_normal((nh, D.M, dh)).astype(bfloat16)
    before = rng.standard_normal((T, K)).astype(np.float32)
    want = before.astype(bfloat16)
    want[:, h0 * dh : (h0 + nh) * dh] = o[:, :T].transpose(1, 0, 2).reshape(T, -1)
    om = np.ascontiguousarray(blocks(o)).reshape(o.shape)
    p = dict(om=om, hg=nh, g=g._replace(dh=dh))
    a16 = a_buf("bf16")
    H.tile_a(before, a16, D.TM, D.TK)
    bf.o_read(p, T, h0, K, a16)
    assert same(a16, tiled(want)), dh
    a8 = a_buf("bfp16")
    H.tile_a(before, a8, D.TM, D.TK)
    bf.o_read(p, T, h0, K, a8)
    want8 = a_buf("bfp16")
    H.tile_a(want.astype(np.float32), want8, D.TM, D.TK)
    assert np.array_equal(a8, want8), dh
print("o blocks ok")
