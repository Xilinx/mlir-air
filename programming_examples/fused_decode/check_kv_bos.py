# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device-free check of KVBuffers with fake BOs.

A cache split over several buffers must hold the same bytes, in the same
order, as one buffer would; a hybrid's ShortConv state must follow group 0's
slabs; and locate() must agree with write_layers(). FakeFD stands in for
fused_decode.kv_layout with tiny slabs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_bos import KVBuffers  # noqa: E402

SLAB, CONV = 8, 4


class FakeBO:
    def __init__(self, nbytes):
        self.b = bytearray(nbytes)

    def write(self, a, off):
        a = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        assert off + a.size <= len(self.b), "write past the buffer"
        self.b[off : off + a.size] = a.tobytes()

    def sync(self, to_dev, nbytes=0, off=0):
        assert off + nbytes <= len(self.b), "sync past the buffer"

    def i16(self, off, n):
        return np.frombuffer(bytes(self.b[off : off + 2 * n]), np.int16)


class FakeFD:
    def __init__(self, n_layers, group, conv):
        self.n, self.g, self.conv = n_layers, group, conv

    def kv_layout(self, attn_maxl):
        n_bufs = -(-self.n // self.g)
        sizes = [min(self.g, self.n - b * self.g) * SLAB for b in range(n_bufs)]
        sizes[0] += self.n * CONV if self.conv else 0
        return SLAB, self.g, sizes, [4] + [8 + b for b in range(n_bufs - 1)]


def check(n_layers, group, conv):
    kvb = KVBuffers(FakeFD(n_layers, group, conv), 0, lambda nb, a: FakeBO(nb))
    kv = np.arange(n_layers * SLAB, dtype=np.int16).reshape(n_layers, SLAB)
    st = np.arange(n_layers * CONV, dtype=np.int16) + 20000 if conv else None
    kvb.write_layers(kv, 0, conv=st)
    flat = np.concatenate(
        [
            bo.i16(0, min(group, n_layers - b * group) * SLAB)
            for b, bo in enumerate(kvb.bos)
        ]
    )
    ok = np.array_equal(flat, kv.reshape(-1))
    if conv:
        ok &= np.array_equal(kvb.bos[0].i16(group * SLAB * 2, n_layers * CONV), st)
    for L in range(n_layers):
        bo, off = kvb.locate(L)
        ok &= np.array_equal(bo.i16(off, SLAB), kv[L])
    return bool(ok) and len(kvb.extra) == len(kvb.bos) - 1


print("one buffer:", check(16, 16, False))
print("split, uneven last group:", check(36, 8, False))
print("split with ShortConv state:", check(16, 8, True))
