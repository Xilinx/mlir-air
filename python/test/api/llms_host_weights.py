# ./python/test/api/llms_host_weights.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch
# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""Each helper in llms/shared/infra/host_weights.py must return the same bits
as the plain numpy expression in its docstring.

The helpers are checked on random data with all-zero, negative-zero and
constant groups, on shapes that are not a multiple of their row block, and with
float64 inputs, which must stay on the numpy path.
"""

import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

SRC = Path(sys.argv[1])
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(SRC / "programming_examples")
]
from air_examples.llms.shared.infra import host_weights as hw  # noqa: E402

G = 32


def same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    )


def numpy_unpack_q4nx(i16, nbi, nbj):
    """Codec B unpack in numpy: within a block, byte k of column c in row
    group g holds rows g*16 + 2k (low nibble) and g*16 + 2k + 1 (high nibble)
    of that column."""
    rb, cb, ng, par = 32, 256, 8, 16
    g = rb // par
    even = np.array([r * par + 2 * k for r in range(g) for k in range(par // 2)])
    nb = nbi * nbj
    m = nbi * rb

    def f32(a):
        return a.copy().view(bfloat16).astype(np.float32).reshape(nb, ng, rb)

    sc, mn = f32(i16[:, 0:256]), f32(i16[:, 256:512])
    qb = i16[:, 512:2560].copy().view(np.uint8).reshape(nb, g, cb, par // 2)
    q = np.zeros((nb, rb, cb), np.uint8)
    q[:, even, :] = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, rb // 2, cb)
    q[:, even + 1, :] = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, rb // 2, cb)

    def logical(a):
        c = a.shape[2]
        return a.reshape(nbi, nbj, rb, c).transpose(0, 2, 1, 3).reshape(m, -1)

    return logical(q), logical(sc.transpose(0, 2, 1)), logical(mn.transpose(0, 2, 1))


rng = np.random.default_rng(0)
failed = []
for m, k in [(3, 64), (300, 256), (513, 1024)]:
    w = rng.standard_normal((m, k), dtype=np.float32) * 0.03
    w[0, :G], w[1, :G], w[2, :G] = 0.0, -0.0, 0.5
    wb = w.astype(bfloat16)
    q = rng.integers(0, 16, (m, k), dtype=np.uint8)
    qs = rng.integers(-8, 8, (m, k), dtype=np.int8)
    sc = rng.random((m, k // G), dtype=np.float32)
    off = rng.standard_normal((m, k // G), dtype=np.float32)
    S, O = np.repeat(sc, G, axis=1), np.repeat(off, G, axis=1)

    wg = w.reshape(m, k // G, G)
    lo, hi = wg.min(2), wg.max(2)
    s4 = np.where((hi - lo) / 15.0 <= 0, 1.0, (hi - lo) / 15.0).astype(np.float32)
    q4 = np.clip(np.round((wg - lo[..., None]) / s4[..., None]), 0, 15)
    q4 = q4.astype(np.uint8).reshape(m, k)
    amax = np.abs(wg).max(2)
    s0 = np.where(amax <= 0, 1.0, amax / 8.0).astype(np.float32)
    q0 = np.clip(np.rint(wg / s0[..., None]), -8, 7).astype(np.int8).reshape(m, k)
    r4 = hw.requant_q4k(w, G)
    r0 = hw.requant_q4_0(w, G)

    checks = {
        "bf16_bytes_to_f32": same(
            hw.bf16_bytes_to_f32(wb.tobytes(), (m, k)), wb.astype(np.float32)
        ),
        "to_f32": same(hw.to_f32(wb), wb.astype(np.float32)),
        "transpose f32": same(hw.transpose(w), np.ascontiguousarray(w.T)),
        "transpose bf16": same(hw.transpose(wb), np.ascontiguousarray(wb.T)),
        "transpose int8": same(hw.transpose(qs), np.ascontiguousarray(qs.T)),
        "transpose_bf16 f32": same(
            hw.transpose_bf16(w), np.ascontiguousarray(w.T, dtype=bfloat16)
        ),
        "transpose_bf16 bf16": same(hw.transpose_bf16(wb), np.ascontiguousarray(wb.T)),
        "dequant additive": same(hw.dequant_q4(q, sc, off, G), S * q + O),
        "dequant subtractive": same(
            hw.dequant_q4(q, sc, off, G, additive=False), S * (q - O)
        ),
        "dequant no offset": same(
            hw.dequant_q4(qs, sc, None, G), qs.astype(np.float32) * S
        ),
        "dequant_q4_transpose_bf16": same(
            hw.dequant_q4_transpose_bf16(q, sc, off, G),
            np.ascontiguousarray((S * q + O).T, dtype=bfloat16),
        ),
        "requant_q4k": same(r4[0], q4)
        and same(r4[1], s4)
        and same(r4[2], lo.astype(np.float32)),
        "requant_q4k_dequant": same(
            hw.requant_q4k_dequant(w, G), np.repeat(s4, G, 1) * q4 + np.repeat(lo, G, 1)
        ),
        "requant_q4_0": same(r0[0], q0.view(np.uint8)) and same(r0[1], s0),
    }
    failed += [f"{name} {m}x{k}" for name, ok in checks.items() if not ok]

for nbi, nbj in [(1, 1), (3, 5), (17, 2)]:
    i16 = rng.integers(-32768, 32767, (nbi * nbj, 2560), dtype=np.int16)
    i16[:, :512] = (
        (rng.standard_normal((nbi * nbj, 512), dtype=np.float32) * 0.01)
        .astype(bfloat16)
        .view(np.int16)
    )
    got = hw.unpack_q4nx(i16, nbi, nbj)
    want = numpy_unpack_q4nx(i16, nbi, nbj)
    if not all(same(a, b) for a, b in zip(got, want)):
        failed.append(f"unpack_q4nx {nbi}x{nbj}")

# Other dtypes go through the numpy expression itself, so float64 stays float64.
w64, sc64 = w.astype(np.float64), sc.astype(np.float64)
S64 = np.repeat(sc64, G, axis=1)
fallbacks = {
    "dequant float64 scale": same(
        hw.dequant_q4(q, sc64, None, G), q.astype(np.float32) * S64
    ),
    "requant_q4k float64": hw.requant_q4k(w64, G)[0].dtype == np.uint8
    and same(
        hw.requant_q4k(w64, G)[2], w64.reshape(-1, k // G, G).min(2).astype(np.float32)
    ),
    "requant_q4_0 float64": same(
        hw.requant_q4_0(w64, G)[1],
        np.where(
            np.abs(w64.reshape(-1, k // G, G)).max(2) <= 0,
            1.0,
            np.abs(w64.reshape(-1, k // G, G)).max(2) / 8.0,
        ).astype(np.float32),
    ),
    "to_f32 float64": same(hw.to_f32(w64), w64.astype(np.float32)),
}
failed += [name for name, ok in fallbacks.items() if not ok]

for f in failed:
    print("MISMATCH", f)
print("PASS" if not failed else "FAIL")
# CHECK-NOT: MISMATCH
# CHECK: PASS
