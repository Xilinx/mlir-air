# ./python/test/api/llms_fused_prefill_q4k_layout.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root %t | FileCheck %s

"""The fused prefill's layout of a Q4_K block (hostops.q4k_prefill) holds the
same weights as the bundle's.

Each block's weights, w = S s_j q + M m_j with the device's bf16 roundings
and float32 sums, are taken twice: from the bundle's block as FastFlowLM
stores it, and from the laid-out block the way gemm_bfp16.cc's dq4k_bfp16
reads it (its tile and constant offsets, the low nibble the even lane). The
two must agree bit for bit, and the constants must be the bf16 products.
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

FP = SRC / "programming_examples/llms/shared/fused_prefill"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["gcc", "-O2", "-fopenmp", "-shared", "-fPIC", "-DNT=4"]
    + [str(FP / "kernels/hostops.c"), "-o", str(OUT / "libhostops.so"), "-lm"],
    check=True,
)
H.load(str(OUT))
rng = np.random.default_rng(0)

NB = 6
blocks = rng.integers(0, 256, (NB, D.Q4K_BYTES), dtype=np.uint8)
# S and M bf16 of a plausible range (M stored negated), not random bit patterns
sm = (rng.standard_normal((NB, 64)) * 2.0**-10).astype(bfloat16)
blocks[:, 4608:] = sm.view(np.uint8).reshape(NB, 128)
laid = H.q4k_prefill(blocks)
assert laid.shape == (NB, D.Q4K_PF_BYTES)


def bf16(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def weights(sc, mn, q):
    """float32 sc, mn and q broadcast together: the device's c = mn - 128 sc,
    then c + (128 + q) sc."""
    c = (mn - np.float32(128) * sc).astype(np.float32)
    return (c + (np.float32(128) + q) * sc).astype(np.float32)


def from_bundle(b):
    """[32 n][256 k] float32 of a bundle block."""
    S = b[4608:4672].view(bfloat16).astype(np.float32)
    M = b[4672:4736].view(bfloat16).astype(np.float32)
    s = b[0:256].reshape(8, 32).astype(np.float32)
    m = b[256:512].reshape(8, 32).astype(np.float32)
    sc, mn = bf16(s * S), bf16(m * M)  # [group][n]
    nib = b[512:4608].reshape(256, 16)
    q = np.stack([nib & 15, nib >> 4], -1).reshape(256, 32).astype(np.float32)
    g = np.arange(256) // 32
    return weights(sc[g].T, mn[g].T, q.T)


def from_laid(b):
    """[32 n][256 k] float32 of a laid-out block, read as dq4k_bfp16 reads it."""
    cst = b[4096:5120].view(bfloat16).astype(np.float32).reshape(8, 2, 2, 16)
    w = np.empty((32, 256), np.float32)
    for h in range(2):
        for p in range(2):
            for gl in range(4):
                g = h * 4 + gl
                for kq in range(4):
                    for o in range(2):
                        t = b[((((h * 2 + p) * 4 + gl) * 4 + kq) * 2 + o) * 32 :][:32]
                        lanes = np.stack([t & 15, t >> 4], -1).reshape(8, 8)
                        n0, k0 = p * 16 + o * 8, h * 128 + gl * 32 + kq * 8
                        sc = cst[g, p, 0, o * 8 : o * 8 + 8][:, None]
                        mn = cst[g, p, 1, o * 8 : o * 8 + 8][:, None]
                        w[n0 : n0 + 8, k0 : k0 + 8] = weights(
                            sc, mn, lanes.astype(np.float32)
                        )
    return w


# CHECK: same weights
for i in range(NB):
    a, b = from_bundle(blocks[i]), from_laid(laid[i])
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), i
print("same weights")

# CHECK: constants are the bf16 products
S = blocks[:, 4608:4672].view(bfloat16).astype(np.float32)
s = blocks[:, 0:256].reshape(NB, 8, 32).astype(np.float32)
want = (s * S[:, None, :]).astype(bfloat16).reshape(NB, 8, 2, 16)
got = laid[:, 4096:5120].view(bfloat16).reshape(NB, 8, 2, 2, 16)[:, :, :, 0]
assert np.array_equal(got.view(np.uint16), want.view(np.uint16))
print("constants are the bf16 products")
