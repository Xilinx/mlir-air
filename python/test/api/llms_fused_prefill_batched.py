# ./python/test/api/llms_fused_prefill_batched.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root %t | FileCheck %s

"""Several of the fused prefill's ops in one dispatch, and the host helpers
that feed them in one parallel region.

The "x" op, a GEMM over a run-time run of expert blocks, must keep its launch
loop rolled and take each block's expert and buffers at run time.
tile_a_rows_n, glu_tile_a_n and scatter_add_rows_n must write what the single
calls do, byte for byte; scatter_add_rows_n's jobs share rows, so its sums
must also come in the single calls' order.
"""

import re
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

FP = SRC / "programming_examples/llms/shared/fused_prefill"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["gcc", "-O2", "-fopenmp", "-shared", "-fPIC", "-DNT=4"]
    + [str(FP / "kernels/hostops.c"), "-o", str(OUT / "libhostops.so"), "-lm"],
    check=True,
)
H.load(str(OUT))
rng = np.random.default_rng(0)


# CHECK: x op ok
# The "x" op: the GEMM on a run of expert blocks, one launch wave per block. Its
# wave count, pair count, first blocks and experts are run-time scalars, so the
# launch loop stays rolled, the wave's expert is picked by an index_switch over
# the expert scalars, and the pairs' and single chunks' feeds are the two arms
# of an scf.if on wave < n1, each on its own buffers of blocks.
cfg = D.Config(
    attn={"a0": D.Attn(col=0, dh=128, lkp=32, kv_heads=2, max_blocks=4)},
    act="silu",
    q4k=True,
    experts=D.Experts(kmax=3, cap=5, stride=7, experts=4),
)
x = str(D.build(cfg, "x", D.Q4K_COLS, D.NR, 0, 3, name="x", own=True).build("npu2"))
sig = re.search(r"func.func @x\((.*?)\) \{", x).group(1)
args = [t.split(": ")[1] for t in sig.split(", %")]
# A and C of pairs, B, then the single chunks' (and the attention's) buffers,
# then nb, n1, i1, i0 and one expert per block
ks = D.k_pad(D.Q4K_COLS) // D.TK
for q, a, c in ((1, 0, 2), (0, 3, 4)):
    assert args[a] == f"memref<5x{D.HR}x{ks}x{1 + q}x{D.A_BFP}xi8>", args[a]
    assert args[c] == f"memref<5x{1 + q}x{D.M}x1x{D.GNC}x{D.TN}xbf16>", args[c]
# one layer's experts: the last one's panels, 3 experts' strides past the first
rows = (D.NR // (2 * D.TN) - 1) + ks // 2 + 3 * 7
assert args[1] == f"memref<{rows}x4x{D.Q4K_PF_BYTES}xi8>", args[1]
assert args[-7:] == ["i32"] * 7, args
loop = re.search(r"scf.for %\S+ = %c0 to (%\S+) step", x)
# the bound is nb, not a constant
nb = len(args) - 7
assert loop and not loop.group(1).startswith("%c"), loop
assert f"arith.index_cast %arg{nb} : i32 to index" in x
# kmax 3: cases 0 and 1, the last expert the default
assert "scf.index_switch" in x and "case 1 {" in x and "case 2 {" not in x
# pairs or single chunks: one scf.if, which keeps the stream free of
# queue-space polls (an scf.index_switch arm gets one per transfer)
assert len(re.findall(r"scf.if %\S+ \{", x)) == 1, re.findall(r"scf.if .*", x)
assert len(re.findall(r"air.channel.put  @gAIn", x)) == 2 * D.HR
assert len(re.findall(r"air.channel.get  @gCOut", x)) == 2 * D.GNC
print("x op ok")

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
