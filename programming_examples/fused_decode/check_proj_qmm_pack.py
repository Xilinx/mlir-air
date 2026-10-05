# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""pack_q4k_cascade against the one-block reference, pack_q4k_block.

Device-free. Every emission order, scales in float32 and float64, and nibble
inputs with their high bits set, which the packer must ignore.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proj_qmm_pack as P  # noqa: E402


def by_block(q, scale, mn, NCX, NCY, **order):
    """pack_q4k_cascade's emission order, one pack_q4k_block per block."""
    M, K = q.shape
    nbi, nbj = M // P.ROW_BLOCK, K // P.COL_BLOCK
    nbi_pc = nbi // (NCX * NCY)
    steps = [(i, j) for i in range(nbi_pc) for j in range(nbj)]
    if order.get("dual_chan"):
        seq = [
            (i, j, cy)
            for h in range(2)
            for (i, j) in steps
            for cy in range(h * (NCY // 2), (h + 1) * (NCY // 2))
        ]
    else:
        seq = [(i, j, cy) for (i, j) in steps for cy in range(NCY)]
    out = []
    for cx in range(NCX):
        for i, j, cy in seq:
            if order.get("iter_major"):
                gi = i * (NCX * NCY) + cx * NCY + cy
            elif order.get("core_major"):
                gi = (cx * NCY + cy) * nbi_pc + i
            else:
                gi = cx * (NCY * nbi_pc) + i * NCY + cy
            r, c, g = gi * P.ROW_BLOCK, j * P.COL_BLOCK, j * P.N_GROUPS
            out.append(
                P.pack_q4k_block(
                    q[r : r + P.ROW_BLOCK, c : c + P.COL_BLOCK],
                    scale[r : r + P.ROW_BLOCK, g : g + P.N_GROUPS],
                    mn[r : r + P.ROW_BLOCK, g : g + P.N_GROUPS],
                )
            )
    return np.concatenate(out)


rng = np.random.default_rng(0)
M, K, NCX, NCY = 512, 768, 4, 4
q = rng.integers(0, 256, (M, K), dtype=np.uint8)
for name, order in (
    ("default", {}),
    ("core_major", dict(core_major=True)),
    ("iter_major", dict(iter_major=True)),
    ("dual_chan", dict(dual_chan=True)),
    # The requantizers combine a row order with the dual-channel split.
    ("iter_major+dual_chan", dict(iter_major=True, dual_chan=True)),
    ("core_major+dual_chan", dict(core_major=True, dual_chan=True)),
):
    ok = True
    for dt in (np.float32, np.float64):
        sc = rng.standard_normal((M, K // P.GROUP)).astype(dt)
        mn = rng.standard_normal((M, K // P.GROUP)).astype(dt)
        got = P.pack_q4k_cascade(q, sc, mn, NCX, NCY, **order)
        want = by_block(q, sc, mn, NCX, NCY, **order)
        ok &= got.dtype == want.dtype and np.array_equal(got, want)
    print(f"{name}: cascade matches the per-block packer: {ok}")
