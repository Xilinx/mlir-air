# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B on the fused prefill device: full attention (head_dim 512) on
column 0, sliding-window attention (head_dim 256) on column 7, GELU drain."""

import gemma4_e2b_q4nx_weights as gw

from ...shared.fused_prefill import device as D
from ...shared.fused_prefill import lm_gemv
from ...shared.fused_prefill.build import attn_points

WINDOW = 512
CFG = D.Config(
    attn={
        "f": D.Attn(col=0, dh=512, lkp=16, dvt=256),
        "s": D.Attn(col=7, dh=256, lkp=32, dvt=128, window=WINDOW),
    },
    act="gelu",
)
ATTN_POINTS = attn_points(gw.N_Q_HEADS)


def gemm_shapes():
    """(k, n_pad, act, wq) of every Gemma4-E2B prefill GEMM."""
    n = lambda v: -(-v // D.NR) * D.NR  # noqa: E731
    d, s = gw.D, set()
    for dh in (gw.DH_SLIDING, gw.DH_GLOBAL):
        dq = gw.N_Q_HEADS * dh
        s |= {(d, n(dq), 0, 1), (d, n(dh * gw.N_KV_HEADS), 0, 1), (dq, d, 0, 1)}
    for inter in (gw.INTER, 2 * gw.INTER):
        s |= {(d, inter, 1, 1), (d, inter, 0, 1), (inter, d, 0, 1)}
    s |= {
        (d, n(gw.PLI_D), 1, 0),
        (gw.PLI_D, d, 0, 0),
        (d, n(gw.NUM_LAYERS * gw.PLI_D), 0, 0),
    }
    return sorted(s)


def lm_build():
    return lm_gemv.build()
