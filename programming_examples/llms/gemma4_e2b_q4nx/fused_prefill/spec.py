# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4 on the fused prefill device: full attention (head_dim 512) on
column 0, sliding-window attention (head_dim 256) on column 7, GELU drain.

E2B's LM head is a GEMV on a device of its own (lm_gemv, built for K=1536);
E4B's K=2560 does not fit that GEMV's L1, so its LM head is Q4 GEMMs on this
device, as the dense models' is."""

import gemma4_e2b_q4nx_weights as gw

from ...shared.fused_prefill import device as D
from ...shared.fused_prefill import lm_gemv
from ...shared.fused_prefill.build import attn_points

WINDOW = 512
# Longest prompt with more than one KV head: their records sit max_blocks apart.
MAX_LEN = 2048
_KV = gw.N_KV_HEADS
_attn = dict(kv_heads=_KV)
CFG = D.Config(
    attn={
        "f": D.Attn(col=0, dh=512, lkp=16, dvt=256, **_attn),
        "s": D.Attn(col=7, dh=256, lkp=32, dvt=128, window=WINDOW, **_attn),
    },
    act="gelu",
)
if _KV > 1:
    CFG = CFG._replace(
        attn={k: a._replace(max_blocks=MAX_LEN // a.lkp) for k, a in CFG.attn.items()}
    )
    max_len = MAX_LEN
# The LM head's device: its own GEMV where it fits, else GEMMs here.
LM_GEMM = gw.D != lm_gemv.K
ATTN_POINTS = attn_points(gw.N_Q_HEADS)


def gemm_shapes():
    """(k, n_pad, act, wq) of every Gemma4 prefill GEMM."""
    n = lambda v: -(-v // D.NR) * D.NR  # noqa: E731
    d, s = gw.D, set()
    for dh in (gw.DH_SLIDING, gw.DH_GLOBAL):
        dq = gw.N_Q_HEADS * dh
        s |= {(d, n(dq), 0, 1), (d, n(dh * gw.N_KV_HEADS), 0, 1), (dq, n(d), 0, 1)}
    for inter in (gw.INTER, 2 * gw.INTER) if gw.DOUBLE_WIDE_MLP else (gw.INTER,):
        s |= {(d, n(inter), 1, 1), (d, n(inter), 0, 1), (inter, n(d), 0, 1)}
    s |= {
        (d, n(gw.PLI_D), 1, 0),
        (gw.PLI_D, n(d), 0, 0),
        (d, n(gw.NUM_LAYERS * gw.PLI_D), 0, 0),
    }
    if LM_GEMM:
        s |= {(d, n(n1 - n0), 0, 1) for n0, n1 in D.n_split(gw.VOCAB)}
    return sorted(s)


if not LM_GEMM:

    def lm_build():
        return lm_gemv.build()
