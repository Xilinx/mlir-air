# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4 on the fused prefill device: full attention (head_dim 512) on
column 0, sliding-window attention (head_dim 256) on column 7, GELU drain.

E2B's LM head is a GEMV on a device of its own (lm_gemv, built for K=1536);
E4B's K=2560 does not fit that GEMV's L1, so its LM head is Q4 GEMMs on this
device, as the dense models' is."""

import types

import gemma4_e2b_q4nx_weights as gw

from ...shared.fused_prefill import device as D
from ...shared.fused_prefill import lm_gemv
from ...shared.fused_prefill.build import attn_points

WINDOW = 512
# Longest prompt with more than one KV head: their records sit max_blocks apart.
MAX_LEN = 2048


def spec(cfg):
    """The fused prefill spec of one Gemma4 model (shared/fused_prefill/build.py)."""
    kv = cfg.n_kv_heads
    dev = D.Config(
        attn={
            "f": D.Attn(col=0, dh=512, lkp=16, dvt=256, kv_heads=kv),
            "s": D.Attn(col=7, dh=256, lkp=32, dvt=128, window=WINDOW, kv_heads=kv),
        },
        act="gelu",
    )
    if kv > 1:
        dev = dev._replace(
            attn={
                k: a._replace(max_blocks=MAX_LEN // a.lkp) for k, a in dev.attn.items()
            }
        )
    # The LM head's device: its own GEMV where it fits, else GEMMs here.
    lm_gemm = cfg.d != lm_gemv.K

    n = lambda v: -(-v // D.NR) * D.NR  # noqa: E731
    d, s = cfg.d, set()
    for dh in (gw.DH_SLIDING, gw.DH_GLOBAL):
        dq = gw.N_Q_HEADS * dh
        s |= {(d, n(dq), 0, 1), (d, n(dh * kv), 0, 1), (dq, n(d), 0, 1)}
    widths = (cfg.inter, 2 * cfg.inter) if cfg.double_wide_mlp else (cfg.inter,)
    for inter in widths:
        s |= {(d, n(inter), 1, 1), (d, n(inter), 0, 1), (inter, n(d), 0, 1)}
    s |= {
        (d, n(gw.PLI_D), 1, 0),
        (gw.PLI_D, n(d), 0, 0),
        (d, n(cfg.num_layers * gw.PLI_D), 0, 0),
    }
    if lm_gemm:
        s |= {(d, n(n1 - n0), 0, 1) for n0, n1 in D.n_split(gw.VOCAB)}
    shapes = sorted(s)

    return types.SimpleNamespace(
        CFG=dev,
        ATTN_POINTS=attn_points(gw.N_Q_HEADS),
        gemm_shapes=lambda: shapes,
        max_len=MAX_LEN if kv > 1 else None,
        lm_gemm=lm_gemm,
        **({} if lm_gemm else {"lm_build": lm_gemv.build}),
    )


_E2B = spec(gw.E2B)
CFG, ATTN_POINTS, gemm_shapes, lm_build = (
    _E2B.CFG,
    _E2B.ATTN_POINTS,
    _E2B.gemm_shapes,
    _E2B.lm_build,
)
