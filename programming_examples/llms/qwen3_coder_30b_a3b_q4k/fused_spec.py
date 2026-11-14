# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-Coder-30B-A3B on the shared fused prefill device (shared/fused_prefill):
the spec build.py reads.

Every GEMM reads the bundle's Q4_K blocks where they lie (Config.q4k): the
dense projections from the prefill's own copy, the routed experts from the
decode's WE buffer. Attention: 32 query heads in two groups of 16 on columns 0
and 7, two KV heads each, head_dim 128, causal."""

import sys
import types
from pathlib import Path

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.build import attn_points  # noqa: E402

D_MODEL, DQ, DH = 2048, 4096, 128
HEADS, KV_HEADS = 32, 4
I = 768  # expert width
max_len = 4096


def npad(v):
    return -(-v // D.NR) * D.NR


# site -> (depth, output width)
SHAPES = {
    "qkv": (D_MODEL, DQ + 2 * KV_HEADS * DH),  # q | k | v
    "o": (DQ, D_MODEL),
    # experts' up | gate, interleaved per 128 rows as the decode's WE holds
    # them, and down
    "gu": (D_MODEL, 2 * I),
    "down": (I, D_MODEL),
}

_attn = D.Attn(
    col=0,
    dh=DH,
    lkp=32,
    kv_heads=KV_HEADS // 2,
    # the cores read an even number of blocks, so one past the last
    max_blocks=-(-max_len // 32) + 1,
)
# an expert's column blocks in the decode's WE: up | gate, then down
E_GROUPS = (2 * I // 128) * (D_MODEL // D.Q4K_COLS) + (D_MODEL // 128) * (
    I // D.Q4K_COLS
)
N_EXPERTS, TOP_K = 128, 8
CFG = D.Config(
    attn={"a0": _attn, "a1": _attn._replace(col=7)},
    act="silu",
    ops={"a": ("a0", "a1")},
    q4k=True,
    # kmax: a group of small experts can hold a single chunk of every
    # expert. cap, per kind: one single chunk per expert at most, and pairs
    # for max_len * TOP_K rows plus one per expert.
    experts=D.Experts(
        kmax=N_EXPERTS,
        cap=max_len * TOP_K // (2 * D.M) + N_EXPERTS,
        stride=E_GROUPS,
        experts=N_EXPERTS,
    ),
)
ATTN_POINTS = attn_points(HEADS)


def gemm_shapes():
    return sorted({(k, npad(n), 0, 3) for k, n in SHAPES.values()})


def expert_shapes():
    """The experts' GEMMs, run as "x" ops on runs of blocks."""
    return sorted({(k, npad(n), 0, 3) for k, n in (SHAPES["gu"], SHAPES["down"])})
