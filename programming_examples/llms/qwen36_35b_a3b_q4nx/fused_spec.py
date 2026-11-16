# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.6-35B-A3B on the shared fused prefill device (shared/fused_prefill):
the spec build.py reads.

Every GEMM reads the bundle's Q4_K blocks in the device's layout of them
(Config.q4k, hostops.q4k_prefill), from the prefill's own copy of the dense
projections and of the routed experts. Attention: 16 query heads in two groups of 8 on columns 0
and 7, one KV head each, head_dim 256, causal."""

import sys
import types
from pathlib import Path

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.build import attn_points  # noqa: E402

D_MODEL, DQ, DH = 2048, 4096, 256
HEADS, KV_HEADS = 16, 2
QKV, Z = 8192, 4096  # Gated DeltaNet q | k | v, and z
I = 512  # routed and shared expert width
max_len = 4096


def npad(v):
    return -(-v // D.NR) * D.NR


# site -> (depth, output width)
SHAPES = {
    "lin_in": (D_MODEL, QKV + Z),  # q | k | v | z
    "lin_out": (Z, D_MODEL),
    "qgkv": (D_MODEL, 2 * DQ + 2 * KV_HEADS * DH),  # q | gate | k | v
    "o": (DQ, D_MODEL),
    # experts' up | gate, interleaved per 128 rows as the bundle's WE holds
    # them, and down; the shared expert takes the same shapes
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
CFG = D.Config(
    attn={"a0": _attn, "a1": _attn._replace(col=7)},
    act="silu",
    ops={"a": ("a0", "a1")},
    q4k=True,
)
ATTN_POINTS = attn_points(HEADS)


def gemm_shapes():
    return sorted({(k, npad(n), 0, 3) for k, n in SHAPES.values()})
