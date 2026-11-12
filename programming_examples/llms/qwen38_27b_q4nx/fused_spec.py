# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.8-27B on the shared fused prefill device (shared/fused_prefill):
the spec build.py reads.

Every GEMM reads the bundle's Q4_K blocks where they lie (Config.q4k), from
the prefill's own copy. A GEMM op is at most D.MAX_N (12288) columns, so the
wider projections are split: the Gated DeltaNet's q | k | v and z, the
attention's q | gate and k | v, and the MLP's interleaved up | gate in three
pieces. Attention: 24 query heads in two groups of 12 on columns 0 and 7, two
KV heads each, head_dim 256, causal."""

import sys
import types
from pathlib import Path

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.build import attn_points  # noqa: E402

D_MODEL, DQ, DH = 5120, 6144, 256
HEADS, KV_HEADS = 24, 4
QKV, Z = 10240, 6144  # Gated DeltaNet q | k | v, and z
I = 17408  # the MLP's width
max_len = 4096

# the interleaved up | gate in pieces of whole 128-row up / gate pairs
GU_PIECES = [(0, 12288), (12288, 24576), (24576, 2 * I)]


def npad(v):
    return -(-v // D.NR) * D.NR


# site -> (depth, output width)
SHAPES = {
    "qkv": (D_MODEL, QKV),
    "z": (D_MODEL, Z),
    "lin_out": (Z, D_MODEL),
    "qg": (D_MODEL, 2 * DQ),  # q | gate
    "kv": (D_MODEL, 2 * KV_HEADS * DH),
    "o": (DQ, D_MODEL),
    **{f"gu{i}": (D_MODEL, c1 - c0) for i, (c0, c1) in enumerate(GU_PIECES)},
    "down": (I, D_MODEL),
}
assert all(n <= D.MAX_N for _, n in SHAPES.values())

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
