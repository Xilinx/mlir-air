# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPT-OSS-20B on the shared fused prefill device (shared/fused_prefill): the
spec build.py reads.

Attention: 64 query heads in two groups of 32 on columns 0 and 7, four KV
heads each, head_dim 64, with the sinks; one op per layer type, the sliding
one with a 128-token window. GEMMs: q | k | v and o on Q4_1 weights packed at
load, and the experts' gate | up (interleaved per 128 rows, as the bundle
stores them) and down on the MXFP4 blocks the decode holds (wq=2)."""

import sys
import types
from pathlib import Path

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.build import attn_points  # noqa: E402

D_MODEL, D_PAD = 2880, 2944  # the bundle pads K to whole 128-column blocks
HEADS, KV_HEADS, DH = 64, 8, 64
WINDOW = 128
max_len = 4096


def _npad(v):
    return -(-v // D.NR) * D.NR


# site -> (depth, output width, wq)
SHAPES = {
    "qkv": (D_PAD, HEADS * DH + 2 * KV_HEADS * DH, 1),
    "o": (HEADS * DH, D_MODEL, 1),
    "gate_up": (D_PAD, 2 * 23 * 128, 2),  # 23 row groups of gate, of up
    "down": (D_PAD, D_MODEL, 2),
}

_attn = D.Attn(
    col=0,
    dh=DH,
    lkp=32,
    kv_heads=KV_HEADS // 2,
    # the cores read an even number of blocks, so one past the last
    max_blocks=-(-max_len // 32) + 1,
    # the q / k biases
    split_scores=True,
    sinks=True,
)
CFG = D.Config(
    attn={"a0": _attn, "a1": _attn._replace(col=7)},
    act="silu",
    ops={"f": ("a0", "a1"), "s": ("a0", "a1")},
    windows={"s": WINDOW},
    mxfp4=True,
)
ATTN_POINTS = attn_points(HEADS)


def gemm_shapes():
    return sorted({(k, _npad(n), 0, wq) for k, n, wq in SHAPES.values()})
