# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused prefill GEMM device spec of Qwen3.5-0.8B Q4NX (shared/fused_prefill).

The GEMM ops all take int4 weights (wq=1: the Q4NX values, dequantized by the
cores to BFP16) and BFP16 compute. A Q8 tensor of the bundle (beta/alpha,
lm_head) is one GEMM of depth 2K over its two Q4 halves
(q35_weights.q8_as_q4) and [x | x]; a Q4K tensor enters with its per-group
scale and min bf16(d * sc), bf16(dmin * m) (q35_weights.Q4nx.q4).

The device also holds the fused prefill's attention (shared/fused_prefill/
qwen35.py): attn_bfp16 with head_dim 256, the 8 query heads in two groups of
4 on columns 0 and 7, one KV head each, causal, KV records for max_len tokens.
"""

import sys
import types
from pathlib import Path

# llms/shared is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.models import Desc, spec  # noqa: E402

D_MODEL, INTER = 1024, 3584
HEADS, KV_HEADS, DH = 8, 2, 256
LIN_HEADS, LIN_DK, LIN_DV = 16, 128, 128
LIN_Z = LIN_HEADS * LIN_DV  # 2048
VOCAB = 248320
NIN = 8448  # lin_in GEMM width: 16 heads x 400 (q | k | v | pad), then z


def n(v):
    return -(-v // D.NR) * D.NR


_base = spec(
    Desc(
        "Qwen/Qwen3.5-0.8B",
        D_MODEL,
        DH,
        HEADS,
        KV_HEADS,
        INTER,
        24,
        VOCAB,
        qk_norm=True,
        tied=False,
        max_len=4096,
    )
)
CFG = _base.CFG
ATTN_POINTS = _base.ATTN_POINTS
# lm_head pieces: the device's widest GEMM op
HEAD_PIECES = [(n0, min(VOCAB, n0 + D.MAX_N)) for n0 in range(0, VOCAB, D.MAX_N)]
# site -> (device depth, output width)
SHAPES = {
    "qkv": (D_MODEL, 2 * HEADS * DH + 2 * KV_HEADS * DH),  # q (with gate) | k | v
    "lin_in": (D_MODEL, NIN),  # per head q | k | v | (b, a: zero) | pad, then z
    "ba": (2 * D_MODEL, 2 * LIN_HEADS),  # Q8: beta | alpha projections
    "o": (HEADS * DH, D_MODEL),  # attention o_proj (Q4)
    "lo": (LIN_Z, D_MODEL),  # Gated DeltaNet out_proj (Q4K)
    "gu": (D_MODEL, 2 * INTER),  # MLP gate | up, 128-row blocks interleaved
    "up": (D_MODEL, INTER),  # fused prefill: up, and gate with SiLU in the drain
    "down": (INTER, D_MODEL),
    "head": (2 * D_MODEL, D.MAX_N),  # lm_head (Q8), per piece
}
max_len = 4096


def gemm_shapes():
    """(device depth, n_pad, act, wq)."""
    return sorted(
        {(k, n(m), 0, 1) for k, m in SHAPES.values()} | {(D_MODEL, n(INTER), 1, 1)}
    )
