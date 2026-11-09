# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused prefill device spec of Whisper large-v3-turbo
(shared/fused_prefill), read by build.py.

The projection GEMMs take the bundle's affine Q4 weights (wq=1), the conv
front end's its BF16 ones (wq=0), all with BFP16 compute. The device's
attention is the decoder's: attn_bfp16 at head dim 64, noncausal with a
run-time key count, the 20 heads in two groups of 10 on columns 0 and 7, each
head its own KV head, KV records for up to 1536 keys (the 1500 encoder frames
of the cross-attention, padded).
"""

import sys
import types
from pathlib import Path

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.build import attn_points  # noqa: E402

D_MODEL, HEADS, DH, FF, DEC_LAYERS = 1280, 20, 64, 5120, 4
FRAMES = 1500
max_len = -(-FRAMES // D.M) * D.M  # 1536


def n(v):
    return -(-v // D.NR) * D.NR


_a = D.Attn(
    col=0,
    dh=DH,
    lkp=32,
    kv_heads=HEADS // 2,
    # the cores read an even number of blocks, so one past the last
    max_blocks=max_len // 32 + 1,
    noncausal=True,
)
CFG = D.Config(attn={"a0": _a, "a1": _a._replace(col=7)}, ops={"a": ("a0", "a1")})
ATTN_POINTS = attn_points(HEADS)
# site -> (device depth, output width)
SHAPES = {
    "qkv": (D_MODEL, 3 * D_MODEL),
    "o": (D_MODEL, D_MODEL),
    "fc1": (D_MODEL, FF),
    "fc2": (FF, D_MODEL),
    # the decoder's cross-attention K | V, all layers in one op
    "xkv": (D_MODEL, 2 * DEC_LAYERS * D_MODEL),
}
# the decoder's sites are qkv, o and fc1/fc2 shapes; its head is the bundle's
# Q4 lm_head (VOCAB_PAD rows) in pieces of at most MAX_N columns
VOCAB, VOCAB_PAD = 51866, 51872
HEAD_PIECES = D.n_split(VOCAB_PAD)


# The conv front end as GEMMs on the bundle's BF16 conv weights (wq=0): a row
# of A is 3 consecutive frames of the conv's input, 3 x channels deep.
N_MELS = 128
CONV_SHAPES = {
    "conv1": (3 * N_MELS, D_MODEL),
    "conv2": (3 * D_MODEL, D_MODEL),
}


def gemm_shapes():
    """(device depth, n_pad, act, wq)."""
    heads = {(D_MODEL, n(n1 - n0), 0, 1) for n0, n1 in HEAD_PIECES}
    convs = {(k, n(m), 0, 0) for k, m in CONV_SHAPES.values()}
    return sorted({(k, n(m), 0, 1) for k, m in SHAPES.values()} | heads | convs)
