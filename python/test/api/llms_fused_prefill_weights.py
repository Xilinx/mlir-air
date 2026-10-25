# ./python/test/api/llms_fused_prefill_weights.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch
# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""The fused prefill's weight readers (llms/shared/fused_prefill/packing.py)
return each int4 matrix as raw Q4NX (q, scale, min):

  Bundle      reads a Q4NX bundle, taking the [rows, K] corner of a matrix the
              bundle stores with one dimension zero-padded;
  Q40Bundle   reads a bundle of Q4_0 matrices (signed nibbles, w = q * scale)
              as q + 8 with min = -8 scale, the same values exactly;
  HFBundle    reads a full-precision checkpoint and rounds it onto the grid of
              host_weights.requant_q4k, scale and min in bf16;

and reads/codec pick which of them a weight source gets.

Synthetic files only: matrices packed with the fused decode's own packer.
"""

import json
import struct
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

SRC = Path(sys.argv[1])
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(SRC / "programming_examples")
]
from air_examples.fused_decode.proj_qmm_pack import pack_q4k_blocks  # noqa: E402
from air_examples.llms.shared.fused_prefill import packing as P  # noqa: E402
from air_examples.llms.shared.fused_prefill.models import MODELS  # noqa: E402
from air_examples.llms.shared.infra import host_weights  # noqa: E402

rng = np.random.default_rng(0)
work = Path(tempfile.mkdtemp())


def bf(a):
    return np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)


def write_tensors(path, tensors):
    """A safetensors file of {name: (dtype, shape, bytes)}."""
    hdr, blobs, off = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        hdr[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [off, off + len(raw)],
        }
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def i8(q, scale, mn):
    """A [M, K] Q4NX matrix as the bundle's I8 tensor."""
    blocks = pack_q4k_blocks(q, scale, mn)
    nb = blocks.shape[0] * blocks.shape[1]
    raw = blocks.reshape(nb, -1).view(np.int8)
    return ("I8", list(raw.shape), raw.tobytes())


def q4nx(m, k):
    q = rng.integers(0, 16, (m, k), dtype=np.uint8)
    scale = bf(rng.uniform(0.01, 0.1, (m, k // 32)))
    mn = bf(rng.uniform(-1, 0, (m, k // 32)))
    return q, scale, mn


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


# Bundle: a plain matrix, then the corners of a column- and a row-padded one.
full = q4nx(64, 512)
kpad = q4nx(96, 768)
rpad = q4nx(96, 256)
both = q4nx(128, 256)
bundle = work / "model.q4nx"
write_tensors(
    bundle,
    {"full": i8(*full), "kpad": i8(*kpad), "rpad": i8(*rpad), "both": i8(*both)},
)
b = P.Bundle(bundle)
print("plain:", same(b.q4nx("full", 64, 512), full))
# CHECK: plain: True
print(
    "K padded:",
    same(
        b.q4nx("kpad", 96, 512),
        (kpad[0][:, :512],) + tuple(a[:, :16] for a in kpad[1:]),
    ),
)
# CHECK: K padded: True
print("rows padded:", same(b.q4nx("rpad", 64, 256), tuple(a[:64] for a in rpad)))
# CHECK: rows padded: True
try:
    b.q4nx("full", 64, 768)
    print("too small: no error")
except ValueError:
    print("too small: ValueError")
# CHECK: too small: ValueError
# [128, 256] read as [64, 256]: four blocks could be K padded to 512 or rows
# padded to 128, so the reader refuses rather than guess.
try:
    b.q4nx("both", 64, 256)
    print("ambiguous: no error")
except ValueError:
    print("ambiguous: ValueError")
# CHECK: ambiguous: ValueError

# Which reader a weight source gets. Q4_0 (qwen25_3b_q4) reads only bundles;
# an "hf" model (qwen25_7b_q4nx) reads a checkpoint, or a Q4NX bundle as one.
q40, hf = MODELS["qwen25_3b_q4"], MODELS["qwen25_7b_q4nx"]
ckpt = work / "ckpt"
ckpt.mkdir()
(ckpt / "config.json").write_text("{}")
print("Q4_0 model reads a bundle:", P.reads(q40, work), P.codec(q40, bundle))
# CHECK: Q4_0 model reads a bundle: True q4_0
print("Q4_0 model reads a checkpoint:", P.reads(q40, ckpt))
# CHECK: Q4_0 model reads a checkpoint: False
print(
    "hf model reads both:",
    P.reads(hf, ckpt),
    P.codec(hf, ckpt / "config.json"),
    P.reads(hf, work),
    P.codec(hf, bundle),
)
# CHECK: hf model reads both: True hf True q4nx

# Q40Bundle: signed q, negative scales allowed, mins stored as zero.
qs = rng.integers(-8, 8, (64, 256)).astype(np.int8)
sc = bf(rng.uniform(-0.1, 0.1, (64, 8)))
q40 = work / "q40.q4nx"
write_tensors(q40, {"w": i8(qs.view(np.uint8) & 0xF, sc, np.zeros_like(sc))})
q, s, m = P.Q40Bundle(q40).q4nx("w", 64, 256)
deq = q.astype(np.float32) * np.repeat(s, 32, 1) + np.repeat(m, 32, 1)
print(
    "Q4_0 values exact:",
    np.array_equal(deq, qs.astype(np.float32) * np.repeat(sc, 32, 1)),
)
# CHECK: Q4_0 values exact: True
print(
    "Q4_0 min is -8 scale in bf16:",
    np.array_equal(m, bf(m)) and np.array_equal(m, -8 * s),
)
# CHECK: Q4_0 min is -8 scale in bf16: True

# HFBundle: a BF16 checkpoint, quantized on read.
w = rng.standard_normal((64, 256)).astype(bfloat16)
emb = rng.standard_normal((10, 256)).astype(bfloat16)
hf = work / "hf"
hf.mkdir()
write_tensors(
    hf / "model.safetensors",
    {
        "w": ("BF16", [64, 256], w.tobytes()),
        "emb": ("BF16", [10, 256], emb.tobytes()),
    },
)
hb = P.HFBundle(hf)
want = host_weights.requant_q4k(w.astype(np.float32), 32)
print(
    "HF on the requant_q4k grid:",
    same(hb.q4nx("w", 64, 256), (want[0], bf(want[1]), bf(want[2]))),
)
# CHECK: HF on the requant_q4k grid: True
print(
    "HF rows:",
    np.array_equal(hb.rows("emb", [3, 0, 9]), emb.astype(np.float32)[[3, 0, 9]]),
)
# CHECK: HF rows: True
