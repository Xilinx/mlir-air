# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM's Whisper large-v3-turbo `model.q4nx`
(FastFlowLM/Whisper-V3-Turbo-NPU2 @ 594eecd2d80b20cbb04ef0099162335d1dd1899a),
read by tensor name.

The file is a safetensors container. BF16 tensors hold the conv front end,
positional and token embeddings, every LayerNorm and every bias. Each linear
weight (and lm_head, a quantized copy of the tied embedding) is an I8 blob of
affine Q4 blocks, 32 rows x 128 columns, 2560 bytes each, row-block major:

  scale bf16 [4][32], min bf16 [4][32]  (per group of 32 columns, per row)
  nibbles [2 planes][128 columns][16 rows / 2]  (block row = 16 plane + r,
          low nibble = even r)
  w = scale * q + min, q in [0, 15].

That is the canonical AIR affine Q4 (group 32 along K, bf16 scale and min),
so q4() hands the logical values over unchanged.
"""

import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = "FastFlowLM/Whisper-V3-Turbo-NPU2"
REVISION = "594eecd2d80b20cbb04ef0099162335d1dd1899a"
SHA256 = "8fb97604bf5762ee26efa696cfc9eb70724110358c7b4ca628a7973bf8a16291"
RB, CB, G = 32, 128, 32
BLOCK = 2560

D, H, DH, FF = 1280, 20, 64, 5120
ENC_LAYERS, DEC_LAYERS = 32, 4
N_MELS, N_FRAMES, N_CTX = 128, 3000, 1500
MAX_TOKENS, VOCAB = 448, 51866


def bundle_path(source=None):
    """The model.q4nx to load: `source` (else $Q4NX_MODEL_SOURCE) when it is a
    file or a directory holding model.q4nx, else the Hub download of REPO at
    REVISION. Its sha256 must be SHA256."""
    source = source or os.environ.get("Q4NX_MODEL_SOURCE")
    if source and os.path.isdir(source):
        source = os.path.join(source, "model.q4nx")
    if source and os.path.isfile(source):
        path = source
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, "model.q4nx", revision=REVISION)
    got = sha256(path)
    if got != SHA256:
        raise ValueError(
            f"{path}: sha256 {got}, expected {SHA256} ({REPO} @ {REVISION})"
        )
    return path


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Q4nx:
    def __init__(self, path=None):
        self.path = Path(bundle_path(path))
        self.raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        n = int(self.raw[:8].view(np.uint64)[0])
        self.header = json.loads(bytes(self.raw[8 : 8 + n]))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def _bytes(self, name):
        e = self.header[name]
        a, b = e["data_offsets"]
        return self.raw[self.base + a : self.base + b]

    def kind(self, name):
        return self.header[name]["dtype"]

    def bf16(self, name):
        e = self.header[name]
        assert e["dtype"] == "BF16", name
        return self._bytes(name).view(bfloat16).reshape(e["shape"])

    def q4(self, name, rows, k):
        """Affine grid (q uint8 [rows, k], scale bf16 [rows, k/32],
        min bf16 [rows, k/32]) of the [rows, k] weight `name`."""
        assert self.kind(name) == "I8", name
        assert rows % RB == 0 and k % CB == 0, (name, rows, k)
        blk = np.asarray(self._bytes(name)).reshape(rows // RB, k // CB, BLOCK)
        sm = blk[..., :512].copy().view(bfloat16).reshape(rows // RB, k // CB, 2, 4, RB)
        # [rb, kb, group, row] -> [rb, row, kb, group]
        scale = sm[:, :, 0].transpose(0, 3, 1, 2).reshape(rows, k // G)
        mn = sm[:, :, 1].transpose(0, 3, 1, 2).reshape(rows, k // G)
        nb = blk[..., 512:].reshape(rows // RB, k // CB, 2, CB, RB // 4)
        nib = np.stack([nb & 15, nb >> 4], -1).reshape(nb.shape[:-1] + (RB // 2,))
        # [rb, kb, plane, col, r] -> [rb, plane, r, kb, col]
        q = nib.transpose(0, 2, 4, 1, 3).reshape(rows, k)
        return q, scale, mn

    def tokenizer_path(self):
        """The bundle's tokenizer.json: beside model.q4nx when it is there,
        else from the Hub at REVISION."""
        beside = self.path.parent / "tokenizer.json"
        if beside.is_file():
            return beside
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(REPO, "tokenizer.json", revision=REVISION))

    def dequant(self, name, rows, k):
        q, s, m = self.q4(name, rows, k)
        s = np.repeat(s.astype(np.float32), G, 1)
        m = np.repeat(m.astype(np.float32), G, 1)
        return q.astype(np.float32) * s + m
