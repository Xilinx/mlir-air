# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Affine-Q4 decoders (shared/AFFINE_Q4_IMPLEMENTATION.md) on the fused
prefill device.

Their weights are the canonical group-32 tensors of shared/affine_q4.py: 4-bit
codes with a bf16 scale and min per 32 input columns, w = q * scale + min. That
is the grid the device's int4 GEMM dequantizes (packing.pack_q4), so the codes,
scales and mins load unchanged. Norm eps and rope come from the config.json of
the checkpoint the tensors were prepared from, and so do Granite's scalar
multipliers when the config has them:

  embedding_multiplier  the embedding rows, rounded to bf16
  attention_multiplier  replaces the attention's 1 / sqrt(dh) (q_scale)
  residual_multiplier   each sublayer's output, rounded to bf16 (res_scale)
  logits_scaling        divides the logits
"""

import json
import os
from pathlib import Path

import numpy as np

from ..affine_q4 import CanonicalWeights, bf16_float, file_hash, safe_child
from .dense import DensePrefill, _bf, config


class Weights:
    """Canonical affine-Q4 tensors behind packing.Bundle's interface."""

    def __init__(self, root):
        self.cw = CanonicalWeights(root)
        self.manifest = self.cw.manifest
        self._rows = {}

    def q4nx(self, name, rows, K):
        """(q [rows, K] uint8, scale, min [rows, K/32] float32), as stored."""
        q, scale, mn = self.cw.codes(name)
        if q.shape != (rows, K):
            raise ValueError(f"{name}: {q.shape} is not [{rows}, {K}]")
        return q, bf16_float(scale), bf16_float(mn)

    def bf16(self, name):
        return self.cw.bf16(name)

    def rows(self, name, ids):
        """Rows ids of a 2-D bf16 tensor, float32."""
        if name not in self._rows:
            rec = self.cw.entries[name]["arrays"]["value"]
            path = safe_child(self.cw.root, rec["path"])
            self._rows[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        a = self._rows[name]
        return bf16_float(np.ascontiguousarray(a[np.asarray(ids, np.int64)]))


def checkpoint_config(desc, weights):
    """config.json of the checkpoint the canonical tensors came from, checked
    against the hash their manifest records."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_OFFLINE

    m = weights.manifest
    if m["checkpoint"] != desc.repo:
        raise ValueError(f"Q4 tensors are of {m['checkpoint']}, not {desc.repo}")
    local = os.environ.get("AIR_MODEL_CHECKPOINT")
    if local:
        # a local checkpoint (CHECKPOINT=...) carries its own config.json
        path = Path(local) / "config.json"
    else:
        d = snapshot_download(
            desc.repo,
            revision=m["revision"],
            allow_patterns=["config.json"],
            local_files_only=HF_HUB_OFFLINE,
        )
        path = Path(d) / "config.json"
    if file_hash(path) != m["config_sha256"]:
        raise ValueError(f"{path} differs from the one the Q4 tensors came from")
    return path


class AffinePrefill(DensePrefill):
    """DensePrefill on canonical affine-Q4 weights. load_weights takes the
    directory of the canonical tensors."""

    emb_scale = 1.0
    logit_scale = 1.0

    def load_weights(self, model=None):
        if model is None:
            raise ValueError("an affine-Q4 model loads its canonical Q4 directory")
        w = Weights(model)
        path = checkpoint_config(self.desc, w)
        self.eps, self.inv_freq, self.rot, self.rope_scale = config(
            self.desc, path, self.context
        )
        c = json.loads(path.read_text())
        self.emb_scale = float(c.get("embedding_multiplier", 1.0))
        self.res_scale = float(c.get("residual_multiplier", 1.0))
        self.logit_scale = float(c.get("logits_scaling", 1.0))
        if c.get("attention_multiplier") is not None:
            self.q_scale = float(c["attention_multiplier"]) * np.sqrt(self.desc.dh)
        self.load_bundle(w)

    def _embed(self, ids):
        x = super()._embed(ids)
        return _bf(x * self.emb_scale) if self.emb_scale != 1.0 else x

    def prefill(self, ids, logits_from=None):
        z = super().prefill(ids, logits_from)
        return z / self.logit_scale if self.logit_scale != 1.0 else z

    def kv_bf16(self):
        """[layers, P, kv_heads * dh] bf16 roped K and raw V, as kv_stack()
        without the float32 copy."""
        return tuple(
            np.stack([np.concatenate(kv[i]) for kv in self.kv]) for i in range(2)
        )
