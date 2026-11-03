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

OpenELM (OpenELMPrefill) varies its head counts and FFN width per layer; Phi-2
has its own layer (phi2.py).
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
        c = self.read_config(checkpoint_config(self.desc, w))
        self.emb_scale = float(c.get("embedding_multiplier", 1.0))
        self.res_scale = float(c.get("residual_multiplier", 1.0))
        self.logit_scale = float(c.get("logits_scaling", 1.0))
        if c.get("no_rope_layers") is not None:
            self.rope_layers = [bool(v) for v in c["no_rope_layers"]]
        if c.get("attention_multiplier") is not None:
            self.q_scale = float(c["attention_multiplier"]) * np.sqrt(self.desc.dh)
        self.load_bundle(w)

    def read_config(self, path):
        """Set norm eps and rope from config.json at `path`; return it."""
        self.eps, self.inv_freq, self.rot, self.rope_scale = config(
            self.desc, path, self.context
        )
        return json.loads(path.read_text())

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


def _pad(t, rows, cols):
    """Q4 triple t zero-padded to [rows, cols]: the padding's weights are 0."""
    out = []
    for a, n in zip(t, (cols, cols // 32, cols // 32)):
        z = np.zeros((rows, n), a.dtype)
        z[: a.shape[0], : a.shape[1]] = a
        out.append(z)
    return tuple(out)


def _rows(t, r0, r1):
    return tuple(a[r0:r1] for a in t)


class OpenELMPrefill(AffinePrefill):
    """OpenELM, whose layers differ in q / KV head counts and FFN width. The
    device is built for the widest layer (the desc's shape), and each layer's
    weights are zero-padded up to it. The padding changes no output: padded
    q, k, v and gate / up rows give zeros, and the o and down columns they
    feed are zero."""

    EMBED = "transformer.token_embeddings.weight"
    FINAL_NORM = "transformer.norm.weight"

    def read_config(self, path):
        c = json.loads(path.read_text())
        e = self.desc
        div = c["ffn_dim_divisor"]

        def width(m):
            # OpenELM's make_divisible
            v = m * c["model_dim"]
            w = max(div, int(v + div / 2) // div * div)
            return w + div if w < 0.9 * v else w

        self.geometry = [
            (q, k, width(m))
            for q, k, m in zip(
                c["num_query_heads"], c["num_kv_heads"], c["ffn_multipliers"]
            )
        ]
        q, k, f = (max(g[i] for g in self.geometry) for i in range(3))
        got = (c["model_dim"], c["head_dim"], q, k, f, len(self.geometry))
        got += (c["vocab_size"],)
        if got != tuple(e[1:8]) or c["num_transformer_layers"] != e.layers:
            raise ValueError(f"{path}: config {got} does not match the build's {e}")
        if any(q * e.kv_heads != k * e.heads for q, k, _ in self.geometry):
            raise ValueError(f"{path}: a layer's q / KV head ratio differs")
        # the norm eps is fixed in OpenELM's modeling code, not its config
        self.eps = 1e-6
        rot = e.dh
        theta = c["rope_freq_constant"]
        self.inv_freq = {
            "a": 1.0 / theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
        }
        self.rot, self.rope_scale = rot, 1.0
        return c

    def layer_tensors(self, b, L):
        e = self.desc
        qh, kh, f = self.geometry[L]
        dh, dq, dk = e.dh, e.heads * e.dh, e.kv_heads * e.dh
        p = f"transformer.layers.{L}."
        qkv = b.q4nx(p + "attn.qkv_proj.weight", (qh + 2 * kh) * dh, e.d)
        gu = b.q4nx(p + "ffn.proj_1.weight", 2 * f, e.d)
        r = (0, qh * dh, (qh + kh) * dh, (qh + 2 * kh) * dh)
        return {
            "qkv": [
                _pad(_rows(qkv, r[i], r[i + 1]), n, e.d)
                for i, n in enumerate((dq, dk, dk))
            ],
            "o": _pad(b.q4nx(p + "attn.out_proj.weight", e.d, qh * dh), e.d, dq),
            # proj_1 is gate | up
            "gate": _pad(_rows(gu, 0, f), e.inter, e.d),
            "up": _pad(_rows(gu, f, 2 * f), e.inter, e.d),
            "down": _pad(b.q4nx(p + "ffn.proj_2.weight", e.d, f), e.d, e.inter),
            "norms": {
                "input": b.bf16(p + "attn_norm.weight"),
                "post_attn": b.bf16(p + "ffn_norm.weight"),
                "q_norm": b.bf16(p + "attn.q_norm.weight"),
                "k_norm": b.bf16(p + "attn.k_norm.weight"),
            },
        }
