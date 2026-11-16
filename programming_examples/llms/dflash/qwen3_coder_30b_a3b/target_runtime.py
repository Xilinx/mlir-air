# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The target (verify_b8's verify pass and LM head) behind the shared
decoder interface, and load_pair's decoder constructor.

The target's K and V cache is [layer][K or V][position][512] BF16 on the
device. One target is loaded per process; the NPU prefill and the context
prepass use it rather than a copy of its 20 GB of weights.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

from paths import target_imports
import model_paths
import model_spec

_LOADED = []


def loaded_target():
    return _LOADED[-1] if _LOADED else None


class _Taps(dict):
    """The tap layers' rows of one verify, each passed to the listeners as
    it arrives; the work they return runs in the dispatch's wait."""

    def __init__(self, target, position):
        super().__init__(dict.fromkeys(target.tap_layers))
        self.target, self.position = target, position

    def __setitem__(self, layer, rows):
        super().__setitem__(layer, rows)
        i = self.target.tap_layers.index(layer)
        for listen in self.target.tap_listeners:
            self.target.verify_pass.work += listen(i, rows, self.position)


class CoderTarget:
    def __init__(self, artifacts, model_source=None, tiled=None):
        """tiled: the weights' cache in the device's layout."""
        artifacts = Path(artifacts).resolve()
        manifest = json.loads((artifacts / "manifest.json").read_text())
        if manifest.get("status") != "compiled":
            raise ValueError(f"{artifacts} holds no compiled target")
        cfg = manifest["config"]
        self.config = SimpleNamespace(**cfg, taps=True)
        self.batch, self.capacity = cfg["batch"], cfg["capacity"]
        self.layers = cfg["layers"]
        self.tap_layers = list(model_spec.TAP_LAYERS)
        self.tap_listeners = []
        with target_imports():
            import verify_b8

        W = verify_b8.W
        self.model = W.Qwen3CoderQ4k(
            model_paths.resolve_target_model(model_source or model_paths.TARGET_MODEL)
        )
        self.verify_pass = verify_b8.VerifyB8(
            self.model,
            artifacts,
            self.capacity,
            cfg["reads"],
            tiled=tiled or model_paths.default_target_cache(),
        )
        self.embed = (
            np.asarray(self.model.raw("model.embed_tokens.weight"))
            .view(bfloat16)
            .reshape(-1, verify_b8.D)
        )
        self._taps = None
        # the prefill whose K and V the cache holds below its length
        self.holds = None
        _LOADED.append(self)

    def seed_kv(self, k, v, position):
        """A prompt's K and V [layer, position, 512] into the cache."""
        if not 0 <= position <= self.capacity - self.batch:
            raise ValueError("Prefill leaves no room for a whole verification block")
        if self.holds is k:
            return
        self.verify_pass.seed(
            {l: (k[l][:position], v[l][:position]) for l in range(self.layers)},
            position,
        )
        self.holds = k

    def read_kv(self, length=None):
        """[layer, K or V, position, 512] BF16 words of the first length."""
        length = self.capacity if length is None else int(length)
        vp = self.verify_pass
        vp.KV.sync(vp.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        m = np.frombuffer(vp.KV.map(), np.uint16)
        m = m.reshape(self.layers, 2, self.capacity, -1)
        return m[:, :, :length].copy()

    def verify(self, tokens, position):
        ids, p = [int(t) for t in tokens], int(position)
        if len(ids) != self.batch or not 0 <= p <= self.capacity - self.batch:
            raise ValueError("Verification block exceeds the artifact")
        self._taps = _Taps(self, p)
        h = self.model.embed(ids).astype(np.float32)
        h = self.verify_pass.forward(h, p, taps=self._taps)
        return SimpleNamespace(logits=self.verify_pass.logits(h))

    def hidden(self, boundaries):
        """BF16 words [len(boundaries), batch, hidden] of the last verify:
        the outputs of layers boundary - 1."""
        rows = [self._taps[b - 1] for b in boundaries]
        return np.stack(rows).astype(bfloat16).view(np.uint16)

    def prefill(self, input_ids):
        """The prompt through the verify pass from position 0: (the first
        token, K, V [layer, n, 512], taps [n, taps x hidden] float32)."""
        n = len(input_ids)
        first, taps = self.verify_pass.prefill(input_ids, self.tap_layers)
        kv = self.read_kv(n).view(bfloat16)
        k, v = kv[:, 0], kv[:, 1]
        self.holds = k
        taps = np.concatenate([taps[l] for l in self.tap_layers], -1)
        return first, k, v, taps


def target_decoder(
    artifact, *, model_weights=None, packed_cache=None, model_source=None, **_
):
    """load_pair's decoder constructor: the target, or with model_weights
    (draft_weights.DraftWeights) the drafter."""
    if model_weights is None:
        return CoderTarget(artifact, model_source, tiled=packed_cache)
    from draft_weights import coder_drafter

    return coder_drafter(artifact, model_weights, packed_cache)
