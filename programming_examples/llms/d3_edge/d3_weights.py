# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The weights of the two d3 configurations, pinned.

  edge  vllm-sr/d3-edge: the BF16 checkpoint (model-00001-of-00001.safetensors)
  lite  FastFlowLM/D3-Lite-NPU2, FastFlowLM's package of vllm-sr/d3-lite:
        model.q4nx, the same layout as FastFlowLM's Qwen3.5-0.8B
        flm_q4k_high_precision bundle (qwen35_0_8b_q4nx/q35_weights.py),
        Q4K projections, Q8 beta / alpha and lm_head (the tied embedding)

Both take the upstream package's tokenizer, decision_config.json, FP32
readout.safetensors and its d3_runtime (prompt format, reference), which
are the same files in the FastFlowLM package.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The model directories are imported as the air_examples package, not through
# sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.qwen35_0_8b_q4nx import q35_weights as Q  # noqa: E402

MODELS = {
    "edge": ("vllm-sr/d3-edge", "c454ee39b862248d70abd7f84195d2f54ffe6d43"),
    "lite": ("vllm-sr/d3-lite", "03560c26f5199c3c04129a2ccb38960c913635e0"),
}
EDGE_WEIGHTS = "model-00001-of-00001.safetensors"
EDGE_SHA256 = "91bb8618450a7f628f1432bbced76bc19ee7346c93d191a228fdd33c25a57a9d"
LITE_REPO = "FastFlowLM/D3-Lite-NPU2"
LITE_REVISION = "51563157b724a1bf69c5c177e000dd7038acd63c"
LITE_SHA256 = "04a20706f44dcc8f142a1774034fc2a01110bef14031fc8ccf26295db00bef42"
# the upstream files the NPU path reads (lite: all but the BF16 weights)
PACKAGE_FILES = ["*.json", "*.py", "*.jinja", "readout.safetensors"]


def _check(path, want):
    got = Q.sha256(path)
    if got != want:
        raise ValueError(f"{path}: sha256 {got}, expected {want}")


def model_dir(model, weights=None):
    """The upstream package of `model` ("edge" / "lite") at its pinned
    revision. edge: $D3_EDGE_SOURCE, a directory holding the package, else
    the Hub download (cached under HF_HOME); its weights' sha256 is checked.
    weights (lite): also fetch the package's BF16 weights."""
    from huggingface_hub import snapshot_download

    repo, rev = MODELS[model]
    if model == "edge":
        source = os.environ.get("D3_EDGE_SOURCE")
        if source:
            root = Path(source)
        else:
            pats = PACKAGE_FILES + [EDGE_WEIGHTS]
            root = Path(snapshot_download(repo, revision=rev, allow_patterns=pats))
        _check(root / EDGE_WEIGHTS, EDGE_SHA256)
        return root
    pats = PACKAGE_FILES + (["*.safetensors"] if weights else [])
    return Path(snapshot_download(repo, revision=rev, allow_patterns=pats))


def lite_bundle_path(source=None):
    """model.q4nx of LITE_REPO at LITE_REVISION ($D3_LITE_SOURCE: a file or a
    directory holding it, else the Hub download), sha256-checked."""
    source = source or os.environ.get("D3_LITE_SOURCE")
    if source and os.path.isdir(source):
        source = os.path.join(source, "model.q4nx")
    if source and os.path.isfile(source):
        path = source
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(LITE_REPO, "model.q4nx", revision=LITE_REVISION)
    _check(path, LITE_SHA256)
    return path


class LiteBundle(Q.Q4nx):
    """The d3-lite bundle, read as q35_weights.Q4nx reads Qwen3.5-0.8B's."""

    def __init__(self, path=None):
        self.path = Path(lite_bundle_path(path))
        with open(self.path, "rb") as f:
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n
        self.mm = np.memmap(self.path, np.uint8, "r")


class EdgeCheckpoint:
    """The BF16 safetensors of d3-edge, by HF tensor name (without the
    "language_model." prefix)."""

    def __init__(self, root):
        path = Path(root) / EDGE_WEIGHTS
        with open(path, "rb") as f:
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n
        self.mm = np.memmap(path, np.uint8, "r")

    def get(self, name):
        h = self.hdr["language_model." + name]
        assert h["dtype"] == "BF16", (name, h["dtype"])
        s, e = h["data_offsets"]
        a = np.asarray(self.mm[self.off + s : self.off + e]).view(bfloat16)
        return a.reshape(h["shape"])

    def f32(self, name):
        return self.get(name).astype(np.float32)
