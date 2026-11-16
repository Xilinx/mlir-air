# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The pinned z-lab drafter: its tensors, and the drafter on the shared B8
decoder (all 8 layers and the head in one dispatch). Each of its 4 KV heads is
stored twice, which gives the decoder's 8-head attention."""

import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
from ml_dtypes import bfloat16

from paths import HERE
from model_paths import default_draft_snapshot, draft_snapshot
import model_spec

# prepare_draft_8b imports this name
SNAPSHOT = default_draft_snapshot()
# the geometry the drafter's build has
LAYERS, D, DH, NKV, THETA = 8, 2048, 128, 4, 1e7


class Safetensors:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.base = n + 8

    def tensor(self, name):
        e = self.header[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        lo, _ = e["data_offsets"]
        return np.memmap(
            self.path,
            mode="r",
            dtype=bfloat16,
            offset=self.base + lo,
            shape=tuple(e["shape"]),
        )


def source_identity(draft_path, bundle):
    """What a packed drafter is made from: the checkpoint, the target bundle
    its vocabulary head comes from, and the packer."""

    def sha(path):
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    st = Path(bundle).stat()
    return dict(
        draft=sha(Path(draft_path) / "model.safetensors"),
        head=f"{Path(bundle).resolve()} {st.st_size} {st.st_mtime_ns}",
        packer=sha(HERE / "prepare_coder_draft.py"),
    )


def rope_row(p):
    inv = 1.0 / (THETA ** (np.arange(0, DH // 2, dtype=np.float64) / (DH // 2)))
    a = p * inv
    return np.concatenate([np.cos(a), np.sin(a)]).astype(bfloat16).astype(np.float32)


class DraftWeights:
    """The checkpoint's metadata, and the tensors the decoder and the context
    prepass read; the input embedding is the target's."""

    def __init__(self, target_model=None, snapshot=None, *, shared_embedding=None):
        self.snapshot = draft_snapshot(snapshot)
        self.config = cfg = json.loads((self.snapshot / "config.json").read_text())
        geometry = tuple(
            cfg[k]
            for k in (
                "num_hidden_layers",
                "hidden_size",
                "head_dim",
                "num_key_value_heads",
                "rope_theta",
            )
        )
        if geometry != (LAYERS, D, DH, NKV, THETA):
            raise ValueError(f"{self.snapshot} is not the drafter this build is for")
        taps = cfg["dflash_config"]["target_layer_ids"]
        if tuple(taps) != model_spec.TAP_LAYERS:
            raise ValueError(f"drafter taps {taps}, expected {model_spec.TAP_LAYERS}")
        self.tap_slots = [layer + 1 for layer in taps]
        self.mask_token_id = cfg["dflash_config"]["mask_token_id"]
        assert self.mask_token_id == model_spec.MASK_TOKEN_ID
        self.target_model = target_model
        if shared_embedding is not None and (
            shared_embedding.shape != (model_spec.VOCAB_SIZE, D)
            or shared_embedding.dtype != bfloat16
        ):
            raise ValueError(f"Shared embedding must be [vocabulary, {D}] BF16")
        self.shared_embedding = shared_embedding
        self.st = Safetensors(self.snapshot / "model.safetensors")

    def tensor(self, name):
        return self.st.tensor(name)

    def bf16(self, name):
        if name == "model.norm.weight":
            return np.asarray(self.tensor("norm.weight"), bfloat16)
        if name == "model.embed_tokens.weight" and self.shared_embedding is not None:
            return self.shared_embedding
        if name == "model.embed_tokens.weight" and self.target_model is not None:
            raw = self.target_model.raw(name)
            return np.asarray(raw).view(bfloat16).reshape(-1, D)
        raise KeyError(name)

    def layer_qk_norm(self, layer):
        return tuple(
            np.asarray(
                self.tensor(f"layers.{layer}.self_attn.{k}_norm.weight"), bfloat16
            )
            for k in "qk"
        )


class _Spin:
    """A kernel whose runs poll for completion: a blocked host wakes late and
    clocks down."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, *args):
        return _SpinRun(self.kernel(*args))

    def __getattr__(self, name):
        return getattr(self.kernel, name)


class _SpinRun:
    def __init__(self, run):
        self.run = run

    def wait(self, timeout_ms):
        done = __import__("pyxrt").ert_cmd_state.ERT_CMD_STATE_COMPLETED
        end = time.perf_counter() + timeout_ms / 1e3
        while self.run.state() != done and time.perf_counter() < end:
            pass
        return self.run.wait(timeout_ms)


def coder_drafter(artifacts, weights, packed):
    """The drafter's build (build_draft.py) with its packed weights
    (prepare_coder_draft.py), which must come from weights' checkpoint and
    the target's bundle."""
    from common.decoder_8b import TargetDecoder

    made = json.loads((Path(packed) / "manifest.json").read_text())
    bundle = weights.target_model.path
    if made.get("source") != source_identity(weights.snapshot, bundle):
        raise ValueError(
            f"{packed} was packed from another checkpoint or bundle; rerun "
            "prepare_coder_draft.py"
        )
    dec = TargetDecoder(artifacts, packed_cache=packed, model_weights=weights)
    dec.rope_row = rope_row
    dec.kernel = _Spin(dec.kernel)
    return dec
