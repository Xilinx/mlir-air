# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Read the pinned official 8B DFlash BF16 tensors without remote code."""

import json
from pathlib import Path
import struct

import numpy as np
from ml_dtypes import bfloat16
from model_paths import default_draft_snapshot, default_draft_cache

SNAPSHOT = default_draft_snapshot()
PACKED_CACHE = default_draft_cache()


class DraftWeights:
    def __init__(self, target_model=None, snapshot=SNAPSHOT, *, shared_embedding=None):
        self.snapshot = Path(snapshot)
        self.path = self.snapshot / "model.safetensors"
        self.config = json.loads((self.snapshot / "config.json").read_text())
        expected = dict(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=5,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            rms_norm_eps=1e-6,
            rope_theta=1000000,
            block_size=16,
        )
        for name, value in expected.items():
            if self.config[name] != value:
                raise ValueError((name, self.config[name], value))
        assert self.config["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
        self.tap_slots = [2, 10, 18, 26, 34]
        self.mask_token_id = self.config["dflash_config"]["mask_token_id"]
        assert self.mask_token_id == 151669
        self.target_model = target_model
        self.shared_embedding = shared_embedding
        if shared_embedding is not None:
            if (
                shared_embedding.shape != (151936, 4096)
                or shared_embedding.dtype != bfloat16
            ):
                raise ValueError("Shared embedding must be [151936,4096] BF16")
        with self.path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.base = n + 8

    def tensor(self, name):
        entry = self.header[name]
        if entry["dtype"] != "BF16":
            raise ValueError((name, entry["dtype"]))
        lo, hi = entry["data_offsets"]
        assert hi - lo == np.prod(entry["shape"]) * 2
        return np.memmap(
            self.path,
            mode="r",
            dtype=bfloat16,
            offset=self.base + lo,
            shape=tuple(entry["shape"]),
        )

    def bf16(self, name):
        # The target has untied embedding/head. Only the input embedding is
        # needed here; prepare_draft copies the target's actual packed head.
        if name == "model.norm.weight":
            return self.tensor("norm.weight")
        if name == "model.embed_tokens.weight" and self.shared_embedding is not None:
            return self.shared_embedding
        if name == "model.embed_tokens.weight" and self.target_model is not None:
            return self.target_model.bf16(name)
        raise ValueError(("Unsupported shared tensor", name))

    def layer_qk_norm(self, layer):
        return tuple(
            self.tensor(f"layers.{layer}.self_attn.{kind}_norm.weight")
            for kind in ("q", "k")
        )

    def layer_rms(self, layer):
        return tuple(
            self.tensor(f"layers.{layer}.{kind}.weight")
            for kind in ("input_layernorm", "post_attention_layernorm")
        )
