# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reuse AIR's Llama Q4NX reader with explicit projection dimensions."""

from paths import EXAMPLES, HERE
import sys
import types
import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(HERE.parents[1] / "llama31_8b_q4nx"))
from llama31_8b_q4nx_weights import Q4nxModel as _Model, llama3_inv_freq
from llama32_1b_q4nx_weights import BLOCK_BF16


class Q4nxModel(_Model):
    _PROJ = {
        "q": ("self_attn.q_proj", 4096, 4096),
        "k": ("self_attn.k_proj", 1024, 4096),
        "v": ("self_attn.v_proj", 1024, 4096),
        "o": ("self_attn.o_proj", 4096, 4096),
        "gate": ("mlp.gate_proj", 14336, 4096),
        "up": ("mlp.up_proj", 14336, 4096),
        "down": ("mlp.down_proj", 4096, 14336),
    }

    def layer_weights(self, layer):
        return {
            key: self.dequant(f"model.layers.{layer}.{name}.weight", rows, width)
            .astype(bfloat16)
            .T.copy()
            for key, (name, rows, width) in self._PROJ.items()
        }

    def layer_qk_norm(self, layer):
        return None


_TARGET_INV = llama3_inv_freq(dim=128, theta=500000.0)
_DRAFT_INV = 1.0 / (1000000.0 ** (np.arange(64, dtype=np.float64) / 64))


def rope_row(position, role):
    inv = _TARGET_INV if role == "target" else _DRAFT_INV
    angles = position * inv
    return np.concatenate((np.cos(angles), np.sin(angles))).astype(bfloat16)


sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(EXAMPLES)
]
from air_examples.llms.llama32_3b.llama32_3b_weights import LayerWeights as PrefillLayer
