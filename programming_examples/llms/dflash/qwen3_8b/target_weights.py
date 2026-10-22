# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Native Qwen3-8B Q4NX weights and per-position RoPE."""

from paths import HERE
import sys

sys.path.insert(0, str(HERE.parents[1] / "qwen3_8b_q4nx"))
from qwen3_8b_q4nx_weights import Q4nxModel, BLOCK_BF16, generate_rope_lut


def rope_row(position, role):
    return generate_rope_lut(position)


from qwen3_4b_weights import LayerWeights as PrefillLayer
