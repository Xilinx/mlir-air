# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The names the shared decoder and the drafter packer import as the target
role's weights (Qwen3-8B's). The drafter takes its own weights and RoPE from
drafter.py."""

from paths import HERE
import sys

sys.path.insert(0, str(HERE.parents[1] / "qwen3_8b_q4nx"))
from qwen3_8b_q4nx_weights import Q4nxModel, generate_rope_lut


def rope_row(position, role):
    return generate_rope_lut(position)
