# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phi-2 fixed identities and local artifact paths."""

from pathlib import Path
import os
from ..shared.deployment import file_hash, save_json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("AIR_MODEL_BUILD_DIR", str(HERE / "build")))
REVISION = "810d367871c1d460086d9f82db8696f2e0a0fcd0"
H, HP, F, FP, D, DP, HEADS, BLOCKS, VOCAB = (
    2560,
    3072,
    10240,
    10752,
    80,
    128,
    32,
    32,
    51200,
)


def read_task():
    from ..shared.deployment import read_task

    return read_task("phi2_2_7b_q4nx")
