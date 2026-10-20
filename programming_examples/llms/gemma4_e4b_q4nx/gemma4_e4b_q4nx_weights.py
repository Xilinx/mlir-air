# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B weight loader and CPU reference:
../gemma4_e2b_q4nx/gemma4_e2b_q4nx_weights.py, with CFG the E4B config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma4_e2b_q4nx"))
from gemma4_e2b_q4nx_weights import *  # noqa: E402,F401,F403
from gemma4_e2b_q4nx_weights import E4B as CFG  # noqa: E402,F401
