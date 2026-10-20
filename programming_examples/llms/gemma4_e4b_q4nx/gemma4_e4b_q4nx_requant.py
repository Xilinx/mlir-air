# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B fused decode weight packer:
../gemma4_e2b_q4nx/gemma4_e2b_q4nx_requant.py, which packs any Q4nxModel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma4_e2b_q4nx"))
from gemma4_e2b_q4nx_requant import *  # noqa: E402,F401,F403
