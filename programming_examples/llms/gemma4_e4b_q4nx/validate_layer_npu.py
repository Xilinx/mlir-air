# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B on-device per-layer gate:
../gemma4_e2b_q4nx/validate_layer_npu.py at GEMMA4_VARIANT=e4b."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_e4b_variant import delegate  # noqa: E402

delegate(__name__, "validate_layer_npu")
