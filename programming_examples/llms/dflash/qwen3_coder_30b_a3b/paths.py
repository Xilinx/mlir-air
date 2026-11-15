# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Locate the example and shared package when invoked as a standalone script."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DFLASH = HERE.parent
COMMON = DFLASH / "common"
EXAMPLES = HERE.parents[2]
REPO = EXAMPLES.parent
FUSED = EXAMPLES / "fused_decode"
if str(DFLASH) not in sys.path:
    sys.path.insert(0, str(DFLASH))
