# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Locate the example and shared package when invoked as a standalone script."""

import contextlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DFLASH = HERE.parent
COMMON = DFLASH / "common"
EXAMPLES = HERE.parents[2]
REPO = EXAMPLES.parent
FUSED = EXAMPLES / "fused_decode"
TARGET = EXAMPLES / "llms" / "qwen3_coder_30b_a3b_q4k"
if str(DFLASH) not in sys.path:
    sys.path.insert(0, str(DFLASH))


@contextlib.contextmanager
def target_imports():
    """Import the target example's modules: they bind a `common` module of
    their own, which shares its name with the DFlash package."""
    held = sys.modules.pop("common", None)
    try:
        yield
    finally:
        sys.modules.pop("common", None)
        if held is not None:
            sys.modules["common"] = held
