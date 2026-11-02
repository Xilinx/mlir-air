# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the lite device: python3 device_build.py BUILD_DIR [-j N]."""

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

from air_examples.llms.shared.fused_prefill.d3 import build_device  # noqa: E402

if __name__ == "__main__":
    build_device("lite", sys.argv[1:])
