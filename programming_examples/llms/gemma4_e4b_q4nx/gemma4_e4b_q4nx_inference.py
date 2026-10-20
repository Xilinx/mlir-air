# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B NPU prefill + fused decode:
../gemma4_e2b_q4nx/gemma4_e2b_q4nx_inference.py with the E4B config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma4_e2b_q4nx"))
from gemma4_e2b_q4nx_weights import E4B  # noqa: E402
from gemma4_e2b_q4nx_inference import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(E4B))
