# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B sub-layer ground truth from the full-precision checkpoint:
../gemma4_e2b_q4nx/dump_sublayer_reference.py, which reads its model from
GEMMA4_FP_CHECKPOINT and FLM_REFERENCE."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma4_e2b_q4nx"))
from dump_sublayer_reference import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
