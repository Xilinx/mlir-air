# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check this model's production B8 attention graph and compiled object."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import HERE
from target_config import TargetConfig, load_layout
from common.attention_check import main

if __name__ == "__main__":
    main(TargetConfig, load_layout)
