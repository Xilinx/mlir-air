# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""B8 full outputs and state gate, one model resident and bounded KV scratch."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash_model import MODEL
from model_paths import resolve_target_model
from target_weights import Q4nxModel
from common.check_decoder import main

if __name__ == "__main__":
    main(MODEL, lambda: Q4nxModel(resolve_target_model()))
