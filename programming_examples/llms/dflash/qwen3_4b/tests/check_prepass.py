# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""BF16/BFP prepass: reference error, isolation, repeatability and wall timing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash_model import MODEL
from common.check_prepass import main

if __name__ == "__main__":
    main(MODEL)
