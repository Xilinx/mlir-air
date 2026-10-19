# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Gemma4-E4B example: the E2B adapter with
GEMMA4_VARIANT=e4b. Pointed at via `--runner=gemma4_e4b_q4nx.verify_adapter`."""

import os
import sys
from pathlib import Path

# Before any Gemma4 module is imported: they read their dims at import time.
os.environ["GEMMA4_VARIANT"] = "e4b"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma4_e2b_q4nx.verify_adapter import *  # noqa: E402,F401,F403
from gemma4_e2b_q4nx.verify_adapter import (  # noqa: E402,F401
    _N_LAYERS,
    _SEQ_LEN,
)
