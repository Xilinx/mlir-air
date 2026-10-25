# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""DFlash token captures compared with llms/verify's top-5 comparator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash_model import MODEL
from prepare_inputs import load_prefill
from resources import prepare
from common.check_model import main

if __name__ == "__main__":
    main(MODEL, prepare=prepare, load_prefill=load_prefill)
