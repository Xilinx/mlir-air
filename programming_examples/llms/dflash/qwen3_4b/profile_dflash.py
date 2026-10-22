# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixed-prompt profile of NPU prefill and the DFlash decode loop."""

from dflash_model import MODEL
from common.profile import main

if __name__ == "__main__":
    main(MODEL, "qwen3_4b_dflash_b8")
