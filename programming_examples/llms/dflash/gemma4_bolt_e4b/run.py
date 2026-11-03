# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Streaming greedy generation and resident, thinking-disabled chat on NPU."""

from dflash_model import MODEL
from common.run import main

if __name__ == "__main__":
    main(MODEL)
