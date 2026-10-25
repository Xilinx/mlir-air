# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pair the native B1 baseline and DFlash on the same saved prompt states."""

from paths import HERE
from benchmark_data import verify_inputs
from common.paired_benchmark import main

if __name__ == "__main__":
    main(model="qwen3_8b_dflash_b8", model_dir=HERE, verify_inputs=verify_inputs)
