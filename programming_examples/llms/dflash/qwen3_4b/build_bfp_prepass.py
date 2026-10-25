# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile this model's four-stage NPU context prepass."""

from paths import HERE
from common.build_prepass import build_module, main

if __name__ == "__main__":
    main(HERE, 2560)
