# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare cached decode and prefill weights from the pinned model pair."""

from paths import HERE
import model_paths
from common.prepare_weights_8b import main

if __name__ == "__main__":
    main(HERE, model_paths)
