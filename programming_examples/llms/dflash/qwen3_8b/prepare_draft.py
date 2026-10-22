# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare this model using the shared 8B packing path."""

from paths import HERE
from common.prepare_draft_8b import main, quantized_blocks

if __name__ == "__main__":
    main()
