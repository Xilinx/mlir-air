# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bounded BF16 prefill cache using the native model dequantizer."""

from paths import HERE
from common.prefill_weights_8b import MappedWeights, convert, main, verify

if __name__ == "__main__":
    main()
