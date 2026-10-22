# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU diagnostic for finite BF16 -> BFP16 operands under conv_even.

This is a validation aid, not the CPU timing control. Groups follow the K axis
of each matrix row (the AIE mmul helper transposes each B operand tile first).
The shared exponent must grow when a positive rounded mantissa reaches 128;
negative -128 is representable. This differs from the truncating byte-packing
helper used by the upstream explicit-BFP input example.
"""

import numpy as np
from ml_dtypes import bfloat16


def bfp_operand(values):
    values = np.asarray(values, bfloat16).astype(np.float32)
    if values.shape[-1] % 8 or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite BF16 rows with K divisible by eight")
    groups = values.reshape(-1, 8)
    exponent = ((np.abs(groups).view(np.uint32) >> 23) & 255).max(axis=1)
    step = np.exp2(exponent.astype(np.float64) - 127 - 6).astype(np.float32)
    mantissa = np.rint(groups / step[:, None])
    overflow = np.any(mantissa > 127, axis=1)
    step[overflow] *= 2
    mantissa[overflow] = np.rint(groups[overflow] / step[overflow, None])
    return (np.clip(mantissa, -128, 127) * step[:, None]).reshape(values.shape)
