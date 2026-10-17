# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fast bfp16ebs8 packer for an array of 8-element rows.

pack_b_bfp16ebs8(B [8, N], 8, 8) emits one 9-byte record per column of B, i.e. per
row of B.T: [shared exponent, 8 int8 mantissas]. This does that directly on a
[R, 8] array of bf16-valued f32: two 64K-entry tables map a bf16 bit pattern to its
exponent and its 8-bit mantissa, then one int8 shift aligns the row to its max
exponent. Bit-identical to pack_b_bfp16ebs8 for every non-NaN bf16 value, including
the reference's quirks (mantissa cut to 7 bits before the shift, rounding away from
zero for negatives; -0.0 under a >=32 exponent gap packs as -1). Test:
bfp16_rows_pack_test.py. NaN payloads are not preserved by the reference either.
"""

import numpy as np

_u = np.arange(65536, dtype=np.uint16)
_EXP = ((_u >> 7) & 0xFF).astype(np.uint8)
_m7 = (_u & 0x7F).astype(np.int16)
_mag = (_m7 >> 1) | (
    (_EXP != 0).astype(np.int16) << 6
)  # implicit bit + 6 mantissa bits
# floor(-m / 2^17) of the reference's 32-bit two's complement = -(mag + dropped bit)
_B8 = np.where(_u >> 15 != 0, -(_mag + (_m7 & 1)), _mag).astype(np.int8)
del _u, _m7, _mag


def pack_rows_bfp16ebs8(rows_f32):
    """[R, 8] f32 holding exact bf16 values -> [R, 9] uint8 records."""
    assert rows_f32.shape[1] == 8
    u = (np.ascontiguousarray(rows_f32, np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )
    exp, b8 = _EXP[u], _B8[u]
    max_exp = exp.max(axis=1)
    shift = max_exp[:, None] - exp  # uint8
    aligned = b8 >> np.minimum(shift, 7).astype(np.int8)
    neg0 = u == 0x8000
    if (
        neg0.any()
    ):  # reference: shift >= 32 gives sign ? -1 : 0, which differs only for -0.0
        aligned = np.where(neg0 & (shift >= 32), np.int8(-1), aligned)
    out = np.empty((len(u), 9), np.uint8)
    out[:, 0] = max_exp
    out[:, 1:] = aligned.view(np.uint8)
    return out
