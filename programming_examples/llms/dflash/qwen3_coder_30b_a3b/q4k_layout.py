# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The Q4_K block layout q4k_b8.cc reads: the bundle's nibbles [256 columns]
# [32 rows] pre-tiled as [8 groups][4 col tiles][4 row tiles][8 rows][8 cols],
# each tile the matmul's operand as it is. Scales, mins, S and M are unchanged.
import numpy as np

BB = 4736
QS, QN = 2 * 8 * 32, 32 * 256 // 2
# the bundle's [256 columns][32 rows] in the tiled order
_ORDER = (
    np.arange(256 * 32)
    .reshape(8, 4, 8, 4, 8)  # g, k, c, rt, r
    .transpose(0, 1, 3, 4, 2)
    .ravel()
)
_INVERSE = np.argsort(_ORDER)


def _permute(blocks, order, n=4096):
    assert blocks.shape[-1] == BB and blocks.dtype == np.uint8
    out = np.array(blocks, copy=True, order="C")
    flat = out.reshape(-1, BB)
    for i in range(0, len(flat), n):
        q = flat[i : i + n, QS : QS + QN]
        nib = np.stack([q & 15, q >> 4], -1).reshape(len(q), -1)[:, order]
        q[:] = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return out


def tiled(blocks):
    """Q4_K blocks [..., BB] in the bundle's layout -> the kernel's."""
    return _permute(blocks, _ORDER)


def untiled(blocks):
    return _permute(blocks, _INVERSE)
