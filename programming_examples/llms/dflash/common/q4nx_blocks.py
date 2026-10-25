# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Q4NX storage primitives; all matrix dimensions and offsets are explicit."""

import numpy as np
from ml_dtypes import bfloat16


def pack_blocks(q, scale, minimum):
    """Logical [...,32,256] nibbles, [...,8,32] scales/mins -> raw words."""
    shape = q.shape[:-2]
    assert q.shape[-2:] == (32, 256) and np.all(q < 16)
    qr = q.reshape(-1, 2, 16, 256).transpose(0, 1, 3, 2).reshape(-1, 8192)
    qb = np.ascontiguousarray(qr[:, 0::2] | (qr[:, 1::2] << 4))
    result = np.empty((qb.shape[0], 2560), np.uint16)
    result[:, :256] = np.asarray(scale, bfloat16).reshape(-1, 256).view(np.uint16)
    result[:, 256:512] = np.asarray(minimum, bfloat16).reshape(-1, 256).view(np.uint16)
    result[:, 512:] = qb.view(np.uint16)
    return result.reshape(*shape, 2560)


def unpack_blocks(packed):
    """(q, scale, mins) for every block in a packed slab, vectorized.

    `pack_q4k_block` is a per-block Python triple loop; inverting it the same
    way takes minutes on fc's 4000 blocks, and a gate nobody runs is not a gate.
    Layout (proj_qmm_pack.py:5-9):
        bf16  scales[8 groups][32 rows]   offset 0
        bf16  mins  [8 groups][32 rows]   offset 512
        uint8 qs[2 rowgrp][256 col][8]    offset 1024, byte = (w_odd<<4)|w_even
    """
    from ml_dtypes import bfloat16

    u8 = np.ascontiguousarray(packed).view(np.uint8).reshape(-1, 2560 * 2)
    nb = u8.shape[0]
    sm = 2 * 8 * 32
    sc = u8[:, :sm].view(bfloat16).reshape(nb, 8, 32)
    mn = u8[:, sm : 2 * sm].view(bfloat16).reshape(nb, 8, 32)
    qs = u8[:, 2 * sm :].reshape(nb, 32 // 16, 256, 16 // 2)
    # row = g*16 + kk*2 + p, p=0 low nibble (even row), p=1 high (odd)
    pair = np.stack([qs & 0xF, qs >> 4], axis=-1)  # [nb, g, col, kk, p]
    q = pair.transpose(0, 1, 3, 4, 2).reshape(nb, 32, 256)
    # scales/mins come back group-major; the caller wants [block, row, group]
    return q, sc.transpose(0, 2, 1), mn.transpose(0, 2, 1)
