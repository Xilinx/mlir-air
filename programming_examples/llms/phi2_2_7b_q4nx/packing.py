# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Lossless repacking of the fixed affine group-32 Q4 assets.

Device block/cascade layout follows fused_decode/proj_qmm_pack.py (AMD, MIT).
No extrema, code selection, or requantization occurs here.
"""

import numpy as np
from ml_dtypes import bfloat16


def cascade_indices(rows, cols, ncx=4, ncy=4):
    assert rows % (32 * ncx * ncy) == 0 and cols % 256 == 0
    ni, nj = rows // (32 * ncx * ncy), cols // 256
    # [column, shim channel, row iteration, col block, cascade row in half]
    cx, half, i, j, cy = np.indices((ncx, 2, ni, nj, ncy // 2))
    rb = i * ncx * ncy + cx * ncy + half * (ncy // 2) + cy
    return rb.ravel(), j.ravel()


def pack_cascade(q, scale_bits, min_bits, *, verify=True):
    """Vectorized version of iter_major=True, dual_chan=True in the seed."""
    m, k = q.shape
    rb, cb = m // 32, k // 256
    assert scale_bits.shape == min_bits.shape == (m, k // 32)
    blocks = np.empty((rb, cb, 2560), np.uint16)
    blocks[:, :, :256] = (
        scale_bits.reshape(rb, 32, cb, 8).transpose(0, 2, 3, 1).reshape(rb, cb, 256)
    )
    blocks[:, :, 256:512] = (
        min_bits.reshape(rb, 32, cb, 8).transpose(0, 2, 3, 1).reshape(rb, cb, 256)
    )
    qblocks = (
        q.reshape(rb, 32, cb, 256).transpose(0, 2, 1, 3).reshape(rb, cb, 2, 16, 256)
    )
    packed = (
        (qblocks[:, :, :, 0::2, :] | (qblocks[:, :, :, 1::2, :] << 4))
        .transpose(0, 1, 2, 4, 3)
        .copy()
    )
    blocks[:, :, 512:] = packed.reshape(rb, cb, 4096).view(np.uint16)
    ri, ci = cascade_indices(m, k)
    stream = blocks[ri, ci].reshape(-1)
    if verify:
        # Invert the actual stream and all metadata; zero allowed bit mismatches.
        inverse = np.empty_like(blocks)
        inverse[ri, ci] = stream.reshape(-1, 2560)
        sc = (
            inverse[:, :, :256]
            .reshape(rb, cb, 8, 32)
            .transpose(0, 3, 1, 2)
            .reshape(m, k // 32)
        )
        mn = (
            inverse[:, :, 256:512]
            .reshape(rb, cb, 8, 32)
            .transpose(0, 3, 1, 2)
            .reshape(m, k // 32)
        )
        raw = inverse[:, :, 512:].copy().view(np.uint8).reshape(rb, cb, 2, 256, 8)
        qq = np.empty((rb, cb, 2, 16, 256), np.uint8)
        qq[:, :, :, 0::2, :] = (raw & 15).transpose(0, 1, 2, 4, 3)
        qq[:, :, :, 1::2, :] = (raw >> 4).transpose(0, 1, 2, 4, 3)
        qq = qq.reshape(rb, cb, 32, 256).transpose(0, 2, 1, 3).reshape(m, k)
        assert (
            np.array_equal(qq, q)
            and np.array_equal(sc, scale_bits)
            and np.array_equal(mn, min_bits)
        )
    return stream


def check_seed_packer():
    from air_examples.fused_decode.proj_qmm_pack import pack_q4k_cascade

    rng = np.random.default_rng(9401)
    q = rng.integers(0, 16, (1024, 512), dtype=np.uint8)
    sc = rng.uniform(0.001, 0.2, (1024, 16)).astype(bfloat16)
    mn = rng.uniform(-1, 0, (1024, 16)).astype(bfloat16)
    actual = pack_cascade(q, sc.view(np.uint16), mn.view(np.uint16))
    seed = pack_q4k_cascade(
        q,
        sc.astype(np.float32),
        mn.astype(np.float32),
        4,
        4,
        iter_major=True,
        dual_chan=True,
    )
    assert np.array_equal(actual, seed.view(np.uint16))
    return {
        "seed_packer_bitwise_equal": True,
        "test_shape": [1024, 512],
        "uint16_elements_compared": actual.size,
    }


def pad(tensor, rows, cols):
    q, sc, mn = tensor
    m, k = q.shape
    assert m <= rows and k <= cols and cols % 32 == 0
    return (
        np.pad(q, ((0, rows - m), (0, cols - k))),
        np.pad(sc, ((0, rows - m), (0, (cols - k) // 32))),
        np.pad(mn, ((0, rows - m), (0, (cols - k) // 32))),
    )


def slice_rows(tensor, lo, hi):
    return tuple(a[lo:hi] for a in tensor)


def dequant(tensor):
    q, s, m = tensor
    return (
        q.astype(np.float32) * s.view(bfloat16).astype(np.float32).repeat(32, axis=1)
        + m.view(bfloat16).astype(np.float32).repeat(32, axis=1)
    ).astype(bfloat16)
