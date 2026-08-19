# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# NumPy port of the bfp16ebs8 host helpers in mlir-aie
# programming_examples/ml/block_datatypes/helper.h.
#
# These are used by the Python dev loop only. The authoritative correctness
# gate for this example is mlir-aie's own bfp_test.cpp, which uses the C++
# originals; see the Makefile's `verify` target.
#
# bfp16ebs8 layout: every 8 consecutive scalars become 9 bytes -- one shared
# 8-bit exponent (the block max) followed by eight 8-bit two's-complement
# mantissas. So a [H, W] float matrix becomes a [H, W*9//8] uint8 matrix.

import numpy as np

BLOCK = 8  # scalars per block
BYTES_PER_BLOCK = 9  # 1 shared exponent + 8 mantissas


def nbytes(n_elems):
    """Byte count of n_elems bfp16ebs8 scalars."""
    assert n_elems % BLOCK == 0, f"{n_elems} is not a multiple of {BLOCK}"
    return n_elems // BLOCK * BYTES_PER_BLOCK


def float_to_bfp16ebs8(x):
    """[..., W] float -> [..., W*9//8] uint8, blocking along the last axis.

    Mirrors floatToBfp16() in helper.h, including its truncation rounding
    (AIE2P's only mantissa rounding mode).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    lead, W = x.shape[:-1], x.shape[-1]
    assert W % BLOCK == 0, f"last axis {W} must be a multiple of {BLOCK}"

    blocks = x.reshape(-1, BLOCK)
    bits = blocks.view(np.uint32)
    sign = (bits & np.uint32(0x80000000)) != 0
    exp = ((bits >> 23) & np.uint32(0xFF)).astype(np.int32)
    # Restore the implicit leading 1 for normals; subnormals keep exp == 0.
    mant = (bits & np.uint32(0x007FFFFF)) | np.where(
        exp != 0, np.uint32(0x00800000), np.uint32(0)
    )
    max_exp = exp.max(axis=1, keepdims=True)  # shared exponent per block

    signed_mant = np.where(
        sign, (~mant + np.uint32(1)) & np.uint32(0xFFFFFFFF), mant
    ).astype(np.int64)

    # helper.h truncates to 8 bits FIRST, then arithmetic-shifts the int8 by
    # (max_exp - exp). Doing the shift before the truncation gives different
    # answers once the in-block exponent spread reaches 8, so keep this order.
    b8 = (signed_mant >> (23 - 7 + 1)).astype(np.uint8).view(np.int8).astype(np.int32)
    shift = (max_exp - exp).astype(np.int32)
    aligned = np.where(
        shift >= 32,  # shifting by >= width is UB in C++; helper.h saturates
        np.where(sign, -1, 0),
        b8 >> np.minimum(shift, 31),
    ).astype(np.int8)

    recs = np.empty((blocks.shape[0], BYTES_PER_BLOCK), dtype=np.uint8)
    recs[:, 0] = max_exp.ravel().astype(np.uint8)
    recs[:, 1:] = aligned.view(np.uint8)
    return recs.reshape(*lead, nbytes(W))


def bfp16ebs8_to_float(b):
    """[..., W*9//8] uint8 -> [..., W] float32. Inverse of the above."""
    b = np.ascontiguousarray(b, dtype=np.uint8)
    lead, WB = b.shape[:-1], b.shape[-1]
    assert WB % BYTES_PER_BLOCK == 0, f"{WB} is not a multiple of {BYTES_PER_BLOCK}"

    recs = b.reshape(-1, BYTES_PER_BLOCK)
    shared_exp = recs[:, 0].astype(np.int32)
    mant = recs[:, 1:].view(np.int8).astype(np.float32)
    # helper.h: multiplier = 2^(e-127) / 64
    mult = np.ldexp(np.float32(1.0), shared_exp - 127 - 6).astype(np.float32)
    vals = mant * mult[:, None]
    return vals.reshape(*lead, WB // BYTES_PER_BLOCK * BLOCK)


def shuffle_bfp16ebs8(mat, tile_h, tile_w_elems, unshuffle=False):
    """Reorder 8x8 sub-tiles to be contiguous within each tile box.

    Mirrors shuffleMatrixForBfp16ebs8() in helper.h. `mat` is the packed
    [H, W*9//8] uint8 matrix; the transform is intra-tile-box only, so the
    global matrix stays row-major and the L3 buffer shape is unchanged.
    """
    mat = np.ascontiguousarray(mat, dtype=np.uint8)
    H, MW = mat.shape
    tw = nbytes(tile_w_elems)  # tile width in bytes
    assert tile_w_elems % 64 == 0, "tile width must be a multiple of 64 elements"
    assert tile_h % BLOCK == 0, "tile height must be a multiple of 8"
    assert H % tile_h == 0 and MW % tw == 0

    ty, tx = H // tile_h, MW // tw
    # [ty, r, tx, c] -> per-tile-box [n_boxes, tile_h, tw]
    boxes = (
        mat.reshape(ty, tile_h, tx, tw).transpose(0, 2, 1, 3).reshape(-1, tile_h, tw)
    )
    if not unshuffle:
        # [sy, i, sx, j] -> [sy, sx, i, j]: sub-tiles become 72B-contiguous.
        boxes = (
            boxes.reshape(-1, tile_h // 8, 8, tw // BYTES_PER_BLOCK, BYTES_PER_BLOCK)
            .transpose(0, 1, 3, 2, 4)
            .reshape(-1, tile_h, tw)
        )
    else:
        boxes = (
            boxes.reshape(-1, tile_h // 8, tw // BYTES_PER_BLOCK, 8, BYTES_PER_BLOCK)
            .transpose(0, 1, 3, 2, 4)
            .reshape(-1, tile_h, tw)
        )
    out = boxes.reshape(ty, tx, tile_h, tw).transpose(0, 2, 1, 3).reshape(H, MW)
    return np.ascontiguousarray(out)


def nearly_equal(a, e, rel_tol=0.05, abs_tol=0.5):
    """matmul_common::nearly_equal from mlir-aie basic/matrix_multiplication/common.h.

    Note this is `|a-e| < max(abs_tol, rel_tol*(|a|+|e|))` -- a max, and the
    norm is the SUM of magnitudes. It is NOT np.isclose's `atol + rtol*|e|`.
    """
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    diff = np.abs(a - e)
    norm = np.abs(a) + np.abs(e)
    return (a == e) | (diff < np.maximum(abs_tol, rel_tol * norm))
