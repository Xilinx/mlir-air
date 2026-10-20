# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side weight conversion for the LLM examples, on torch.

Loading a model dequantizes or requantizes every projection and lays it out
transposed in bf16. These helpers do that with torch, which runs on all cores,
and work through large matrices in row blocks so that intermediates stay in
cache. Every function returns exactly the bits of the plain numpy expression in
its docstring; inputs of other dtypes than the ones the torch path handles go
through that numpy expression unchanged.
"""

import warnings

import numpy as np
from ml_dtypes import bfloat16

from ....fused_decode.proj_qmm_pack import (
    BLOCK_BF16,
    COL_BLOCK,
    N_GROUPS,
    PARALLEL,
    ROW_BLOCK,
)


class _Torch:
    """torch, imported on first use, so that importing this module does not
    need torch."""

    def __getattr__(self, name):
        import torch

        return getattr(torch, name)


torch = _Torch()

# Rows per block. Small enough for one block's float32 intermediates to stay in
# cache, large enough to amortize the per-op overhead.
_ROWS = 256


def _t(a):
    """numpy array -> torch tensor sharing its memory (bf16 as torch.bfloat16).
    The input may be read-only (a memory-mapped file); it is only read."""
    a = np.asarray(a)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        if a.dtype == bfloat16:
            return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(a)


def _np_bf16(t):
    """torch.bfloat16 tensor -> numpy bfloat16 array sharing its memory."""
    return t.view(torch.int16).numpy().view(bfloat16)


def bf16_bytes_to_f32(buf, shape):
    """`np.frombuffer(buf, bfloat16).astype(np.float32).reshape(shape)`."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        t = torch.frombuffer(memoryview(buf), dtype=torch.bfloat16)
    return t.float().reshape(shape).numpy()


def to_f32(a):
    """`np.asarray(a).astype(np.float32)`."""
    a = np.asarray(a)
    if a.dtype not in (np.float32, bfloat16, np.float16):
        return a.astype(np.float32)
    return _t(a).float().numpy()


def transpose(w):
    """`np.ascontiguousarray(np.asarray(w).T)` for a 2-D `w` (same dtype)."""
    w = np.asarray(w)
    if w.dtype not in (np.float32, bfloat16):
        return np.ascontiguousarray(w.T)
    out = _t(w).t().contiguous()
    return _np_bf16(out) if w.dtype == bfloat16 else out.numpy()


def transpose_bf16(w):
    """`np.ascontiguousarray(np.asarray(w).T, dtype=bfloat16)` for a 2-D `w`."""
    w = np.asarray(w)
    if w.dtype == bfloat16:
        return transpose(w)
    if w.dtype != np.float32:
        return np.ascontiguousarray(w.T, dtype=bfloat16)
    # Converting a row block at a time keeps it in cache; a plain transpose
    # (above) is faster when there is nothing to convert.
    src = _t(w)
    m, k = src.shape
    out = torch.empty((k, m), dtype=torch.bfloat16)
    for r in range(0, m, _ROWS):
        out[:, r : r + _ROWS] = src[r : r + _ROWS].t()
    return _np_bf16(out)


def _dequant_rows(q, scale, offset, group, additive, r):
    """Rows [r, r + _ROWS) of dequant_q4, float32 [rows, K]."""
    k = q.shape[1]
    qg = q[r : r + _ROWS].reshape(-1, k // group, group).float()
    s = scale[r : r + _ROWS, :, None]
    if offset is None:
        blk = s * qg
    elif additive:
        blk = s * qg + offset[r : r + _ROWS, :, None]
    else:
        blk = s * (qg - offset[r : r + _ROWS, :, None])
    return blk.reshape(-1, k)


def _np_dequant_q4(q, scale, offset, group, additive):
    s = np.repeat(scale, group, axis=1)
    if offset is None:
        return q.astype(np.float32) * s
    o = np.repeat(offset, group, axis=1)
    return s * q + o if additive else s * (q - o)


def _torch_dequant_ok(q, scale, offset):
    """Whether the torch path reproduces numpy's dtypes for these operands:
    a 4-bit code array and float32 scales and offsets."""
    return (
        np.asarray(q).dtype in (np.uint8, np.int8)
        and np.asarray(scale).dtype == np.float32
        and (offset is None or np.asarray(offset).dtype == np.float32)
    )


def dequant_q4(q, scale, offset, group, additive=True):
    """Group dequant of `q` [M, K] with `scale`/`offset` [M, K/group]. With
    S = np.repeat(scale, group, 1), O = np.repeat(offset, group, 1): `S * q + O`
    if `additive`, `S * (q - O)` if not, and `q.astype(np.float32) * S` if
    `offset` is None.
    """
    if not _torch_dequant_ok(q, scale, offset):
        return _np_dequant_q4(np.asarray(q), scale, offset, group, additive)
    q, scale = _t(q), _t(scale)
    offset = None if offset is None else _t(offset)
    m, k = q.shape
    out = torch.empty((m, k), dtype=torch.float32)
    for r in range(0, m, _ROWS):
        out[r : r + _ROWS] = _dequant_rows(q, scale, offset, group, additive, r)
    return out.numpy()


def dequant_q4_transpose_bf16(q, scale, offset, group, additive=True):
    """`transpose_bf16(dequant_q4(q, scale, offset, group, additive))`."""
    if not _torch_dequant_ok(q, scale, offset):
        return transpose_bf16(dequant_q4(q, scale, offset, group, additive))
    q, scale = _t(q), _t(scale)
    offset = None if offset is None else _t(offset)
    m, k = q.shape
    out = torch.empty((k, m), dtype=torch.bfloat16)
    for r in range(0, m, _ROWS):
        out[:, r : r + _ROWS] = _dequant_rows(q, scale, offset, group, additive, r).t()
    return _np_bf16(out)


def _np_requant_q4k(w, group):
    m, k = w.shape
    wg = w.reshape(m, k // group, group)
    mn = wg.min(2)
    mx = wg.max(2)
    sc = (mx - mn) / 15.0
    sc = np.where(sc <= 0, 1.0, sc).astype(np.float32)
    q = np.clip(np.round((wg - mn[..., None]) / sc[..., None]), 0, 15).astype(np.uint8)
    return q.reshape(m, k), sc, mn.astype(np.float32)


def _minmax_q4(wg):
    """Affine 4-bit quant of one row block [m, K/group, group] (see requant_q4k)."""
    mn = wg.amin(2)
    mx = wg.amax(2)
    sc = (mx - mn) / 15.0
    sc = torch.where(sc <= 0, torch.ones_like(sc), sc)
    q = torch.clamp(torch.round((wg - mn[..., None]) / sc[..., None]), 0, 15)
    return q, sc, mn


def requant_q4k(w, group):
    """Per-group affine 4-bit quant of `w` [M, K] -> (q uint8 [M, K], scale
    float32 [M, K/group], min float32 [M, K/group]), w ~= scale * q + min:

        Wg = w.reshape(M, K // group, group)
        mn, mx = Wg.min(2), Wg.max(2)
        sc = np.where((mx - mn) / 15.0 <= 0, 1.0, (mx - mn) / 15.0)
        q = np.clip(np.round((Wg - mn[..., None]) / sc[..., None]), 0, 15)
    """
    w = np.asarray(w)
    if w.dtype != np.float32:
        return _np_requant_q4k(w, group)
    w = _t(w)
    m, k = w.shape
    q = torch.empty((m, k), dtype=torch.uint8)
    sc = torch.empty((m, k // group), dtype=torch.float32)
    mn = torch.empty((m, k // group), dtype=torch.float32)
    for r in range(0, m, _ROWS):
        qb, sb, mb = _minmax_q4(w[r : r + _ROWS].reshape(-1, k // group, group))
        q[r : r + _ROWS] = qb.reshape(-1, k).to(torch.uint8)
        sc[r : r + _ROWS] = sb
        mn[r : r + _ROWS] = mb
    return q.numpy(), sc.numpy(), mn.numpy()


def requant_q4k_dequant(w, group):
    """`w` rounded onto its requant_q4k grid, float32 [M, K]:
    dequant_q4(*requant_q4k(w, group), group)."""
    w = np.asarray(w)
    if w.dtype != np.float32:
        return dequant_q4(*requant_q4k(w, group), group)
    w = _t(w)
    m, k = w.shape
    out = torch.empty((m, k), dtype=torch.float32)
    for r in range(0, m, _ROWS):
        q, sc, mn = _minmax_q4(w[r : r + _ROWS].reshape(-1, k // group, group))
        out[r : r + _ROWS] = (q * sc[..., None] + mn[..., None]).reshape(-1, k)
    return out.numpy()


def requant_q4_0(w, group):
    """Symmetric signed 4-bit per-group quant of `w` [M, K] -> (q uint8 [M, K]
    holding int8 two's complement, scale float32 [M, K/group]), w ~= q * scale:

        Wg = w.reshape(M, K // group, group)
        sc = np.where(abs(Wg).max(2) <= 0, 1.0, abs(Wg).max(2) / 8.0)
        q = np.clip(np.rint(Wg / sc[..., None]), -8, 7).astype(np.int8)
    """
    w = np.asarray(w)
    if w.dtype != np.float32:
        m, k = w.shape
        wg = w.reshape(m, k // group, group)
        amax = np.abs(wg).max(2)
        sc = np.where(amax <= 0, 1.0, amax / 8.0).astype(np.float32)
        q = np.clip(np.rint(wg / sc[..., None]), -8, 7).astype(np.int8)
        return q.reshape(m, k).view(np.uint8), sc
    w = _t(w)
    m, k = w.shape
    q = torch.empty((m, k), dtype=torch.int8)
    sc = torch.empty((m, k // group), dtype=torch.float32)
    for r in range(0, m, _ROWS):
        wg = w[r : r + _ROWS].reshape(-1, k // group, group)
        amax = wg.abs().amax(2)
        s = torch.where(amax <= 0, torch.ones_like(amax), amax / 8.0)
        qb = torch.clamp(torch.round(wg / s[..., None]), -8, 7)
        q[r : r + _ROWS] = qb.reshape(-1, k).to(torch.int8)
        sc[r : r + _ROWS] = s
    return q.numpy().view(np.uint8), sc.numpy()


# Q4NX "Codec B" layout: a block covers ROW_BLOCK rows x COL_BLOCK columns and
# holds the bf16 scales and mins of its N_GROUPS column groups per row, then the
# nibbles. Within a block, byte k of column c in row group g holds rows
# g*PARALLEL + 2k (low nibble) and g*PARALLEL + 2k + 1 (high nibble).


def unpack_q4nx(i16, nbi, nbj):
    """Block-major Codec B tensor `i16` [nbi*nbj, BLOCK_BF16] int16 -> (q uint8
    [nbi*ROW_BLOCK, nbj*COL_BLOCK], scale and min float32 [nbi*ROW_BLOCK,
    nbj*N_GROUPS]) in logical row-major order; block (i, j) covers row block i
    and column block j."""
    t = _t(np.asarray(i16).reshape(nbi, nbj, BLOCK_BF16))
    m, kq, kg = nbi * ROW_BLOCK, nbj * COL_BLOCK, nbj * N_GROUPS
    q = torch.empty((m, kq), dtype=torch.uint8)
    sc = torch.empty((m, kg), dtype=torch.float32)
    mn = torch.empty((m, kg), dtype=torch.float32)
    per = N_GROUPS * ROW_BLOCK  # scales (and mins) per block
    step = max(1, _ROWS // ROW_BLOCK)
    g = ROW_BLOCK // PARALLEL
    for i in range(0, nbi, step):
        blk = t[i : i + step]
        r = blk.shape[0]
        rows = slice(i * ROW_BLOCK, (i + r) * ROW_BLOCK)

        def groups(a):  # [r, nbj, per] bf16 bits -> [r*ROW_BLOCK, kg] float32
            a = a.contiguous().view(torch.bfloat16).float()
            a = a.reshape(r, nbj, N_GROUPS, ROW_BLOCK).permute(0, 3, 1, 2)
            return a.reshape(r * ROW_BLOCK, kg)

        sc[rows] = groups(blk[..., 0:per])
        mn[rows] = groups(blk[..., per : 2 * per])
        qb = blk[..., 2 * per :].contiguous().view(torch.uint8)
        qb = qb.reshape(r, nbj, g, COL_BLOCK, PARALLEL // 2).transpose(3, 4)
        rows_q = torch.stack((qb & 0xF, qb >> 4), dim=4)  # low, high nibble rows
        rows_q = rows_q.reshape(r, nbj, ROW_BLOCK, COL_BLOCK).permute(0, 2, 1, 3)
        q[rows] = rows_q.reshape(r * ROW_BLOCK, kq)
    return q.numpy(), sc.numpy(), mn.numpy()
