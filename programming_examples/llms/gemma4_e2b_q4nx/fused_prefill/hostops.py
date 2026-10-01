# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ctypes front end of kernels/hostops.c (built by build.py into the build
dir as libhostops.so): the host side of the fused prefill, in C."""

import ctypes
import os

import numpy as np

_lib = None
_p = ctypes.c_void_p
_i = ctypes.c_int
_f = ctypes.c_float


def load(build_dir):
    global _lib
    _lib = ctypes.CDLL(os.path.join(build_dir, "libhostops.so"))
    _lib.tile_a.argtypes = [_p, _i, _i, _i, _i, _p]
    _lib.bf16_to_f32.argtypes = [_p, _i, _i, _i, _p]
    _lib.rms.argtypes = [_p, _i, _i, _p, _f, _p]
    _lib.add_rms.argtypes = [_p, _p, _i, _i, _p, _f, _p]
    _lib.rope.argtypes = [_p, _i, _i, _i, _i, _p, _p]
    _lib.glu_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _i, _p]
    _lib.mul_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _p]


def tile_a(x, dst, tm, tk):
    """x [t, K] float32 (C order) -> dst, a bf16 [HR, K/tk, tm, tk] array."""
    x = np.ascontiguousarray(x, np.float32)
    _lib.tile_a(x.ctypes.data, x.shape[0], x.shape[1], tm, tk, dst.ctypes.data)


def bf16_to_f32(src, rows, n):
    """src [R, ld] bf16 (C order) -> new float32 [rows, n]."""
    out = np.empty((rows, n), np.float32)
    _lib.bf16_to_f32(src.ctypes.data, rows, src.shape[1], n, out.ctypes.data)
    return out


def _c(a):
    return np.ascontiguousarray(a, np.float32)


def rms(x, w, eps):
    x = _c(x)
    n = x.shape[-1]
    y = np.empty_like(x)
    _lib.rms(
        x.ctypes.data,
        x.size // n,
        n,
        None if w is None else _c(w).ctypes.data,
        eps,
        y.ctypes.data,
    )
    return y


def add_rms(res, x, w, eps):
    res, x = _c(res), _c(x)
    n = x.shape[-1]
    y = np.empty_like(x)
    _lib.add_rms(
        res.ctypes.data,
        x.ctypes.data,
        x.size // n,
        n,
        _c(w).ctypes.data,
        eps,
        y.ctypes.data,
    )
    return y


def rope(x, cs, sn, rot):
    """x [T, H, dh] float32 C-order, in place; cs / sn [T, rot/2]."""
    t, h, dh = x.shape
    _lib.rope(x.ctypes.data, t, h, dh, rot, _c(cs).ctypes.data, _c(sn).ctypes.data)
    return x


def glu_tile(g, u, t, n, dst, tm, tk):
    """dst (tiled A of K = n) = bf16(g[:t, :n] * u[:t, :n]); g, u bf16 [R, ld]."""
    _lib.glu_tile(
        g.ctypes.data,
        g.shape[1],
        u.ctypes.data,
        u.shape[1],
        t,
        n,
        tm,
        tk,
        dst.ctypes.data,
    )


def mul_tile(g, p, dst, tm, tk):
    """dst (tiled A of K = n) = bf16(g[:t, :n] * p); g bf16 [R, ld], p f32 [t, n]."""
    p = _c(p)
    _lib.mul_tile(
        g.ctypes.data,
        g.shape[1],
        p.ctypes.data,
        p.shape[0],
        p.shape[1],
        tm,
        tk,
        dst.ctypes.data,
    )
