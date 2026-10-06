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
    # spinning OpenMP workers would compete with the XRT waits between ops
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    _lib = ctypes.CDLL(os.path.join(build_dir, "libhostops.so"))
    _lib.tile_a.argtypes = [_p, _i, _i, _i, _i, _p]
    _lib.bf16_to_f32.argtypes = [_p, _i, _i, _i, _p]
    _lib.rms.argtypes = [_p, _i, _i, _p, _f, _p]
    _lib.add_rms.argtypes = [_p, _p, _i, _i, _p, _f, _p]
    _lib.rope.argtypes = [_p, _i, _i, _i, _i, _p, _p]
    _lib.glu_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _i, _p]
    _lib.mul_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _p]
    _lib.q_pack.argtypes = [_p, _i, _i, _i, _i, _f, _p]
    _lib.o_unpack.argtypes = [_p, _i, _i, _i, _i, _p]
    _lib.rms_tile.argtypes = [_p, _i, _i, _p, _f, _i, _i, _p]
    _lib.add_rms_tile.argtypes = [_p, _p, _i, _p, _f, _i, _i, _p, _i, _i, _p]
    _l = ctypes.c_long
    _lib.head_post.argtypes = [_p, _i, _i, _i, _i, _i, _p, _p, _f, _p, _p, _i, _f]
    _lib.head_post.argtypes += [_p, _l, _l]
    _lib.kv_rec.argtypes = [_p, _p, _i, _i, _i, _i, _i, _i, _p]
    _lib.o_tile.argtypes = [_p, _i, _i, _i, _i, _i, _i, _i, _i, _p]


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


def q_pack(q, scale, dst):
    """dst [H, M, dh] bf16 = q [T, H, dh] * scale, head-first, rows T.. zeroed."""
    q = _c(q)
    t, h, dh = q.shape
    _lib.q_pack(q.ctypes.data, t, h, dh, dst.shape[1], scale, dst.ctypes.data)


def o_unpack(o, t):
    """o [H, M, dh] bf16 -> new float32 [t, H * dh]."""
    h, m, dh = o.shape
    out = np.empty((t, h * dh), np.float32)
    _lib.o_unpack(o.ctypes.data, t, h, dh, m, out.ctypes.data)
    return out


def _ptr(a):
    return None if a is None else a.ctypes.data


def rms_tile(x, w, eps, dst, tm, tk):
    """dst (tiled A) = bf16(rms(x) * w); x [t, n] float32."""
    t, n = x.shape
    _lib.rms_tile(_c(x).ctypes.data, t, n, _ptr(w), eps, tm, tk, dst.ctypes.data)


def add_rms_tile(x, c, t, post, eps, w, dst, tm, tk):
    """x[:t] += post ? rms(c) * post : c (c bf16 [R, ld]), in place; then
    dst (tiled A) = bf16(rms(x) * w) if w is given."""
    n = x.shape[1]
    _lib.add_rms_tile(
        x.ctypes.data,
        c.ctypes.data,
        c.shape[1],
        _ptr(post),
        eps,
        t,
        n,
        _ptr(w),
        tm,
        tk,
        None if dst is None else dst.ctypes.data,
    )


def head_post(src, t, col0, nh, dh, bias, norm, eps, rope, scale, dst, hs, rs):
    """Heads [col0, col0 + nh * dh) of src bf16 [R, ld]: + bias, rms * norm,
    rotary (rope = (cos, sin, rot) or None), * scale, to bf16 dst[h * hs + r *
    rs + i] (strides in elements)."""
    cs, sn, rot = rope or (None, None, 0)
    _lib.head_post(
        src.ctypes.data,
        src.shape[1],
        t,
        col0,
        nh,
        dh,
        _ptr(bias),
        _ptr(norm),
        eps,
        _ptr(cs),
        _ptr(sn),
        rot,
        scale,
        dst.ctypes.data,
        hs,
        rs,
    )


def kv_rec(k, v, c0, t, dh, lkp, dvt, dst):
    """KV records of the head at column c0 of k / v bf16 [t, ld] into dst."""
    _lib.kv_rec(
        k.ctypes.data, v.ctypes.data, k.shape[1], c0, t, dh, lkp, dvt, dst.ctypes.data
    )


def o_tile(o, t, h0, k, dst, tm, tk):
    """Heads of o bf16 [nh, M, dh] into heads [h0, h0 + nh) of dst (tiled A, K = k)."""
    nh, m, dh = o.shape
    _lib.o_tile(o.ctypes.data, t, nh, dh, m, h0, k, tm, tk, dst.ctypes.data)
