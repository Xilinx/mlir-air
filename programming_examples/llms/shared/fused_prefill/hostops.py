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
    _l = ctypes.c_long
    _lib.tiles_bfp.argtypes = [_p, _i, _i, _i, _i, _i, _i, _p, _l, _l]
    _lib.bfp_rows.argtypes = [_p, ctypes.c_long, _p]
    _lib.bf16_to_f32.argtypes = [_p, _i, _i, _i, _p]
    _lib.rms.argtypes = [_p, _i, _i, _p, _f, _p]
    _lib.add_rms.argtypes = [_p, _p, _i, _i, _p, _f, _p]
    _lib.rope.argtypes = [_p, _i, _i, _i, _i, _p, _p]
    _lib.glu_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _i, _p]
    _lib.mul_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _p]
    _lib.q_pack.argtypes = [_p, _i, _i, _i, _i, _f, _p]
    _lib.rms_tile.argtypes = [_p, _i, _i, _p, _f, _i, _i, _p]
    _lib.add_rms_tile.argtypes = [_p, _p, _i, _p, _f, _i, _i, _p, _i, _i, _p]
    _l = ctypes.c_long
    _lib.head_post.argtypes = [_p, _i, _i, _i, _i, _i, _p, _p, _f, _p, _p, _i, _f]
    _lib.head_post.argtypes += [_p, _l, _l]
    _lib.o_tile.argtypes = [_p, _i, _i, _i, _i, _i, _i, _i, _i, _i, _p]
    _lib.kv_rec_bfp.argtypes = [_p, _p, _i, _i, _i, _i, _i, _p]
    _lib.q_bfp.argtypes = [_p, _i, _i, _i, _i, _p]


_tiles = {}


def _bf16_tiles(dst, tm, tk):
    """The bf16 tiles [M/tm, K/tk, tm, tk] the tilers fill for dst, the GEMM's
    A; reused per K."""
    key = (dst.shape[0], dst.shape[1], tm, tk)
    if key not in _tiles:
        _tiles[key] = np.zeros(key, np.uint16)
    return _tiles[key]


def _to_bfp(tiles, t, dst, k0=0, k1=None):
    """Rows [0, t), columns [k0, k1) of tiles into dst: bfp16 bytes
    [M/tm, K/tk, tile], or bf16 tiles [M/tm, K/tk, tm, tk] for a GEMM that
    reads bf16 (packing.a_layout). Either may be a strided view
    (Engine.a_chunk)."""
    _, ks, tm, tk = tiles.shape
    k1 = ks * tk if k1 is None else k1
    if dst.ndim == 4:
        nb = -(-t // tm)  # row blocks with rows below t
        src = tiles[:nb].view(dst.dtype)
        # whole K steps in one copy, a partial step at either end on its own
        s0, s1 = -(-k0 // tk), k1 // tk
        dst[:nb, s0:s1] = src[:, s0:s1]
        for s in {k0 // tk, s1} - set(range(s0, s1)):
            c0, c1 = max(k0 - s * tk, 0), min(k1 - s * tk, tk)
            if c0 < c1:
                dst[:nb, s, :, c0:c1] = src[:, s, :, c0:c1]
        return
    assert dst.strides[2] == 1, dst.strides
    dr, dk = (v // dst.shape[2] for v in dst.strides[:2])
    _lib.tiles_bfp(
        tiles.ctypes.data, t, ks * tk, tm, tk, k0, k1, dst.ctypes.data, dr, dk
    )


def tile_a(x, dst, tm, tk):
    """x [t, K] float32 (C order) -> dst, the GEMM's A (bfp16 tiles)."""
    x = np.ascontiguousarray(x, np.float32)
    a = _bf16_tiles(dst, tm, tk)
    _lib.tile_a(x.ctypes.data, x.shape[0], x.shape[1], tm, tk, a.ctypes.data)
    _to_bfp(a, x.shape[0], dst)


def bfp_rows(src):
    """src [..., 8] bf16 -> bfp16 groups [..., 9] uint8."""
    src = np.ascontiguousarray(src)
    out = np.empty(src.shape[:-1] + (9,), np.uint8)
    _lib.bfp_rows(src.ctypes.data, src.size // 8, out.ctypes.data)
    return out


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
    """dst (A of K = n) = bf16(g[:t, :n] * u[:t, :n]); g, u bf16 [R, ld]."""
    a = _bf16_tiles(dst, tm, tk)
    _lib.glu_tile(
        g.ctypes.data,
        g.shape[1],
        u.ctypes.data,
        u.shape[1],
        t,
        n,
        tm,
        tk,
        a.ctypes.data,
    )
    _to_bfp(a, t, dst)


def mul_tile(g, p, dst, tm, tk):
    """dst (A of K = n) = bf16(g[:t, :n] * p); g bf16 [R, ld], p f32 [t, n]."""
    p = _c(p)
    a = _bf16_tiles(dst, tm, tk)
    _lib.mul_tile(
        g.ctypes.data,
        g.shape[1],
        p.ctypes.data,
        p.shape[0],
        p.shape[1],
        tm,
        tk,
        a.ctypes.data,
    )
    _to_bfp(a, p.shape[0], dst)


def q_pack(q, scale, dst):
    """dst [H, M, dh] bf16 = q [T, H, dh] * scale, head-first, rows T.. zeroed."""
    q = _c(q)
    t, h, dh = q.shape
    _lib.q_pack(q.ctypes.data, t, h, dh, dst.shape[1], scale, dst.ctypes.data)


def _ptr(a):
    return None if a is None else a.ctypes.data


def rms_tile(x, w, eps, dst, tm, tk):
    """dst (A) = bf16(rms(x) * w); x [t, n] float32."""
    t, n = x.shape
    a = _bf16_tiles(dst, tm, tk)
    _lib.rms_tile(_c(x).ctypes.data, t, n, _ptr(w), eps, tm, tk, a.ctypes.data)
    _to_bfp(a, t, dst)


def add_rms_tile(x, c, t, post, eps, w, dst, tm, tk):
    """x[:t] += post ? rms(c) * post : c (c bf16 [R, ld]), in place; then
    dst (A) = bf16(rms(x) * w) if w is given."""
    n = x.shape[1]
    a = None if dst is None else _bf16_tiles(dst, tm, tk)
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
        _ptr(a),
    )
    if a is not None:
        _to_bfp(a, t, dst)


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


def o_tile(o, t, h0, k, dst, tm, tk, tq):
    """Heads of o bf16 [nh, M, dh], in tiles of tq rows as the attention cores
    write them, into heads [h0, h0 + nh) of dst (A, K = k)."""
    nh, m, dh = o.shape
    a = _bf16_tiles(dst, tm, tk)
    _lib.o_tile(o.ctypes.data, t, nh, dh, m, tq, h0, k, tm, tk, a.ctypes.data)
    _to_bfp(a, t, dst, h0 * dh, (h0 + nh) * dh)


def kv_rec_bfp(k, v, c0, t, dh, lkp, dst):
    """kv_rec, as bfp16 (attn_bfp16.cc's records)."""
    _lib.kv_rec_bfp(
        k.ctypes.data, v.ctypes.data, k.shape[1], c0, t, dh, lkp, dst.ctypes.data
    )


def q_bfp(q, tq, dst):
    """q bf16 [nh, M, dh] as bfp16 tiles of tq rows into dst (bytes)."""
    nh, m, dh = q.shape
    _lib.q_bfp(q.ctypes.data, nh, m, dh, tq, dst.ctypes.data)
