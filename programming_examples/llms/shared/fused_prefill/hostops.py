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
    _lib.tile_a_rows.argtypes = [_p, _i, _p, _i, _i, _i, _i, _p, _l, _l]
    _lib.glu_tile_a.argtypes = [_p, _i, _i, _i, _i, _i, _p, _l, _l]
    _lib.bf16_to_f32.argtypes = [_p, _i, _i, _i, _p]
    _lib.rms.argtypes = [_p, _i, _i, _p, _f, _p]
    _lib.add_rms.argtypes = [_p, _p, _i, _i, _p, _f, _p]
    _lib.rope.argtypes = [_p, _i, _i, _i, _i, _p, _p]
    _lib.glu_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _i, _p]
    _lib.oss_glu_tile.argtypes = [_p, _i, _p, _p, _i, _i, _i, _i, _p]
    _lib.scatter_add_rows.argtypes = [_p, _i, _p, _i, _p, _p, _p, _i, _i]
    _lib.mul_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _p]
    _lib.q_pack.argtypes = [_p, _i, _i, _i, _i, _f, _p]
    _lib.rms_tile.argtypes = [_p, _i, _i, _p, _f, _i, _i, _p]
    _lib.add_rms_tile.argtypes = [_p, _p, _i, _p, _f, _i, _i, _p, _i, _i, _p]
    _lib.add_rms_tile_s.argtypes = _lib.add_rms_tile.argtypes + [_f]
    _lib.ln_tile.argtypes = [_p, _i, _i, _p, _p, _f, _i, _i, _p]
    _lib.sub_row_bf16.argtypes = [_p, _i, _i, _p, _p]
    _lib.bias_gelu_tile.argtypes = [_p, _i, _p, _i, _i, _i, _i, _p]
    _lib.bias_gelu_erf_tile.argtypes = _lib.bias_gelu_tile.argtypes
    _lib.bias_gelu_erf_rows.argtypes = [_p, _i, _p, _i, _i, _p, _i]
    _l = ctypes.c_long
    _lib.head_post.argtypes = [_p, _i, _i, _i, _i, _i, _p, _p, _f, _p, _p, _i, _f]
    _lib.head_post.argtypes += [_p, _l, _l]
    _lib.o_tile.argtypes = [_p, _i, _i, _i, _i, _i, _i, _i, _i, _i, _p]
    _lib.kv_rec_bfp.argtypes = [_p, _p, _i, _i, _i, _i, _i, _p]
    _lib.q_bfp.argtypes = [_p, _i, _i, _i, _i, _p]
    _lib.o_tiles.argtypes = [_p, _i, _i, _i, _i, _i, _i, _i, _i, _p]
    _lib.kv_rec_bf16.argtypes = [_p, _p, _i, _i, _i, _i, _i, _i, _p]
    _lib.q_tiles.argtypes = [_p, _i, _i, _i, _p]


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


def _tile_strides(dst):
    """dst's row-block and K-step strides in tiles (tiles_bfp)."""
    assert dst.ndim == 3 and dst.strides[2] == 1, dst.strides
    return [v // dst.shape[2] for v in dst.strides[:2]]


def tile_a_rows(x, rows, dst, tm, tk):
    """tile_a of x[rows], read in place: x float32 [*, K] (C order)."""
    x = np.ascontiguousarray(x, np.float32)
    rows = np.ascontiguousarray(rows, np.int64)
    _lib.tile_a_rows(
        x.ctypes.data,
        x.shape[1],
        rows.ctypes.data,
        len(rows),
        x.shape[1],
        tm,
        tk,
        dst.ctypes.data,
        *_tile_strides(dst),
    )


def glu_tile_a(c, t, n, dst, tm, tk):
    """bf16(u * SiLU(g)) of c's first t rows into dst, the GEMM's A: c bf16
    [*, ld] holds u and g in turns of tk columns, n of each."""
    _lib.glu_tile_a(
        c.ctypes.data, c.shape[1], t, n, tm, tk, dst.ctypes.data, *_tile_strides(dst)
    )


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


def oss_glu_tile(gu, gb, ub, t, n, dst, tm, tk):
    """dst (A of K = n) = gpt-oss's clamped SwiGLU of gu bf16 [R, ld], gate
    and up in turns of 128 columns, with their biases gb, ub [n]."""
    a = _bf16_tiles(dst, tm, tk)
    _lib.oss_glu_tile(
        gu.ctypes.data,
        gu.shape[1],
        _c(gb).ctypes.data,
        _c(ub).ctypes.data,
        t,
        n,
        tm,
        tk,
        a.ctypes.data,
    )
    _to_bfp(a, t, dst)


def scatter_add_rows(out, y, rows, w, b, n):
    """out[rows[i], :n] += w[i] * (y[i, :n] + b): out float32 [*, ld] (C
    order), y bf16 [t, ld], rows (distinct) and w [t], b [n]."""
    rows = np.ascontiguousarray(rows, np.int64)
    w, b = _c(w), _c(b)
    _lib.scatter_add_rows(
        out.ctypes.data,
        out.shape[1],
        y.ctypes.data,
        y.shape[1],
        rows.ctypes.data,
        w.ctypes.data,
        b.ctypes.data,
        len(rows),
        n,
    )


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


def add_rms_tile(x, c, t, post, eps, w, dst, tm, tk, scale=1.0):
    """x[:t] += post ? rms(c) * post : c (c bf16 [R, ld]), in place; then
    dst (A) = bf16(rms(x) * w) if w is given. A scale other than 1 takes
    bf16(c * scale) for c."""
    n = x.shape[1]
    a = None if dst is None else _bf16_tiles(dst, tm, tk)
    args = [
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
    ]
    if scale == 1.0:
        _lib.add_rms_tile(*args)
    else:
        _lib.add_rms_tile_s(*args, scale)
    if a is not None:
        _to_bfp(a, t, dst)


def sub_row(src, row, dst):
    """dst = bf16(src - row): src, dst bf16 [t, n] (C order), row [n]."""
    t, n = src.shape
    row = _c(row)
    _lib.sub_row_bf16(src.ctypes.data, t, n, row.ctypes.data, dst.ctypes.data)


def ln_tile(x, w, b, eps, dst, tm, tk):
    """dst (A) = bf16(layernorm(x) * w + b); x [t, n] float32."""
    t, n = x.shape
    a = _bf16_tiles(dst, tm, tk)
    _lib.ln_tile(_c(x).ctypes.data, t, n, _ptr(w), _ptr(b), eps, tm, tk, a.ctypes.data)
    _to_bfp(a, t, dst)


def bias_gelu_tile(g, bias, t, n, dst, tm, tk):
    """dst (A) = bf16(gelu_tanh(g + bias)), g bf16 [R, ld] (rows [0, t),
    columns [0, n)), bias float32 [n]."""
    a = _bf16_tiles(dst, tm, tk)
    _lib.bias_gelu_tile(
        g.ctypes.data, g.shape[1], _ptr(bias), t, n, tm, tk, a.ctypes.data
    )
    _to_bfp(a, t, dst)


def bias_gelu_erf_tile(g, bias, t, n, dst, tm, tk):
    """bias_gelu_tile with the erf form of gelu."""
    a = _bf16_tiles(dst, tm, tk)
    _lib.bias_gelu_erf_tile(
        g.ctypes.data, g.shape[1], _ptr(bias), t, n, tm, tk, a.ctypes.data
    )
    _to_bfp(a, t, dst)


def bias_gelu_erf_rows(g, bias, t, n, dst):
    """dst[:t, :n] = bf16(gelu_erf(g + bias)); g and dst bf16 [R, ld] (C
    order rows), bias float32 [n]."""
    _lib.bias_gelu_erf_rows(
        g.ctypes.data, g.shape[1], _ptr(bias), t, n, dst.ctypes.data, dst.shape[1]
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


def o_tile(o, t, h0, k, dst, tm, tk, tq):
    """Heads of o bf16 [nh, M, dh], in tiles of tq rows as the attention cores
    write them, into heads [h0, h0 + nh) of dst (A, K = k)."""
    nh, m, dh = o.shape
    a = _bf16_tiles(dst, tm, tk)
    _lib.o_tile(o.ctypes.data, t, nh, dh, m, tq, h0, k, tm, tk, a.ctypes.data)
    _to_bfp(a, t, dst, h0 * dh, (h0 + nh) * dh)


def o_tiles(o, t, h0, k, dst, tm, tk):
    """Heads of o bf16 [nh, M, dh] in 8x8 blocks (q_tiles' order) into heads
    [h0, h0 + nh) of dst (A, K = k)."""
    nh, m, dh = o.shape
    a = _bf16_tiles(dst, tm, tk)
    _lib.o_tiles(o.ctypes.data, t, nh, dh, m, h0, k, tm, tk, a.ctypes.data)
    _to_bfp(a, t, dst, h0 * dh, (h0 + nh) * dh)


def kv_rec_bfp(k, v, c0, t, dh, lkp, dst):
    """kv_rec, as bfp16 (attn_bfp16.cc's records)."""
    _lib.kv_rec_bfp(
        k.ctypes.data, v.ctypes.data, k.shape[1], c0, t, dh, lkp, dst.ctypes.data
    )


def kv_rec_bf16(k, v, c0, t, dh, lkp, dvt, dst):
    """Columns [c0, c0 + dh) of k and v bf16 [t, ld] as bf16 records into dst:
    per lkp rows the K tile's 8x8 blocks [lkp/8, dh/8], then V's in column
    chunks dvt wide, each [lkp/8, dvt/8]; each block row-major, rows past t
    zero."""
    _lib.kv_rec_bf16(
        k.ctypes.data, v.ctypes.data, k.shape[1], c0, t, dh, lkp, dvt, dst.ctypes.data
    )


def q_tiles(q, dst):
    """q bf16 [nh, M, dh] into dst as 8x8 blocks, per head [M/8, dh/8], each
    row-major."""
    nh, m, dh = q.shape
    _lib.q_tiles(q.ctypes.data, nh, m, dh, dst.ctypes.data)


def q_bfp(q, tq, dst):
    """q bf16 [nh, M, dh] as bfp16 tiles of tq rows into dst (bytes)."""
    nh, m, dh = q.shape
    _lib.q_bfp(q.ctypes.data, nh, m, dh, tq, dst.ctypes.data)
