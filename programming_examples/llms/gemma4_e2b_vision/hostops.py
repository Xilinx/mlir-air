# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ctypes front end of hostops.c, built next to it on first use."""

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "hostops.c"
_p, _i, _l, _f = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_float
_lib = None


def build_dir():
    """Where generated files go: $G4_VISION_BUILD_DIR, else .build next to the
    sources."""
    return Path(os.environ.get("G4_VISION_BUILD_DIR") or _HERE / ".build")


def _load():
    global _lib
    if _lib is not None:
        return _lib
    _LIB = build_dir() / "libvishostops.so"
    if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
        _LIB.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LIB.with_suffix(f".{os.getpid()}.so")
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC"]
            + [str(_SRC), "-o", str(tmp), "-lm"],
            check=True,
        )
        os.replace(tmp, _LIB)
    # Keep idle OpenMP threads from spinning between calls. Read when libgomp
    # loads.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    lib = ctypes.CDLL(str(_LIB))
    lib.rms_clip_bf16.argtypes = [_p, _i, _i, _i, _p, _f, _f, _f, _p]
    lib.add_rms_clip.argtypes = [_p, _p, _i, _i, _i, _p, _f, _f, _f]
    lib.qkv_prep.argtypes = [_p, _i, _i, _i, _i, _i, _i, _p, _p, _f, _p, _p, _p]
    lib.qkv_prep.argtypes += [_f, _f, _p, _p, _p]
    lib.glu_clip.argtypes = [_p, _p, _l, _f, _f, _f, _f, _f, _p]
    lib.clip_bf16.argtypes = [_p, _l, _f, _f]
    _lib = lib
    return lib


def _f32(a):
    return np.ascontiguousarray(a, np.float32)


def _bf(a):
    return np.ascontiguousarray(a, bfloat16)


def _ptr(a):
    return None if a is None else a.ctypes.data


def rms_clip_bf16(x, valid, w, eps, lo, hi, out):
    """out = bf16(clip(rmsnorm(x) * w)) on the first `valid` rows, 0 after."""
    x, w = _f32(x), _f32(w)
    rows, n = x.shape
    assert out.shape == (rows, n) and out.dtype == bfloat16, (out.shape, out.dtype)
    _load().rms_clip_bf16(
        x.ctypes.data, rows, valid, n, w.ctypes.data, eps, lo, hi, out.ctypes.data
    )
    return out


def add_rms_clip(h, y, valid, w, eps, lo, hi):
    """h += rmsnorm(clip(y)) * w in place on the first `valid` rows, 0 after."""
    assert h.flags.c_contiguous and h.dtype == np.float32
    y, w = _bf(y), _f32(w)
    rows, n = h.shape
    _load().add_rms_clip(
        h.ctypes.data, y.ctypes.data, rows, valid, n, w.ctypes.data, eps, lo, hi
    )
    return h


def qkv_prep(
    qkv, valid, heads, dh, dk, qn, kn, eps, cos, sin, bounds, qscale, big, Q, K, V
):
    """The fused qkv GEMM output -> the attention's Q, K, V (see hostops.c)."""
    assert dh <= 256, dh  # hostops.c keeps one head in a 256-float buffer
    qkv = _bf(qkv)
    rows, kv_rows = Q.shape[0], K.shape[0]
    assert qkv.shape[0] >= max(rows, kv_rows), (qkv.shape, rows, kv_rows)
    assert Q.shape[1] == K.shape[1] == heads * dk and V.shape[1] == heads * dh
    cos, sin = _f32(cos), _f32(sin)
    assert cos.shape[0] >= max(rows, kv_rows), (cos.shape, rows, kv_rows)
    b = _f32(bounds)
    _load().qkv_prep(
        qkv.ctypes.data,
        rows,
        valid,
        kv_rows,
        heads,
        dh,
        dk,
        _ptr(_f32(qn)),
        _ptr(_f32(kn)),
        eps,
        cos.ctypes.data,
        sin.ctypes.data,
        b.ctypes.data,
        qscale,
        big,
        Q.ctypes.data,
        K.ctypes.data,
        V.ctypes.data,
    )


def glu_clip(g, u, g_hi, u_lo, u_hi, lo, hi, out):
    """out = bf16(clip(min(g, g_hi) * clip(u, u_lo, u_hi), lo, hi))."""
    g, u = _bf(g), _bf(u)
    assert out.shape == g.shape == u.shape and out.dtype == bfloat16
    _load().glu_clip(
        g.ctypes.data, u.ctypes.data, g.size, g_hi, u_lo, u_hi, lo, hi, out.ctypes.data
    )
    return out


def clip_bf16(x, lo, hi):
    """x = bf16(clip(x)) in place, x bf16."""
    assert x.flags.c_contiguous and x.dtype == bfloat16
    _load().clip_bf16(x.ctypes.data, x.size, lo, hi)
    return x
