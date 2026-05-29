"""Single-launch standalone MLIR modules for the decode rms_gemv_rope kernel-group.

Each function returns a ready-to-compile mlir.Module containing exactly one
air.launch (or launch+segment for sub-builders that emit bare herds) at
production decode shape (single-token, emb_dim=2048, kv_dim=512,
n_heads=32, n_kv_heads=8, head_dim=64).

These are the Cell-A/B/C inputs. Cell D reuses the production merged
build_rms_gemv_rope_module from multi_launch_builder/rms_gemv_rope_multi.py.

The 6 sub-launches mirror the production stitch-spec in
multi_launch_builder/rms_gemv_rope_multi.py.
"""

from ml_dtypes import bfloat16

from multi_launch_builder.rms_gemv_rope_multi import (
    _build_rms_1d,
    _build_rope_1d,
)


def build_rmsnorm(emb_dim=2048):
    """RMSNorm 1D: (x_in[emb_dim], norm_w[emb_dim]) -> normed[emb_dim]."""
    return _build_rms_1d(emb_dim, bfloat16, 16)


def build_gemv(out_dim, in_dim, tile_m=8, m_input=4, herd_m=8):
    """Generic decode GEMV: (W[out_dim, in_dim], x[in_dim]) -> y[out_dim].

    Covers Q (out=emb=2048), K/V (out=kv=512).
    """
    from matvec import build_module as _build_gemv

    return _build_gemv(out_dim, in_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)


def build_rope(n_rows, head_dim=64, herd_x=1):
    """RoPE 1D: (x_flat[n_rows*head_dim], lut[head_dim]) -> y_flat[n_rows*head_dim].

    Covers RoPE Q (n_rows=n_heads=32) and RoPE K (n_rows=n_kv_heads=8).
    """
    return _build_rope_1d(n_rows, head_dim, bfloat16, herd_x)


# Full registry of standalones for this kernel-group.
# Each entry: (name, build_fn, build_kwargs)
STANDALONES = [
    ("rmsnorm", build_rmsnorm, {"emb_dim": 2048}),
    ("q_gemv", build_gemv, {"out_dim": 2048, "in_dim": 2048}),
    ("k_gemv", build_gemv, {"out_dim": 512, "in_dim": 2048}),
    ("v_gemv", build_gemv, {"out_dim": 512, "in_dim": 2048}),
    ("rope_q", build_rope, {"n_rows": 32, "head_dim": 64}),
    ("rope_k", build_rope, {"n_rows": 8, "head_dim": 64}),
]
