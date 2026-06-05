# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Thin GEMM wrapper for bf16 x bfp16ebs8 GEMM, callable from the bfp16
prefill stitchers."""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_BFP_GEMM_DIR = os.path.normpath(
    os.path.join(_THIS, "..", "..", "matrix_multiplication", "bf16_x_bfp16")
)
if _BFP_GEMM_DIR not in sys.path:
    sys.path.insert(0, _BFP_GEMM_DIR)


def _build_bfp16_gemm_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
):
    """Build a bf16 A x bfp16ebs8 B GEMM MLIR module."""
    from matmul_bf16_x_bfp16 import build_module as build_bfp16_gemm

    return build_bfp16_gemm(
        m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n
    )
