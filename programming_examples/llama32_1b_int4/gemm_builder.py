# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEMM module builder for the int4-AWQ LLAMA-3.2-1B example on NPU2.

Mirrors `llama32_1b/gemm_builder.py:_build_gemm_module` but produces an
int4-AWQ packed GEMM (dequant + mmul + f32 accumulator) instead of a
bf16 one. No `transform.IR` is needed: `matmul_int4_packed.build_module`
already returns directly-callable MLIR; the kernel-side .o
(`mv_int4_bf16.o`) does the work via `CallOp`.

The function is a thin re-export with the same positional signature as
`matrix_multiplication/int4_awq/matmul_int4_packed.build_module`, so the
int4 prefill stitchers can call it identically.
"""

import os
import sys


_THIS = os.path.dirname(os.path.abspath(__file__))
_INT4_GEMM_DIR = os.path.normpath(
    os.path.join(_THIS, "..", "matrix_multiplication", "int4_awq")
)
if _INT4_GEMM_DIR not in sys.path:
    sys.path.insert(0, _INT4_GEMM_DIR)


def _build_int4_gemm_module(
    m, k, n, gs, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n,
    m_per_segment=1,
):
    """Build an int4-AWQ packed GEMM MLIR module.

    Signature matches `matmul_int4_packed.build_module` positionally so
    callers can swap a single import line.
    """
    from matmul_int4_packed import build_module as build_int4_gemm
    return build_int4_gemm(
        m, k, n, gs,
        tile_m, tile_k_l2, tile_k_l1, tile_n,
        herd_m, herd_n,
        m_per_segment=m_per_segment,
    )
