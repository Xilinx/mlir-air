# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B on the fused prefill device: ../gemma4_e2b_q4nx's spec() with
the E4B config."""

from ...gemma4_e2b_q4nx.fused_prefill.spec import gw, spec

_E4B = spec(gw.E4B)
CFG, ATTN_POINTS, gemm_shapes, max_len = (
    _E4B.CFG,
    _E4B.ATTN_POINTS,
    _E4B.gemm_shapes,
    _E4B.max_len,
)
