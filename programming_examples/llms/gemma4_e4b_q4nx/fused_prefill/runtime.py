# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B prefill on the fused prefill device: ../gemma4_e2b_q4nx's
FusedPrefill with the E4B config."""

from ...gemma4_e2b_q4nx.fused_prefill import runtime as _rt


class FusedPrefill(_rt.FusedPrefill):
    def __init__(self, build_dir, max_len=2048):
        super().__init__(build_dir, _rt.gw.E4B, max_len)
