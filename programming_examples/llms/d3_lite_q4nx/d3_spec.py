# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused prefill device spec of vllm-sr/d3-lite on FastFlowLM's Q4K bundle:
the lite configuration of shared/fused_prefill/d3.py (device_spec)."""

import sys
import types
from pathlib import Path

# llms/shared is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.fused_prefill.d3 import device_spec  # noqa: E402

_spec = device_spec("lite")
CFG = _spec.CFG
ATTN_POINTS = _spec.ATTN_POINTS
max_len = _spec.max_len
gemm_shapes = _spec.gemm_shapes
