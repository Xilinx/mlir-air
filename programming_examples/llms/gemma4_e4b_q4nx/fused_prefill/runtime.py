# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B prefill on the fused prefill device: ../gemma4_e2b_q4nx's
runtime at GEMMA4_VARIANT=e4b."""

from .. import gemma4_e4b_variant  # noqa: F401
from ...gemma4_e2b_q4nx.fused_prefill.runtime import FusedPrefill  # noqa: F401
