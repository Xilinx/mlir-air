# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B on the fused prefill device: ../gemma4_e2b_q4nx's spec at
GEMMA4_VARIANT=e4b."""

from .. import gemma4_e4b_variant  # noqa: F401
from ...gemma4_e2b_q4nx.fused_prefill.spec import *  # noqa: F401,F403
