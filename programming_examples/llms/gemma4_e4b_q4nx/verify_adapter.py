# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Verify adapter for the Q4NX Gemma4-E4B example: ../gemma4_e2b_q4nx's at
GEMMA4_VARIANT=e4b. Pointed at via `--runner=gemma4_e4b_q4nx.verify_adapter`."""

from .gemma4_e4b_variant import delegate

delegate(__name__, "verify_adapter")
