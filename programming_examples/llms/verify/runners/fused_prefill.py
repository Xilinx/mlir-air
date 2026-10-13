# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The fused prefill (llms/shared/fused_prefill) for the q4nx verify adapters."""

import os
import sys
import types
from pathlib import Path

_PE = Path(__file__).resolve().parents[3]  # programming_examples


def build_dir():
    """$FUSED_PREFILL_DIR (the model's Makefile sets it) if it holds a build."""
    d = os.environ.get("FUSED_PREFILL_DIR")
    return d if d and os.path.isfile(os.path.join(d, "manifest.json")) else None


def load(name, model=None):
    """name's fused prefill on build_dir(), or None: the per-op prefill."""
    d = build_dir()
    if d is None:
        return None
    sys.modules.setdefault(
        "air_examples", types.ModuleType("air_examples")
    ).__path__ = [str(_PE)]
    if name == "gemma4_e2b_q4nx":
        from air_examples.llms.gemma4_e2b_q4nx.fused_prefill.runtime import (
            FusedPrefill,
        )

        pf = FusedPrefill(d)
        pf.load_weights(model=model)
        return pf
    from air_examples.llms.shared.fused_prefill import dense

    return dense.load(name, d, model)
