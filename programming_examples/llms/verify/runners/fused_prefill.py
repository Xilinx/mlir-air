# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The fused prefill (llms/shared/fused_prefill) for the q4nx verify adapters."""

import os
import sys
import types
from pathlib import Path

import numpy as np

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
    if name.startswith("gemma4_"):
        import importlib

        rt = importlib.import_module(f"air_examples.llms.{name}.fused_prefill.runtime")
        pf = rt.FusedPrefill(d)
        pf.load_weights(model=model)
        return pf
    from air_examples.llms.shared.fused_prefill import dense, packing
    from air_examples.llms.shared.fused_prefill.models import MODELS

    if not packing.reads(MODELS[name], model):
        return None
    return dense.load(name, d, model)


def prefill_logits(prefiller, ids, logits_from):
    """[len(ids) - logits_from, vocab] logits of the prompt's rows from
    `logits_from` on. Only the fused prefill can produce them; the per-op
    prefills compute the last row alone."""
    from air_examples.llms.shared.fused_prefill.dense import DensePrefill

    if not isinstance(prefiller, DensePrefill):
        raise NotImplementedError(
            "the prefill path needs the fused prefill (make compile-fused-prefill)"
        )
    prefiller.clear_context()
    ids = [int(t) for t in np.asarray(ids).reshape(-1)]
    return np.asarray(prefiller.prefill(ids, logits_from=logits_from), np.float32)
