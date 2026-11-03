# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The single-token (B1) Gemma4-E4B decoder (llms/gemma4_e4b_q4nx) with
this model's weights, adapted to the saved-prefill benchmark."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

from paths import HERE
from model_paths import cache_root, resolve_target_model
from target_runtime import unpack_prefill_kv


def load_baseline(artifacts):
    artifacts = Path(artifacts).resolve()
    if not (artifacts / "manifest.json").exists():
        raise ValueError(f"No baseline build at {artifacts}; run make compile-baseline")
    # Read when the driver is imported: its program, and a weight cache of
    # this model's own (the cache is keyed on geometry, not on the weights).
    os.environ["Q4NX_GEMMA4_DECODE_DIR"] = str(artifacts)
    os.environ["Q4NX_GEMMA4_WCACHE_DIR"] = str(cache_root() / "baseline_wcache")
    os.environ.pop("DECODE_ELF", None)
    sys.path.insert(0, str(HERE.parents[1] / "gemma4_e4b_q4nx"))
    from gemma4_e4b_q4nx_inference import FusedDecoder

    decoder = FusedDecoder(model=resolve_target_model(), verbose=False)

    class Baseline:
        batch = 1
        capacity = decoder.maxL

        def seed_kv(self, k, v, position):
            ks, vs = unpack_prefill_kv(k, v)
            decoder.seed_kv(ks, vs, position)

        def verify(self, tokens, position):
            return SimpleNamespace(
                logits=decoder.dispatch(int(tokens[0]), position)[None, :]
            )

    return Baseline()
