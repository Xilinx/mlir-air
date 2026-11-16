# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The single-token (B1) decode of llms/qwen3_coder_30b_a3b_q4k with this
model's weights, adapted to the saved-prefill benchmark."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

from paths import TARGET, target_imports
from model_paths import resolve_target_model


def load_baseline(artifacts):
    artifacts = Path(artifacts).resolve()
    manifest = artifacts / "manifest.json"
    if not manifest.exists():
        raise ValueError(f"No baseline build at {artifacts}; run make compile-baseline")
    made = json.loads(manifest.read_text())
    # the builds are restored from the cache compile-baseline filled
    os.environ["AIR_COMPILE_CACHE"] = str(artifacts / "compile_cache")
    with target_imports():
        sys.path.insert(0, str(TARGET))
        import qwen3_coder_30b_a3b_q4k_decode as DEC
        import lm_head  # noqa: F401  (the decode imports it later)
    w = DEC.W.Qwen3CoderQ4k(resolve_target_model())
    work = artifacts / "work"
    work.mkdir(exist_ok=True)
    # The decode builds into the working directory; the caller's paths are
    # relative to its own.
    caller = os.getcwd()
    os.chdir(work)
    try:
        decoder = DEC.load(w, made["capacity"], reads=made["reads"])
    finally:
        os.chdir(caller)

    class Baseline:
        batch = 1
        capacity = made["capacity"]

        def seed_kv(self, k, v, position):
            decoder.seed(
                [(k[l][:position], v[l][:position]) for l in range(len(k))], position
            )

        def verify(self, tokens, position):
            x = np.asarray(w.embed([int(tokens[0])])[0]).astype(bfloat16)
            return SimpleNamespace(logits=decoder.step(x, position + 1)[None, :])

    return Baseline()
