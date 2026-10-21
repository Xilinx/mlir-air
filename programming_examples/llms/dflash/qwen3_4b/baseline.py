# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Adapt the single-token (B1) Qwen3-4B decoder to the saved-prefill benchmark."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

from build_identity import build_identity, check_existing
from model_paths import default_target_cache, resolve_target_model


def load_baseline(artifacts):
    artifacts = Path(artifacts).resolve()
    check_existing(artifacts, build_identity("baseline"))
    os.environ["Q4NX_QWEN3_4B_DECODE_DIR"] = str(artifacts)
    # B1 decodes from the same requantized weight cache as B8.
    os.environ["Q4NX_QWEN3_4B_DECODE_NPZ"] = str(default_target_cache())
    os.environ["DECODE_ELF"] = "0"
    os.environ["W_DUAL_CHAN"] = "1"
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1] / "qwen3_4b_q4nx"))
    from qwen3_4b_q4nx_inference import FusedDecoder

    decoder = FusedDecoder(model=resolve_target_model(), max_L=4096)

    class Baseline:
        batch = 1
        capacity = decoder.ATTN_MAXL
        seed_kv = decoder.seed_kv

        def verify(self, tokens, position):
            return SimpleNamespace(
                logits=decoder.dispatch(int(tokens[0]), position)[None, :]
            )

    return Baseline()
