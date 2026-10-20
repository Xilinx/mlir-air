# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Verify adapter for the Q4NX Gemma4-E4B example: ../gemma4_e2b_q4nx's
hf_config and NpuRunner with the E4B config.

Pointed at via `--runner=gemma4_e4b_q4nx.verify_adapter`.
"""

import importlib.util
import sys
from pathlib import Path

_E2B = Path(__file__).resolve().parents[1] / "gemma4_e2b_q4nx" / "verify_adapter.py"
_spec = importlib.util.spec_from_file_location("gemma4_e2b_verify_adapter", _E2B)
_a = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _a
_spec.loader.exec_module(_a)

CFG = _a.gw.E4B
MODEL_CHOICES = {"instruct": CFG.hf_repo}
DEFAULT_MODEL = _a.DEFAULT_MODEL
PROMPT_BOS = _a.PROMPT_BOS
hf_reference = _a.hf_reference


def build_config():
    return _a.hf_config(CFG)


def resolve_model(model_choice_or_id):
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def build_runner(
    model_name, config, max_seq, tokenizer, *, npu_attn=True, lite_mode=False
):
    return _a.NpuRunner(CFG, tokenizer=tokenizer, lite_mode=lite_mode)
