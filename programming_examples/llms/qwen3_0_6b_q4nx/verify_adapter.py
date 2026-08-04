# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Qwen3-0.6B example.

The NPU runs FastFlowLM's 4-bit `model.q4nx` weights (dequant'd to bf16); the
reference is the HF **bf16** checkpoint. The shared verify gate therefore compares
NPU-q4nx vs HF-bf16 (top-k token-set inclusion) — the apples-to-bf16 signal for a
4-bit model.

The NPU runner is weight-source-agnostic, so we reuse `NpuRunner` from the bf16
qwen3_0_6b adapter verbatim and only swap the weight loader.

Pointed at via `--runner=qwen3_0_6b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_QWEN3 = _LLMS_DIR / "qwen3_0_6b"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_QWEN3), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Reuse the bf16 runner + config verbatim.
from qwen3_0_6b.verify_adapter import NpuRunner, build_config  # noqa: E402
from qwen3_0_6b_q4nx_weights import load_q4nx_weights  # noqa: E402

MODEL_CHOICES = {
    "base": "Qwen/Qwen3-0.6B",
    "instruct": "Qwen/Qwen3-0.6B",
}
DEFAULT_MODEL = "instruct"

# NPU weight source (model.q4nx bundle). `model_name` selects tokenizer + HF bf16
# reference; the NPU weights always come from here.
Q4NX_MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-0.6B-NPU2")


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """Reference is the HF bf16 checkpoint (NPU q4nx vs HF bf16)."""
    return npu_model_name


def build_runner(
    model_name, config, max_seq, tokenizer, *, npu_attn=True, lite_mode=False
):
    """Load Q4NX weights (dequant->bf16), compile NPU kernels, return NpuRunner."""
    weights = load_q4nx_weights(Q4NX_MODEL_SOURCE, config=config)
    return NpuRunner(
        weights=weights,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=npu_attn,
        lite_mode=lite_mode,
    )
