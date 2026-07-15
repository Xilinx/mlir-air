# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared Record dataclasses returned by all Runner implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrefillRecord:
    layer_intermediates: list[dict[str, np.ndarray]]  # len == n_layers
    # final_hidden after the model's final RMSNorm — the value that feeds
    # into the LM-head matmul. HF transformers exposes this as
    # output_hidden_states[n_layers] (which is post-final-norm by HF v5.3
    # convention; see hf_runner for the empirical confirmation). NPU
    # produces it natively in non-lite mode (the same array used to
    # compute final_logits). Diagnosis pairs this NPU vs HF cell as the
    # "layer 15" probe so the last layer is not silently skipped.
    final_hidden_normed: np.ndarray
    logits_at_pred: np.ndarray
    top1_token: int


@dataclass
class DecodeStepRecord:
    lm_head_logits: np.ndarray
    top1_token: int
