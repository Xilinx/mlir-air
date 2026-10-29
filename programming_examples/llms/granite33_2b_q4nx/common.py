# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed task geometry and shared, production-independent capture utilities."""

from dataclasses import dataclass
from pathlib import Path

from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

from ..shared.deployment import file_hash, save_json


@dataclass
class Config:
    n_layers: int = 40
    emb_dim: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 64
    hidden_dim: int = 8192
    vocab_size: int = 49159
    rope_base: float = 10000000.0
    rms_norm_eps: float = 1e-5
    attention_multiplier: float = 0.015625
    residual_multiplier: float = 0.22
    embedding_multiplier: float = 12.0
    logits_scaling: float = 8.0
    dtype: object = bfloat16


def read_task():
    from ..shared.deployment import read_task

    return read_task("granite33_2b_q4nx")
