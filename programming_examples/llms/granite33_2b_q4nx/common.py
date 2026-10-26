# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os

"""Fixed task geometry and shared, production-independent capture utilities."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "llama32_1b", HERE.parent / "verify"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from air_examples.llms.verify.comparators import topk_token_ids


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
    from air_examples.llms.shared.deployment import read_task

    return read_task("granite33_2b_q4nx")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)
