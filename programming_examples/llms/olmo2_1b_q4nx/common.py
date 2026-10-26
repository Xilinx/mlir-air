# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Fixed task geometry and shared, production-independent capture utilities."""

from dataclasses import dataclass
from pathlib import Path
import os
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
    n_layers: int = 16
    emb_dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 16
    head_dim: int = 128
    hidden_dim: int = 8192
    logical_hidden_dim: int = 8192
    vocab_size: int = 100352
    rope_base: float = 500000.0
    rms_norm_eps: float = 1e-6
    attention_multiplier: float = 128**-0.5
    residual_multiplier: float = 1.0
    dtype: object = bfloat16
    gemm_bfp16: bool = True
    attn_lkp: int = 64


RUN = Path(os.environ.get("AIR_MODEL_BUILD_DIR", str(HERE / "build")))
REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"


def read_task():
    from air_examples.llms.shared.deployment import read_task

    return read_task("olmo2_1b_q4nx")


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
