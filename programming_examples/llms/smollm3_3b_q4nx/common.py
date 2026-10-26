# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Fixed task geometry and shared, production-independent capture utilities."""

from dataclasses import dataclass
from pathlib import Path
import os

from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

from ..shared.deployment import file_hash, save_json


@dataclass
class Config:
    n_layers: int = 36
    emb_dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    hidden_dim: int = 11264
    logical_hidden_dim: int = 11008
    vocab_size: int = 128256
    rope_base: float = 5000000.0
    rms_norm_eps: float = 1e-6
    attention_multiplier: float = 128**-0.5
    residual_multiplier: float = 1.0
    dtype: object = bfloat16
    gemm_bfp16: bool = True
    attn_lkp: int = 64
    attn_omit_while_true_loop: bool = False


RUN = Path(os.environ.get("AIR_MODEL_BUILD_DIR", str(HERE / "build")))
REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"


def read_task():
    from ..shared.deployment import read_task

    return read_task("smollm3_3b_q4nx")
