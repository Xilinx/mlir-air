# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Record the host sources and model revisions a run used."""

import hashlib
from pathlib import Path

from model_paths import DRAFT_REVISION, TARGET_REVISION, TOKENIZER_REVISION

HERE = Path(__file__).resolve().parent
NAMES = (
    "dflash_model.py",
    "target_runtime.py",
    "verify_b8.py",
    "layers_b8.py",
    "lm_head_b8.py",
    "model_b8.py",
    "q4k_layout.py",
    "draft_weights.py",
    "context_prepass.py",
    "npu_prefill.py",
    "model_paths.py",
    "model_spec.py",
    "paths.py",
    "provenance.py",
)
SHARED = (
    "dflash/common/model.py",
    "dflash/common/runtime.py",
    "dflash/common/run.py",
    "dflash/common/speculative.py",
    "dflash/common/benchmark_runner.py",
    "dflash/common/paired_benchmark.py",
    "dflash/common/streaming.py",
    "dflash/common/decoder.py",
    "dflash/common/decoder_8b.py",
    "qwen3_coder_30b_a3b_q4k/qwen3_coder_30b_a3b_q4k_weights.py",
    "qwen3_coder_30b_a3b_q4k/attn_layer.py",
    "qwen3_coder_30b_a3b_q4k/moe_block.py",
    "qwen3_coder_30b_a3b_q4k/lm_head.py",
)
BASELINE = (
    "qwen3_coder_30b_a3b_q4k/qwen3_coder_30b_a3b_q4k_decode.py",
    "shared/moe/wave_decode.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity(engine="dflash"):
    llms = HERE.parents[1]
    names = NAMES + (("baseline.py",) if engine == "b1" else ())
    shared = SHARED + (BASELINE if engine == "b1" else ())
    return dict(
        source_sha256={n: _sha(HERE / n) for n in names},
        shared_source_sha256={n: _sha(llms / n) for n in shared},
        model_revisions=dict(
            target=TARGET_REVISION, tokenizer=TOKENIZER_REVISION, drafter=DRAFT_REVISION
        ),
    )
