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
    "weights_b8.py",
    "b8_config.py",
    "draft_weights.py",
    "draft_cpu.py",
    "context_prepass.py",
    "npu_prefill.py",
    "target.py",
    "model_paths.py",
    "model_spec.py",
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
    "dflash/qwen3_4b/target_runtime.py",
    "gemma4_e2b_q4nx/gemma4_e2b_q4nx_weights.py",
    "gemma4_e2b_q4nx/fused_prefill/runtime.py",
)
BASELINE = ("gemma4_e2b_q4nx/gemma4_e2b_q4nx_inference.py",)


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
