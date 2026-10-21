# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Identify the actual host sources and model revisions used by a run."""

import hashlib
from pathlib import Path
from model_paths import TARGET_REVISION, TOKENIZER_REVISION, DRAFT_REVISION


def runtime_identity(engine="dflash"):
    here = Path(__file__).resolve().parent
    names = (
        "runtime.py",
        "target_runtime.py",
        "speculative.py",
        "decode_dynseq.py",
        "draft_weights.py",
        "bfp_context_prepass.py",
        "npu_context_prepass.py",
        "npu_prefill.py",
        "model_paths.py",
        "weight_io.py",
        "provenance.py",
    )
    llms = here.parents[1]
    shared_names = (
        "qwen3_4b_q4nx/qwen3_4b_q4nx_weights.py",
        "qwen3_8b_q4nx/qwen3_8b_q4nx_weights.py",
        "qwen3_4b_q4nx/qwen3_4b_q4nx_prefill.py",
        "qwen3_4b/qwen3_4b_prefill.py",
        "qwen3_4b/qwen3_4b_weights.py",
        "shared/infra/cache.py",
    )
    if engine == "b1":
        names += ("baseline.py",)
        shared_names += ("qwen3_4b_q4nx/qwen3_4b_q4nx_inference.py",)
    fused = llms.parent / "fused_decode"
    return dict(
        source_sha256={
            name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in names
        },
        shared_source_sha256={
            name: hashlib.sha256((llms / name).read_bytes()).hexdigest()
            for name in shared_names
        },
        baseline_source_sha256=(
            {
                name: hashlib.sha256((fused / name).read_bytes()).hexdigest()
                for name in ("decode_insts_gen.py", "decode_staircase.py")
            }
            if engine == "b1"
            else {}
        ),
        model_revisions=dict(
            target=TARGET_REVISION, tokenizer=TOKENIZER_REVISION, drafter=DRAFT_REVISION
        ),
    )
