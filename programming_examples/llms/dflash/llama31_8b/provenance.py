# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Record the host sources and model revisions a run used."""

import hashlib
from pathlib import Path
from model_paths import TARGET_REVISION, TOKENIZER_REVISION, DRAFT_REVISION


def runtime_identity(engine="dflash"):
    here = Path(__file__).resolve().parent
    names = (
        "runtime.py",
        "target_runtime.py",
        "draft_weights.py",
        "bfp_context_prepass.py",
        "npu_prefill.py",
        "prefill_weights.py",
        "model_paths.py",
        "paths.py",
        "provenance.py",
        "model_spec.py",
        "target_weights.py",
    )
    llms = here.parents[1]
    shared_names = (
        "dflash/common/speculative.py",
        "dflash/common/benchmark_runner.py",
        "dflash/common/paired_benchmark.py",
        "dflash/common/streaming.py",
        "dflash/common/weight_io.py",
        "dflash/common/decoder.py",
        "dflash/common/decoder_8b.py",
        "dflash/common/b8_config.py",
        "dflash/common/q4nx_blocks.py",
        "dflash/common/pipeline_8b.py",
        "dflash/common/draft_weights_8b.py",
        "dflash/common/prefill_weights_8b.py",
        "dflash/common/prefill_cache.py",
        "dflash/common/run_8b.py",
        "dflash/common/prepass.py",
        "dflash/common/prepass_runtime.py",
        "dflash/common/context_reference.py",
        "dflash/common/decode_dynseq.py",
        "dflash/common/buffers.py",
        "dflash/common/prefill_reference.py",
        "llama31_8b_q4nx/llama31_8b_q4nx_weights.py",
        "llama31_8b_q4nx/llama31_8b_q4nx_prefill.py",
        "llama32_3b/llama32_3b_prefill.py",
        "llama32_3b/llama32_3b_weights.py",
        "llama32_1b_q4nx/llama32_1b_q4nx_weights.py",
        "shared/infra/cache.py",
    )
    if engine == "b1":
        names += ("baseline.py",)
        shared_names += (
            "llama31_8b_q4nx/q4nx_decode_8b.py",
            "llama32_1b_q4nx/q4nx_requant.py",
            "shared/infra/host_weights.py",
        )
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
                for name in (
                    "decode_insts_gen.py",
                    "decode_staircase.py",
                    "decode_elf.py",
                    "kv_bos.py",
                )
            }
            if engine == "b1"
            else {}
        ),
        model_revisions=dict(
            target=TARGET_REVISION, tokenizer=TOKENIZER_REVISION, drafter=DRAFT_REVISION
        ),
    )
