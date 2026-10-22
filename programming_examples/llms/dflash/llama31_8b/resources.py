# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare pinned HF inputs before offline compilation/verification/profiling."""

from paths import HERE
from model_paths import (
    DRAFT_MODEL,
    DRAFT_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
    TOKENIZER_REVISION,
)
from common import resources

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


def prepare(*, reference=False, offline=False):
    return resources.prepare(
        (TARGET_REPO, TARGET_REVISION),
        (DRAFT_MODEL, DRAFT_REVISION),
        ("NousResearch/Meta-Llama-3.1-8B-Instruct", TOKENIZER_REVISION),
        TOKENIZER_FILES,
        reference=reference,
        offline=offline,
    )


if __name__ == "__main__":
    resources.main(prepare)
