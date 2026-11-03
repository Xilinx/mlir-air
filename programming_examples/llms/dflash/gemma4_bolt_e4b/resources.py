# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare pinned HF inputs before offline compilation/verification/profiling."""

from pathlib import Path

import paths  # noqa: F401
from common import resources
from model_paths import (
    DRAFT_FILE,
    DRAFT_MODEL,
    DRAFT_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
)

TARGET_FILES = (
    "model.q4nx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def prepare(*, reference=False, offline=False):
    """The target package (weights, tokenizer, chat template) and the
    drafter's GGUF. The model check's reference runs on the target package's
    own weights, so `reference` adds nothing."""
    from huggingface_hub import snapshot_download

    records = []
    for repo, revision, files in (
        (TARGET_REPO, TARGET_REVISION, list(TARGET_FILES)),
        (DRAFT_MODEL, DRAFT_REVISION, [DRAFT_FILE]),
    ):
        root = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=files,
                local_files_only=offline,
            )
        )
        missing = [
            n
            for n in files
            if not (root / n).is_file() or not (root / n).stat().st_size
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete checkpoint {repo}@{revision}: {missing}. "
                "Run make download with network access to seed the persistent cache."
            )
        records.append(
            dict(repo=repo, revision=revision, snapshot=str(root), files=files)
        )
    return records


if __name__ == "__main__":
    resources.main(prepare)
