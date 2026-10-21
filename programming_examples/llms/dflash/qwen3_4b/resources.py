# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare pinned HF inputs before offline compilation/verification/profiling."""

import argparse
import json
from pathlib import Path

from model_paths import (
    DRAFT_MODEL,
    DRAFT_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
    TOKENIZER_REVISION,
)


def prepare(*, reference=False, offline=False):
    from huggingface_hub import snapshot_download

    sources = [
        (TARGET_REPO, TARGET_REVISION, ["model.q4nx"]),
        (DRAFT_MODEL, DRAFT_REVISION, ["config.json", "model.safetensors"]),
        (
            "Qwen/Qwen3-4B",
            TOKENIZER_REVISION,
            [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "merges.txt",
                "vocab.json",
            ],
        ),
    ]
    records = []
    for repo, revision, files in sources:
        patterns = files + (
            ["*.safetensors", "model.safetensors.index.json"]
            if reference and repo == "Qwen/Qwen3-4B"
            else []
        )
        root = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=patterns,
                local_files_only=offline,
            )
        )
        required = list(files)
        if reference and repo == "Qwen/Qwen3-4B":
            index = root / "model.safetensors.index.json"
            required.append(index.name)
            if index.is_file():
                required += sorted(
                    set(json.loads(index.read_text())["weight_map"].values())
                )
        missing = [
            name
            for name in required
            if not (root / name).is_file() or (root / name).stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete checkpoint {repo}@{revision}: {missing}. "
                "Run make download with network access to seed the persistent cache."
            )
        records.append(
            dict(repo=repo, revision=revision, snapshot=str(root), files=required)
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also cache HF BF16 verification weights",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(reference=args.reference, offline=args.offline), indent=2))
