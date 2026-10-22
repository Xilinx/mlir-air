# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare pinned HF inputs before offline compilation/verification/profiling."""

import argparse
import json
from pathlib import Path


def prepare(
    target, draft, reference_model, tokenizer_files, *, reference=False, offline=False
):
    """Resolve the pinned snapshots, each given as a (repo, revision) pair.

    `reference_model` supplies the tokenizer files; with `reference`, also its
    BF16 weights for verification.
    """
    from huggingface_hub import snapshot_download

    sources = [
        (*target, ["model.q4nx"]),
        (*draft, ["config.json", "model.safetensors"]),
        (*reference_model, ["config.json", *tokenizer_files]),
    ]
    records = []
    for repo, revision, files in sources:
        patterns = files + (
            ["*.safetensors", "model.safetensors.index.json"]
            if reference and repo == reference_model[0]
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
        if reference and repo == reference_model[0]:
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


def main(prepare):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also cache HF BF16 verification weights",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(reference=args.reference, offline=args.offline), indent=2))
