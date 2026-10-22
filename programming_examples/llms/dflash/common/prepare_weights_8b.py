# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare cached decode and prefill weights from the pinned model pair."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from types import ModuleType

from common.prepare_target_8b import sha


def main(model_dir: Path, model_paths: ModuleType):
    """Prepare and check the caches of the model in `model_dir`.

    `model_paths` is the model's module of checkpoint and cache locations.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    model = Path(model_paths.resolve_target_model(offline=args.offline)).resolve()
    target = model_paths.default_target_cache()
    draft = model_paths.default_draft_cache()
    snapshot = model_paths.default_draft_snapshot()
    prefill = model_paths.default_prefill_cache()
    steps = (
        (target, "prepare_target.py", ["--model", str(model), "--output", str(target)]),
        (
            draft,
            "prepare_draft.py",
            [
                "--snapshot",
                str(snapshot),
                "--target-cache",
                str(target),
                "--output",
                str(draft),
            ],
        ),
        (
            prefill,
            "prefill_weights.py",
            ["--model", str(model), "--cache", str(prefill)],
        ),
    )
    model_sha = sha(model)
    draft_sha = sha(snapshot / "model.safetensors")
    records = []
    for directory, script, options in steps:
        if not directory.exists():
            subprocess.run(
                [sys.executable, str(model_dir / script), *options],
                check=True,
            )
        receipt = json.loads((directory / "manifest.json").read_text())
        if receipt["status"] != "prepared":
            raise ValueError(
                f"Incomplete weight cache: {directory}; use a fresh cache directory"
            )
        expected = draft_sha if directory == draft else model_sha
        actual = receipt.get("source_model_sha256", receipt.get("model_sha256"))
        if actual != expected:
            raise ValueError(
                f"Weight cache belongs to a different checkpoint: {directory}"
            )
        if directory == draft and receipt["target_cache_manifest_sha256"] != sha(
            target / "manifest.json"
        ):
            raise ValueError("Drafter vocabulary head belongs to another target cache")
        files = receipt.get("files", {})
        if "tensors" in receipt:
            files = {
                name: value["sha256"] for name, value in receipt["tensors"].items()
            }
        for name, digest in files.items():
            if sha(directory / name) != digest:
                raise ValueError(f"Changed packed weight file: {directory / name}")
        records.append(
            dict(
                directory=str(directory.resolve()),
                manifest_sha256=sha(directory / "manifest.json"),
            )
        )
        print(f"Verified weight cache: {directory}", flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(dict(status="prepared", caches=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
