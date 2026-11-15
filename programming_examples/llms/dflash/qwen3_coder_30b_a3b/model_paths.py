# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Where the drafter's checkpoint and its packed weights are found. The
shared B8 decoder also imports its target's names; only its drafter role
runs here."""

from __future__ import annotations

import os
from pathlib import Path

DRAFT_MODEL = "z-lab/Qwen3-Coder-30B-A3B-DFlash"
DRAFT_REVISION = "98ca0e3e2e6a372f2789d3a5e146566194084317"
TARGET_MODEL = TARGET_REPO = "FastFlowLM/Qwen3-8B-NPU2"
TARGET_REVISION = "941c148c228496682e95c413cb90025f73200af2"


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_CACHE_DIR",
            Path.home() / ".cache/mlir-air/dflash/qwen3_coder_30b_a3b",
        )
    ).expanduser()


def default_draft_cache() -> Path:
    return Path(
        os.environ.get("DFLASH_DRAFT_CACHE", cache_root() / "draft_q4nx")
    ).expanduser()


default_target_cache = default_draft_cache


def default_draft_snapshot() -> Path:
    configured = os.environ.get("DFLASH_DRAFT_SNAPSHOT")
    if configured:
        return Path(configured).expanduser()
    from huggingface_hub.constants import HF_HUB_CACHE

    return (
        Path(HF_HUB_CACHE)
        / ("models--" + DRAFT_MODEL.replace("/", "--"))
        / "snapshots"
        / DRAFT_REVISION
    )


def draft_snapshot(path=None) -> Path:
    """path, else the pinned checkpoint, downloaded if it is not cached."""
    if path is not None:
        return Path(path)
    snapshot = default_draft_snapshot()
    if not snapshot.is_dir():
        from huggingface_hub import snapshot_download

        snapshot_download(DRAFT_MODEL, revision=DRAFT_REVISION)
    return snapshot


def resolve_target_model(model=TARGET_MODEL, *, offline=False):
    if model != TARGET_REPO:
        return model
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        TARGET_REPO, "model.q4nx", revision=TARGET_REVISION, local_files_only=offline
    )
