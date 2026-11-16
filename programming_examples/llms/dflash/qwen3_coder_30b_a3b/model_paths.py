# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pinned checkpoints and packed-cache defaults for the DFlash example."""

from __future__ import annotations

import os
from pathlib import Path

# The target's Q4_K bundle is quantized from this checkpoint by the target
# example (llms/qwen3_coder_30b_a3b_q4k), which also carries the tokenizer.
TARGET_MODEL = TARGET_REPO = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
TARGET_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
TOKENIZER_MODEL = TARGET_REPO
TOKENIZER_REVISION = TARGET_REVISION
DRAFT_MODEL = "z-lab/Qwen3-Coder-30B-A3B-DFlash"
DRAFT_REVISION = "98ca0e3e2e6a372f2789d3a5e146566194084317"


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


def default_target_cache() -> Path:
    """The target's weights in the verify pass's layout."""
    return Path(
        os.environ.get("DFLASH_TARGET_CACHE", cache_root() / "target_tiled")
    ).expanduser()


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
    """The target's Q4_K bundle: `model` itself unless it names the pinned
    checkpoint."""
    from paths import TARGET
    import sys

    sys.path.insert(0, str(TARGET))
    import qwen3_coder_30b_a3b_q4k_weights as W

    return W.bundle_path(None if model == TARGET_REPO else model)


def load_tokenizer(model=TOKENIZER_MODEL):
    from transformers import AutoTokenizer

    options = {"revision": TOKENIZER_REVISION} if model == TARGET_REPO else {}
    return AutoTokenizer.from_pretrained(model, **options)
