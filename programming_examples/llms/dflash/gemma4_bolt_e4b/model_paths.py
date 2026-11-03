# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pinned checkpoints and packed-cache defaults for the DFlash example."""

from __future__ import annotations

import os
from pathlib import Path

TARGET_REPO = "FastFlowLM/Nemotron-3.5-Bolt-E4B-NPU2"
TARGET_REVISION = "f9ee21728399b0d570dd40273abe01f7662cbab8"
TARGET_MODEL = os.environ.get("DFLASH_TARGET_MODEL", TARGET_REPO)
# The target package carries the tokenizer and chat template.
TOKENIZER_MODEL = os.environ.get("DFLASH_TOKENIZER_MODEL", TARGET_REPO)
TOKENIZER_REVISION = TARGET_REVISION
DRAFT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash"
DRAFT_REVISION = "0949a2eb7152ec6fda1a91ff9302aecbcd464fbf"
DRAFT_FILE = "Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf"


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_CACHE_DIR", Path.home() / ".cache/mlir-air/dflash/gemma4_bolt_e4b"
        )
    ).expanduser()


def default_draft_cache() -> Path:
    return Path(
        os.environ.get("DFLASH_DRAFT_CACHE", cache_root() / "draft_b8")
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


def resolve_target_model(model=TARGET_MODEL, *, offline=False):
    """The target's model.q4nx: `model` itself unless it names the pinned repo."""
    if model != TARGET_REPO:
        return model
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        TARGET_REPO, "model.q4nx", revision=TARGET_REVISION, local_files_only=offline
    )


def load_tokenizer(model=TOKENIZER_MODEL):
    from transformers import AutoTokenizer

    options = {"revision": TOKENIZER_REVISION} if model == TARGET_REPO else {}
    return AutoTokenizer.from_pretrained(model, **options)
