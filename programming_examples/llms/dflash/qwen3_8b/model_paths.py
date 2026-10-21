# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Portable model and packed-cache defaults for the DFlash example."""

from __future__ import annotations

import os
from pathlib import Path

TARGET_MODEL = os.environ.get("DFLASH_TARGET_MODEL", "FastFlowLM/Qwen3-8B-NPU2")
TOKENIZER_MODEL = os.environ.get("DFLASH_TOKENIZER_MODEL", "Qwen/Qwen3-8B")
TARGET_REPO = "FastFlowLM/Qwen3-8B-NPU2"
TARGET_REVISION = "941c148c228496682e95c413cb90025f73200af2"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
DRAFT_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_CACHE_DIR",
            Path.home() / ".cache/mlir-air/dflash/qwen3_8b",
        )
    ).expanduser()


def default_target_cache() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_TARGET_CACHE",
            cache_root() / "target_q4nx",
        )
    ).expanduser()


def default_draft_cache() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_DRAFT_CACHE",
            cache_root() / "draft_q4nx",
        )
    ).expanduser()


def default_prefill_cache() -> Path:
    return Path(
        os.environ.get("DFLASH_PREFILL_WEIGHTS", cache_root() / "prefill_bf16")
    ).expanduser()


def default_draft_snapshot() -> Path:
    configured = os.environ.get("DFLASH_DRAFT_SNAPSHOT")
    if configured:
        return Path(configured).expanduser()
    from huggingface_hub.constants import HF_HUB_CACHE

    return (
        Path(HF_HUB_CACHE)
        / "models--z-lab--Qwen3-8B-DFlash-b16/snapshots"
        / DRAFT_REVISION
    )


def resolve_target_model(model=TARGET_MODEL, *, offline=False):
    if model != TARGET_REPO:
        return model
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        TARGET_REPO, "model.q4nx", revision=TARGET_REVISION, local_files_only=offline
    )


def load_tokenizer(model=TOKENIZER_MODEL):
    from transformers import AutoTokenizer

    options = {"revision": TOKENIZER_REVISION} if model == "Qwen/Qwen3-8B" else {}
    return AutoTokenizer.from_pretrained(model, **options)
