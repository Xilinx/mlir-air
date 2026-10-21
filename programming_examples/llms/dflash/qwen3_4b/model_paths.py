# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Portable model and packed-cache defaults for the DFlash example."""

from __future__ import annotations

import os
from pathlib import Path

TARGET_MODEL = os.environ.get("DFLASH_TARGET_MODEL", "FastFlowLM/Qwen3-4B-NPU2")
TOKENIZER_MODEL = os.environ.get("DFLASH_TOKENIZER_MODEL", "Qwen/Qwen3-4B")
TARGET_REPO = "FastFlowLM/Qwen3-4B-NPU2"
TARGET_REVISION = "07651f11cac70ff5b7a81731643dee6f6e7baee1"
TOKENIZER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DRAFT_MODEL = "z-lab/Qwen3-4B-DFlash-b16"
DRAFT_REVISION = "b74e3a329c4d963783143b1e970d95b002be72bd"


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_CACHE_DIR",
            Path.home() / ".cache/mlir-air/dflash/qwen3_4b",
        )
    ).expanduser()


def default_target_cache() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_TARGET_CACHE",
            cache_root() / "target_requant_w2ch.npz",
        )
    ).expanduser()


def default_draft_cache() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_DRAFT_CACHE",
            cache_root() / "draft_requant_w2ch.npz",
        )
    ).expanduser()


def default_draft_snapshot() -> Path:
    configured = os.environ.get("DFLASH_DRAFT_SNAPSHOT")
    if configured:
        return Path(configured).expanduser()
    from huggingface_hub.constants import HF_HUB_CACHE

    return (
        Path(HF_HUB_CACHE)
        / "models--z-lab--Qwen3-4B-DFlash-b16/snapshots"
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

    options = {"revision": TOKENIZER_REVISION} if model == "Qwen/Qwen3-4B" else {}
    return AutoTokenizer.from_pretrained(model, **options)
