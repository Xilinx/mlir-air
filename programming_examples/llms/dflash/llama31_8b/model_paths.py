# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Portable model and packed-cache defaults for the DFlash example."""

from __future__ import annotations

import os
from pathlib import Path

TARGET_MODEL = os.environ.get("DFLASH_TARGET_MODEL", "FastFlowLM/Llama-3.1-8B-NPU2")
TOKENIZER_MODEL = os.environ.get(
    "DFLASH_TOKENIZER_MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct"
)
TARGET_REPO = "FastFlowLM/Llama-3.1-8B-NPU2"
TARGET_REVISION = "527006483dca6c9d75474ff92b8a6bdd8f257914"
TOKENIZER_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
DRAFT_MODEL = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
DRAFT_REVISION = "d3af30def9601abdd10810aba220d692f0e803f0"


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_CACHE_DIR",
            Path.home() / ".cache/mlir-air/dflash/llama31_8b",
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
        / "models--z-lab--LLaMA3.1-8B-Instruct-DFlash-UltraChat/snapshots"
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

    options = (
        {"revision": TOKENIZER_REVISION}
        if model == "NousResearch/Meta-Llama-3.1-8B-Instruct"
        else {}
    )
    return AutoTokenizer.from_pretrained(model, **options)
