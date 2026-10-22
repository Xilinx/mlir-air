# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host API for prompt preparation and the NPU DFlash target/drafter pair."""

from __future__ import annotations

import time

import numpy as np
from ml_dtypes import bfloat16

from bfp_context_prepass import BFPContextPrepass, BF16ContextReference
from draft_weights import DraftWeights
from model_paths import (
    TARGET_MODEL,
    load_tokenizer,
    TOKENIZER_MODEL,
    default_draft_cache,
    default_draft_snapshot,
    default_target_cache,
)
from selected_config import require_selected
from paths import HERE, COMMON
from common.speculative import DFlash
from target_runtime import TargetDecoder
import model_spec


def load_pair(
    target_artifacts,
    draft_artifacts,
    *,
    prepass_artifacts=None,
    target_model=TARGET_MODEL,
    target_cache=None,
    draft_cache=None,
    draft_snapshot=None,
):
    target_cache = target_cache or default_target_cache()
    draft_cache = draft_cache or default_draft_cache()
    draft_snapshot = draft_snapshot or default_draft_snapshot()
    target = TargetDecoder(
        target_artifacts,
        packed_cache=target_cache,
        model_source=target_model,
        stream_weights=True,
    )
    require_selected(target.config, "target")
    weights = DraftWeights(
        target_model=target.model,
        snapshot=draft_snapshot,
        shared_embedding=target.embed,
    )
    drafter = TargetDecoder(
        draft_artifacts,
        packed_cache=draft_cache,
        model_weights=weights,
        stream_weights=True,
    )
    require_selected(drafter.config, "drafter")
    from pathlib import Path

    prepass_artifacts = (
        prepass_artifacts or Path(target_artifacts).resolve().parent / "prepass"
    )
    prepass = BFPContextPrepass(
        prepass_artifacts, BF16ContextReference(weights), capacity=target.capacity
    )
    return DFlash(target, drafter, prepass, weights.tap_slots, weights.mask_token_id)


def load_pipeline(
    target_artifacts,
    draft_artifacts,
    *,
    prefill_context=2048,
    prefill_artifacts=None,
    prepass_artifacts=None,
    target_model=TARGET_MODEL,
    target_cache=None,
    draft_cache=None,
    draft_snapshot=None,
):
    """Prepare prefill before resident decoders to bound setup memory.

    Prefill maps its weight cache before decoder BOs are allocated and streams
    one layer per call. The pipeline remains loaded for subsequent prompts.
    """
    from npu_prefill import NpuPrefill

    metadata = DraftWeights(snapshot=draft_snapshot or default_draft_snapshot())
    prefill = NpuPrefill(
        seq_len=prefill_context,
        cache_dir=prefill_artifacts,
        tap_slots=metadata.tap_slots,
    )
    prefill.load_weights(model=target_model)
    loop = load_pair(
        target_artifacts,
        draft_artifacts,
        prepass_artifacts=prepass_artifacts,
        target_model=target_model,
        target_cache=target_cache,
        draft_cache=draft_cache,
        draft_snapshot=draft_snapshot,
    )
    prefill.share_embedding(loop.target.embed)
    return loop, prefill


def tokenize_prompt(prompt: str, tokenizer_model=TOKENIZER_MODEL, *, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **model_spec.CHAT_OPTIONS,
    )
    return tokenizer, tokenizer.encode(text, add_special_tokens=False)
