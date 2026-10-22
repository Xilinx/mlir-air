# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host API for prompt preparation and the NPU DFlash target/drafter pair."""

from __future__ import annotations

import time

import numpy as np
from ml_dtypes import bfloat16

from pathlib import Path

from common.model import DFlashModel
from common.prefill_reference import forward_prompt_with_taps
from common.speculative import DFlash


def load_pair(
    model: DFlashModel,
    target_artifacts,
    draft_artifacts,
    *,
    prepass_artifacts=None,
    target_model=None,
    target_cache=None,
    draft_cache=None,
    draft_snapshot=None,
):
    target_model = target_model or model.target_model
    target_cache = target_cache or model.default_target_cache()
    draft_cache = draft_cache or model.default_draft_cache()
    draft_snapshot = draft_snapshot or model.default_draft_snapshot()
    target = model.target_decoder(
        target_artifacts,
        packed_cache=target_cache,
        model_source=target_model,
        stream_weights=True,
    )
    model.require_selected(target.config, "target")
    weights = model.draft_weights(
        target_model=target.model,
        snapshot=draft_snapshot,
        shared_embedding=target.embed,
    )
    drafter = model.target_decoder(
        draft_artifacts,
        packed_cache=draft_cache,
        model_weights=weights,
        stream_weights=True,
    )
    model.require_selected(drafter.config, "drafter")
    prepass_artifacts = (
        prepass_artifacts or Path(target_artifacts).resolve().parent / "prepass"
    )
    prepass = model.context_prepass(
        prepass_artifacts,
        model.context_reference(weights),
        capacity=target.capacity,
    )
    return DFlash(target, drafter, prepass, weights.tap_slots, weights.mask_token_id)


def load_pipeline(
    model: DFlashModel,
    target_artifacts,
    draft_artifacts,
    *,
    prefill_context=2048,
    prefill_artifacts=None,
    prepass_artifacts=None,
    target_model=None,
    target_cache=None,
    draft_cache=None,
    draft_snapshot=None,
):
    """Prepare prefill before resident decoders to bound setup memory.

    Prefill reuses its uploaded MLP weights before the decoder weight BOs are
    allocated. All three model programs remain resident for subsequent prompts.
    """
    target_model = target_model or model.target_model
    metadata = model.draft_weights(
        snapshot=draft_snapshot or model.default_draft_snapshot()
    )
    prefill = model.npu_prefill()(
        seq_len=prefill_context,
        cache_dir=prefill_artifacts,
        tap_slots=metadata.tap_slots,
    )
    prefill.load_weights(model=target_model)
    loop = load_pair(
        model,
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


def encode_prompt(model: DFlashModel, tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **model.chat_options,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_prompt(
    model: DFlashModel, prompt: str, tokenizer_model=None, *, tokenizer=None
):
    if tokenizer is None:
        tokenizer = model.load_tokenizer(tokenizer_model or model.tokenizer_model)
    return tokenizer, encode_prompt(model, tokenizer, prompt)


class PrefillWeights:
    """Keep the existing CPU reference's decoded weights resident for chat.

    The weight reader normally dequantizes each layer on every prompt. This
    adapter eagerly stores the exact same arrays once, before the first input.
    It changes preparation cost and memory use, not prefill arithmetic.
    """

    def __init__(self, target):
        model = target.model
        self._embed = model.embed_norm_lmhead()
        self._layers = [model.layer_weights(i) for i in range(target.layers)]
        self._rms = [model.layer_rms(i) for i in range(target.layers)]
        self._qk = [model.layer_qk_norm(i) for i in range(target.layers)]
        arrays = list(self._embed)
        arrays += [w for layer in self._layers for w in layer.values()]
        arrays += [w for pair in self._rms + self._qk for w in pair]
        self.nbytes = sum(w.nbytes for w in {id(w): w for w in arrays}.values())

    def embed_norm_lmhead(self):
        return self._embed

    def layer_weights(self, layer):
        return self._layers[layer]

    def layer_rms(self, layer):
        return self._rms[layer]

    def layer_qk_norm(self, layer):
        return self._qk[layer]


def prepare_prompt(
    target,
    input_ids: list[int],
    tap_slots: list[int],
    *,
    prefill_model=None,
):
    started = time.perf_counter()
    model = target.model if prefill_model is None else prefill_model
    k, v, logits, boundaries = forward_prompt_with_taps(model, input_ids, tap_slots)
    taps = np.stack([boundaries[index] for index in tap_slots], axis=1).reshape(
        len(input_ids), -1
    )
    result = {
        "input_ids": list(input_ids),
        "k": k.astype(bfloat16),
        "v": v.astype(bfloat16),
        "taps": taps.astype(bfloat16),
        "first": int(logits[-1].argmax()),
    }
    return result, time.perf_counter() - started
