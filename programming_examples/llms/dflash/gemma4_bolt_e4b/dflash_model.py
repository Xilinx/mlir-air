# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Nemotron-3.5-Bolt-E4B checkpoints, caches and decoders for the shared
DFlash code."""

from paths import HERE
from common.model import DFlashModel
from context_prepass import ContextPrepass
from draft_weights import DraftWeights
import model_paths
import model_spec
from provenance import runtime_identity
from target_runtime import target_decoder


def require_selected(config, role):
    """Reject an artifact that is not the shipped mapping for its role."""
    expected = dict(
        role=role,
        batch=model_spec.BLOCK_SIZE,
        capacity=model_spec.CAPACITY,
        layers=(
            model_spec.TARGET_LAYERS if role == "target" else model_spec.DRAFT_LAYERS
        ),
        causal=role == "target",
    )
    observed = {k: getattr(config, k, None) for k in expected}
    if observed != expected:
        raise ValueError(f"Artifact is not the selected {role}: {observed}")


def _npu_prefill():
    from npu_prefill import NpuPrefill

    return NpuPrefill


MODEL = DFlashModel(
    directory=HERE,
    target_model=model_paths.TARGET_MODEL,
    tokenizer_model=model_paths.TOKENIZER_MODEL,
    tokenizer_revision=model_paths.TOKENIZER_REVISION,
    eos_ids=model_spec.EOS_IDS,
    chat_options=model_spec.CHAT_OPTIONS,
    cpu_prefill=model_spec.CPU_PREFILL,
    load_tokenizer=model_paths.load_tokenizer,
    # The target packs its weights from model.q4nx at load.
    default_target_cache=lambda: None,
    default_draft_cache=model_paths.default_draft_cache,
    default_draft_snapshot=model_paths.default_draft_snapshot,
    target_decoder=target_decoder,
    draft_weights=DraftWeights,
    context_prepass=ContextPrepass,
    context_reference=lambda weights: weights,
    require_selected=require_selected,
    npu_prefill=_npu_prefill,
    runtime_identity=runtime_identity,
)
