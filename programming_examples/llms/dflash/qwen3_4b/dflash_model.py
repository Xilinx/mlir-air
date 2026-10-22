# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-4B checkpoints, caches and decoder classes for the shared DFlash code."""

from paths import HERE
from bfp_context_prepass import BFPContextPrepass, BF16ContextReference
from common.model import DFlashModel
from common.speculative import EOS_IDS
from draft_weights import DraftWeights
import model_paths
from provenance import runtime_identity
from selected_config import require_selected
from target_runtime import TargetDecoder


def _npu_prefill():
    from npu_prefill import NpuPrefill

    return NpuPrefill


MODEL = DFlashModel(
    directory=HERE,
    target_model=model_paths.TARGET_MODEL,
    tokenizer_model=model_paths.TOKENIZER_MODEL,
    tokenizer_revision=model_paths.TOKENIZER_REVISION,
    eos_ids=EOS_IDS,
    chat_options={"enable_thinking": False},
    cpu_prefill=True,
    load_tokenizer=model_paths.load_tokenizer,
    default_target_cache=model_paths.default_target_cache,
    default_draft_cache=model_paths.default_draft_cache,
    default_draft_snapshot=model_paths.default_draft_snapshot,
    target_decoder=TargetDecoder,
    draft_weights=DraftWeights,
    context_prepass=BFPContextPrepass,
    context_reference=BF16ContextReference,
    require_selected=require_selected,
    npu_prefill=_npu_prefill,
    runtime_identity=runtime_identity,
)
