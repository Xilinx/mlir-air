# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What a model example provides to the shared DFlash host code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DFlashModel:
    """A model's sources, weight caches and decoder classes.

    Each model directory builds one in its ``dflash_model.py``, and its entry
    scripts pass it to the shared runtime, generation and verification code.
    """

    # The model's example directory, which holds its default build outputs.
    directory: Path
    target_model: str
    tokenizer_model: str
    tokenizer_revision: str
    # Token IDs that end a response.
    eos_ids: tuple[int, ...]
    # Extra arguments to the tokenizer's chat template.
    chat_options: dict
    # Whether the CPU prompt reference can stand in for NPU prefill.
    cpu_prefill: bool
    load_tokenizer: Callable[[str], Any]
    default_target_cache: Callable[[], Path]
    default_draft_cache: Callable[[], Path]
    default_draft_snapshot: Callable[[], Path]
    target_decoder: type
    draft_weights: type
    context_prepass: type
    context_reference: type
    # Rejects a decoder whose compiled configuration is not the selected one.
    require_selected: Callable[[Any, str], None]
    # Returns the NPU prefill class; deferred because it loads the prefill
    # builders.
    npu_prefill: Callable[[], type]
    # Identifies the sources and tools a measurement ran with.
    runtime_identity: Callable[[], dict]
