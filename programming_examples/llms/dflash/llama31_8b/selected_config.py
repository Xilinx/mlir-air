# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The supported B8 and B16 NPU mappings for this example."""

from __future__ import annotations

from dataclasses import asdict

from target_config import TargetConfig

CAPACITY = 4096
BATCH = 8
from model_spec import TARGET_LAYERS, DRAFT_LAYERS


def selected_config(role: str, block_size: int = BATCH) -> TargetConfig:
    if role not in ("target", "drafter"):
        raise ValueError("role must be target or drafter")
    if block_size not in (8, 16):
        raise ValueError("Block size must be 8 or 16")
    return TargetConfig(
        role=role,
        batch=block_size,
        projection_schedule="2x2" if block_size == 16 else "groups",
        attention_native_key=block_size == 16,
        attention_early_key_release=block_size == 16,
        attention_pv_reuse=block_size == 16,
        capacity=CAPACITY,
        layers=TARGET_LAYERS if role == "target" else DRAFT_LAYERS,
        causal=role == "target",
    )


def require_selected(config: TargetConfig, role: str) -> None:
    expected = asdict(selected_config(role, config.batch))
    observed = asdict(config)
    if observed != expected:
        differences = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if expected[key] != observed[key]
        }
        raise ValueError(
            f"Artifact is not the selected {role} configuration: {differences}"
        )
