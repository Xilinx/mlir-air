# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The one validated NPU configuration delivered by this example."""

from __future__ import annotations

from dataclasses import asdict

from target_config import TargetConfig

CAPACITY = 4096
BATCH = 8
from model_spec import TARGET_LAYERS, DRAFT_LAYERS


def selected_config(role: str) -> TargetConfig:
    if role not in ("target", "drafter"):
        raise ValueError("role must be target or drafter")
    return TargetConfig(
        role=role,
        capacity=CAPACITY,
        layers=TARGET_LAYERS if role == "target" else DRAFT_LAYERS,
        causal=role == "target",
    )


def require_selected(config: TargetConfig, role: str) -> None:
    expected = asdict(selected_config(role))
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
