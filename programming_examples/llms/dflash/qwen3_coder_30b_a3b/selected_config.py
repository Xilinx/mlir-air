# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The drafter's B8 mapping on the shared decoder, the one this example builds."""

from __future__ import annotations

from dataclasses import asdict

from target_config import TargetConfig

from model_spec import BLOCK_SIZE as BATCH, CAPACITY, DRAFT_LAYERS


def selected_config(role: str, block_size: int = BATCH) -> TargetConfig:
    if role != "drafter":
        raise ValueError("only the drafter runs on the shared decoder here")
    if block_size != BATCH:
        raise ValueError(f"the drafter is built for blocks of {BATCH}")
    return TargetConfig(
        role=role,
        batch=BATCH,
        projection_schedule="groups",
        attention_native_key=False,
        attention_early_key_release=False,
        attention_pv_reuse=False,
        capacity=CAPACITY,
        layers=DRAFT_LAYERS,
        causal=False,
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
