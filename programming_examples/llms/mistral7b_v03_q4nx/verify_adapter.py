# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""verify/ adapter: the production session on the fused prefill
(shared/affine_verify.py). Pointed at by --runner=mistral7b_v03_q4nx.verify_adapter."""

from pathlib import Path
import sys
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.affine_verify import adapter  # noqa: E402

globals().update(adapter("mistral7b_v03_q4nx"))
