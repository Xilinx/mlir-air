# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Selected two-owner B8 layout and ABI, bound to model_spec.py."""

from paths import HERE, COMMON, EXAMPLES, REPO, FUSED
from common.b8_config import (
    ABI_VERSION,
    TargetConfig,
    builder_environment,
    load_geometry,
    load_layout,
    abi,
    build_module,
)
