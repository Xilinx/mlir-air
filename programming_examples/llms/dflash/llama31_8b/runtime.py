# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NPU prefill, context preparation, drafting and verification."""

from paths import HERE
from common.pipeline_8b import load_pair, load_pipeline, tokenize_prompt
