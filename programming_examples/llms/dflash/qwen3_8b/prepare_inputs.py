# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Load one saved benchmark prefill and check its prompt token identity."""

from pathlib import Path
import sys
import numpy as np
from ml_dtypes import bfloat16
from target_config import HERE

sys.path.insert(0, str(HERE.parents[1] / "qwen3_8b_q4nx"))
from qwen3_8b_q4nx_weights import Q4nxModel as Q4nxModel


def load_prefill(root, case):
    with np.load(Path(root) / case["cache_file"]) as z:
        if not np.array_equal(z["input_ids"], case["input_ids"]):
            raise ValueError("Prefill token IDs differ from the frozen case")
        return {
            "input_ids": z["input_ids"].tolist(),
            "k": z["k"].view(bfloat16),
            "v": z["v"].view(bfloat16),
            "taps": z["taps"].view(bfloat16),
            "first": int(z["first"]),
        }
