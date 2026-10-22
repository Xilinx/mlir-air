# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile the B8 target or drafter and save its ABI."""

from paths import HERE
import target_config
from selected_config import selected_config
from common.build_target import main


def kernel_defines(cfg):
    return ["-DMODEL_TYPE=QWEN3_4B"]


if __name__ == "__main__":
    main(target_config, selected_config, kernel_defines=kernel_defines)
