# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile the selected B8/B16 target or drafter and save its ABI."""

from paths import HERE
import model_spec
import target_config
from selected_config import selected_config
from common.build_target import main


def kernel_defines(cfg):
    target = cfg.role == "target"
    eps = model_spec.TARGET_RMS_EPS if target else model_spec.DRAFT_RMS_EPS
    return [
        "-DMODEL_TYPE=" + (model_spec.TARGET_CXX if target else model_spec.DRAFT_CXX),
        f"-DRMS_NORM_EPS={eps}f",
        "-DNORM_CACHE_ROWS=4",
        "-DGLU_BATCH_WAVE_LAYOUT=1",
        f"-DGLU_BATCH={cfg.batch}",
    ]


if __name__ == "__main__":
    main(target_config, selected_config, kernel_defines=kernel_defines)
