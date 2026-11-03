#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the Nemotron-3.5-Bolt-E4B DFlash draft on the NPU: dflash/qwen3_4b's
B8 drafter (its graph, kernels and build), at the draft's geometry
(fused_decode's "gemma4-e4b-dflash-draft": FFN 5120, the target's 262144-row
LM head). The draft's attention, norms and heads are Qwen3-4B's.

  python3 build_draft.py --output DIR [--capacity C] [--check-existing]
"""

import contextlib
import os
import sys
from pathlib import Path

import paths  # noqa: E402,F401  (dflash on sys.path)

# Last, so that this directory's modules win where qwen3_4b has one of the
# same name.
QWEN = Path(__file__).resolve().parents[1] / "qwen3_4b"
sys.path.append(str(QWEN))
import target_config  # noqa: E402
from selected_config import selected_config  # noqa: E402
from common.build_target import main  # noqa: E402


@contextlib.contextmanager
def draft_environment(config, length):
    saved = dict(os.environ)
    try:
        for name in list(os.environ):
            if name.startswith(("DECODE_", "RMS_", "PROJ_", "Q4K_", "UNI_")):
                del os.environ[name]
        os.environ.update(
            DECODE_MODEL="gemma4-e4b-dflash-draft",
            VOCAB_CHUNK_I2="20",
            DECODE_GOLDEN="1",
            DECODE_GOLDEN_L=str(length),
            DECODE_UNI_DEC=str(config.layers),
            DECODE_STACK=str(config.stack_bytes),
            DECODE_HIDDEN_TAPS=str(int(config.taps)),
            W_DUAL_CHAN="1",
            NLAYERS="1",
            LM_HEAD="0",
            DECODE_DYNSEQ=str(int(config.dynamic_context)),
        )
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


target_config.builder_environment = draft_environment


def kernel_defines(cfg):
    return ["-DMODEL_TYPE=QWEN3_4B"]


if __name__ == "__main__":
    if "--role" not in sys.argv:
        sys.argv += ["--role", "drafter"]
    main(target_config, selected_config, kernel_defines=kernel_defines)
