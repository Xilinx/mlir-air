# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repeated NPU prefill, then DFlash against its sequential oracle and committed KV."""

from dflash_model import MODEL
from common.verify_loop import main

if __name__ == "__main__":
    main(MODEL)
