# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the fused prefill device and the encoder's attention ELF:
python3 device_build.py BUILD_DIR [-j N]."""

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

from air_examples.llms.shared.fused_prefill.build import main  # noqa: E402

if __name__ == "__main__":
    from air_examples.llms.whisper_v3_turbo_q4nx import whisper_attn

    main("air_examples.llms.whisper_v3_turbo_q4nx.whisper_spec", sys.argv[1:])
    whisper_attn.build(sys.argv[1])
