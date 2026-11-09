# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Transcribe audio files on the NPU:
python3 whisper.py --build BUILD_DIR AUDIO [AUDIO ...]"""

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

from air_examples.llms.whisper_v3_turbo_q4nx.whisper_npu import main  # noqa: E402

if __name__ == "__main__":
    main()
