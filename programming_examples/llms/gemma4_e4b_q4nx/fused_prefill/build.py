# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the Gemma4-E4B fused prefill: kernels and every op's insts
(shared/fused_prefill/build.py).

  python3 build.py BUILD_DIR [-j N]
"""

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The model directories go on sys.path, this one does not: its module names
# are too generic to publish. The package is reached as air_examples.*.
# gemma4_e2b_q4nx holds the Gemma4 code this model's spec runs.
sys.path[:] = [str(HERE.parent), str(HERE.parents[1] / "gemma4_e2b_q4nx")] + [
    p for p in sys.path if Path(p or ".").resolve() != HERE
]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[2])
]

from air_examples.llms.shared.fused_prefill.build import main  # noqa: E402

if __name__ == "__main__":
    main("air_examples.llms.gemma4_e4b_q4nx.fused_prefill.spec")
