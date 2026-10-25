# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the Gemma4-E4B fused prefill: kernels and every op's insts
(shared/fused_prefill/build.py).

  python3 build.py BUILD_DIR [-j N] [--max-len N]
"""

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The model's directory goes on sys.path, this one does not: its module names
# are too generic to publish. The package is reached as air_examples.*.
sys.path[:] = [str(HERE.parent)] + [
    p for p in sys.path if Path(p or ".").resolve() != HERE
]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[2])
]

from air_examples.llms.gemma4_e2b_q4nx.fused_prefill.build import run  # noqa: E402

if __name__ == "__main__":
    run("air_examples.llms.gemma4_e4b_q4nx.fused_prefill.spec")
