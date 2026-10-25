# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Time the Gemma4-E4B fused prefill at each of a list of prompt lengths:
../gemma4_e2b_q4nx/fused_prefill/bench.py at GEMMA4_VARIANT=e4b.

  python3 bench.py BUILD_DIR [options as there]
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemma4_e4b_variant  # noqa: E402

runpy.run_path(
    str(gemma4_e4b_variant.E2B / "fused_prefill" / "bench.py"), run_name="__main__"
)
