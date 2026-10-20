# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gate and time the Gemma4-E4B fused prefill against the CPU reference:
../gemma4_e2b_q4nx/fused_prefill/verify.py with the E4B config.

  python3 verify.py BUILD_DIR [options as there]
"""

import importlib.util
import sys
from pathlib import Path

_E2B = Path(__file__).resolve().parents[2] / "gemma4_e2b_q4nx" / "fused_prefill"
_spec = importlib.util.spec_from_file_location(
    "gemma4_fused_verify", _E2B / "verify.py"
)
_v = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _v
_spec.loader.exec_module(_v)

if __name__ == "__main__":
    sys.exit(_v.main(_v.gw.E4B))
