# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Runs this directory's entry points on the gemma4_e2b_q4nx code with
GEMMA4_VARIANT=e4b. The Gemma4 modules read their dims at import time, so this
is imported before any of them."""

import importlib
import os
import runpy
import sys
from pathlib import Path

E2B = Path(__file__).resolve().parents[1] / "gemma4_e2b_q4nx"

os.environ["GEMMA4_VARIANT"] = "e4b"
if str(E2B) not in sys.path:
    sys.path.insert(0, str(E2B))
_w = sys.modules.get("gemma4_e2b_q4nx_weights")
if _w is not None and _w.VARIANT != "e4b":
    raise ImportError(f"gemma4_e2b_q4nx_weights is already loaded as {_w.VARIANT}")


def delegate(name, module):
    """Run gemma4_e2b_q4nx's `module` as the script, or alias `name` to it."""
    if name == "__main__":
        runpy.run_path(str(E2B / f"{module}.py"), run_name="__main__")
    else:
        sys.modules[name] = importlib.import_module(module)
