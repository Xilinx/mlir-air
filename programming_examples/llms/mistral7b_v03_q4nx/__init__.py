# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from pathlib import Path
import sys
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
# The shared Llama prefill still imports its sibling helpers by module name.
p = str(Path(__file__).resolve().parent.parent / "llama32_1b")
if p not in sys.path:
    sys.path.insert(0, p)

# The fused-decode driver also uses sibling imports when loaded as a package.
p = str(Path(__file__).resolve().parents[2] / "fused_decode")
if p not in sys.path:
    sys.path.insert(0, p)
