"""Pytest config for full-decode ablation tests.

Inserts paths so tests can import:
- llama32_1b/ packages (kernel_builder, multi_launch_builder)
- llama32_1b/ablation/ (Plan 0's standalone_builders + validate.py)
- llama32_1b/ablation/prefill/ (Plan 1's cells, specs, common helpers)
- llama32_1b/ablation/decode/ (this package)
- programming_examples/ (matvec, weighted_rms_norm, ffn_swiglu)
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_DECODE = os.path.dirname(_THIS)
_ABLATION = os.path.dirname(_DECODE)
_LLAMA = os.path.dirname(_ABLATION)
_PROG_EXAMPLES = os.path.dirname(_LLAMA)

for p in (
    _PROG_EXAMPLES,
    _LLAMA,
    _ABLATION,
    os.path.join(_ABLATION, "prefill"),
    _DECODE,
):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pytest may have already inserted other paths or pre-imported a `cells` package
# from prefill/. Force _DECODE to sys.path[0] AND drop any cached `cells*` modules
# so subsequent `from cells.X import Y` resolves to decode/cells/.
if sys.path[0] != _DECODE:
    if _DECODE in sys.path:
        sys.path.remove(_DECODE)
    sys.path.insert(0, _DECODE)

for _stale in [m for m in list(sys.modules) if m == "cells" or m.startswith("cells.")]:
    del sys.modules[_stale]
for _stale in [m for m in list(sys.modules) if m == "specs" or m.startswith("specs.")]:
    del sys.modules[_stale]
for _stale in [
    m
    for m in list(sys.modules)
    if m == "standalone_builders" or m.startswith("standalone_builders.")
]:
    del sys.modules[_stale]
