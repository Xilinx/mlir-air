"""Pytest config for prefill ablation tests.

Inserts paths so tests can import:
- llama32_1b/ packages (kernel_builder, multi_launch_builder)
- llama32_1b/ablation/ (Plan 1's validate.py and shared helpers)
- llama32_1b/ablation/prefill/ (this package)
- programming_examples/ (matvec, weighted_rms_norm, ffn_swiglu)
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PREFILL = os.path.dirname(_THIS)
_ABLATION = os.path.dirname(_PREFILL)
_LLAMA = os.path.dirname(_ABLATION)
_PROG_EXAMPLES = os.path.dirname(_LLAMA)

for p in (_PROG_EXAMPLES, _LLAMA, _ABLATION, _PREFILL):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pytest's package-import mode inserts the package parent (ablation/) into sys.path[0]
# before this conftest runs, which can shadow prefill/validate.py with ablation/validate.py.
# Guarantee that prefill/ is at index 0 so prefill-local modules take priority.
if sys.path[0] != _PREFILL:
    sys.path.remove(_PREFILL) if _PREFILL in sys.path else None
    sys.path.insert(0, _PREFILL)
