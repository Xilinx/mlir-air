# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the Gemma4-E2B fused prefill: kernels, every op's insts, the LM-head
GEMV (shared/fused_prefill/build.py).

  python3 build.py BUILD_DIR [-j N] [--max-len N]

--max-len is the longest prompt the build takes. Only a variant with more than
one KV head fixes it at build time (spec.MAX_LEN).
"""

import argparse
import os
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

from air_examples.llms.shared.fused_prefill.build import main  # noqa: E402


def run(spec_name):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=None)
    a, rest = ap.parse_known_args()
    # Set before the spec is imported: it builds its configuration on import.
    if a.max_len is not None:
        # 32 is the larger of the attention groups' record lengths (lkp).
        if a.max_len <= 0 or a.max_len % 32:
            sys.exit("--max-len must be a positive multiple of 32")
        os.environ["GEMMA4_FUSED_MAX_LEN"] = str(a.max_len)
    main(spec_name, rest)


if __name__ == "__main__":
    run("air_examples.llms.gemma4_e2b_q4nx.fused_prefill.spec")
