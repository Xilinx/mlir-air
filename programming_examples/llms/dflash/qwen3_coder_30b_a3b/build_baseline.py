# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the single-token (B1) decode of llms/qwen3_coder_30b_a3b_q4k for a
4096-position cache, with the same cache reads as the target, into a compile
cache of its own.

  python3 build_baseline.py --output DIR
"""

import argparse
import json
import os
from pathlib import Path
import sys

from paths import TARGET, target_imports
import model_spec

READS = (256, 512, 1024, 2048)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    (out / "work").mkdir(parents=True, exist_ok=True)
    os.environ["AIR_COMPILE_CACHE"] = str(out / "compile_cache")
    with target_imports():
        sys.path.insert(0, str(TARGET))
        import qwen3_coder_30b_a3b_q4k_decode as DEC
        import lm_head  # noqa: F401  (the decode imports it later)
    os.chdir(out / "work")
    DEC.compile_all(DEC.W.N_LAYERS // DEC.PER_WAVE, model_spec.CAPACITY, READS)
    manifest = dict(
        status="compiled", capacity=model_spec.CAPACITY, batch=1, reads=list(READS)
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"B1 baseline compiled in {out}")


if __name__ == "__main__":
    main()
