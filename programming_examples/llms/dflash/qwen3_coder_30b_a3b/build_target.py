# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile the target's verify pass (one build per cache read) and its LM
head, and record the target's configuration.

  python3 build_target.py --output DIR [--reads 256,512,1024,2048,4096]

A build is kept while its sources, module and toolchain are unchanged.
"""

import argparse
import json
from pathlib import Path

from paths import target_imports
import model_spec

READS = "256,512,1024,2048,4096"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reads", default=READS, help="cache positions each build reads")
    a = p.parse_args()
    out = a.output.resolve()
    reads = sorted(int(r) for r in a.reads.split(","))
    if reads[-1] != model_spec.CAPACITY:
        raise SystemExit(f"the longest read must be the capacity {model_spec.CAPACITY}")
    with target_imports():
        import verify_b8 as V
    layers = model_spec.TARGET_DEPTH
    for r in reads:
        V.compile_layers(out, layers, r, model_spec.CAPACITY)
    V.compile_head(out)
    config = dict(
        role="target",
        batch=model_spec.BLOCK_SIZE,
        capacity=model_spec.CAPACITY,
        layers=layers,
        causal=True,
        reads=reads,
    )
    manifest = dict(status="compiled", config=config)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"compiled {out}")


if __name__ == "__main__":
    main()
