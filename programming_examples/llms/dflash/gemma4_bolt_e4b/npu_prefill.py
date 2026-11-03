# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prompt prefill on the NPU for the DFlash loop: gemma4_e4b_q4nx's fused
prefill (target.Target), returning the prompt's K/V, the drafter's tap
features and the first token.

  python3 npu_prefill.py --cache-dir DIR [-j N]    build the prefill
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import paths  # noqa: F401
import model_spec
from model_paths import TARGET_MODEL, resolve_target_model
from target import Target
from target_runtime import pack_prefill_kv

BUILD = paths.HERE.parents[1] / "gemma4_e4b_q4nx/fused_prefill/build.py"


class NpuPrefill:
    def __init__(self, seq_len=2048, cache_dir=None, tap_slots=model_spec.TAP_SLOTS):
        if cache_dir is None:
            raise ValueError("NPU prefill needs its build directory")
        self.seq = seq_len
        self.taps = [b - 1 for b in tap_slots]
        self.engine = Target(str(cache_dir), max_len=seq_len)

    def load_weights(self, model=None):
        self.engine.load(resolve_target_model(model or TARGET_MODEL))

    def share_embedding(self, embedding):
        """The prefill reads its own embedding rows."""

    def prepare(self, input_ids):
        ids = [int(t) for t in input_ids]
        if len(ids) > self.seq:
            raise ValueError(f"Prompt exceeds the prefill length {self.seq}")
        started = time.perf_counter()
        e = self.engine
        e.clear_context()
        e.keep_kv, e.kv = True, {}
        try:
            logits, taps = e.forward(ids, 0, self.taps, 1)
        finally:
            e.keep_kv = False
        k, v = pack_prefill_kv(*e.kv_stack(), len(ids))
        result = dict(
            input_ids=ids,
            k=k.astype(bfloat16),
            v=v.astype(bfloat16),
            taps=taps.astype(bfloat16),
            first=int(np.argmax(logits[-1])),
            logits=logits[-1],
        )
        return result, time.perf_counter() - started


def build_identity():
    """The prefill's sources and the tools that compile it."""
    from common.build_identity import build_identity as shared, digest

    llms = paths.HERE.parents[1]
    sources = []
    for d in (
        "shared/fused_prefill",
        "gemma4_e2b_q4nx/fused_prefill",
        "gemma4_e4b_q4nx/fused_prefill",
    ):
        sources += [p for p in sorted((llms / d).rglob("*")) if p.is_file()]
    sources = [p for p in sources if "__pycache__" not in p.parts]
    ident = shared("decoder", paths.HERE)
    return dict(
        sources={str(p.relative_to(llms)): digest(p) for p in sources},
        tools=ident["tools"],
        dependencies=ident["dependencies"],
    )


def main():
    p = argparse.ArgumentParser(description="Build the NPU prefill.")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("-j", "--jobs", type=int, default=4)
    a = p.parse_args()
    stamp = a.cache_dir / "build_identity.json"
    ident = build_identity()
    if stamp.exists() and json.loads(stamp.read_text()) == ident:
        print("up to date", a.cache_dir)
        return
    stamp.unlink(missing_ok=True)
    subprocess.run(
        [sys.executable, str(BUILD), str(a.cache_dir), "-j", str(a.jobs)], check=True
    )
    stamp.write_text(json.dumps(ident, indent=2) + "\n")


if __name__ == "__main__":
    main()
