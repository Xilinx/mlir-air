#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phase 2 sweep: run dflash_phase2_device.py + dflash_phase2_replay.py over a
set of diverse prompts and aggregate the real acceptance-rate measurement.

Chosen to avoid docs/DFlashFeasibility.md's own caveat about the first
(Paris/capitals) measurement: that prompt's continuation is a repetitive
enumeration, and the drafter's specific failure mode there (correct on
templated connective tokens, wrong on the one high-entropy fact) might not
generalize. These prompts are ordinary, non-enumerative continuations across
different registers (narrative, technical/code, conversational, factual,
instructional) -- deliberately NOT chosen to make the result look better or
worse either way.

Each prompt runs in its own two subprocesses (device dispatch, then CPU
replay) -- see hidden_taps_verify.py's module docstring for why torch and an
open XRT session cannot share a process.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

PROMPTS = [
    "The weather today is quite",
    "In the middle of the night, she heard a",
    "def fibonacci(n):\n    if n <= 1:",
    "The meeting was scheduled for",
    "According to recent studies, climate change has",
    "He picked up the phone and said",
    "The recipe calls for two cups of",
    "My favorite thing about summer is",
]

N_TOKENS = 96


def run_device(idx, prompt_text, out_npz):
    print(f"\n[sweep] === prompt {idx}: {prompt_text!r} -- device ===", flush=True)
    r = subprocess.run(
        [
            sys.executable,
            str(_HERE / "dflash_phase2_device.py"),
            str(out_npz),
            str(N_TOKENS),
            prompt_text,
        ],
        cwd=str(_HERE),
    )
    if r.returncode != 0 and not out_npz.exists():
        print(
            f"[sweep] prompt {idx} device recording FAILED (exit {r.returncode}), skipping",
            flush=True,
        )
        return False
    if r.returncode != 0:
        print(
            f"[sweep] prompt {idx} device subprocess exited {r.returncode} "
            "(known post-decode teardown segfault) but the npz was written -- continuing",
            flush=True,
        )
    return True


def run_replay(idx, out_npz):
    print(f"[sweep] === prompt {idx} -- replay ===", flush=True)
    r = subprocess.run(
        [sys.executable, str(_HERE / "dflash_phase2_replay.py"), str(out_npz)],
        cwd=str(_HERE),
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print(
            f"[sweep] prompt {idx} replay FAILED (exit {r.returncode}):\n{r.stderr[-2000:]}",
            flush=True,
        )
        return None
    m = re.search(r"accepted lengths: \[([^\]]*)\]", r.stdout)
    if not m:
        print(
            f"[sweep] prompt {idx}: could not parse accepted lengths from replay output",
            flush=True,
        )
        return None
    lens = [int(x) for x in m.group(1).split(",") if x.strip()]
    return lens


def main():
    work_dir = _HERE / "dflash_phase2_sweep_data"
    work_dir.mkdir(exist_ok=True)

    all_results = {}
    for i, prompt_text in enumerate(PROMPTS):
        out_npz = work_dir / f"prompt_{i}.npz"
        if not run_device(i, prompt_text, out_npz):
            continue
        lens = run_replay(i, out_npz)
        if lens is not None:
            all_results[prompt_text] = lens

    print("\n" + "=" * 70)
    print("[sweep] SUMMARY")
    print("=" * 70)
    grand_all = []
    for prompt_text, lens in all_results.items():
        arr = np.array(lens)
        grand_all.extend(lens)
        print(
            f"  {prompt_text[:45]:<45} n_blocks={len(arr):>3} "
            f"mean={arr.mean():.2f}/16 ({100*arr.mean()/16:.1f}%)"
        )
    if grand_all:
        g = np.array(grand_all)
        print("-" * 70)
        print(
            f"  {'OVERALL':<45} n_blocks={len(g):>3} "
            f"mean={g.mean():.2f}/16 ({100*g.mean()/16:.1f}%)  "
            f"median={np.median(g):.1f}  max={g.max()}"
        )
    else:
        print("  no successful prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
