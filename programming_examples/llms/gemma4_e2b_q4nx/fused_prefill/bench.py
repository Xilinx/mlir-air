# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Time the fused prefill at each of a list of prompt lengths, with no
reference (verify.py is the correctness gate).

  python3 bench.py BUILD_DIR --lengths 512,2048 [--repeat R] [--model M]

Prints one "[bench] prefill L=<n> median <ms> ms min <ms> ms" line per length,
the line bench/sweep_prefill.py --fused reads.
"""

import argparse
import sys
import time
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

import gemma4_e2b_q4nx_weights as gw  # noqa: E402
from air_examples.llms.gemma4_e2b_q4nx.fused_prefill.runtime import (  # noqa: E402
    FusedPrefill,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--lengths", required=True, help="comma-separated prompt lengths")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    pf = FusedPrefill(a.build_dir, max_len=max(lengths))
    pf.load_weights(gw.Q4nxModel(a.model))
    for n in lengths:
        # Synthetic ids: timing does not depend on the tokens.
        ids = [(7 + 13 * i) % gw.VOCAB for i in range(n)]
        pf.prefill(ids)
        ts = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            pf.prefill(ids)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        print(
            f"[bench] prefill L={n} median {ts[len(ts) // 2]:.1f} ms min {ts[0]:.1f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
