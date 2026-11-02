# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Vision tower gate: NPU soft tokens vs HF's Gemma4VisionModel + embedder.

python run_vision.py [--image PATH] [--max-soft-tokens N] [--min-cosine C]
python run_vision.py --diagnosis    # per-layer cosine vs the numpy tower
python run_vision.py --cpu-ops      # device ops emulated in numpy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gemma4_vision_reference as R  # noqa: E402
from gemma4_vision_encoder import CpuOps, NpuOps, VisionEncoder  # noqa: E402
from gemma4_vision_weights import NUM_LAYERS, VisionWeights, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-soft-tokens", type=int, default=None)
    ap.add_argument("--min-cosine", type=float, default=0.99)
    ap.add_argument("--layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--diagnosis", action="store_true")
    ap.add_argument("--cpu-ops", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="encodes to time")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    kw = {"max_soft_tokens": args.max_soft_tokens} if args.max_soft_tokens else {}
    pv, pid, n_soft = preprocess(R.load_image(args.image), **kw)
    w = VisionWeights()
    ops = CpuOps() if args.cpu_ops else NpuOps(verbose=args.verbose)
    enc = VisionEncoder(ops, w, n_layers=args.layers)
    sh = enc.shape_of(pid)
    print(
        f"patches {sh.L} ({sh.L_valid} valid), GEMM rows {sh.L_pad}, "
        f"keys {sh.kv_rows}, soft tokens {n_soft}"
    )

    taps = {} if args.diagnosis else None
    t0 = time.perf_counter()
    out = enc.encode(pv, pid, taps=taps)
    print(
        f"first encode (incl. any compile + weight upload): {time.perf_counter() - t0:.2f}s"
    )
    for _ in range(args.repeat):
        if isinstance(ops, NpuOps):
            ops.op_t = {}
        t0 = time.perf_counter()
        out = enc.encode(pv, pid)
        dt = time.perf_counter() - t0
        dev = ops.timings()
        print(
            f"encode {dt * 1e3:.1f} ms (host {enc.host_t * 1e3:.1f} ms); device "
            + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in sorted(dev.items()))
        )

    if args.diagnosis:
        ref_taps = {}
        R.tower(w, pv, pid, n_layers=args.layers, taps=ref_taps)
        for i in range(args.layers):
            a, b = taps[f"L{i}.out"][: sh.L_valid], ref_taps[f"L{i}.out"][: sh.L_valid]
            print(f"  layer {i:2d} out cosine vs numpy {R.cosine(a, b):.6f}")

    if args.layers != NUM_LAYERS:
        print("partial layer count: no HF comparison")
        return 0
    ref = R.cached_hf_tower(pv, pid)
    assert ref.shape == out.shape, (ref.shape, out.shape)
    cos = R.cosine(out, ref)
    print(f"soft tokens {out.shape} vs HF: cosine {cos:.6f} (gate {args.min_cosine})")
    ok = cos >= args.min_cosine
    print("PASS!" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
