#!/usr/bin/env python3
"""Dump ONE batch-8 dispatch's layer output to .npy, for A/B-ing two builds.

batch_equiv.py compares batch 8 against a batch-1 REFERENCE, and that reference
does not exist at every configuration -- at DECODE_HIDDEN_TAPS=1 with
VOCAB_CHUNK_I2=50 the batch-1 template times out on its own, with and without
RMS_MEMTILE_REFEED=3. So there is no way to ask "are these answers right" there.

But "are these two builds' answers the SAME" is answerable, and for a change
that is supposed to be bit-exact it is the stronger question anyway. Same seed
=> same input row (batch_equiv builds it from np.random.default_rng(seed)), so
two runs differ only in the xclbin.

ONE DEVICE CONTEXT PER PROCESS, so this dumps one build and exits; run it twice
and diff the files. Raw int16, like batch_equiv's own dispatch(): a bit-level
difference a float compare would round away is exactly what is being looked for.

    python dump_layer_output.py --prefix taps --batch 8 --L 128 --out ctrl.npy
    python dump_layer_output.py --prefix taps --batch 8 --L 128 --out ring.npy
    python dump_layer_output.py --diff ctrl.npy ring.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", nargs=2, help="compare two dumps and exit")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="taps")
    ap.add_argument("--wait", type=int, default=60000)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.diff:
        a = np.load(args.diff[0])
        b = np.load(args.diff[1])
        if a.shape != b.shape:
            print(f"SHAPE {a.shape} vs {b.shape}")
            return 1
        d = a != b
        n = int(d.sum())
        print(f"{a.size} elements, {n} differ ({n / a.size:.2%})")
        if n:
            idx = np.flatnonzero(d)
            print(f"  first differing index {idx[0]}, last {idx[-1]}")
            print(f"  first 8 differing indices: {idx[:8].tolist()}")
            print(f"  A at those: {a[idx[:8]].tolist()}")
            print(f"  B at those: {b[idx[:8]].tolist()}")
            # A one-block shift is the classic ping-pong drain failure: the
            # stream is right but consumed one buffer early. Look for it.
            for sh in (1, 2, 256, 512, 2048, 2560):
                if a.size > sh and np.array_equal(a[sh:], b[: a.size - sh]):
                    print(f"  B IS A SHIFTED BY {sh} ELEMENTS")
        else:
            print("  IDENTICAL")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import batch_equiv as BE

    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("pyxrt not importable: this needs the NPU")

    g = BE.geom(args.model, args.vocab_chunk_i2, args.L, args.batch, args.n_layers)
    g1 = BE.geom(args.model, args.vocab_chunk_i2, args.L, 1, args.n_layers)
    rng = np.random.default_rng(args.seed)
    row = BE.bf16(rng.uniform(-1.0, 1.0, size=g1["k"]))
    bn, bni = BE.template(args.prefix, args.batch, args.L)
    if not bn.exists():
        sys.exit(f"{bn.name} not found")
    y, _probe, _kv = BE.dispatch(
        bn, bni, g, args.batch, row, args.seed, xrt, args.wait, None
    )
    print(f"dispatch COMPLETED, {y.size} elements, {len(set(y.tolist()))} distinct")
    if args.out:
        np.save(args.out, y)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
