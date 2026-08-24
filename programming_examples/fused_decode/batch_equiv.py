#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""THE device gate for batching: B identical tokens must give B identical rows.

WHY THIS AND NOT A REFERENCE MODEL. Every gate up to here is either
"byte-identical at DECODE_BATCH=1" -- which by construction says nothing about
batch 8 -- or numpy (batch_path_check.py), which models the data path and cannot
see DMA ordering, lock counts, cascade or backpressure. Those are exactly where
this engine has failed before, and they fail as deadlocks or as plausible wrong
answers, not as compile errors.

The trick that removes the need for a reference: dispatch the batched build with
ALL B TOKENS THE SAME. Then every token's output row must equal every other's,
AND must equal what the batch-1 build produces from that same input. No golden
data, no requant cache, no model -- and any layout, ordering or aliasing error
shows up as a specific token's row differing, which localises it.

    row t != row 0            a per-token bug: append slot, LUT index, q feed
    all rows equal, != b1     a whole-batch bug: the feed transpose or a gather
    b1 != b1 (--self)         nondeterminism, and nothing else is meaningful yet

SYNTHETIC WEIGHTS, DELIBERATELY. bench_decode.cpp already establishes that decode
time and dataflow are data-independent here (every trip count is fixed except the
attention block loop, bounded by the RTP-L word). This gate needs the two
dispatches to see IDENTICAL bytes, not realistic ones, so it fills the weight /
rms / KV BOs from a seeded PRNG. It is a DATAFLOW gate, not a numerics gate --
numerics are q4k_mm_gate.py's and proj_qmm_gate.py's job, on device, and they
pass.

WHAT IT NEEDS. Two compiled templates at the same ATTN_MAXL and the same model,
one per batch size:

    make compile-decode                          # DECODE_BATCH unset -> b1
    DECODE_BATCH=8 make compile-decode           # -> b8

then rename/point at them with --b1 / --bn. Both must come from the SAME source
tree, or the comparison is between two designs rather than two batch sizes.

    python3 batch_equiv.py --self                # b1 vs b1: plumbing check
    python3 batch_equiv.py --batch 8             # THE gate

Exit code is the gate: 0 equivalent.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llms" / "bench"))


def _geom(model, vocab_i2, ctx, batch, n_layers=1):
    """BO element counts for one batch size, read off the builder.

    decode_geometry.geometry() imports fused_decode.py under a given env and
    reads its sizes back, which is what keeps this from restating them. The
    batch is passed the same way -- through the environment the builder reads.

    n_layers defaults to 1 because that is what the Makefile's DECODE_ENV
    builds (NLAYERS=1); a template built with a different NLAYERS needs the
    same value here or the weight BO is sized for a different design.
    """
    import decode_geometry as dg

    os.environ["DECODE_BATCH"] = str(batch)
    try:
        return dg.geometry(model, vocab_i2, ctx, n_layers=n_layers)
    finally:
        os.environ.pop("DECODE_BATCH", None)


def _fill(bo, n_elems, seed, xrt, dtype=np.int16):
    """Deterministic bytes into a BO. Same seed -> same bytes, both dispatches."""
    rng = np.random.default_rng(seed)
    buf = rng.integers(-2048, 2048, size=n_elems, dtype=np.int16)
    bo.write(buf, 0)
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return buf


def dispatch(xclbin, insts, geom, batch, x_row, seed, xrt):
    """One dispatch. Returns the y BO contents as int16 (raw bytes, not floats).

    Raw int16 rather than bf16 floats on purpose: this compares BYTES. A
    bit-level difference that a float compare would round away is exactly the
    kind of layout error the gate exists to catch.
    """
    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kn = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
    kern = xrt.kernel(ctx, kn.get_name())

    ib = np.fromfile(str(insts), dtype=np.uint32)
    i_bo = xrt.bo(dev, ib.nbytes, xrt.bo.cacheable, kern.group_id(1))
    i_bo.write(ib, 0)
    i_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # decode_geometry's own key names: k (X), w_elems, rms_size, ny (Y),
    # kv_elems. Not restated -- read straight off the builder.
    bos, g = {}, kern.group_id
    for i, (name, n) in enumerate(
        (
            ("x", geom["k"]),
            ("w", geom["w_elems"]),
            ("r", geom["rms_size"]),
            ("y", geom["ny"]),
            ("kv", geom["kv_elems"]),
        ),
        start=3,
    ):
        bos[name] = xrt.bo(dev, n * 2, xrt.bo.host_only, g(i))

    # Weights / norms / KV: identical bytes for both dispatches, so any
    # difference in y is attributable to the batching and nothing else.
    _fill(bos["w"], geom["w_elems"], seed + 1, xrt)
    _fill(bos["r"], geom["rms_size"], seed + 2, xrt)
    _fill(bos["kv"], geom["kv_elems"], seed + 3, xrt)
    # X: the SAME row, B times. That is the whole trick.
    xbuf = np.tile(x_row, batch)[: geom["k"]]
    bos["x"].write(xbuf, 0)
    bos["x"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    st = kern(3, i_bo, ib.size, bos["x"], bos["w"], bos["r"], bos["y"], bos["kv"]).wait(
        60000
    )
    if not str(st).endswith("COMPLETED"):
        raise RuntimeError(f"dispatch state={st} ({xclbin.name})")
    bos["y"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bos["y"].map(), dtype=np.int16, count=geom["ny"]).copy()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=30)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--b1", default="decode_L2048.xclbin")
    ap.add_argument(
        "--bn", default=None, help="batched xclbin (default decode_b<N>...)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="NLAYERS the templates were built with (Makefile uses 1)",
    )
    ap.add_argument(
        "--self",
        dest="selftest",
        action="store_true",
        help="dispatch the batch-1 build TWICE and compare. Checks this "
        "harness and the determinism it assumes, nothing about batching.",
    )
    args = ap.parse_args()

    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("pyxrt not importable: this gate needs the NPU")

    g1 = _geom(args.model, args.vocab_chunk_i2, args.ctx, 1, args.n_layers)
    rng = np.random.default_rng(args.seed)
    x_row = rng.integers(-2048, 2048, size=g1["k"], dtype=np.int16)

    b1 = Path(args.b1)
    i1 = b1.with_suffix("").with_suffix(".insts.bin")
    if not b1.exists():
        sys.exit(f"{b1} not found -- build it with `make compile-decode`")

    print(f"\nbatch equivalence  [{args.model}, ctx {args.ctx}]")
    y1 = dispatch(b1, i1, g1, 1, x_row, args.seed, xrt)
    print(f"  batch 1: {y1.size} elements")

    if args.selftest:
        y1b = dispatch(b1, i1, g1, 1, x_row, args.seed, xrt)
        ok = np.array_equal(y1, y1b)
        print(
            f"  batch 1 again: {'identical' if ok else 'DIFFERS -- nondeterministic'}"
        )
        print(
            "\n  --self checks the harness and the determinism every other\n"
            "  comparison here assumes. It says nothing about batching."
        )
        return 0 if ok else 1

    bn = Path(args.bn) if args.bn else Path(f"decode_b{args.batch}_L{args.ctx}.xclbin")
    inn = bn.with_suffix("").with_suffix(".insts.bin")
    if not bn.exists():
        sys.exit(
            f"{bn} not found. Build it from THIS tree with\n"
            f"    DECODE_BATCH={args.batch} make compile-decode\n"
            f"and rename decode_L{args.ctx}.* to {bn.stem}.*  -- both templates "
            "must come from the same source or this compares two designs."
        )
    gN = _geom(args.model, args.vocab_chunk_i2, args.ctx, args.batch, args.n_layers)
    yN = dispatch(bn, inn, gN, args.batch, x_row, args.seed, xrt)
    print(f"  batch {args.batch}: {yN.size} elements")

    # The y BO is (HOST_ROUNDS+LAYER_RNDS)*PAYLOAD + head; at batch B every
    # PAYLOAD row becomes B rows, token-major, which is what egress_bd.py's
    # descriptors were built to guarantee. So row t of the batched output sits
    # at the batch-1 offset scaled by B, plus t.
    row = g1["ny"]
    bad = []
    for t in range(args.batch):
        got = yN[t * row : (t + 1) * row]
        if got.size != y1.size:
            sys.exit(
                f"y geometry mismatch: batch-1 {y1.size}, batched row {got.size}. "
                "decode_geometry did not scale the y BO with the batch."
            )
        if not np.array_equal(got, y1):
            n = int((got != y1).sum())
            first = int(np.argmax(got != y1))
            bad.append((t, n, first))

    print(f"\n  {'token':>6}{'vs batch 1':>14}")
    print(f"  {'-' * 20}")
    for t in range(args.batch):
        hit = next((b for b in bad if b[0] == t), None)
        print(f"  {t:6d}{'EQUAL' if hit is None else f'{hit[1]} differ':>14}")
    if bad:
        t, n, first = bad[0]
        print(f"\n  first mismatch: token {t}, element {first}, {n} elements differ")
        print(
            "  Read WHICH tokens differ before anything else:\n"
            "    only some tokens   -> per-token: KV append slot, rope LUT index,\n"
            "                          q feed, or the causal L for that token\n"
            "    every token, same  -> whole-batch: the @xnorm chunk-major feed,\n"
            "                          the tile-blocked broadcast, or a gather"
        )
        return 1
    print(f"\n  all {args.batch} rows identical to batch 1 -- GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
