#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""THE device gate for batching. Batch-8 token t must equal batch-1 at position P+t.

WHAT CHANGED FROM THE FIRST VERSION OF THIS FILE, and why it matters. The first
version asserted that B IDENTICAL tokens give B IDENTICAL rows. That is false,
and the reason is the whole point of the batch: a block of B tokens occupies B
CONSECUTIVE positions, so token t attends to t more keys than token 0 and
rotates by a different RoPE angle. Identical inputs are supposed to give
DIFFERENT outputs. An "all rows equal" gate would have passed on an engine that
gave every token position P's context -- exactly the failure batch_attn_mask.py
was written to warn about.

THE PROPERTY THAT IS ACTUALLY TRUE, and that DFlash rests on:

    one batch-B dispatch at position P
      ==
    B batch-1 dispatches at positions P, P+1, ... P+B-1, same X each time

Both append the same K/V at the same positions and both mask token t to keys
0..P+t. They differ only in WHEN the appends happen -- the block does all B up
front -- and a key past a token's own L is masked, so that cannot show. If this
holds, speculative verify is lossless; if it does not, the batch is scoring
tokens against a context they should not see, and nothing downstream would say
so.

    --tokens 0        token 0 only: needs ONE batch-1 template, at the same L
                      as the batched build. Already covers the X feed, the
                      batched mmul, both egress gathers, the rms chunk
                      regeneration and residual accumulate, the glu row loop,
                      the QKV L2 transpose, rope, the q broadcast, the first
                      attention pass, the o gather and the layer-out drain.
    --tokens all      every token: needs a batch-1 template per position
                      (decode_b1_L{P+1+t}), because a non-DYNSEQ template bakes
                      L. What the extra templates buy is the ONE thing token 0
                      cannot see: that token t gets a DIFFERENT and correct
                      answer rather than a copy of token 0's.

WHAT IS READ BACK. Nothing -- the decode drains its layer output IN PLACE into
the X buffer (arg0), which is what makes layer chaining work. So the gate writes
X, dispatches, and reads X.

SYNTHETIC EVERYTHING, DELIBERATELY. This is a DATAFLOW gate: it asks whether
the engine moves the right bytes to the right places, which is where this design
has failed before and where the failures are silent. Numerics have their own
device gates (q4k_mm_gate.py, proj_qmm_gate.py) and they pass.

BUT NOT RANDOM BYTES. The first attempt filled every BO from a PRNG and the
device returned 0x7F81 in every element -- one uniform NaN. A gate whose output
is constant passes on anything. So the weights are REAL q4k blocks
(proj_qmm_pack), the norms are near 1, X is order 1, and the KV cache starts at
zero -- zero rather than random because the two builds pad the cache to
different lengths, so "the same random bytes" would not be the same cache.
The gate checks the output is not constant before it compares anything.

    python3 batch_equiv.py --batch 8 --L 128
    python3 batch_equiv.py --batch 8 --L 128 --tokens all

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


def geom(model, vocab_i2, L, batch, n_layers=1):
    """BO element counts, read off the builder rather than restated.

    decode_geometry.geometry() imports fused_decode.py under a given env and
    reads its sizes back. The batch and the context length are passed the same
    way the build passed them -- through the environment -- so a mismatch
    between what was built and what is dispatched is not expressible here.
    """
    import decode_geometry as dg

    old = {k: os.environ.get(k) for k in ("DECODE_BATCH", "DECODE_GOLDEN_L")}
    os.environ["DECODE_BATCH"] = str(batch)
    os.environ["DECODE_GOLDEN_L"] = str(L)
    try:
        return dg.geometry(model, vocab_i2, L, n_layers=n_layers)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bf16(x):
    """float array -> its int16 bit pattern, which is what a BO holds."""
    from ml_dtypes import bfloat16

    return np.asarray(x, dtype=np.float32).astype(bfloat16).view(np.int16)


def weight_bo(n_elems, seed, n_distinct=64):
    """A weight BO of REAL q4k blocks, not random bytes.

    Random bytes are random bf16 SCALES, which is how the first version of this
    gate got a NaN in every output element. These are packed the way the
    kernel's dequant expects (w = q*scale + min) with scale and min small enough
    that a 2048-wide dot product stays in range.

    n_distinct blocks are packed and then tiled to fill the buffer: packing all
    39680 of them takes minutes and buys nothing, but ONE would make every
    row-block of the output identical and hide a whole class of permutation
    error. 64 is enough that neighbours differ.
    """
    import proj_qmm_pack as pk

    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_distinct):
        q = rng.integers(0, 16, size=(pk.ROW_BLOCK, pk.COL_BLOCK), dtype=np.uint8)
        scale = rng.uniform(0.002, 0.01, size=(pk.ROW_BLOCK, pk.N_GROUPS))
        mn = rng.uniform(-0.05, 0.05, size=(pk.ROW_BLOCK, pk.N_GROUPS))
        blocks.append(pk.pack_q4k_block(q, scale, mn))
    slab = np.concatenate(blocks)
    reps = -(-n_elems // slab.size)
    return np.tile(slab, reps)[:n_elems]


def rms_bo(g, batch, seed):
    """The norm/LUT buffer, with EVERY token's rope LUT set to the same words.

    The LUT is per position, and the batched build carries B of them. Giving
    them all the same value is what makes token t's rotation comparable to a
    batch-1 dispatch at position P+t -- that batch-1 build has exactly one LUT,
    and it has to be this one.
    """
    rng = np.random.default_rng(seed)
    buf = bf16(rng.uniform(0.8, 1.2, size=g["rms_size"])).copy()
    lut_off, lut_len = g["rms_lut_off"], g["rope_w_len"]
    # cos/sin, so in [-1, 1]. The SAME block for every position: that is what
    # makes token t comparable to a batch-1 dispatch, whose single LUT has to be
    # this one. It does mean the gate cannot see a LUT INDEX error that lands on
    # another slot -- which is what --tokens all is for, since the causal L
    # still differs per token.
    lut = bf16(np.random.default_rng(seed + 77).uniform(-1.0, 1.0, size=lut_len))
    for t in range(batch):
        buf[lut_off + t * lut_len : lut_off + (t + 1) * lut_len] = lut
    return buf


def dispatch(xclbin, insts, g, batch, x_row, seed, xrt):
    """One dispatch. Returns the X buffer afterwards -- the layer output.

    Raw int16, not floats: this compares BYTES. A bit-level difference a float
    compare would round away is exactly the kind of layout error the gate is
    for.
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
    # kv_elems -- not restated, read straight off the builder.
    names = ("x", "w", "r", "y", "kv")
    sizes = (g["k"], g["w_elems"], g["rms_size"], g["ny"], g["kv_elems"])
    bos = {}
    for i, (name, n) in enumerate(zip(names, sizes), start=3):
        bos[name] = xrt.bo(dev, n * 2, xrt.bo.host_only, kern.group_id(i))

    fills = {
        # Identical bytes in both dispatches, so any difference in the output is
        # attributable to the batching and nothing else.
        "w": weight_bo(g["w_elems"], seed + 1),
        "r": rms_bo(g, batch, seed + 2),
        "kv": np.zeros(g["kv_elems"], np.int16),
        "y": np.zeros(g["ny"], np.int16),
        # The SAME row, B times: the block is B copies of one token, so any
        # difference between rows comes from the POSITION, which is the thing
        # under test.
        "x": np.tile(x_row, batch)[: g["k"]],
    }
    for name, buf in fills.items():
        bos[name].write(buf, 0)
        bos[name].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    st = kern(3, i_bo, ib.size, bos["x"], bos["w"], bos["r"], bos["y"], bos["kv"]).wait(
        60000
    )
    for b in ("x", "y", "kv"):
        bos[b].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(bos["x"].map(), dtype=np.int16, count=g["k"]).copy()
    if not str(st).endswith("COMPLETED"):
        # HOW FAR DID IT GET. A timeout says nothing about where; the output
        # buffers do. The layer output is written IN PLACE over X, the KV cache
        # is written by the rope append, and Y takes the host drains -- so
        # whether each one moved brackets the hang between the phases that write
        # them. Cheap, and the only progress signal a hung dispatch leaves.
        def moved(name, n, ref):
            got = np.frombuffer(bos[name].map(), dtype=np.int16, count=n)
            ix = np.nonzero(got != ref[:n])[0]
            if not ix.size:
                return f"{name}: nothing written"
            # Contiguous runs, not just a count: WHICH regions moved is what
            # localises the stall. The KV cache is region-major, so a run tells
            # you which token and which of K / V got there.
            brk = np.nonzero(np.diff(ix) != 1)[0]
            runs = np.split(ix, brk + 1)
            shown = ", ".join(f"[{r[0]}..{r[-1]}]" for r in runs[:8])
            more = f" (+{len(runs) - 8} more)" if len(runs) > 8 else ""
            return f"{name}: {ix.size} of {n} in {len(runs)} runs: {shown}{more}"

        print("  TIMEOUT. what the device managed to write:")
        for name, n in (("x", g["k"]), ("y", g["ny"]), ("kv", g["kv_elems"])):
            print("    " + moved(name, n, fills[name]))
        bos.clear()
        del i_bo, kern, ctx, xb, dev
        raise RuntimeError(f"dispatch state={st} ({Path(xclbin).name})")
    # pyxrt's objects have to go before the context and the device or the
    # interpreter segfaults at exit -- the shipping driver keeps an explicit
    # release order for the same reason.
    bos.clear()
    del i_bo, kern, ctx, xb, dev
    return out


def template(prefix, batch, L):
    xb = HERE / f"{prefix}_b{batch}_L{L}.xclbin"
    return xb, xb.with_suffix("").with_suffix(".insts.bin")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=128, help="token 0's context length")
    ap.add_argument("--prefix", default="decode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument(
        "--tokens",
        default="0",
        choices=["0", "all"],
        help="'0' needs one batch-1 template; 'all' needs one per position",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="dispatch the batched build only and report that it completed. "
        "No reference, no comparison -- answers 'does it hang', which is "
        "the first thing a new wire fails at.",
    )
    args = ap.parse_args()

    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("pyxrt not importable: this gate needs the NPU")

    gN = geom(args.model, args.vocab_chunk_i2, args.L, args.batch, args.n_layers)
    g1 = geom(args.model, args.vocab_chunk_i2, args.L, 1, args.n_layers)
    rng = np.random.default_rng(args.seed)
    row = rng.integers(-2048, 2048, size=g1["k"], dtype=np.int16)

    bn, bni = template(args.prefix, args.batch, args.L)
    if not bn.exists():
        sys.exit(
            f"{bn.name} not found. Build it from THIS tree with\n"
            f"    DECODE_BATCH={args.batch} DECODE_GOLDEN_L={args.L} "
            f"make compile-decode\n"
            f"and rename decode.* to {bn.stem}.*"
        )

    print(f"\nbatch equivalence  [{args.model}, batch {args.batch}, L {args.L}]")
    yN = dispatch(bn, bni, gN, args.batch, row, args.seed, xrt)
    print(f"  batch {args.batch}: dispatch COMPLETED, {yN.size} elements back")
    # A constant output makes every comparison below trivially true. This is not
    # hypothetical: random-byte weights produced 0x7F81 -- one NaN -- in every
    # element, and the gate would have "passed".
    uniq = np.unique(yN).size
    print(
        f"  {uniq} distinct values in the output" + ("" if uniq > 8 else "  <-- FLAT")
    )
    if uniq <= 8:
        print(
            "\n  The output carries no information, so nothing below would mean\n"
            "  anything. Check the weight fill (real q4k blocks, bounded scales)\n"
            "  before reading any comparison."
        )
        return 1
    if args.smoke:
        print(
            "\n  --smoke says the wire does not hang and nothing more. It does\n"
            "  NOT say the answers are right; run without it for that."
        )
        return 0

    K = g1["k"]
    positions = range(args.batch if args.tokens == "all" else 1)
    bad, missing = [], []
    for t in positions:
        b1, b1i = template(args.prefix, 1, args.L + t)
        if not b1.exists():
            missing.append(b1.name)
            continue
        y1 = dispatch(b1, b1i, g1, 1, row, args.seed, xrt)
        got = yN[t * K : (t + 1) * K]
        if not np.array_equal(got, y1):
            n = int((got != y1).sum())
            bad.append((t, n, int(np.argmax(got != y1))))
        else:
            bad.append(None) if False else None
        print(
            f"  token {t} (L {args.L + t}): "
            f"{'EQUAL' if not bad or bad[-1][0] != t else f'{bad[-1][1]} of {K} differ'}"
        )

    if missing:
        print(
            "\n  missing batch-1 references: "
            + ", ".join(missing)
            + "\n  Each position needs its own template -- a non-DYNSEQ build\n"
            "  bakes L. Build them with DECODE_GOLDEN_L=<L> and rename."
        )
        return 1
    if bad:
        t, n, first = bad[0]
        print(f"\n  first mismatch: token {t}, element {first}, {n} elements differ")
        print(
            "  Read WHICH tokens differ before anything else:\n"
            "    token 0 too        -> the batched PATH: the @xnorm chunk feed,\n"
            "                          the tile-blocked broadcast, a gather, or\n"
            "                          the rms chunk regeneration\n"
            "    only t > 0         -> the batched POSITION: the KV append slot,\n"
            "                          the rope LUT index, or the causal L\n"
            "    every t > 0 equal\n"
            "    to token 0         -> the position is not reaching the token at\n"
            "                          all; every one got position P"
        )
        return 1
    n = len(list(positions))
    print(
        f"\n  {n} token{'s' if n > 1 else ''} match batch 1 at their own "
        f"position -- GATE PASS"
    )
    if args.tokens == "0":
        print(
            "  Token 0 only. That covers the whole batched data path but NOT\n"
            "  that tokens 1.. get their own context; --tokens all does."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
