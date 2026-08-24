#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does the decode wire survive batch > 1?

`batch_l1_budget.py` answers the L1 capacity question. This answers the other
half of step 2: what the batch does to the data moving in and out of the proj
cores.

OUT. Every proj core packs its 32-row output block into a packet, four cores'
packets are gathered into a group memtile buffer, four of those are gathered
into one main buffer, and the assembled packet is emitted on a single MM2S
whose routing header picks the consumer (rope / rms / glu).

There are two ways to carry a batch of B tokens through that:

  WIDEN   one packet per round, B times longer   <- what this models
  REPEAT  B packets per round, unchanged length

WIDEN is the one to build. It leaves N_ROUNDS, the BD count and the host
instruction stream alone, and it amortises the 2-element routing header over B
times the payload. REPEAT multiplies every one of those by B, and the builder
already documents shim BD exhaustion as a live constraint on round count.

WIDEN is not free, and the cost is exactly one thing: the assembled packet is
emitter-major but the consumer wants token-major, so the group gather picks up
a second dimension. This script checks that the resulting descriptors are
legal, against the AIE2p limits in mlir-aie's target model
(`AIETargetModel.h`, AIE2pTargetModel): BD length in 32-bit words, ND dimension
count, and the per-dimension wrap field.

    memtile   48 BDs   4 dims   len <= 2^17-1 words   wrap <= 2^10   step <= 2^17
    core      16 BDs   3 dims   len <= 2^14-1 words   wrap <= 2^8    step <= 2^13

IN. The other direction is the one that bites. A proj core's inner loop pairs
one X chunk with one weight block, and only the X chunk grows with the batch --
so the ratio of activation to weight bytes arriving at the core moves by a
factor of B. This reports where it crosses over and what it costs per cycle,
using the cycles/block that bench_q4k_mm.py measured.

Sizes come from fused_decode.py and limits from the numbers above; the only
inputs are the stream width and the cycles/block, both flags.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF16 = 2
WORD = 4

# AIE2p, from mlir-aie include/aie/Dialect/AIE/IR/AIETargetModel.h.
LIMITS = {
    "core": dict(bds=16, dims=3, len_words=(1 << 14) - 1, wrap=1 << 8, step=1 << 13),
    "memtile": dict(
        bds=48, dims=4, len_words=(1 << 17) - 1, wrap=1 << 10, step=1 << 17
    ),
}
L2_BYTES = 512 * 1024  # memtile data memory on AIE2p


def load_builder(model, vocab_chunk_i2, ctx=2048, env_extra=None):
    os.environ.update(
        DECODE_MODEL=model,
        VOCAB_CHUNK_I2=str(vocab_chunk_i2),
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        UNIFIED="1",
        DECODE_GOLDEN_L=str(ctx),
        W_DUAL_CHAN="1",
    )
    os.environ.update(env_extra or {})
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("_fd_egr", HERE / "fused_decode.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def geometry(fd, b):
    """Egress element counts at batch b. b=1 must reproduce the builder."""
    pair_pay = fd.PAIR_ROWS * fd.ROW_BLOCK * b
    grp = fd.HDR + fd.LEADS_PER_GRP * pair_pay
    main = grp + (fd.N_GRP - 1) * fd.LEADS_PER_GRP * pair_pay
    return dict(
        pair_pay=pair_pay,
        grp=grp,
        main=main,
        payload=fd.N_PAIRS * pair_pay,
    )


def descriptors(fd, b):
    """Every BD on the egress path at batch b.

    (name, tile, length in elements, [wrap sizes outer->inner], [strides])
    The single-dimension entries are contiguous runs: length is carried in the
    BD's buffer_length field and no wrap is used.
    """
    g = geometry(fd, b)
    rb = fd.PAIR_ROWS * fd.ROW_BLOCK  # one token's block from one emitter
    lead_span = fd.LEADS_PER_GRP * rb  # token stride in the group buffer
    d = [
        # Proj core -> wire. ypair is contiguous [16 | B*rb], so this stays 1D
        # however the batch grows; only its length moves.
        ("outA put", "core", fd.HDR + g["pair_pay"], [], []),
        # Group memtile <- wire. THIS is the descriptor the batch changes: the
        # emitter sends its B blocks back to back, and they have to land strided
        # so that token t's row of the group buffer is contiguous across
        # emitters. One extra dimension, iterating tokens.
        ("outA get k=0", "memtile", fd.HDR + g["pair_pay"], [b, rb], [lead_span, 1]),
        ("outA get k>0", "memtile", g["pair_pay"], [b, rb], [lead_span, 1]),
        # Group -> main and main -> consumer are contiguous at any batch.
        ("toMain put", "memtile", g["grp"], [], []),
        ("toMain get g=0", "memtile", g["grp"], [], []),
        ("toMain get g>0", "memtile", fd.LEADS_PER_GRP * g["pair_pay"], [], []),
        ("outY put", "memtile", g["main"], [], []),
    ]
    return d


def check(name, tile, elems, wraps, strides):
    """[(limit, value, cap, ok)] for one descriptor."""
    lim = LIMITS[tile]
    words = (elems * BF16 + WORD - 1) // WORD
    out = [("len words", words, lim["len_words"])]
    if wraps:
        out.append(("dims", len(wraps), lim["dims"]))
        for i, w in enumerate(wraps[:-1]):  # innermost dim is the contiguous run
            out.append((f"wrap[{i}]", w, lim["wrap"]))
        for i, s in enumerate(strides[:-1]):
            out.append((f"step[{i}]", s, lim["step"]))
    return [(k, v, c, v <= c) for k, v, c in out]


def max_batch(fd, cap_batch):
    """Largest b <= cap_batch with every egress descriptor legal."""
    b = 1
    while b <= cap_batch:
        if any(not ok for d in descriptors(fd, b + 1) for *_, ok in check(*d)):
            return b
        b += 1
    return cap_batch


def l2_egress(fd, b):
    """Egress L2 buffers, per pinned memtile column. Bytes if all live at once.

    The builder allocs and deallocs these inside the round loop, so the true
    peak depends on the ring depth air-to-aie picks. This is the upper bound.
    RELAY_COLS mirrors the builder-local list of the same name (fused_decode.py,
    just after the relay_l2 declaration) -- it is not module scope, so it is the
    one number here that is restated rather than read.
    """
    g = geometry(fd, b)
    relay_cols = [3, 5, 4][: fd.NDEST]
    rows = [(f"grp_l2 (group {i})", c, g["grp"]) for i, c in enumerate(fd.GRP_PCOL)]
    rows.append(("main_l2", fd.MAIN_PCOL, g["main"]))
    rows += [
        (f"relay_l2 (dest {i})", c, g["payload"]) for i, c in enumerate(relay_cols)
    ]
    return rows


def ingress(fd, b, mfold=1):
    """What one proj core pulls in per layer, and per inner-loop step.

    The `_gemv` inner loop is one `inX` get paired with one `wL2ToL1` get, so
    the per-step ratio is just the two chunk sizes -- and only X scales:

        W chunk = BLOCK_BF16 elements      (a packed q4k block, fixed)
        X chunk = COL_BLOCK * b elements   (b tokens of one column block)

    N_XBLK is the builder's count of those steps for a whole layer on one core.

    mfold is the number of weight blocks a step consumes IN THE ROW DIRECTION,
    i.e. q4k_mmul's MROWS / 32. That is the only fold that helps this ratio: the
    same X serves mfold blocks, so X per weight byte drops by mfold. Folding in
    the CONTRACTION direction (bench_q4k_mm.py --kcol 512) does not help here --
    two column blocks need two different X chunks, so W and X both double and
    the ratio is unchanged.
    """
    w_step = fd.BLOCK_BF16 * BF16 * mfold
    x_step = fd.COL_BLOCK * BF16 * b
    steps = fd.N_XBLK // mfold
    return dict(
        w_step=w_step,
        x_step=x_step,
        w_layer=steps * w_step,
        x_layer=steps * x_step,
        steps=steps,
        # b at which the X chunk overtakes the weight chunk.
        crossover=fd.BLOCK_BF16 * mfold / fd.COL_BLOCK,
        xmt=2 * fd.COL_BLOCK * b,  # xmt_l2 has to widen with it
    )


def report_in(fd, b, cyc0, cyc1, stream_bpc, mfold=1):
    """cyc0 + cyc1*b is the measured cycles/STEP line from bench_q4k_mm.py.

    A step is one inner-loop iteration: mfold weight blocks against one X chunk.
    Pass the line that matches the mfold being modelled -- they are not related
    by a factor of mfold (see the --cyc0/--cyc1 help).
    """
    cycles_per_block = cyc0 + cyc1 * b
    i1, i = ingress(fd, 1, mfold), ingress(fd, b, mfold)
    fold = f", MROWS fold {mfold}" if mfold > 1 else ""
    print(f"\n  in-feed per proj core (X grows with the batch, W does not{fold}):")
    print(f"  {'':22s}{'batch 1':>12}{'batch %d' % b:>12}")
    print(f"  {'-'*46}")
    print(f"  {'W chunk (bytes)':22s}{i1['w_step']:12d}{i['w_step']:12d}")
    print(f"  {'X chunk (bytes)':22s}{i1['x_step']:12d}{i['x_step']:12d}")
    print(
        f"  {'X/W per step':22s}{i1['x_step']/i1['w_step']:12.2f}"
        f"{i['x_step']/i['w_step']:12.2f}"
    )
    print(
        f"  {'X / layer / core (MB)':22s}{i1['x_layer']/1e6:12.2f}"
        f"{i['x_layer']/1e6:12.2f}"
    )
    print(
        f"  {'W / layer / core (MB)':22s}{i1['w_layer']/1e6:12.2f}"
        f"{i['w_layer']/1e6:12.2f}"
    )
    print(
        f"  {'xmt_l2 (KB)':22s}{i1['xmt']*BF16/1024:12.2f}"
        f"{i['xmt']*BF16/1024:12.2f}"
    )
    print(
        f"\n    X overtakes W at batch {i['crossover']:.0f}"
        f"  (BLOCK_BF16 {fd.BLOCK_BF16} * {mfold} / COL_BLOCK {fd.COL_BLOCK})"
    )
    # Same X goes to every core in the column group, so the memtile emits one
    # copy and the switchbox fans it out: the stream that has to carry it is the
    # broadcast source, not one per core.
    x_bpc = i["x_step"] / cycles_per_block
    w_bpc = i["w_step"] / cycles_per_block
    print(
        f"    at {cyc0:.0f} + {cyc1:.1f}*b = {cycles_per_block:.0f} cycles/block"
        f" [measured, bench_q4k_mm.py]:  X {x_bpc:.2f} B/cycle, W {w_bpc:.2f} B/cycle"
    )
    tag = "fits" if x_bpc <= stream_bpc else "OVER"
    print(
        f"    X broadcast needs {x_bpc:.2f} of {stream_bpc:.1f} B/cycle per stream"
        f"  [{tag}]   (stream width is an input, not measured here)"
    )
    # X bytes and cycles are both linear in b, so demand approaches an asymptote
    # rather than growing without bound: COL_BLOCK*2 / cyc1 B/cycle.
    # cyc1 is already the per-step slope for this mfold, so no mfold factor here.
    per_b = fd.COL_BLOCK * BF16
    ceiling = per_b / cyc1
    if ceiling <= stream_bpc:
        print(
            f"    X demand tops out at {ceiling:.2f} B/cycle as b -> inf, under"
            f" {stream_bpc:.1f}: the broadcast never becomes the limit"
        )
    else:
        cap = stream_bpc * cyc0 / (per_b - stream_bpc * cyc1)
        print(
            f"    X demand tops out at {ceiling:.2f} B/cycle as b -> inf, over"
            f" {stream_bpc:.1f}: the broadcast saturates at batch {cap:.0f}"
        )


def report(fd, model, b, verbose):
    g1, g = geometry(fd, 1), geometry(fd, b)
    print(f"\n=== {model}  batch {b}   [WIDEN: one packet per round, B x longer]")
    print(
        f"    PAIR_ROWS={fd.PAIR_ROWS} ROW_BLOCK={fd.ROW_BLOCK} "
        f"N_PAIRS={fd.N_PAIRS} N_GRP={fd.N_GRP} LEADS_PER_GRP={fd.LEADS_PER_GRP} "
        f"HDR={fd.HDR}"
    )
    print(f"    N_ROUNDS={fd.N_ROUNDS} -- unchanged by batching, that is the point\n")

    print(f"  {'':16s}{'batch 1':>10}{'batch %d' % b:>10}{'':4}{'bytes':>9}")
    print(f"  {'-'*49}")
    for k, label in (
        ("pair_pay", "emitter block"),
        ("grp", "group packet"),
        ("main", "main packet"),
        ("payload", "round payload"),
    ):
        print(f"  {label:16s}{g1[k]:10d}{g[k]:10d}{'':4}{g[k]*BF16:9d}")
    hdr1 = fd.HDR / g1["main"]
    hdrb = fd.HDR / g["main"]
    print(
        f"\n  routing header is {100*hdr1:.2f}% of the packet at batch 1, "
        f"{100*hdrb:.3f}% at batch {b}"
    )

    print("\n  buffer descriptors:")
    bad = 0
    for name, tile, elems, wraps, strides in descriptors(fd, b):
        res = check(name, tile, elems, wraps, strides)
        ok = all(o for *_, o in res)
        bad += not ok
        shape = f"[{'x'.join(str(w) for w in wraps)}]" if wraps else "contiguous"
        print(
            f"    {name:16s} {tile:8s} {elems:7d} elem  {shape:12s} "
            f"[{'OK' if ok else 'ILLEGAL'}]"
        )
        if verbose or not ok:
            for k, v, c, o in res:
                print(f"        {k:10s} {v:8d} / {c:<8d} {'' if o else '<-- OVER'}")

    print(
        f"\n  egress L2, per pinned memtile column (upper bound, {L2_BYTES//1024} KB each):"
    )
    per_col = {}
    for name, col, elems in l2_egress(fd, b):
        per_col[col] = per_col.get(col, 0) + elems * BF16
        if verbose:
            print(f"      col {col}  {name:22s} {elems*BF16/1024:8.2f} KB")
    for col in sorted(per_col):
        pct = 100 * per_col[col] / L2_BYTES
        print(
            f"      col {col}  {per_col[col]/1024:8.2f} KB  ({pct:.1f}% of L2)"
            f"  [{'FITS' if per_col[col] <= L2_BYTES else 'OVER'}]"
        )

    cap = max_batch(fd, 4096)
    print(
        f"\n  verdict: {'all descriptors legal' if not bad else '%d ILLEGAL' % bad}"
        f"; largest legal batch on this path: {cap}"
    )


MODELS = {
    "llama-3.2-1b": ("18", None),
    "qwen3-4b": ("30", None),
    "llama-3.2-3b": ("9", None),
    "qwen3-8b": ("8", {"DECODE_STACK": "6144", "DECODE_WGROUP": "9"}),
    "gemma3-4b": ("5", None),
    "phi4-mini": ("18", None),
    "qwen2.5-7b": ("7", None),
    "llama-3.1-8b": ("16", {"DECODE_STACK": "6144", "DECODE_WGROUP": "8"}),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(MODELS))
    ap.add_argument("--batch", type=int, default=16)
    # cycles/step = cyc0 + cyc1*batch, from bench_q4k_mm.py [measured]:
    #   mfold 1 (32x256): 1438 + 109.1*b   -- fitted to 3184 @16 and 4930 @32
    #   mfold 2 (64x256): 2876 + 194.4*b   -- 5986 @16, intercept = 2x the above
    # Note the mfold-2 slope is 194.4, not 2x109.1 = 218.2: the longer row block
    # gives the 2x2 register blocking more to reuse. So the lines are NOT related
    # by a factor of mfold, and the tool does not try to derive one from the other.
    ap.add_argument("--cyc0", type=float, default=1438.0, help="unpack intercept")
    ap.add_argument("--cyc1", type=float, default=109.1, help="mmul slope per batch")
    ap.add_argument(
        "--stream-bpc",
        type=float,
        default=4.0,
        help="bytes/cycle a single AIE2 stream carries (INPUT, not measured here)",
    )
    ap.add_argument(
        "--mfold",
        type=int,
        default=1,
        help="weight blocks per step in the ROW direction (q4k_mmul MROWS/32); "
        "the same X serves all of them. Contraction folding does not help here.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    i2, extra = MODELS[args.model]
    fd = load_builder(args.model, i2, env_extra=extra)

    # b=1 has to reproduce what the builder itself computed, or the model above
    # has drifted from the design.
    g1 = geometry(fd, 1)
    assert g1["grp"] == fd.GRP_ROWS, (g1["grp"], fd.GRP_ROWS)
    assert g1["main"] == fd.MAIN_ROWS, (g1["main"], fd.MAIN_ROWS)
    assert g1["payload"] == fd.PAYLOAD, (g1["payload"], fd.PAYLOAD)

    report(fd, args.model, args.batch, args.verbose)
    report_in(fd, args.batch, args.cyc0, args.cyc1, args.stream_bpc, args.mfold)
    print()


if __name__ == "__main__":
    main()
