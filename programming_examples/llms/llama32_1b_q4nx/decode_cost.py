#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Where a decode dispatch's time actually goes, as a function of batch.

THE NUMBER THIS EXISTS FOR. `dflash_blocksize.py` prices a pass as
`max(memory, compute)`, compute = projection + attention, and off measured
kernel cycles it concludes that going from batch 1 to batch 8 costs almost
nothing: the weights are read once either way, so a pass that is memory bound at
batch 1 stays memory bound, and the whole layer body should come in at 1.04x.
Measured on device it is **3.74x**, which is what holds the batch-8 verify to
2.2-2.5x instead of the 8x a shared weight stream would give. Every block-size
conclusion in docs/DFlashFeasibility.md rests on that term.

WHAT IT SEPARATES, and why a single dispatch time cannot. A dispatch is

    t(batch, layers, ctx)  =  fixed
                           +  layers * per_layer(batch)
                           +  lm_head(batch)
                           +  ctx * attn_slope(batch)

and only the last of those had ever been swept. Four template families per batch
pin the rest -- see `decompose()`. Measured, llama-3.2-1b, ctx 8 -> 1800:

    batch    fixed  per layer      x16  attention   lm head    total
        1     1.71      0.827    13.23       1.43      4.32    20.70
        8     2.07      3.089    49.43      11.95     12.34    75.78
    scaling   1.21x      3.74x               8.33x     2.86x    3.66x

At batch 1 the model is right: 0.827 ms/layer measured against a 0.826 ms
memory floor. The failure is specific to batch > 1, it is in the LAYER BODY and
not in attention, and it is large enough to invert the document's headline --
attention is **16%** of a batch-8 dispatch here, not the 69% the roofline says.
Localized, the device spends ~10450 cycles on a 32x256 weight block at batch 8
where `bench_q4k_mm.py` measures 2327. That gap is not yet explained.

CONTEXT IS SYNTHETIC AND THIS IS NEVER A CORRECTNESS GATE. The KV cache is
seeded with arbitrary values to the requested depth -- what matters is how many
keys the attention loop walks, not what is in them. `make verify` is the gate.

    python3 decode_cost.py                       # dispatch cost, batch 1 and 8
    python3 decode_cost.py --decompose 1:nl16_b1_L:nolm5_b1_L:5:lm16_b1_L,\\
                                       8:nolm16_b8_L:nolm5_b8_L:5:decode_b8_L

Every family is a PAIR at <prefix><N> and <prefix><N-1> beside fused_decode.py
(DecodeInstsGen needs the slope). The four the table above came from:

    for L in 2048 2047; do
      UNI_WAVE_HI=16 ./build_template.sh 1 $L        # -> nl16_b1     16 layers
      UNI_WAVE_HI=5  ./build_template.sh 1 $L        # -> nolm5_b1     5 layers
      DECODE_NO_LM_WAVES=0 ./build_template.sh 1 $L  # -> lm16_b1    16 + head
      DECODE_NO_LM_WAVES=1 ./build_template.sh 8 $L  # -> nolm16_b8   16 layers
      UNI_WAVE_HI=5 DECODE_NO_LM_WAVES=1 ./build_template.sh 8 $L
    done                                             # -> nolm5_b8     5 layers
    # RENAME each pair the moment it is built. build_template.sh ALWAYS writes
    # decode_b<B>_L<N>, so leaving one there silently replaces the model
    # template the correctness gates read.

USE UNI_WAVE_HI, NOT DECODE_NO_LM_WAVES, TO DROP THE HEAD. The latter is gated
on `BATCH > 1` in fused_decode.py, so at batch 1 it is a no-op and the build
still runs the vocab waves -- the two batch-1 templates come out byte-identical.
A per-layer number derived from such a pair absorbs the entire LM head into the
layer difference and reads 1.242 ms/layer instead of 0.827.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def time_dispatch(inf, np, prefix, batch, ctx_list, iters, warmup):
    """Median dispatch ms at each context, for one template family.

    One decoder for the whole sweep: constructing it loads 1.8 GB of weight BOs
    and that cost is per session, not per dispatch. The KV is re-seeded per
    context rather than grown, so no point's timing includes another's.
    """
    maxctx = max(ctx_list)
    dec = inf.FusedDecoder(maxctx + batch, batch=batch, template_prefix=prefix or None)
    rng = np.random.default_rng(0)
    out = {}
    try:
        for ctx in ctx_list:
            # Arbitrary K/V: the attention loop walks the same number of keys
            # whatever is in them, and this is a latency measurement. Shape is
            # seed_kv's [16 layers, P, DK_TOT_A].
            fk = rng.standard_normal((16, ctx, dec.DK_TOT_A), dtype=np.float32)
            fv = rng.standard_normal((16, ctx, dec.DK_TOT_A), dtype=np.float32)
            dec.seed_kv(fk, fv, ctx)
            tok = [1000 + t for t in range(batch)] if batch > 1 else 1000
            for _ in range(warmup):
                dec.dispatch(tok, ctx)
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                dec.dispatch(tok, ctx)
                ts.append((time.perf_counter() - t0) * 1e3)
            out[ctx] = statistics.median(ts)
    finally:
        dec.close()
    return out


def slope(pts):
    """(intercept, ms per key) from the two extreme contexts.

    Two points and not a fit: the interesting quantity is the difference
    between the ends, and a least-squares line over four points that are nearly
    collinear reports the same thing with more machinery.
    """
    xs = sorted(pts)
    lo, hi = xs[0], xs[-1]
    if hi == lo:
        return pts[lo], 0.0
    m = (pts[hi] - pts[lo]) / (hi - lo)
    return pts[lo] - m * lo, m


def decompose(args):
    """Split a dispatch into fixed / per-layer / attention / lm-head, per batch.

    Four templates per batch, and each one is there to remove one unknown:

        deep      UNI_DEC layers, NO vocab waves
        shallow   n layers,       NO vocab waves   -> per-layer, then fixed
        withlm    UNI_DEC layers, vocab waves ON   -> lm head = withlm - deep
        the ctx sweep                              -> attention, from the slope

    The vocab waves are the trap. `DECODE_NO_LM_WAVES` is a NO-OP at batch 1 --
    fused_decode.py gates it on `BATCH > 1` -- so a batch-1 "no LM head" build
    made that way still runs the head, and a per-layer number derived from it
    quietly absorbs the whole head into the layer difference. Use
    `UNI_WAVE_HI=<UNI_DEC>`, which works at every batch. Checked here rather
    than trusted: a family whose logits come back all-zero did not run the head.
    """
    import numpy as np
    import llama32_1b_q4nx_inference as inf

    ctxs = [int(c) for c in args.ctx.split(",") if c.strip()]
    lo, hi = min(ctxs), max(ctxs)
    print(f"\ndecode cost decomposed  [llama-3.2-1b, median of {args.iters}]")
    print(f"  attention taken as the ctx {lo} -> {hi} difference\n")
    print(
        f"  {'batch':>5} {'fixed':>8} {'per layer':>10} {'x16':>8} "
        f"{'attention':>10} {'lm head':>9} {'total':>8}"
    )
    rows = []
    for spec in args.decompose.split(","):
        b, deep, shallow, ns, withlm = spec.split(":")
        b, ns = int(b), int(ns)
        td = time_dispatch(inf, np, deep, b, ctxs, args.iters, args.warmup)
        ts = time_dispatch(inf, np, shallow, b, ctxs, args.iters, args.warmup)
        tl = time_dispatch(inf, np, withlm, b, ctxs, args.iters, args.warmup)
        nl = 16
        per = (td[lo] - ts[lo]) / (nl - ns)
        fixed = ts[lo] - ns * per
        lm = tl[lo] - td[lo]
        attn = tl[hi] - tl[lo]
        rows.append((b, fixed, per, per * nl, attn, lm, tl[hi]))
        print(
            f"  {b:>5} {fixed:>8.2f} {per:>10.3f} {per*nl:>8.2f} "
            f"{attn:>10.2f} {lm:>9.2f} {tl[hi]:>8.2f}"
        )
    if len(rows) > 1:
        b0 = rows[0]
        print(
            f"\n  how each term scales from batch {b0[0]} (8x the tokens would be 8.00x):"
        )
        print(
            f"  {'batch':>5} {'fixed':>8} {'layers':>10} {'attention':>10} {'lm head':>9} {'total':>8}"
        )
        for r in rows[1:]:
            print(
                f"  {r[0]:>5} {r[1]/b0[1]:>7.2f}x {r[3]/b0[3]:>9.2f}x "
                f"{r[4]/b0[4]:>9.2f}x {r[5]/b0[5]:>8.2f}x {r[6]/b0[6]:>7.2f}x"
            )
        print(
            f"\n  share of the batch {rows[-1][0]} dispatch: layers "
            f"{rows[-1][3]/rows[-1][6]*100:.0f}%, attention "
            f"{rows[-1][4]/rows[-1][6]*100:.0f}%, lm head "
            f"{rows[-1][5]/rows[-1][6]*100:.0f}%, fixed "
            f"{rows[-1][1]/rows[-1][6]*100:.0f}%"
        )
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--batch",
        default="1,8",
        help="comma-separated batches to time; each needs its own template family",
    )
    ap.add_argument("--ctx", default="8,512,1024,1800")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--prefix",
        default="",
        help="template family override, e.g. decode_b8_L. Default is the "
        "driver's own choice for the batch.",
    )
    ap.add_argument(
        "--layers",
        default="",
        help="prefix:nlayers pairs to separate per-layer cost from the fixed "
        "term, e.g. 'decode_b8_L:16,draft5_b8_L:5'. Both families must be "
        "built at the same ATTN_MAXL and differ ONLY in UNI_WAVE_HI.",
    )
    ap.add_argument(
        "--decompose",
        default="",
        help="batch:deep:shallow:nshallow:withlm -- four template families that "
        "split a dispatch into fixed / per-layer / attention / lm head. Repeat "
        "comma-separated for several batches. See the recipe in the module "
        "docstring; the LM-head family is the ONLY one built with the vocab "
        "waves on.",
    )
    args = ap.parse_args()

    if args.decompose:
        return decompose(args)

    import numpy as np
    import llama32_1b_q4nx_inference as inf

    ctxs = [int(c) for c in args.ctx.split(",") if c.strip()]
    batches = [int(b) for b in args.batch.split(",") if b.strip()]

    if args.layers:
        fams = [
            (p, int(n)) for p, n in (kv.split(":") for kv in args.layers.split(","))
        ]
        print(f"\ndecode cost by LAYER COUNT  [llama-3.2-1b, batch {batches[0]}]")
        rows = []
        for pfx, nl in fams:
            pts = time_dispatch(inf, np, pfx, batches[0], ctxs, args.iters, args.warmup)
            rows.append((pfx, nl, pts))
            print(
                f"  {pfx:18s} {nl:2d} layers  "
                + "  ".join(f"ctx {c}: {pts[c]:7.2f} ms" for c in ctxs)
            )
        if len(rows) >= 2:
            (_, n0, p0), (_, n1, p1) = rows[0], rows[1]
            for c in ctxs:
                per = (p0[c] - p1[c]) / (n0 - n1)
                fixed = p1[c] - n1 * per
                print(
                    f"  ctx {c:5d}: per layer {per:6.3f} ms   "
                    f"fixed + lm head {fixed:7.2f} ms"
                )
        return 0

    print(f"\ndecode dispatch cost  [llama-3.2-1b, median of {args.iters}]")
    per_batch = {}
    for b in batches:
        pts = time_dispatch(inf, np, args.prefix, b, ctxs, args.iters, args.warmup)
        per_batch[b] = pts
        b0, m = slope(pts)
        print(
            f"\n  batch {b}: " + "  ".join(f"ctx {c}: {pts[c]:7.2f} ms" for c in ctxs)
        )
        print(
            f"    context-free {b0:7.2f} ms   attention {m*1e3:6.2f} us/key"
            f"   ({m * max(ctxs) / pts[max(ctxs)] * 100:.0f}% of the dispatch at "
            f"ctx {max(ctxs)})"
        )

    # The decomposition the block-size model gets wrong. Attention is taken out
    # first, from the measured slope, so what is left is the projection path and
    # everything that rides with it.
    if 1 in per_batch and len(per_batch) > 1:
        c = max(ctxs)
        a1 = slope(per_batch[1])[1] * c
        n1 = per_batch[1][c] - a1
        print(f"\n  non-attention cost at ctx {c}, and how it scales:")
        print(f"    batch  1: {n1:7.2f} ms")
        for b in batches:
            if b == 1:
                continue
            ab = slope(per_batch[b])[1] * c
            nb = per_batch[b][c] - ab
            P = (nb - n1) / (b - 1)
            print(
                f"    batch {b:2d}: {nb:7.2f} ms  ->  fixed {n1 - P:6.2f} ms + "
                f"{P:5.2f} ms per token   ({P / n1 * 100:.0f}% of a batch-1 "
                f"decode is paid per token)"
            )
        print(
            "\n  dflash_blocksize.py prices the same term at 0.074 ms per token\n"
            "  (9440 blocks/core/token x (2327-2240) cycles / 1.57 GHz)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
