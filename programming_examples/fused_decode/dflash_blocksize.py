#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What block size should DFlash use? Priced against max(compute, memory).

dflash_traffic.py prices a DFlash iteration by WEIGHT TRAFFIC alone, on the
premise that decode is memory bound. At batch 1 it is. At batch 16 it is not,
and not by a little: bench_attn.py measures the serial compute path at 3.6x the
memory floor, because attention does not amortize over a batch the way the
projections do (section 5e). A traffic-only model therefore reports a speedup
the hardware cannot deliver, and reports it as increasing in the block size
when the real curve turns over.

So: price each pass as max(memory, compute), sweep the block size, and see where
the maximum actually is.

    memory(b)   weight bytes / bandwidth        -- flat in b, that is the point
    compute(b)  proj_blocks * (c0 + c1*b)       -- amortizes over b
                + b * attn_blocks * attn_cost   -- does NOT amortize
    pass(b)     max(memory, compute)

    iteration   verify(b) + draft(b)
    baseline    one target pass at batch 1, per token

ACCEPTANCE IS THE OTHER HALF, and it moves with the block size too -- a bigger
block cannot help if the drafter's tokens are rejected. Modelled the standard
speculative-decoding way, with a per-token acceptance probability alpha:

    E[accepted per iteration] = (1 - alpha^(b+1)) / (1 - alpha)

which saturates at 1/(1-alpha) however large b gets. alpha is an INPUT, not a
prediction -- like tau in dflash_traffic.py, it depends on the drafter and on
how far quantization moved the target, and neither is measured here. The default
is set so that b=16 reproduces the tau=6 the rest of the analysis assumes, which
makes this sweep comparable to it rather than independently optimistic.

EVERY COMPUTE CONSTANT IS MEASURED, and all of them are cited on the flags:
the projection line from bench_q4k_mm.py, the attention cost from bench_attn.py
(UNROLLED -- the rolled number is a loop body), the GEMV from the tree. Sizes
come off the builder, as dflash_traffic.py does it.

WHAT IT INHERITS. Bundle counts are issue slots: no DMA stalls, no L2
backpressure. Compute is taken as proj + attn SERIAL, which is the phase
structure as built. Both bounds are quoted so the conclusion can be checked
against the optimistic end too.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash_traffic as dt

BF16 = 2


def alpha_for(tau, b):
    """Invert E[accepted] = (1-a^(b+1))/(1-a) for a, by bisection.

    Used to calibrate the sweep against the tau the rest of the doc assumes,
    so this file and dflash_traffic.py are talking about the same drafter.
    """
    lo, hi = 1e-6, 1 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1 - mid ** (b + 1)) / (1 - mid) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def accepted(alpha, b):
    return (1 - alpha ** (b + 1)) / (1 - alpha)


def proj_blocks_per_core(fd, extra_bytes=0):
    """Packed weight blocks one proj core walks per token.

    Read off the builder rather than restated: the body is UNI_DEC layers of
    W_LAYER bf16, the head is UNI_LM chunks of VOCAB_W_BLOCKS packed blocks, and
    the cores are the NCX x NCY grid. extra_bytes carries anything outside the
    _MODELS geometry -- the drafter's fc linear, which the builder knows nothing
    about.
    """
    elems = (dt.body_bytes(fd) + dt.head_bytes(fd) + extra_bytes) / BF16
    return elems / fd.BLOCK_BF16 / (fd.NCX * fd.NCY)


def attn_blocks_per_token(fd):
    """attn_qk_blk/attn_kv_blk calls per attention CU per token, all layers."""
    return fd.UNI_DEC * ((fd.ATTN_MAXL + 15) // 16)


class Pass:
    """One model pass -- verify or draft -- priced both ways at any batch."""

    def __init__(self, name, fd, args, extra_bytes=0):
        self.name = name
        self.bytes = dt.body_bytes(fd) + dt.head_bytes(fd) + extra_bytes
        self.pblk = proj_blocks_per_core(fd, extra_bytes)
        self.ablk = attn_blocks_per_token(fd)
        self.a = args

    def mem_ms(self):
        return self.bytes / 1e9 / self.a.bw * 1e3

    def attn_cycles(self, b):
        """Attention cycles per attention CU for a batch-b pass.

        Two terms, because attention has two kinds of work (bench_attn_batch.py
        measures the split):

            per token       b * hoistable-free cost. The online-softmax update
                            and the y accumulator traffic. Nothing shares it.
            per KV block    paid ONCE for the whole batch, if and only if the
                            kernel is restructured to put tokens in the mmul's
                            R dimension. The K and V tile loads.

        --attn-hoistable 0 (the default) is the kernel AS BUILT: nothing is
        shared, so the whole cost multiplies by b. Passing the measured
        hoistable share prices the batched kernel that does not exist yet.
        """
        floor = self.a.attn_cycles_total - self.a.attn_hoistable
        return self.ablk * (b * floor + self.a.attn_hoistable)

    def compute_ms(self, b, serial=True):
        pj = self.pblk * self.a.table[b]
        at = self.attn_cycles(b)
        cy = pj + at if serial else max(pj, at)
        return cy / (self.a.ghz * 1e9) * 1e3

    def ms(self, b, serial=True):
        return max(self.mem_ms(), self.compute_ms(b, serial))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", default="qwen3-4b")
    ap.add_argument("--draft", default="qwen3-4b-draft")
    ap.add_argument("--vocab-chunk-i2", default="30")
    ap.add_argument("--taps", type=int, default=5, help="fc taps, sizes the fc linear")
    ap.add_argument("--bw", type=float, default=46.0, help="sustained GB/s [measured]")
    ap.add_argument("--ghz", type=float, default=1.57)
    ap.add_argument(
        "--proj-table",
        default="1:2240,4:2998,8:2327,16:3763,32:4942",
        help="MEASURED projection cycles per weight block at each batch "
        "[bench_q4k_mm.py --batches 4,8,16,32 --noperm; batch 1 is the GEMV]. "
        "A table, not a fit: the cost is NOT linear in the batch, because the "
        "kernel changes shape -- batch 4 uses aie::mmul<4,8,8> and costs more "
        "in total than batch 8 does. Only the batches listed here are swept, "
        "because those are the ones that have been measured.",
    )
    ap.add_argument(
        "--attn-cycles",
        type=float,
        dest="attn_cycles_total",
        default=4260.0,
        help="attn_qk_blk+attn_kv_blk bundles per 16-key block, UNROLLED "
        "[measured, bench_attn.py --model QWEN3_4B; 2368 for llama-3.2-1b]",
    )
    ap.add_argument(
        "--attn-hoistable",
        type=float,
        default=0.0,
        help="of --attn-cycles, the bundles that are per-KV-BLOCK rather than "
        "per token and so could be shared across the batch by a kernel that "
        "does not exist yet [measured, bench_attn_batch.py: 1503 for qwen3-4b, "
        "635 for llama-3.2-1b]. The default 0 prices the kernel AS BUILT.",
    )
    ap.add_argument(
        "--tau-at",
        type=float,
        nargs=2,
        default=(6.0, 16.0),
        metavar=("TAU", "B"),
        help="calibrate acceptance so block size B yields TAU accepted tokens "
        "(default matches the rest of the analysis)",
    )
    ap.add_argument(
        "--overlap",
        action="store_true",
        help="price compute as "
        "max(proj, attn) instead of proj+attn -- an unattainable bound",
    )
    args = ap.parse_args()
    args.table = {
        int(k): float(v)
        for k, v in (kv.split(":") for kv in args.proj_table.split(","))
    }

    tgt_fd = dt.load(args.target, args.vocab_chunk_i2)
    drf_fd = dt.load(args.draft, args.vocab_chunk_i2)
    fc = dt.fc_bytes(drf_fd, args.taps, q4nx=True)
    verify = Pass("verify", tgt_fd, args, 0)
    draft = Pass("draft", drf_fd, args, fc)
    serial = not args.overlap

    tau0, b0 = args.tau_at
    alpha = alpha_for(tau0, int(b0))

    print(
        f"\nDFlash block size, priced against max(compute, memory)"
        f"  [target {args.target}, draft {args.draft}]"
    )
    print(
        f"  {args.bw} GB/s, {args.ghz} GHz, compute ="
        f" {'proj + attn (serial, as built)' if serial else 'max(proj, attn) (overlap bound)'}"
    )
    print(
        f"  acceptance alpha = {alpha:.4f}, calibrated so block {int(b0)} gives"
        f" tau = {tau0}"
    )
    print(
        f"\n  memory floor per pass:  verify {verify.mem_ms():6.1f} ms"
        f"   draft {draft.mem_ms():6.1f} ms   [flat in block size]"
    )
    print(
        f"  blocks/core/token:      verify {verify.pblk:6.0f}"
        f"      draft {draft.pblk:6.0f}"
    )
    print(
        f"  attn blocks/CU/token:   verify {verify.ablk:6d}"
        f"      draft {draft.ablk:6d}"
    )

    base = verify.ms(1, serial)
    print(f"\n  baseline: one token, batch 1 = {base:.1f} ms  ({1e3/base:.1f} tok/s)")

    print(
        f"\n  {'blk':>4}{'verify':>9}{'draft':>8}{'iter':>9}{'tau':>7}"
        f"{'ms/tok':>9}{'tok/s':>8}{'speedup':>9}{'bound':>8}"
    )
    print(f"  {'-' * 71}")
    best = None
    for b in sorted(args.table):
        v, d = verify.ms(b, serial), draft.ms(b, serial)
        it = v + d
        tau = accepted(alpha, b)
        per = it / tau
        sp = base / per
        bound = "mem" if verify.compute_ms(b, serial) <= verify.mem_ms() else "compute"
        star = ""
        if best is None or sp > best[1]:
            best, star = (b, sp), ""
        print(
            f"  {b:4d}{v:9.1f}{d:8.1f}{it:9.1f}{tau:7.2f}{per:9.2f}"
            f"{1e3/per:8.1f}{sp:8.2f}x{bound:>8}{star}"
        )

    print(f"\n  best block size {best[0]} at {best[1]:.2f}x")
    tb = min(args.table, key=lambda x: abs(x - b0))
    v16 = verify.ms(tb, serial)
    print(
        f"  for comparison, block {tb} gives"
        f" {base / ((verify.ms(tb, serial) + draft.ms(tb, serial)) / accepted(alpha, tb)):.2f}x"
        f" -- and its verify pass is {v16 / verify.mem_ms():.1f}x its memory floor,"
        f"\n  which is the work the traffic-only model does not charge for."
    )
    print(
        "\n  Read the `bound` column: while a pass is memory bound, a bigger block\n"
        "  is nearly free and the speedup climbs. Once it is compute bound the\n"
        "  extra tokens are paid for in full, and the only thing still improving\n"
        "  is acceptance -- which saturates at 1/(1-alpha)."
    )


if __name__ == "__main__":
    main()
