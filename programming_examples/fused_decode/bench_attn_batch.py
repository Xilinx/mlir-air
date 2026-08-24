#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""How much of attention could a BATCH amortize? Measured, by removing pieces.

THE QUESTION. Section 5e measured that attention does not amortize over a batch
and is 71% of batch-16 compute, and concluded that the lever is to put tokens in
the mmul's R dimension so the per-KV-block work -- the K/V tile loads and the
aie::transpose -- is hoisted out of the per-token path. That conclusion was
reasoning, not measurement. It is only worth building if the hoistable work is a
large share of the cost, and nobody had priced it.

So price it. Split each call into two parts:

    PER TOKEN     work that exists once per (token, key block) no matter how the
                  kernel is arranged. The online-softmax update, the score
                  epilogue, the y accumulator traffic, the MACs themselves. A
                  batch CANNOT remove any of it.
    PER KV BLOCK  work that depends only on the key block. The K and V tile
                  loads and the transposes. b tokens sharing one block pay it
                  ONCE, so a batch removes (b-1)/b of it.

Then the whole batching lever, however cleverly implemented, is bounded by

    per-token cost at batch b  =  PER_TOKEN + PER_BLOCK / b
    ceiling as b -> infinity   =  PER_TOKEN
    best possible speedup      =  (PER_TOKEN + PER_BLOCK) / PER_TOKEN

METHOD. bench_attn.py's, with knobs. Compile the kernel with a piece #ifdef'd
out and take the bundle delta as that piece's cost. The knobs live in
aie_kernel_utils.h (ATTN_BENCH_NO_TRANSPOSE / _NO_KLOAD / _NO_VLOAD /
_NO_UPDATE / _NO_CORRECT) plus attn_kv.cc's pre-existing SKIP_CALC_L /
SKIP_ATTN_FV. Each builds a NUMERICALLY WRONG kernel; none is defined by
anything that runs, and check_kernels_inert.py proves the guards changed no
shipping code.

UNROLLED, always. Rolled, the contraction loop reports one loop body and gives
the same answer for DH=64 and DH=128 (bench_attn.py's docstring; it cost real
time once already).

WHICH WAY THE ERROR RUNS. Removing work can only make register allocation and
scheduling easier, so each delta OVERSTATES the piece -- sometimes by a lot. 64
vector loads cannot really cost 615 bundles; what the K-load knob also deletes
is the address arithmetic and the load-use dependence chain that was pacing the
schedule. That inflates PER_BLOCK, which inflates the ceiling. The reported
bound on batching is therefore OPTIMISTIC: the real batched kernel lands below
it, so a small ceiling is a genuinely small opportunity.

A PROBE THAT DOES NOT WORK, recorded so it is not retried: splitting the y
rescale pass into traffic and arithmetic by keeping the load/store and dropping
the multiply. `store(load(p))` is a provable self-copy, so the compiler deletes
the whole pass and the variant measures identically to removing it. Any traffic
probe here has to make the stored value depend on something the compiler cannot
fold.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_attn as ba
import bench_q4k_mm as bench
import batch_attn_mask as mask

# Each entry: (label, kernel, symbol, extra -D flags, bucket).
#   "block" = the delta is per-KV-block work a batch can hoist
#   "token" = the delta is per-token work a batch cannot touch
PIECES = [
    ("K tile loads", "attn_qk", "attn_qk_blk", ["-DATTN_BENCH_NO_KLOAD"], "block"),
    ("K transposes", "attn_qk", "attn_qk_blk", ["-DATTN_BENCH_NO_TRANSPOSE"], "block"),
    ("softmax update", "attn_qk", "attn_qk_blk", ["-DATTN_BENCH_NO_UPDATE"], "token"),
    ("V tile loads", "attn_kv", "attn_kv_blk", ["-DATTN_BENCH_NO_VLOAD"], "block"),
    ("y rescale pass", "attn_kv", "attn_kv_blk", ["-DATTN_BENCH_NO_CORRECT"], "token"),
    ("calculate_l", "attn_kv", "attn_kv_blk", ["-DSKIP_CALC_L"], "token"),
    ("attn_fv (all)", "attn_kv", "attn_kv_blk", ["-DSKIP_ATTN_FV"], "token"),
]


def compile_with(src, outdir, model, defs):
    """bench_attn.compile_attn, plus arbitrary extra -D flags.

    Deliberately a copy of the flag list rather than a call with a hook: these
    are the flags the engine ships (-O1 for attention, -O2 elsewhere) and a
    silent divergence between this tool and bench_attn.py would make their
    numbers incomparable, which is the whole point of quoting both.
    """
    peano, inc = bench._peano(), bench._aie_include()
    tag = "".join(sorted(d.replace("-D", "")[:6] for d in defs)) or "full"
    obj = outdir / f"{src}_{tag}.o"
    cmd = [
        str(peano / "bin" / "clang++"),
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        "-DNDEBUG",
        f"-DMODEL_TYPE={model}",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-I",
        str(inc),
        "-I",
        str(HERE / "kernels"),
        "-I",
        str(HERE / "models"),
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-DATTN_BENCH_UNROLL",
        *defs,
        "-O1",
        "-c",
        str(HERE / "kernels" / f"{src}.cc"),
        "-o",
        str(obj),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"compile failed ({src} {defs}):\n{r.stdout}\n{r.stderr[-4000:]}")
    return obj


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="LLAMA_3_2_1B")
    ap.add_argument("--prefix", type=int, default=2048, help="KV context length P")
    ap.add_argument("--layers", type=int, default=16, help="UNI_DEC")
    ap.add_argument("--ghz", type=float, default=1.57)
    ap.add_argument(
        "--batches",
        default="1,2,4,8,16,32",
        help="block sizes to price the ceiling at",
    )
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cache = {}

        def bundles_for(kern, sym, defs):
            key = (kern, tuple(defs))
            if key not in cache:
                cache[key] = ba.pick(
                    ba.bundles(compile_with(kern, td, args.model, defs)), sym
                )
            return cache[key]

        full_qk = bundles_for("attn_qk", "attn_qk_blk", [])
        full_kv = bundles_for("attn_kv", "attn_kv_blk", [])
        rows = []
        for label, kern, sym, defs, bucket in PIECES:
            full = full_qk if kern == "attn_qk" else full_kv
            rows.append((label, kern, bucket, full - bundles_for(kern, sym, defs)))

    total = full_qk + full_kv
    print(f"\nattention, priced piece by piece  [{args.model}, -O1, UNROLLED]")
    print(
        f"  attn_qk_blk {full_qk:6d}   attn_kv_blk {full_kv:6d}   per block {total:6d}"
    )
    print(f"\n  {'piece':18s}{'kernel':12s}{'bundles':>9}{'of call':>9}  amortizes?")
    print(f"  {'-' * 62}")
    for label, kern, bucket, cost in rows:
        full = full_qk if kern == "attn_qk" else full_kv
        tag = {
            "block": "over a batch",
            "token": "NO -- per token",
            "sub": "(breakdown only)",
        }[bucket]
        note = "  <- free, below the noise" if cost <= 0 else ""
        print(f"  {label:18s}{kern:12s}{cost:9d}{cost / full:8.0%}  {tag}{note}")

    # Only the "block" pieces can be hoisted. Everything else -- measured or
    # not -- stays in the per-token path, so charging the UNMEASURED remainder
    # to per-token is the conservative direction for the hoistable share.
    #
    # NEGATIVE deltas are clamped to 0, not summed. A piece that measures
    # negative got FASTER when it was deleted, which is scheduling noise rather
    # than a saving; letting it subtract would silently shrink the hoistable
    # share and flatter the conclusion this file is trying not to flatter.
    per_block = sum(max(0, c) for _, _, b, c in rows if b == "block")
    per_token = total - per_block
    print(
        f"\n  per KV block (hoistable)  {per_block:6d}   {per_block / total:5.0%}"
        f"\n  per token   (floor)       {per_token:6d}   {per_token / total:5.0%}"
    )

    print(f"\n  attention cost per token, if the hoist were perfect")
    print(f"  {'batch':>6}{'bundles':>10}{'vs batch 1':>13}")
    print(f"  {'-' * 29}")
    for b in batches:
        c = per_token + per_block / b
        print(f"  {b:6d}{c:10.0f}{total / c:12.2f}x")
    print(
        f"\n  ceiling as batch -> infinity: {total / per_token:.2f}x"
        f"  ({total} -> {per_token} bundles/token)"
    )

    P, NL = args.prefix, args.layers
    blocks = mask.rounds(P + 1)
    ms = lambda c: c / (args.ghz * 1e9) * 1e3
    per_tok_ms = ms(NL * blocks * total)
    best_ms = ms(NL * blocks * per_token)
    print(
        f"\n  at P={P}, {NL} layers, {blocks} blocks/layer: attention is"
        f" {per_tok_ms:.2f} ms/token now,\n  and no better than {best_ms:.2f} ms/token"
        " with the hoist, per attention CU."
    )
    print(
        "\n  Read the `amortizes?` column first. The hoist can only ever remove\n"
        "  the `over a batch` rows; every other row is paid once per token per\n"
        "  key block whatever the kernel does. Each delta OVERSTATES its piece\n"
        "  (removing work eases scheduling), so the ceiling above is optimistic."
    )


if __name__ == "__main__":
    main()
