#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Static-cost sweep for the batched q4k matmul (kernels/q4k_mm.h).

THE QUESTION. Decode's projection kernel is a GEMV -- one multiply per unpacked
weight. Speculative decoding (DFlash) needs a batch of tokens per call, so the
cost of a 32x256 weight block splits in two:

    cycles/block = unpack  (fixed, proportional to weights)
                 + mmul    (scales with batch)

The intercept decides whether batching is worth anything: if unpack dominates,
a batch of 16 is nearly free and speculative decoding pays off; if the multiply
dominates, it does not. This measures both by compiling the kernel at several
batches and counting VLIW bundles, which on AIE2P issue at about one per cycle.

METHOD. Each function processes exactly one 32x256 weight block and is compiled
with Q4K_MM_FULL_UNROLL, which straight-lines every loop. Static bundle count is
then the dynamic cycle count, and no trip-count bookkeeping is needed.

(Diffing two builds at different contraction lengths -- the trick DecodeInstsGen
uses to recover per-word slopes -- does NOT work here. That works on runtime
instruction streams; in object code the trip count lives in a register, so both
builds emit an identical body and the difference is zero.)

THE INTERCEPT IS NO LONGER DIRECTLY MEASURABLE. The unpack cannot be fully
unrolled: with the correct two-tile store it crashes the Peano backend
("Register not in mBMs", AIE2P assembly printer), while the rolled form a real
build uses compiles fine. So by default the multiply is unrolled and reported as
cycles, and the unpack is rolled and reported as a static size with an R suffix
-- not a cycle count, and not summable with the multiply. --noperm restores the
old contiguous store, which is numerically WRONG but can be unrolled, and is the
only way to get an exact total / MAC-per-cycle / roofline. Those totals are a
lower bound on the correct kernel, whose rolled unpack is 87 bundles to the
contiguous one's 68.

STATIC COST ONLY. Bundle counts are an issue-slot measure. They do not model
DMA stalls or memory backpressure, so they bound the compute side of the
roofline and nothing else. The kernel also has no numeric gate -- see the
q4k_mm.h header.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_SRC = HERE / "kernels" / "q4k_mm_bench.cc"

# One disassembly line carrying an address is one VLIW bundle.
BUNDLE_RE = re.compile(r"^\s*[0-9a-f]+:\s")
SEC_RE = re.compile(r"^Disassembly of section (\S+):")

BATCHES = (4, 8, 16, 32)


def _peano():
    p = os.environ.get("PEANO_INSTALL_DIR")
    if not p:
        sys.exit("PEANO_INSTALL_DIR not set")
    return Path(p)


def _aie_include():
    """aie_api headers ship inside the mlir_aie wheel."""
    for env in ("AIEOPT_DIR", "MLIR_AIE_INSTALL_DIR"):
        v = os.environ.get(env)
        if v and (Path(v) / "include" / "aie_api").is_dir():
            return Path(v) / "include"
    import importlib.util

    spec = importlib.util.find_spec("aie")
    if spec and spec.submodule_search_locations:
        # The aie package sits at <mlir_aie>/python/aie; the headers at
        # <mlir_aie>/include. Walk up rather than assuming the depth.
        p = Path(list(spec.submodule_search_locations)[0])
        for cand in p.parents:
            if (cand / "include" / "aie_api").is_dir():
                return cand / "include"
    sys.exit("cannot locate aie_api headers; set AIEOPT_DIR")


def compile_bench(
    kcol, outdir, unroll=True, mrows=32, chunks=0, chunk_batch=16, noperm=False
):
    """Compile q4k_mm_bench.cc at a given contraction length -> object path."""
    peano, inc = _peano(), _aie_include()
    obj = outdir / f"q4k_mm_bench_m{mrows}_k{kcol}_c{chunks}_b{chunk_batch}.o"
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
        "-DMODEL_TYPE=LLAMA_3_2_1B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-I",
        str(inc),
        "-I",
        str(HERE / "kernels"),
        "-I",
        str(HERE / "models"),
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DBENCH_KCOL={kcol}",
        f"-DBENCH_MROWS={mrows}",
        "-O2",
        # Wrong layout, but the only unpack that can be fully unrolled.
        *(["-DQ4K_UNPACK_NOPERM"] if noperm else []),
        # One chunked batch per build: the always_inline'd bodies are duplicated
        # rather than shared, so each extra batch is a full extra copy.
        *(
            [f"-DBENCH_CHUNKS={chunks}", f"-DBENCH_CHUNK_BATCH={chunk_batch}"]
            if chunks
            else []
        ),
        "-c",
        str(BENCH_SRC),
        "-o",
        str(obj),
    ]
    if unroll:
        cmd.insert(-4, "-DQ4K_MM_FULL_UNROLL")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(
            f"compile failed (MROWS={mrows} KCOL={kcol} CHUNKS={chunks}):"
            f"\n{r.stdout}\n{r.stderr}"
        )
    return obj


def bundles_per_section(obj):
    """{section name: bundle count} from the disassembly.

    Counting per SECTION, not per symbol: clang outlines the `static inline`
    template instantiations into their own .text.<mangled> sections and leaves
    the extern "C" wrapper as a bare tail-jump. Counting the wrapper would report
    one bundle for the whole kernel. Outlining is convenient here -- it separates
    the unpack cost from the multiply cost without extra work.
    """
    peano = _peano()
    r = subprocess.run(
        [str(peano / "bin" / "llvm-objdump"), "-d", "--no-show-raw-insn", str(obj)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        sys.exit(f"objdump failed:\n{r.stderr}")
    out, cur = {}, None
    for line in r.stdout.splitlines():
        m = SEC_RE.match(line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, 0)
            continue
        if cur and BUNDLE_RE.match(line):
            out[cur] += 1
    return out


def _find_mmul(sections, mrows, kcol, bt):
    """Bundles of the multiply at this batch, wherever clang decided to put it.

    Three places it can land, and which one depends on the body's size against
    clang's outlining heuristic rather than on anything meaningful:

      * .text._ZL9q4k_mmul<MROWS,KCOL,BATCH>        the 2x2 path, outlined
      * .text._ZL15q4k_mmul_small<MROWS,KCOL,BATCH> the 1x4 path, outlined
      * .text.q4k_bench_mmul_b<BATCH>               inlined into the wrapper

    The small-batch bodies are under the threshold and land in the wrapper; the
    batch-16 and -32 bodies are over it and get their own sections. Chasing that
    by hand cost two full sweeps, so look in all three and say which one hit.
    A bare wrapper is ~6 bundles of tail-jump, so anything larger is a real body.
    """
    for nm in (
        f"q4k_mmul_smallILi{mrows}ELi{kcol}ELi{bt}E",
        f"q4k_mmulILi{mrows}ELi{kcol}ELi{bt}E",
    ):
        hits = [v for k, v in sections.items() if nm in k]
        if len(hits) == 1:
            return hits[0]
    w = sections.get(f".text.q4k_bench_mmul_b{bt}", 0)
    if w > 16:
        return w
    sys.exit(
        f"batch {bt}: no outlined mmul section and the wrapper is only {w} "
        f"bundles, so the body is somewhere unexpected"
    )


def _find(sections, *needles):
    """Bundle count of the one section whose name contains every needle."""
    hits = [v for k, v in sections.items() if all(n in k for n in needles)]
    if len(hits) != 1:
        sys.exit(f"expected 1 section matching {needles}, got {len(hits)}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mrows", type=int, default=32, help="weight rows per block")
    # Contraction folded into ONE q4k_mmul call. 256 is a single q4k block; 512
    # folds two, 1024 folds four. More contraction per call gives the 2x2
    # register blocking more to reuse, at the cost of a bigger unpacked bf16 tile
    # in L1 (mrows*kcol*2 bytes: 16 KB at 256, 32 KB at 512, 64 KB at 1024 --
    # the last does not fit a 64 KB tile alongside the activations).
    ap.add_argument(
        "--kcol", type=int, default=256, help="contraction length per mmul call"
    )
    # Full unroll makes compile time grow fast with kcol*batch; a folded sweep
    # may only be affordable at one batch.
    ap.add_argument(
        "--batches",
        default=",".join(str(b) for b in BATCHES),
        help="comma-separated batch sizes to report",
    )
    ap.add_argument("--keep", action="store_true", help="keep the objects")
    # Roofline reference model. Defaults are llama-3.2-1B q4nx decode, all
    # measured: 9440 blocks/core/token is the count in proj_qmm.cc's
    # reduction-cache comment; 2240 cycles is 140 bundles x 16 iterations per
    # block from the fused_decode README; 1.57 GHz is back-solved from the
    # qwen2.5-3B intercept in that README and independently reproduces
    # llama-1B; 19.6 ms/token was measured on a Krackan NPU2 box.
    ap.add_argument("--model", default="llama-3.2-1b q4nx")
    ap.add_argument(
        "--blocks", type=int, default=9440, help="weight blocks per core per token"
    )
    ap.add_argument(
        "--gemv-cycles", type=int, default=2240, help="batch-1 GEMV cycles per block"
    )
    ap.add_argument("--clock", type=float, default=1.57, help="core clock GHz")
    ap.add_argument(
        "--decode-ms",
        type=float,
        default=19.6,
        help="measured batch-1 decode ms/token (the memory floor)",
    )
    ap.add_argument(
        "--noperm",
        action="store_true",
        help="build the unpack with the OLD contiguous store, which is "
        "numerically wrong but can be fully unrolled -- the only way to get an "
        "exact intercept and hence total / MAC-per-cycle. Same standing as "
        "Q4_SFIX_MODE=2 in q4_k.h.",
    )
    ap.add_argument(
        "--chunks",
        type=int,
        default=0,
        help="also bench q4k_mm_chunked with N contraction chunks through ONE "
        "scratch, against N x (unpack + mmul). Settles whether splitting the "
        "contraction costs anything -- see the q4k_mm_chunked header comment.",
    )
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="q4k_mm_"))
    try:
        kcol = args.kcol
        sec = bundles_per_section(
            compile_bench(
                kcol,
                tmp,
                mrows=args.mrows,
                chunks=args.chunks,
                chunk_batch=int(args.batches.split(",")[0]),
                noperm=args.noperm,
            )
        )
        # The unpack is only UNROLLED under --noperm; with the correct two-tile
        # store the unrolled form crashes the Peano backend (see the q4k_mm.h
        # header). Rolled, it is inlined into the extern "C" wrapper instead of
        # being outlined, so the section to match differs -- and, more to the
        # point, the number means a different thing: a rolled static size, not a
        # cycle count. Everything downstream of the intercept is gated on that.
        # Match the FULL mangled template signature where it exists: "Li32E"
        # alone is ambiguous when MROWS and BATCH are both 32.
        if args.noperm:
            unpack = _find(sec, f"q4k_unpack_blockILi{args.mrows}ELi{kcol}E")
        else:
            unpack = _find(sec, "q4k_bench_unpack")
        exact = args.noperm

        print()
        print(f"q4k batched matmul -- per {args.mrows}x{kcol} weight block, one core")
        if exact:
            print(
                "(fully unrolled, so static bundles == dynamic cycles;"
                " ~1 bundle/cycle)"
            )
            print("UNPACK LAYOUT IS WRONG (--noperm): cost attribution only.")
        else:
            print("mmul is unrolled -> cycles. UNPACK IS ROLLED -> static size,")
            print("NOT a cycle count, so no total / MAC-per-cycle is reported.")
            print("Use --noperm for those, at the cost of a wrong unpack layout.")
        print()
        hdr = (
            f"  {'batch':>5}  {'unpack':>7}  {'mmul':>7}  {'total':>7}  "
            f"{'MAC/cyc':>8}  {'cyc/token':>9}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        rows = []
        for bt in [int(x) for x in args.batches.split(",")]:
            mm = _find_mmul(sec, args.mrows, kcol, bt)
            tot = unpack + mm
            macs = args.mrows * kcol * bt
            rows.append((bt, tot, mm))
            if exact:
                print(
                    f"  {bt:5d}  {unpack:7d}  {mm:7d}  {tot:7d}  "
                    f"{macs/tot:8.1f}  {tot/bt:9.1f}"
                )
            else:
                print(
                    f"  {bt:5d}  {unpack:6d}R  {mm:7d}  {'--':>7}  "
                    f"{'--':>8}  {'--':>9}"
                )
        print()
        if not exact:
            print("  R = rolled static size, not cycles.")
            print()
        # A folded run measures a call covering kcol/256 q4k blocks. Normalise so
        # folded and unfolded runs are directly comparable -- otherwise a fold-2
        # run reads as twice as expensive when it is in fact cheaper per weight.
        fold = kcol // 256
        if fold > 1:
            print(f"  per 32x256 q4k block (this run folds {fold} of them per call):")
            for bt, tot, mm in rows:
                print(
                    f"    batch {bt:2d}: unpack {unpack/fold:6.0f}  "
                    f"mmul {mm/fold:6.0f}  total {tot/fold:6.0f}  "
                    f"cycles/token {tot/fold/bt:5.1f}"
                )
            print()

        if args.chunks:
            n = args.chunks
            print(
                f"  {n} contraction chunks through ONE {args.mrows}x{kcol} scratch,"
                f" accumulated into one C:"
            )
            print(
                f"    {'batch':>5}  {'chunked':>8}  {'N x separate':>12}"
                f"  {'delta':>8}"
            )
            # One chunked batch per build (it is the expensive one to compile):
            # the first entry of --batches.
            bt, tot, _mm = rows[0]
            # always_inline puts the whole thing in the extern "C" wrapper's own
            # section, so match that rather than a mangled template.
            ch = _find(sec, "q4k_bench_chunked")
            sep = n * tot
            print(f"    {bt:5d}  {ch:8d}  {sep:12d}  {100*(ch-sep)/sep:+7.1f}%")
            print(
                "    A delta near zero means splitting the contraction is free,"
                " so the\n    large-scratch fold buys nothing the small one cannot."
            )
            print()

        if len(rows) >= 2:
            (b0, t0, m0), (b1, t1, m1) = rows[0], rows[-1]
            slope = (t1 - t0) / (b1 - b0)
            icept = t0 - slope * b0
            print(
                f"  cycles/block = {icept:.0f} + {slope:.1f} x batch   "
                f"[fit over batch {b0}-{b1}]"
            )
            print(f"  unpack-only measured directly = {unpack}")
            print()
            print(f"  multiply overtakes unpack at batch {icept/slope:.1f}")
            print()
            for bt, tot, _ in rows:
                print(
                    f"    batch {bt:2d}: unpack {100*unpack/tot:4.1f}% of block, "
                    f"{tot/bt:.1f} cycles/token vs {args.gemv_cycles} for the batch-1 GEMV"
                )
            print()

            # ---- roofline ----------------------------------------------------
            # Weights are read once per call whatever the batch, so the memory
            # side is flat and the compute side grows with batch. Where they
            # cross is the largest batch that is still free.
            print(
                f"  roofline vs {args.model}  [blocks/core/token {args.blocks}, "
                f"{args.clock} GHz, decode {args.decode_ms} ms/token measured]"
            )
            cyc_to_ms = 1e3 / (args.clock * 1e9)
            # --kcol folds several 32x256 q4k blocks into one call, so there are
            # proportionally fewer calls per token. Normalise or a folded sweep
            # reports a compute time scaled by the fold factor.
            calls = args.blocks / (kcol / 256)
            base_c = args.blocks * args.gemv_cycles * cyc_to_ms
            print(
                f"    batch  1 (today)  compute {base_c:5.1f} ms   "
                f"memory {args.decode_ms:5.1f} ms   -> memory bound"
            )
            for bt, tot, _ in rows:
                c = calls * tot * cyc_to_ms
                bound = "compute" if c > args.decode_ms else "memory"
                print(
                    f"    batch {bt:2d}          compute {c:5.1f} ms   "
                    f"memory {args.decode_ms:5.1f} ms   -> {bound} bound"
                )
            # compute(M) == memory  =>  calls*(icept + slope*M)*cyc_to_ms = decode_ms
            xover = (args.decode_ms / (calls * cyc_to_ms) - icept) / slope
            print()
            print(
                f"    crossover at batch {xover:.1f}  "
                f"(largest batch still memory bound)"
            )
            print()

        print("  Reference [measured, from the tree]:")
        print("    decode GEMV, batch 1    2240 cycles/block   3.7 MAC/cycle/core")
        print("    prefill aie::mmul                          98  MAC/cycle/core")
        print()
    finally:
        if args.keep:
            print(f"objects kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
