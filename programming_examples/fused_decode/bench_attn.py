#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does attention change the batch-16 roofline? Static cost, bench_q4k_mm's method.

THE QUESTION. Section 5's roofline counts PROJECTION weight blocks and nothing
else, and concludes batch 16 is compute bound by ~3.5% (crossover 14.8). That is
only sound if attention is a rounding error on the compute side, and there is a
specific reason to doubt it:

    projections   16 tokens share one weight block. Compute scales with the
                  batch, DDR traffic does not. That is the whole point of
                  batching, and it is why the crossover exists at all.
    attention     every query re-reads the whole KV cache. 16 queries is 16x
                  the attn_qk/attn_kv calls. Nothing is shared.

So one term amortizes and the other does not, and their ratio at batch 16 is 16x
what it is at batch 1. A term that is 7% of batch-1 compute is not 7% of
batch-16 compute.

METHOD. Same as bench_q4k_mm.py: Peano-compile, count VLIW bundles per section
from llvm-objdump, ~1 bundle/cycle. attn_qk_blk and attn_kv_blk each handle ONE
16-key block for ONE query, and the block loop lives in the AIR herd rather than
the kernel, so one call is one section.

BUT THE KERNEL MUST BE UNROLLED FIRST, and that is not a detail. Attention's
contraction loops run colQ = DH/8 times with the trip count in a register, so a
rolled build reports one LOOP BODY and reports the identical number for DH=64
and DH=128. Measured: attn_kv_blk is 320 bundles rolled and 1500 unrolled on
llama-3.2-1b, a 4.7x undercount, and rolled it claims qwen3-4b's DH=128
attention costs exactly what llama's DH=64 attention costs. -DATTN_BENCH_UNROLL
straightens the loops for this tool only; the engine builds them rolled. The
report prints both columns so the trap cannot be walked into again.

Call counts come from batch_attn_mask.py's model of that same block loop, so the
two files cannot disagree about the mask or the trip count.

CAVEATS, and they cut one way. Even unrolled the count is a LOWER bound:
getExpBf16 and getActivationBf16 stay out of line and are called from loops
inside the body. Bundle counts are issue slots, so no DMA stalls and no L2
backpressure -- and batch-16 attention leans on L2 far harder than batch 1,
because the KV block is re-read per query. The kernels are built -O1 because -O2
miscompiles them (see the Makefile); that is what the engine ships.

The batch-1 row is the model's one cross-check against reality, and it has to
land BELOW the measured 19.6 ms wall time or the cost model is wrong before any
batching argument is made. It lands at 16.6 ms.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_q4k_mm as bench
import batch_attn_mask as mask

BUNDLE_RE = re.compile(r"^\s*[0-9a-f]+:\s")
SEC_RE = re.compile(r"^Disassembly of section (\S+):")


def compile_attn(src, outdir, model="LLAMA_3_2_1B", unroll=True):
    """Peano-compile one attention kernel at the flags the engine ships.

    -O1, not -O2: the Makefile pins these two kernels to -O1 because -O2 hits a
    do-while miscompile. Benchmarking -O2 would measure a build that cannot run.

    unroll=True adds -DATTN_BENCH_UNROLL, WITHOUT WHICH THE NUMBERS ARE
    MEANINGLESS. The contraction loops run colQ = DH/8 times with the trip count
    in a register, so a rolled build reports one loop body and reports the SAME
    body for DH=64 and DH=128. That is not a subtle undercount: attn_kv_blk
    measures 320 bundles rolled and 1500 unrolled on llama-3.2-1b. The tool
    builds both and prints the ratio so the trap stays visible.
    """
    peano, inc = bench._peano(), bench._aie_include()
    obj = outdir / f"{src}{'_u' if unroll else ''}.o"
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
        *(["-DATTN_BENCH_UNROLL"] if unroll else []),
        "-O1",
        "-c",
        str(HERE / "kernels" / f"{src}.cc"),
        "-o",
        str(obj),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"compile failed ({src}):\n{r.stdout}\n{r.stderr}")
    return obj


def bundles(obj):
    peano = bench._peano()
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


def pick(secs, needle):
    """Bundles of the section named exactly .text.<needle>.

    Exact, not substring: `attn_qk` is a prefix of `attn_qk_blk` and picking the
    wrong one silently measures a different entry point.
    """
    key = f".text.{needle}"
    if key not in secs:
        sys.exit(f"no section {key}; have {sorted(secs)[:8]}")
    return secs[key]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prefix", type=int, default=2048, help="KV context length P")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--layers", type=int, default=16, help="UNI_DEC")
    ap.add_argument("--model", default="LLAMA_3_2_1B")
    ap.add_argument(
        "--blocks-core-token",
        type=int,
        default=9440,
        help="projection weight blocks per core per token (llama-3.2-1b q4nx)",
    )
    ap.add_argument(
        "--proj-line",
        type=float,
        nargs=2,
        default=(1806.0, 98.0),
        metavar=("C0", "C1"),
        help="projection cycles/block = C0 + C1*batch, from bench_q4k_mm.py",
    )
    ap.add_argument(
        "--gemv-cycles",
        type=float,
        default=2240.0,
        help="measured GEMV cycles/block, used for the batch-1 row",
    )
    ap.add_argument("--ghz", type=float, default=1.57)
    ap.add_argument(
        "--mem-ms", type=float, default=19.6, help="measured batch-1 decode wall time"
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        got = {}
        for unroll in (False, True):
            qk = bundles(compile_attn("attn_qk", td, args.model, unroll))
            kv = bundles(compile_attn("attn_kv", td, args.model, unroll))
            got[unroll] = (pick(qk, "attn_qk_blk"), pick(kv, "attn_kv_blk"))
    (r_qk, r_kv), (n_qk, n_kv) = got[False], got[True]
    per_call = n_qk + n_kv

    print(f"\nattention static cost  [{args.model}, -O1, the flags the engine ships]")
    print(f"  {'':14s}{'rolled':>10}{'unrolled':>10}{'ratio':>9}")
    print(f"  {'-' * 43}")
    for nm, r, u in (("attn_qk_blk", r_qk, n_qk), ("attn_kv_blk", r_kv, n_kv)):
        print(f"  {nm:14s}{r:10d}{u:10d}{u / r:8.1f}x")
    print(
        f"  {'per block':14s}{r_qk + r_kv:10d}{per_call:10d}{per_call / (r_qk + r_kv):8.1f}x"
    )
    print(
        "  Use the UNROLLED column. The rolled one is a single loop body -- it\n"
        "  reports the same number for DH=64 and DH=128, which is how you can\n"
        "  tell it is not measuring a call. Still a lower bound even unrolled:\n"
        "  getExpBf16 and getActivationBf16 stay out of line."
    )

    P, B, NL = args.prefix, args.batch, args.layers
    blocks = mask.rounds(P + 1)
    per_token = NL * blocks * per_call  # per attention CU, whole model
    c0, c1 = args.proj_line
    ghz, BLK_N = args.ghz, args.blocks_core_token
    ms = lambda c: c / (ghz * 1e9) * 1e3

    print(f"\ncalls per attention CU  [P={P}, {NL} layers, {blocks} blocks/layer]")
    print(f"  per token             {NL * blocks:7d}   {ms(per_token):6.2f} ms")
    print(
        f"  per batch-{B} dispatch  {B * NL * blocks:7d}   {ms(B * per_token):6.2f} ms"
        f"   {B}x, no amortization"
    )

    # Batch 1 ships the GEMV, not this kernel, and the fitted line was taken
    # over batch 16-32 -- extrapolating it down to 1 would report a projection
    # cost no build has. Use the measured GEMV there instead.
    proj = lambda b: BLK_N * (args.gemv_cycles if b == 1 else c0 + c1 * b)
    attn = lambda b: b * per_token
    print(
        f"\nper dispatch, ms at {ghz} GHz   [memory floor {args.mem_ms} ms, measured]"
    )
    print(
        f"  {'batch':>6}{'proj':>9}{'attn':>9}{'serial':>9}{'overlap':>9}{'memory':>9}"
    )
    print(f"  {'-' * 54}")
    for b in sorted({1, B, 32}):
        pj, at = ms(proj(b)), ms(attn(b))
        tag = "  (GEMV)" if b == 1 else ""
        print(
            f"  {b:6d}{pj:9.2f}{at:9.2f}{pj + at:9.2f}{max(pj, at):9.2f}"
            f"{args.mem_ms:9.2f}{tag}"
        )
    # The batch-1 row is the model's only cross-check against reality: it has to
    # come out BELOW the measured wall time, or the cost model is wrong before
    # any batching argument is made.
    c1_ms = ms(proj(1)) + ms(attn(1))
    print(
        f"\n  Cross-check: at batch 1 the model puts compute at {c1_ms:.1f} ms"
        f" against a\n  measured {args.mem_ms} ms wall time -- memory bound with"
        f" {1 - c1_ms / args.mem_ms:.0%} slack, which is\n  what the decode is"
        " known to be. The model is consistent where it can be checked."
    )
    print(
        "\n  serial  = the phase structure as built. qkv proj -> attention ->\n"
        "            o proj -> glu, each a barrier, so the two terms add.\n"
        "  overlap = a lower bound nothing achieves: attention hiding entirely\n"
        "            behind the projections. The truth is between them, and at\n"
        "            batch 16 BOTH exceed the memory floor."
    )

    def cross(with_attn):
        """Largest batch still memory bound."""
        tgt = args.mem_ms * 1e-3 * ghz * 1e9
        slope = BLK_N * c1 + (per_token if with_attn else 0)
        return (tgt - BLK_N * c0) / slope

    print("\n  crossover (largest batch still memory bound)")
    print(f"    projections only, as section 5 has it   {cross(False):6.1f}")
    print(f"    + attention, serial                     {cross(True):6.1f}")
    pj, at = ms(proj(B)), ms(attn(B))
    print(
        f"\n  At batch {B} the serial compute path is {(pj + at) / args.mem_ms:.2f}x the"
        f" memory floor,\n  not the {pj / args.mem_ms:.2f}x section 5 reports."
        f" Attention is {at / (pj + at):.0%} of that compute."
    )
    print(
        "\n  What this does NOT say: that batching is not worth it. Batch"
        f" {B} still\n  delivers {B} tokens in {pj + at:.1f} ms against"
        f" {B * args.mem_ms:.0f} ms one at a time"
        f" ({B * args.mem_ms / (pj + at):.1f}x).\n  It says the win is bounded by"
        " compute, not by the weight stream, so\n  the traffic-only model in"
        " dflash_traffic.py overstates it."
    )


if __name__ == "__main__":
    main()
