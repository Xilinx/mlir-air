#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Are the modified production kernels still byte-identical to HEAD? THE GATE.

The DFlash work adds bench-only and batch-only entry points to kernels the
shipping decode compiles. Every one is behind an #ifdef that no shipping build
defines, so every one is supposed to be INERT. "Supposed to" is not a gate: a
stray macro, a header include, or an #ifdef that lands one line off changes the
emitted code without changing any symbol name, and nothing else in this tree
would notice.

WHAT IT COMPARES. For each kernel, compile the worktree source and the `git
show HEAD:` source at the EXACT flags the Makefile ships -- which are not the
same for all of them:

    proj_qmm rms_residual glu rope     -O2   (rope MISCOMPILES at -O1: the
                                              decode hangs on device)
    attn_qk attn_kv                    -O1   (-O2 hits a do-while deadlock)
                                             plus -DDECODE_INLINE_ATTN

then disassemble both and diff. Benchmarking or gating at the wrong -O measures
a build that cannot run, so the flags are read from the Makefile rather than
restated here -- if the Makefile's flags change and this file's parse breaks,
that is the correct failure.

WHY NOT `cmp` ON THE .o. It ALWAYS differs, on unmodified sources, because the
absolute source path is embedded in the object. That false positive is what
made the check get skipped by hand before. Compare `llvm-objdump -d` output
with the header lines (which name the file) dropped.

    python3 check_kernels_inert.py           # exit 0 = all inert
    python3 check_kernels_inert.py -v        # print the first differing lines

Exit code is the gate: 0 inert, 1 something changed.
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

# -O2 kernels, then -O1 attention kernels. Split exactly as the Makefile splits
# them; see the module docstring for why the two levels are not interchangeable.
NONATTN = ["proj_qmm", "rms_residual", "glu", "rope"]
ATTN = ["attn_qk", "attn_kv"]


def makefile_kbase():
    """PEANO_KBASE from the Makefile, with $(...) expanded.

    Parsed rather than duplicated so this gate cannot drift from the build it
    claims to be gating.
    """
    text = (HERE / "Makefile").read_text()
    m = re.search(r"^PEANO_KBASE\s*:=\s*((?:.*\\\n)*.*)$", text, re.M)
    if not m:
        sys.exit("could not find PEANO_KBASE in the Makefile")
    flat = m.group(1).replace("\\\n", " ")
    flat = flat.replace("$(AIEOPT_DIR)", str(bench._aie_include().parent))
    flat = flat.replace("$(srcdir)", str(HERE))
    return flat.split()


def disasm(src_path, out_o, flags):
    peano = bench._peano()
    r = subprocess.run(
        [str(peano / "bin" / "clang++"), *flags, "-c", str(src_path), "-o", str(out_o)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        sys.exit(f"compile failed ({src_path.name}):\n{r.stdout}\n{r.stderr[-4000:]}")
    r = subprocess.run(
        [str(peano / "bin" / "llvm-objdump"), "-d", "--no-show-raw-insn", str(out_o)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        sys.exit(f"objdump failed ({src_path.name}):\n{r.stderr}")
    # Drop the leading header lines: they carry the object's path, which differs
    # for the HEAD copy by construction and says nothing about the code.
    return [ln for ln in r.stdout.splitlines() if not ln.startswith(str(out_o.parent))][
        2:
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--ref", default="HEAD", help="git ref to compare against (default HEAD)"
    )
    args = ap.parse_args()

    kbase = makefile_kbase()
    jobs = [(k, ["-O2"]) for k in NONATTN]
    jobs += [(k, ["-O1", "-DDECODE_INLINE_ATTN"]) for k in ATTN]

    bad = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print(f"\nproduction kernels vs {args.ref}  [Makefile flags, disassembly diff]")
        for kern, extra in jobs:
            rel = f"programming_examples/fused_decode/kernels/{kern}.cc"
            ref_src = subprocess.run(
                ["git", "show", f"{args.ref}:{rel}"],
                capture_output=True,
                text=True,
                cwd=HERE,
            )
            if ref_src.returncode:
                sys.exit(f"git show {args.ref}:{rel} failed:\n{ref_src.stderr}")
            # The reference copy must sit beside the worktree source: these
            # kernels include headers by relative path ("q4k_mm.h",
            # "../models/all_models.h"), so a /tmp copy would not compile -- and
            # a copy that resolves DIFFERENT headers would not be a fair diff.
            ref_path = HERE / "kernels" / f"_head_{kern}.cc"
            ref_path.write_text(ref_src.stdout)
            try:
                a = disasm(ref_path, td / f"{kern}_head.o", kbase + extra)
                b = disasm(
                    HERE / "kernels" / f"{kern}.cc", td / f"{kern}_wt.o", kbase + extra
                )
            finally:
                ref_path.unlink()
            ok = a == b
            opt = extra[0]
            print(
                f"  {kern:14s} {opt:4s} {len(b):6d} lines   {'OK' if ok else 'DIFFERS'}"
            )
            if not ok:
                bad.append(kern)
                if args.verbose:
                    for i, (x, y) in enumerate(zip(a, b)):
                        if x != y:
                            print(f"      first diff at line {i}:")
                            print(f"        {args.ref}: {x}")
                            print(f"        worktree: {y}")
                            break
                    if len(a) != len(b):
                        print(f"      length {len(a)} -> {len(b)}")

    if bad:
        print(f"\n  NOT INERT: {', '.join(bad)}")
        print(
            "  A shipping kernel changed. Either an #ifdef guard is wrong, or the\n"
            "  change is real and belongs in a build the decode does not compile."
        )
        return 1
    print("\n  all inert -- every shipping kernel emits identical code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
