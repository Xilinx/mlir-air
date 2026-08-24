#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Is the batching work still a NO-OP at DECODE_BATCH=1? Run after every step.

The batched decode is being wired into the shipping builder in place. Every
change is supposed to be inert at batch 1 -- that is the only reason it is safe
to keep working on a file the LLM models compile. "Supposed to" is not a gate:
this builds the module from `git show <ref>:fused_decode.py` and from the
worktree, both at batch 1, and diffs the emitted MLIR.

BOTH MODELS, ALWAYS. This has already caught a leaked constant that folded away
on qwen3-4b (PAIR_ROWS 1) and did not on llama-3.2-1b (PAIR_ROWS 2). One model
is not enough, and the one that fails is not predictable.

The reference copy is imported from a temp directory, so the sibling modules it
imports (proj_qmm_pack, xfeed_bd, ...) resolve to the WORKTREE copies. That is
deliberate: those are new files with no HEAD counterpart, and at batch 1 the
builder does not import them at all.

    python3 check_batch1_noop.py             # exit 0 = inert
    python3 check_batch1_noop.py -v          # show the first differing lines

Exit code is the gate: 0 no-op, 1 the batch-1 design moved.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (model, VOCAB_CHUNK_I2). The chunk must divide the model's padded vocab, so it
# is per model -- 18 is the Makefile default (llama), qwen3-4b needs 30.
MODELS = [("llama-3.2-1b", 18), ("qwen3-4b", 30)]

# The Makefile's DECODE_ENV, minus the parts it computes.
BASE_ENV = dict(LM_HEAD="0", NLAYERS="1", DECODE_GOLDEN="1")

EMIT = (
    "import sys; sys.path.insert(0, {d!r}); sys.path.append({here!r});\n"
    "import fused_decode as fd; sys.stdout.write(str(fd.build_module()))\n"
)


def emit(pydir, model, chunk):
    env = dict(os.environ, **BASE_ENV)
    env["DECODE_MODEL"] = model
    env["VOCAB_CHUNK_I2"] = str(chunk)
    env.pop("DECODE_BATCH", None)  # batch 1 is the point
    r = subprocess.run(
        [sys.executable, "-c", EMIT.format(d=str(pydir), here=str(HERE))],
        capture_output=True,
        text=True,
        cwd=HERE,
        env=env,
    )
    if r.returncode:
        sys.exit(f"build failed ({model}, {pydir}):\n{r.stderr[-3000:]}")
    return r.stdout.splitlines()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", default="HEAD", help="git ref to compare against")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    rel = "programming_examples/fused_decode/fused_decode.py"
    src = subprocess.run(
        ["git", "show", f"{args.ref}:{rel}"], capture_output=True, text=True, cwd=HERE
    )
    if src.returncode:
        sys.exit(f"git show {args.ref}:{rel} failed:\n{src.stderr}")

    bad = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "fused_decode.py").write_text(src.stdout)
        print(f"\nDECODE_BATCH=1 vs {args.ref}  [emitted MLIR, both models]")
        for model, chunk in MODELS:
            a = emit(td, model, chunk)
            b = emit(HERE, model, chunk)
            ok = a == b
            print(
                f"  {model:14s} {len(b):6d} lines   {'IDENTICAL' if ok else 'DIFFERS'}"
            )
            if not ok:
                bad.append(model)
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        print(f"      first diff at line {i}:")
                        print(f"        {args.ref}: {x.strip()[:110]}")
                        print(f"        worktree: {y.strip()[:110]}")
                        break
                if len(a) != len(b):
                    print(f"      length {len(a)} -> {len(b)}")
                if args.verbose:
                    n = 0
                    for i, (x, y) in enumerate(zip(a, b)):
                        if x != y and n < 10:
                            print(f"      {i}: {x.strip()[:80]}  |  {y.strip()[:80]}")
                            n += 1

    if bad:
        print(f"\n  NOT A NO-OP: {', '.join(bad)}")
        print(
            "  Something guarded by `if BATCH > 1` is not, or a constant that\n"
            "  used to fold now does not. Both change the shipping decode."
        )
        return 1
    print("\n  batch 1 is untouched on both models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
