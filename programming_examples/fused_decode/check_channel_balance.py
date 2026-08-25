#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Did both sides of every channel scale by the SAME factor? The deadlock, statically.

WHY THIS EXISTS. The first batch-8 template compiled cleanly and then hung on
device -- ERT_CMD_STATE_TIMEOUT, no message, no clue which of thirty channels
stalled. That is this engine's characteristic failure and the builder's comments
are full of it: "the size must match or the fan deadlocks", "the memtile stalls
waiting for chunks the rms never produces". Every one of those is visible in the
emitted IR, and invisible to the compiler, which will happily build a design
that waits forever.

RATIOS, NOT TOTALS, and that is the whole design of this file. Counting elements
absolutely needs a model of scf.parallel fans, index_switch arms, per-tile IfOp
specialisation and herd multiplicity -- and getting any of them wrong makes the
tool report the SHIPPING batch-1 design as broken, which it did on its first
run. So it counts the same (wrong) way twice, at batch 1 and at batch N, and
compares how each SIDE of each channel grew:

    put x8, get x8      fine, whatever the absolute numbers are
    put x8, get x1      the consumer never scaled -- a hang

Batch 1 is the reference because it is the design that runs. Any modelling error
cancels, because it is the same error on both sides of the division.

WHAT IT STILL CANNOT SEE. A channel both sides of which scaled wrongly by the
same factor. Order, and therefore deadlocks between correctly-balanced channels.
And anything under a runtime loop bound, which it reports as UNKNOWN rather than
guessing.

    python3 check_channel_balance.py --batch 8
    python3 check_channel_balance.py --batch 8 -v      # every channel

Exit code is the gate: 0 every channel scaled symmetrically.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = [("llama-3.2-1b", 18), ("qwen3-4b", 30)]

# The emitted MLIR is read back through the air dialect's own parser, so this
# walks real operations rather than matching text. A regex over the put lines
# cannot see which loops enclose one, which is most of the question.
EMIT = r"""
import sys, json
sys.path.insert(0, {here!r})
import fused_decode as fd
from air.ir import Context, Module

mod = fd.build_module()
UNKNOWN = object()


def const_int(v):
    op = getattr(v, "owner", None)
    if op is None:
        return None
    try:
        name = op.name
    except Exception:
        return None
    if name not in ("arith.constant", "air.constant"):
        return None
    try:
        return int(op.attributes["value"].__str__().split(":")[0].strip())
    except Exception:
        return None


def trips(op):
    '''Trip count of an scf.for, or None if its bounds are not constants.'''
    lo, hi, st = (const_int(op.operands[i]) for i in range(3))
    if lo is None or hi is None or st is None or st <= 0:
        return None
    return max(0, -(-(hi - lo) // st))


def sizes_of(op):
    '''Elements one channel op moves: product of its `sizes`, or None.

    The op's operand layout is (indices..., memref, offsets..., sizes...,
    strides...) with the segment sizes attribute saying how many of each.
    '''
    try:
        seg = [int(x) for x in op.attributes["operandSegmentSizes"]]
    except Exception:
        return None
    # put:  [async_deps, indices, src, offsets, sizes, strides]
    # get:  [async_deps, indices, dst, offsets, sizes, strides]
    if len(seg) < 6:
        return None
    base = sum(seg[:4])
    n = seg[4]
    if n == 0:
        return UNKNOWN  # whole-memref form; resolved from the memref below
    tot = 1
    for i in range(base, base + n):
        c = const_int(op.operands[i])
        if c is None:
            return None
        tot *= c
    return tot


def static_sizes(op):
    '''The `sizes` list when it is STATIC, read off the printed form.

    An air.channel op takes offsets/sizes/strides either as SSA operands or as
    static attributes, and the builder mixes the two: `sizes=[258]` prints as
    `[258]`, `sizes=[idx(258)]` prints as `[%c258]`. Missing this made the
    batch-1 side fall back to the whole memref, and the tool reported a
    correctly-scaled channel as growing 4x instead of 8x -- the ratio comparison
    only cancels a modelling error if it is the SAME error on both sides.
    '''
    import re

    txt = str(op).split("(", 1)[-1]
    groups = re.findall(r"\[([^\]]*)\]", txt)
    if len(groups) < 3:
        return None
    body = [g.strip() for g in groups[-3:]]  # offsets, sizes, strides
    parts = [p.strip() for p in body[1].split(",") if p.strip()]
    if not parts or any(p.startswith("%") for p in parts):
        return None
    tot = 1
    for p in parts:
        try:
            tot *= int(p)
        except ValueError:
            return None
    return tot


def memref_elems(op):
    try:
        seg = [int(x) for x in op.attributes["operandSegmentSizes"]]
        mr = op.operands[sum(seg[:2])]
        shp = mr.type.shape
    except Exception:
        return None
    tot = 1
    for d in shp:
        tot *= d
    return tot


acc = {{}}


def walk(op, mult, unknown):
    name = op.name
    if name in ("air.channel.put", "air.channel.get"):
        sym = str(op.attributes["chan_name"]).strip('@" ')
        n = sizes_of(op)
        if n is UNKNOWN:
            n = static_sizes(op)
            if n is None:
                n = memref_elems(op)
        e = acc.setdefault(sym, {{"put": 0, "get": 0, "psite": 0, "gsite": 0,
                                  "unknown": False}})
        k = "put" if name.endswith("put") else "get"
        e[("p" if k == "put" else "g") + "site"] += 1
        if n is None or unknown:
            e["unknown"] = True
        else:
            e[k] += n * mult
    sub, sub_unknown = mult, unknown
    if name == "scf.for":
        t = trips(op)
        if t is None:
            sub_unknown = True
        else:
            sub = mult * t
    if name == "scf.parallel":
        # Spatial: every index runs, so it multiplies like a loop.
        lo, hi, st = (const_int(op.operands[i]) for i in range(3))
        if None in (lo, hi, st) or st <= 0:
            sub_unknown = True
        else:
            sub = mult * max(0, -(-(hi - lo) // st))
    # The vocab/decode ARM select, in either of its two spellings, is a pair of
    # ALTERNATIVES: counting both mixes an lm-head build's traffic into a decode
    # build's. scf.index_switch is one by construction -- region 0 is the
    # default region (SCF.td declares defaultRegion first), which is the decode
    # arm everywhere in this builder. An scf.if is only one when the builder
    # says so, because scf.if is also how per-tile specialisation is written
    # (see _emit), where both regions are wanted; air.arm_select is the builder
    # marking the ones that are arms, and their `then` region is decode.
    arm_if = name == "scf.if" and "air.arm_select" in op.attributes
    regions = (
        [op.regions[0]]
        if (name == "scf.index_switch" or arm_if)
        else list(op.regions)
    )
    for r in regions:
        for b in r.blocks:
            for o in b.operations:
                walk(o, sub, sub_unknown)


for r in mod.operation.regions:
    for b in r.blocks:
        for o in b.operations:
            walk(o, 1, False)
print(json.dumps(acc))
"""


def balance(model, chunk, batch):
    """Emit at `batch` and return the per-channel put/get counts.

    Returns `(counts, batch_used)`. The batch may come back LOWER than asked
    for: the rms core's L1 ceiling is per model (`BATCH_MAX_RMS`), and
    qwen3-4b's is below the 8 this gate defaults to. Clamping is the right
    answer rather than skipping the model — the question here is whether both
    sides of a channel scale together, and any batch above 1 asks it. The
    ceiling is parsed out of the builder's own refusal so the two cannot drift.
    """
    env = dict(os.environ, LM_HEAD="0", NLAYERS="1", DECODE_GOLDEN="1", UNIFIED="1")
    env["DECODE_MODEL"] = model
    env["VOCAB_CHUNK_I2"] = str(chunk)
    if batch > 1:
        env["DECODE_BATCH"] = str(batch)
    else:
        env.pop("DECODE_BATCH", None)
    r = subprocess.run(
        [sys.executable, "-c", EMIT.format(here=str(HERE))],
        capture_output=True,
        text=True,
        cwd=HERE,
        env=env,
    )
    if r.returncode:
        cap = re.search(r"exceeds the rms core's L1 ceiling of (\d+)", r.stderr)
        if cap and int(cap.group(1)) > 1 and int(cap.group(1)) < batch:
            return balance(model, chunk, int(cap.group(1)))
        sys.exit(f"build failed ({model}, batch {batch}):\n{r.stderr[-3000:]}")
    import json

    return json.loads(r.stdout.strip().splitlines()[-1]), batch


def ratio(a, b):
    """b/a as a float, or None when the reference side moved nothing."""
    if not a:
        return None
    return b / a


# A packet's 2-word routing header rides with the payload and is dropped by the
# consumer, so a channel's two ratios differ by a fixed amount that shrinks as
# the payload grows -- outY is put x7.973 against get x8.0 at batch 8 and that is
# correct. The real failure this file exists for was x3.986 against x7.946, so a
# 5% band separates them by a wide margin.
TOL = 0.05


def report(model, batch, one, many, verbose):
    bad, rows = [], []
    for sym in sorted(set(one) | set(many)):
        e1, eN = one.get(sym), many.get(sym)
        if e1 is None:
            # New at this batch -- can't be a ratio, but it still has to balance
            # against itself.
            ok = eN["put"] == eN["get"] and not eN["unknown"]
            rows.append(
                (
                    sym,
                    f"NEW at batch {batch}: put {eN['put']} get {eN['get']}"
                    + ("" if ok else "   UNBALANCED"),
                )
            )
            if not ok:
                bad.append(sym)
            continue
        if eN is None:
            rows.append((sym, "GONE at batch " + str(batch)))
            bad.append(sym)
            continue
        if e1["unknown"] or eN["unknown"]:
            rows.append((sym, "UNKNOWN (runtime loop bound)"))
            continue
        rp, rg = ratio(e1["put"], eN["put"]), ratio(e1["get"], eN["get"])
        if rp is None and rg is None:
            continue
        if rp is None or rg is None or abs(rp - rg) > TOL * max(rp or 0, rg or 0):
            rows.append(
                (
                    sym,
                    f"put x{rp if rp is None else round(rp, 3)}  "
                    f"get x{rg if rg is None else round(rg, 3)}   ASYMMETRIC",
                )
            )
            bad.append(sym)
        elif verbose:
            rows.append((sym, f"put x{round(rp, 3)}  get x{round(rg, 3)}"))
    print(f"\n  {model}: batch {batch} vs batch 1")
    if not rows:
        print("    every channel scaled symmetrically")
    for sym, note in rows:
        print(f"    {sym:16s}{note}")
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=None, help="default: both")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.batch == 1:
        sys.exit("--batch 1 compares the reference to itself; pick 2 or more")

    models = (
        [(m, c) for m, c in MODELS if m == args.model] if args.model else list(MODELS)
    )
    if not models:
        sys.exit(f"unknown model {args.model}")

    print("\nchannel scaling  [how each SIDE grew from batch 1]")
    bad = []
    for model, chunk in models:
        one, _ = balance(model, chunk, 1)
        many, used = balance(model, chunk, args.batch)
        if used != args.batch:
            print(
                f"\n  {model}: capped at batch {used} -- its rms core's L1 "
                f"ceiling is below the requested {args.batch}"
            )
        bad += report(model, used, one, many, args.verbose)

    if bad:
        print(
            f"\n  ASYMMETRIC: {', '.join(sorted(set(bad)))}\n"
            "  One side of these scaled with the batch and the other did not.\n"
            "  Whichever side moves fewer elements blocks forever; that is the\n"
            "  timeout, and it is not visible at build time."
        )
        return 1
    print("\n  every channel's producers and consumers scaled together")
    return 0


if __name__ == "__main__":
    sys.exit(main())
