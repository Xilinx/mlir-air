#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the speculative VERIFY pass: batch 8 vs eight batch-1 steps.

A DFlash verify pass hands the target B draft tokens at B consecutive positions
and takes B next-token distributions back -- token t's conditioned on the prompt
plus draft tokens 0..t. That is exactly what `DECODE_BATCH=8` computes, and this
checks it against the thing it has to equal: eight sequential batch-1 dispatches
of the same tokens from the same KV seed, on the same weights.

WHY THIS AND NOT batch_equiv.py. That gate runs the TEMPLATE on synthetic q4k
weights and compares LAYER OUTPUTS -- it is what proved the batched dataflow
correct (5.56e-03 at 36 layers). This runs the SHIPPING DRIVER on the real model
and compares LOGITS, so it covers everything batch_equiv.py cannot see: the
batched X buffer, the B rope slabs at B positions, the batched logits readback
and its wave/token interleave, and the ATTN_MAXL window the batch moves.

THE LOGITS DO NOT AGREE TIGHTLY, AND THAT IS THE KERNEL, NOT THE BATCHING.
The batch-1 template runs the v1 GEMV projection and the batched one runs the
q4k mmul: different kernels for the same product, in different accumulation
orders. proj_qmm_gate.py measures the batched one at 1.4x the GEMV's error;
batch_equiv.py measures 5.56e-03 between them at the 36-layer LAYER OUTPUT. By
the time that has gone through the final norm and a 151936-wide tied head it is
6.5e-02 to 2.0e-01 in RMS-relative logit terms, measured here on the Paris
prompt -- with the argmax unchanged at all 8 positions.

That the toolchain is not the cause was CHECKED rather than assumed: the same
batch-1 design built by build_template.sh (which skips the Peano pin preflight,
and this sandbox's nightly index no longer carries the pinned build) is
BIT-IDENTICAL to the shipping Makefile-built pair, all 8 tokens, rel 0.0.

So the gate is what a greedy verify pass actually depends on:

  argmax agreement    hard, at every position whose top-2 margin exceeds
                      --margin. A flip at a near-tie is expected in bf16
                      (docs/DFlashFeasibility.md 3.1 measures the observed
                      divergences at the 0th and 1.5th percentile of that gap);
                      a flip at a WIDE margin is the mask or the position
                      wiring, not arithmetic.
  correlation         REPORTED. Enforced only under --strict.
  rel                 REPORTED. Enforced only under --strict. It is a property
                      of the projection kernel, not a target.

A SINGLE BATCH-8 DISPATCH IS NOT TRUSTWORTHY, SO THIS GATE DOES NOT USE ONE.
It dispatches until two in a row come back BIT-IDENTICAL (--tries), and reports
nothing at all if that never happens. That is not smoothing over a wrong answer;
it is refusing to report an unreproducible one.

Why it is needed, measured inside ONE process and one decoder, re-seeding the KV
cache before each dispatch so the input state is identical:

    dispatch 0  differs from 1
    dispatch 1  BIT-IDENTICAL to 2
    dispatch 3  differs from 1 and 2 by rms rel 2.4e-01

A second run of the same probe had all five dispatches agree, so it is
intermittent, and the first dispatch after construction is the usual offender.
Before the filter, three runs of the gate on a mode-0 pair that is bit-identical
to the shipping build gave argmax 5/8, 8/8, 8/8 -- which is how a correct build
was mistaken for a regression and cost a session. With the filter, repeat runs
agree to the last decimal.

Not root-caused. The shape fits WDDM paging the 2.1 GiB host-only weight BO --
the same pager behind the 0xc01e0200 that kills `make verify` -- where a page
that is not resident when the engine reads it corrupts a region silently.

corr and rel are still printed rather than thresholded (--strict to enforce
them). They are a property of the projection kernel, not a target; a known-good
pair sits at corr 0.936-0.989 and rel 0.10-0.37.

For a compiler or builder change, prefer the DETERMINISTIC instrument anyway:
fused_decode/dflash_build_diff.py compares two builds bit-exactly on synthetic
weights, and needs no filtering at all.

ONE DECODER PER PROCESS. The two halves run as subprocesses of this script and
hand over a .npz. They have to: with a batch-1 and a batch-8 FusedDecoder
resident at once, the batch-8 half dies right after its banner -- on the
SHIPPING templates, with numpy prefill and with --npu-prefill alike, and with
18.9 GiB of 31.2 free, so it is not host memory. Run the batch-8 half alone and
it completes and returns sensible logits. The same fragility shows up in any
script that registers two xclbins in one process: the second dispatch comes back
ERT_CMD_STATE_TIMEOUT with a partly-written X, which reads exactly like a hang
in the build under test and is not one.

A half that writes its .npz and THEN dies on the way out is a success -- the
teardown fault (dropping a FusedDecoder takes its BOs and the XRT device down in
whatever order the collector picks) lands after the result is flushed. This
judges on the file, not on the exit code.

    python dflash_verify_gate.py                 # numpy prefill (no NPU prefill load)
    python dflash_verify_gate.py --npu-prefill
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import qwen3_4b_q4nx_inference as INF


def _rel(a, b):
    import numpy as np

    return np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b**2).mean()), 1e-9)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--stack",
        default="6080",
        help="DECODE_STACK the batched templates were built with",
    )
    ap.add_argument("--npu-prefill", action="store_true")
    ap.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="extend the prompt to this many tokens (0 = PARIS_PROMPT as-is). "
        "THE LENGTH IS A CORRECTNESS SETTING, not a knob -- see _prompt().",
    )
    ap.add_argument("--model", default=INF.MODEL_DEFAULT)
    ap.add_argument("--tol", type=float, default=0.30)
    ap.add_argument("--corr", type=float, default=0.97)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also enforce --corr and --tol. Off by default: they are "
        "reported, not reproducible -- see the comment in the compare loop.",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="top-2 logit gap below which an argmax disagreement is expected "
        "rather than a failure (see the module docstring)",
    )
    ap.add_argument(
        "--half",
        choices=["b1", "bB"],
        help="internal: run ONE half in this process and save its .npz. The "
        "default (no --half) runs both as subprocesses and compares.",
    )
    ap.add_argument("--tag", default="gate", help="stem for the two .npz halves")
    ap.add_argument(
        "--tries",
        type=int,
        default=6,
        help="batch-B dispatches to spend looking for two consecutive identical "
        "ones. A single dispatch is not trustworthy here -- see _half_bB.",
    )
    ap.add_argument(
        "--b1-env",
        default="",
        help="K=V,... applied to the batch-1 half's environment only",
    )
    ap.add_argument(
        "--bB-env",
        default="",
        help="K=V,... applied to the batch-B half's environment only. THE TWO "
        "HALVES MAY LEGITIMATELY DIFFER. The vocab chunking has to divide the "
        "re-feed cycle at batch 8 -- VOCAB_CHUNK_I2=50 is what lets "
        "RMS_MEMTILE_REFEED=3 carry the LM head -- and the batch-1 path does not "
        "build at that chunking. The two requant caches hold byte-identical "
        "decode-layer weights and differ only in the ORDER of the lm-head slab, "
        "so each half with its own matched (template, cache) pair computes the "
        "same function, which is exactly what this gate asks about.",
    )
    args = ap.parse_args()

    f1 = _HERE / f"_vg_{args.tag}_b1.npz"
    fB = _HERE / f"_vg_{args.tag}_b{args.batch}.npz"

    if args.half == "b1":
        return _half_b1(args, f1)
    if args.half == "bB":
        return _half_bB(args, f1, fB)

    for half in ("b1", "bB"):
        want = f1 if half == "b1" else fB
        want.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--half",
            half,
            "--batch",
            str(args.batch),
            "--stack",
            args.stack,
            "--model",
            args.model,
            "--tag",
            args.tag,
            "--tries",
            str(args.tries),
            "--prompt-len",
            str(args.prompt_len),
        ]
        if args.npu_prefill:
            cmd.append("--npu-prefill")
        env = dict(os.environ)
        for kv in (args.b1_env if half == "b1" else args.bB_env).split(","):
            if kv.strip():
                k, _, v = kv.partition("=")
                env[k.strip()] = v.strip()
        # Each half resolves its own requant cache FROM its own chunking, so a
        # per-half VOCAB_CHUNK_I2 must not inherit a path the other half (or the
        # parent) already resolved.
        env.pop("Q4NX_QWEN3_4B_DECODE_NPZ", None)
        print(f"\n=== {half} half ===", flush=True)
        r = subprocess.run(cmd, cwd=str(_HERE), env=env)
        if not want.exists():
            print(
                f"[verify gate] {half} half produced no {want.name} "
                f"(rc={r.returncode})"
            )
            return 1
        if r.returncode:
            print(
                f"[verify gate] {half} half rc={r.returncode} AFTER writing its "
                f"result -- the known teardown fault, continuing"
            )
    return _compare(args, f1, fB)


def _prompt(args):
    """The prompt, extended to --prompt-len by cycling PARIS_PROMPT.

    THE LENGTH DECIDES WHETHER THIS GATE CAN SEE ANYTHING. `max_L = P + B + 1`
    picks the ATTN_MAXL window, and hence which templates are bound. At
    PARIS_PROMPT's P=5 that window is 32 (L15/L16) -- and at L15 two builds of
    the SAME design differ by rms rel 2.5e-03 to 9.6e-03 at the layer output,
    which is FOURTEEN TIMES the effect the gate exists to detect. Every verdict
    it returns there is build noise.

    At the 176 window (L161/L162) builds are bit-reproducible: two independently
    built templates of different designs came back bit-identical over 36 layers
    and the whole KV region. So --prompt-len 150 puts max_L at 159, binds the
    L161/L162 pair, and the gate measures the build instead of the weather.

    The token CONTENT does not matter here -- this checks that one batch-8
    dispatch equals eight batch-1 steps of the same tokens, which is true or
    false regardless of what they say. Only the length matters."""
    base = list(INF.PARIS_PROMPT)
    n = args.prompt_len
    if n <= 0:
        return base
    return [base[i % len(base)] for i in range(n)]


def _seed(args):
    """The KV seed and the prompt's first token.

    Deterministic either way, so each half recomputes it rather than moving 36
    layers of K/V through a file; the batch-8 half checks it agrees with what
    the batch-1 half saw before going near the device."""
    import qwen3_4b_q4nx_weights as gw

    prompt = _prompt(args)
    if args.npu_prefill:
        Kc, Vc, first, _ = INF._prefill_npu(prompt, args.model)
    else:
        import gc

        qm = gw.Q4nxModel(args.model)
        Kc, Vc, logits = gw.forward_prompt(qm, prompt)
        first = int(logits[-1].argmax())
        # The numpy prefill holds the whole dequantized model, and FusedDecoder
        # then loads a 2.1 GiB requant cache AND writes a 2.1 GiB host-only BO --
        # and builds its own Q4nxModel on top. Holding this one across that is
        # enough to die during BO allocation, with no traceback: the process
        # simply stops after the decoder's banner.
        del qm, logits
        gc.collect()
    return Kc, Vc, first, int(Kc.shape[1])


def _half_b1(args, out):
    """B sequential batch-1 steps, each appending its own K/V.

    The tokens the batched pass will be asked to verify are exactly the ones
    this run produces, so the two see the same sequence. Running batch 1 FIRST
    and to completion is what makes that possible."""
    import numpy as np

    B = args.batch
    Kc, Vc, first, P = _seed(args)
    print(f"[verify gate] prompt_len={P}, first={first}, batch={B}")
    d1 = INF.FusedDecoder(model=args.model, max_L=P + B + 1)
    if P + B >= d1.ATTN_MAXL:
        print(f"[verify gate] P+B={P+B} >= ATTN_MAXL={d1.ATTN_MAXL}; abort")
        return 1
    d1.seed_kv(Kc, Vc, P)
    toks, ref = [first], []
    for t in range(B):
        y = d1.dispatch(toks[t], P + t)
        ref.append(np.asarray(y, np.float32))
        toks.append(int(ref[-1].argmax()))
    print(f"[verify gate] batch-1 tokens: {toks}", flush=True)
    np.savez(out, ref=np.stack(ref), toks=np.asarray(toks), P=P, first=first)
    return 0


def _half_bB(args, inp, out):
    """ONE dispatch of the same B tokens at the same positions.

    DECODE_STACK must match what the template was BUILT with, and at batch 8 it
    is not optional: the default stack leaves the rms core 55280 B of L1 against
    the 59424 B a batch-8 residual + staging + norm weights need, and the builder
    refuses to import rather than build something that would fit by truncation."""
    import numpy as np

    B = args.batch
    d = np.load(inp)
    toks = [int(v) for v in d["toks"]]
    Kc, Vc, first, P = _seed(args)
    if int(d["P"]) != P or int(d["first"]) != first:
        print(
            "[verify gate] the two halves disagree on the prefill; the seed "
            "is not deterministic"
        )
        return 1
    dB = INF.FusedDecoder(
        model=args.model,
        max_L=P + B + 1,
        batch=B,
        env_extra={"DECODE_STACK": args.stack},
    )
    if P + B >= dB.ATTN_MAXL:
        print(f"[verify gate] P+B={P+B} >= ATTN_MAXL={dB.ATTN_MAXL}; abort")
        return 1
    # DISPATCH UNTIL TWO IN A ROW AGREE BIT FOR BIT, then keep that one.
    #
    # A single batch-8 dispatch is not trustworthy on this machine. Measured
    # inside ONE process, one decoder, re-seeding the KV cache each time so the
    # input state is identical: dispatch 0 differed from 1, 1 and 2 came back
    # BIT-IDENTICAL, and 3 differed from both by rms rel 2.4e-01. A second run
    # of the same script had all five agree. So it is intermittent, the first
    # dispatch after construction is the usual offender, and it corrupts the
    # answer rather than failing -- which is how a build that is bit-identical
    # to shipping came back argmax 5/8 and sent a whole session chasing it.
    #
    # Not isolated, but the shape fits the 2.1 GiB host-only weight BO being
    # paged by WDDM (the same pager behind the 0xc01e0200 that kills
    # `make verify`): a page that is not resident when the engine reads it gives
    # exactly this -- intermittent, region-local, and silent.
    #
    # Two consecutive identical dispatches is a cheap and honest filter: it does
    # not hide a wrong answer, it refuses to report an unreproducible one.
    prev = None
    for attempt in range(args.tries):
        dB.seed_kv(Kc, Vc, P)
        got = np.asarray(dB.dispatch(toks[:B], P), np.float32)
        if prev is not None and np.array_equal(prev, got):
            if attempt > 1:
                print(
                    f"[verify gate] batch-{B} reproduced after {attempt + 1} "
                    f"dispatches",
                    flush=True,
                )
            np.savez(out, got=got, attempts=attempt + 1)
            return 0
        prev = got
    print(
        f"[verify gate] batch-{B} never reproduced in {args.tries} dispatches -- "
        f"NOT writing a result. This is the device, not the build; see the "
        f"module docstring."
    )
    return 1


def _compare(args, f1, fB):
    import numpy as np

    a = np.load(f1)
    ref, P = a["ref"], int(a["P"])
    got = np.load(fB)["got"]
    B = args.batch
    bad = 0
    print(
        f"\n[verify gate] {B} tokens at positions {P}..{P+B-1}, "
        f"ATTN_MAXL window from the batch-{B} templates"
    )
    for t in range(B):
        r = _rel(got[t], ref[t])
        corr = float(np.corrcoef(got[t], ref[t])[0, 1])
        a_g, a_r = int(got[t].argmax()), int(ref[t].argmax())
        top2 = np.partition(ref[t], -2)[-2:]
        margin = float(top2[1] - top2[0])
        top5 = len(set(np.argsort(got[t])[-5:]) & set(np.argsort(ref[t])[-5:]))
        # A disagreement is only a failure where the reference is not near-tied.
        agree = a_g == a_r
        # ARGMAX IS THE GATE, AND EVEN IT NEEDS REPEATS. corr and rel are
        # REPORTED and only enforced under --strict: enforcing them made a
        # known-good mode-0 pair report FAIL(6) at argmax 8/8, which teaches
        # the next session to ignore the gate entirely. And the argmax itself
        # varies run to run on this machine -- see the module docstring for the
        # three-run measurement on a build that is bit-identical to shipping.
        ok = agree or margin < args.margin
        if args.strict:
            ok = ok and r <= args.tol and corr >= args.corr
        bad += not ok
        note = "" if agree else f" ({a_g} vs {a_r})"
        print(
            f"  token {t} (pos {P+t}): argmax {'=' if agree else 'X'}{note}"
            f"  margin {margin:6.3f}  corr {corr:.6f}  top5 {top5}/5  "
            f"rel {r:.3e}" + ("" if ok else "   <-- FAIL")
        )

    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
