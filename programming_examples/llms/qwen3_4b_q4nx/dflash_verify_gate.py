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
  correlation         >= --corr, which catches a systematically wrong logit
                      vector that happens to keep its maximum.
  rel                 reported, and bounded loosely by --tol. It is a property
                      of the projection kernel, not a target.

    python3 dflash_verify_gate.py                 # numpy prefill (no NPU prefill load)
    python3 dflash_verify_gate.py --npu-prefill
"""

import argparse
import os
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
    ap.add_argument("--model", default=INF.MODEL_DEFAULT)
    ap.add_argument("--tol", type=float, default=0.30)
    ap.add_argument("--corr", type=float, default=0.97)
    ap.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="top-2 logit gap below which an argmax disagreement is expected "
        "rather than a failure (see the module docstring)",
    )
    args = ap.parse_args()

    import numpy as np
    import qwen3_4b_q4nx_weights as gw

    B = args.batch
    prompt = INF.PARIS_PROMPT

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
    P = Kc.shape[1]
    print(f"[verify gate] prompt_len={P}, first={first}, batch={B}")

    # ---- batch 1: B sequential steps, each appending its own K/V -----------
    # The tokens the batched pass will be asked to verify are exactly the ones
    # this run produces, so the two see the same sequence. Running batch 1
    # FIRST and to completion is what makes that possible.
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
    # NOT `del d1`. Dropping a FusedDecoder segfaults the process -- its BOs and
    # the XRT device go down in whatever order the collector picks, and the
    # symptom is a bare SIGSEGV after the last flushed line, which reads exactly
    # like a device fault in the dispatch that just succeeded. Both decoders
    # stay alive; that costs a second weight BO and nothing else.

    # ---- batch B: ONE dispatch of the same B tokens at the same positions --
    # DECODE_STACK must match what the template was BUILT with, and at batch 8
    # it is not optional: the default stack leaves the rms core 55280 B of L1
    # against the 59424 B a batch-8 residual + staging + norm weights need, and
    # the builder refuses to import rather than build something that would fit
    # by truncation.
    dB = INF.FusedDecoder(
        model=args.model,
        max_L=P + B + 1,
        batch=B,
        env_extra={"DECODE_STACK": args.stack},
    )
    if P + B >= dB.ATTN_MAXL:
        print(f"[verify gate] P+B={P+B} >= ATTN_MAXL={dB.ATTN_MAXL}; abort")
        return 1
    dB.seed_kv(Kc, Vc, P)
    got = np.asarray(dB.dispatch(toks[:B], P), np.float32)

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
        ok = r <= args.tol and corr >= args.corr and (agree or margin < args.margin)
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
