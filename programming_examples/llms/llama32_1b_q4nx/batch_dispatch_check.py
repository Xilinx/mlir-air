#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""THE gate for the batched HOST DRIVER, on the real model.

`batch_equiv.py` and `batch_lm_equiv.py` gate the DEVICE with synthetic weights.
They cannot see the driver at all -- the embeddings, the B rope LUTs, the KV
seeding, the Y reshape and the bounds check are host code, and every one of them
is a place a batched dispatch can be plausibly wrong. This runs the real
llama-3.2-1B, real weights, a real prefill, and asserts the property the whole
batch rests on:

    one batch-B dispatch of tokens [t0..tB-1] at position P
      ==
    B batch-1 dispatches, token t at position P+t, on the same seeded KV

Both append the same K/V at the same positions; they differ only in WHEN -- the
block does all B up front, and a key past a token's own context is masked, so
that cannot show. If this holds, a speculative verify is lossless.

THE ANSWER IS A MATRIX, not a per-token number. Real logits over one vocabulary
are correlated, so "token 3 is 1% from reference 3" alone does not rule out
"token 3 got token 5's row" -- the off-diagonal does. The diagonal is the
GEMV-vs-mmul kernel difference (proj_qmm_gate.py: 1.7% at the projection
output); the off-diagonal is two different tokens, and it is two orders of
magnitude larger.

DISTINCT TOKENS, DELIBERATELY. Feeding B copies of one token would pass on a
driver that broadcast token 0's embedding to every row -- which is the first
thing a batched X feed gets wrong.

    python3 batch_dispatch_check.py                    # batch 8, ATTN_MAXL 128
    python3 batch_dispatch_check.py --batch 4 --max-l 128

Needs a batch-B decode template pair beside the batch-1 one:

    cd ../../fused_decode
    DECODE_NO_LM_WAVES=0 ./build_template.sh 8 128     # -> decode_b8_L128
    DECODE_NO_LM_WAVES=0 ./build_template.sh 8 127     # -> decode_b8_L127 (slope)

Exit code is the gate: 0 the batched driver computes what B single ones do.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def rel(a, b):
    """rms relative difference between two logit rows, in float64."""
    import numpy as np

    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("inf")
    den = float(np.sqrt(np.mean(b[ok] ** 2)))
    num = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    return num / den if den else num


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--max-l",
        type=int,
        default=128,
        help="the ATTN_MAXL window the DECODE templates were built for",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="the PREFILL length, which is a different thing and is not free "
        "to choose: its GEMM shapes have to be in kernel_registry, and 2048 is "
        "the one llama-3.2-1B has measured. The decode window is --max-l.",
    )
    ap.add_argument(
        "--prompt-ids",
        type=str,
        default="128000,791,6864,315,9822,374",
        help="prefill prompt as token IDs. IDs and not text because this is an "
        "EQUIVALENCE test -- what the prompt means does not enter into it, and "
        "requiring a tokenizer would make the gate need network access it does "
        "not otherwise need. Its length plus --batch must fit --max-l.",
    )
    ap.add_argument(
        "--ref-prefix",
        default="decode_b1_L",
        help="template family for the batch-1 reference. Defaults to the "
        "decode_b1_L pair rather than the shipping decode_L one so both sides "
        "come from the same tree; a toolchain difference between them would "
        "show up here as a batching difference.",
    )
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import llama32_1b_q4nx_inference as inf

    # ---- prefill once; both decoders are seeded from the SAME K/V ----
    kv_path = HERE / "_batch_check_kv.npz"
    ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    inf.run_prefill(ids, args.seq_len, kv_path)
    pf = np.load(kv_path)
    fk, fv = pf["k"].astype(np.float32), pf["v"].astype(np.float32)
    first = int(pf["first"])
    P = fk.shape[1]
    if P + args.batch > args.max_l:
        sys.exit(
            f"prompt is {P} tokens and the block is {args.batch}: "
            f"{P + args.batch} > ATTN_MAXL {args.max_l}. Shorten the prompt or "
            "build a larger template."
        )

    # B DISTINCT tokens. The first is the prefill's own next token, so the block
    # starts on the trajectory the model is actually on; the rest are spread
    # across the vocabulary so no two rows are close by accident.
    rng = np.random.default_rng(args.seed)
    toks = [first]
    while len(toks) < args.batch:
        t = int(rng.integers(1000, 100000))
        if t not in toks:
            toks.append(t)
    print(f"\nbatched dispatch  [llama-3.2-1b, batch {args.batch}, P={P}]")
    print(f"  tokens {toks}")

    # ---- the batched dispatch ----
    dec = inf.FusedDecoder(P + args.batch, batch=args.batch)
    dec.seed_kv(fk, fv, P)
    got = np.asarray(dec.dispatch(toks, P), np.float32)
    dec.close()
    del dec
    print(f"  batch {args.batch}: {got.shape} logits back")

    # ---- B single dispatches, token t at position P+t ----
    ref = inf.FusedDecoder(P + args.batch, batch=1, template_prefix=args.ref_prefix)
    ref.seed_kv(fk, fv, P)
    want = np.stack(
        [
            np.asarray(ref.dispatch(toks[t], P + t), np.float32)
            for t in range(args.batch)
        ]
    )
    ref.close()
    del ref

    cross = np.array(
        [[rel(got[t], want[r]) for r in range(args.batch)] for t in range(args.batch)]
    )
    print("\n  each token vs every reference (own reference on the diagonal):")
    print("        " + "".join(f"{r:>9d}" for r in range(args.batch)))
    for t in range(args.batch):
        print(
            f"    t{t}  " + "".join(f"{cross[t, r]:>9.2e}" for r in range(args.batch))
        )

    # The argmax is what a sampler acts on, so report it separately: a logit
    # difference that never moves the top token is a different kind of pass.
    tops = [(int(got[t].argmax()), int(want[t].argmax())) for t in range(args.batch)]
    agree = sum(a == b for a, b in tops)
    print(f"\n  argmax agrees on {agree} of {args.batch} tokens: {tops}")

    over = [t for t in range(args.batch) if cross[t, t] > args.tol]
    mis = [t for t in range(args.batch) if int(np.argmin(cross[t])) != t]
    if over or mis:
        if mis:
            print(f"  tokens {mis} are closest to ANOTHER token's reference.")
        if over:
            print(f"  tokens {over} exceed --tol {args.tol} against their own.")
        print(
            "\n  NOT EQUIVALENT. A batch-B dispatch is not what B single "
            "dispatches\n  compute, so a speculative verify built on it would "
            "not be lossless."
        )
        return 1
    margin = min(
        cross[t, r] / cross[t, t]
        for t in range(args.batch)
        for r in range(args.batch)
        if r != t and cross[t, t]
    )
    print(
        f"\n  every token matches its own single dispatch (tol {args.tol}) and is "
        f"{margin:.0f}x\n  closer to it than to any other token's -- GATE PASS"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
