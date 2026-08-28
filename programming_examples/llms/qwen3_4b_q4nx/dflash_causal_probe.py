#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does the decode engine attend past its own context length?

A DECODE step at context length L must depend on KV rows 0..L-1 and on nothing
after them. This writes garbage into rows L..L+7 -- rows the mask is supposed to
exclude entirely -- and re-dispatches the SAME token from the SAME state. If the
logits move, the engine read them.

WHAT IT FINDS, on qwen3-4b / NPU2, batch 1, the shipping decode template:

    L % 8 == 0     bit-identical.
    every other L  max|dlogit| 0.48-1.25, corr 0.994-0.999, argmax unchanged.

So attention attends to ceil(L/8)*8 keys, not L. It is a V-SIDE defect, not a
masking one: poisoning only K changes nothing (attn_qk's `aie::le(idx, rem)`
mask is exact per key), poisoning only V changes the output, and the reach is
exactly rows L..ceil(L/8)*8-1 -- rows at or above that boundary never matter,
even though the whole 16-key block containing them is streamed to the core.

WHY IT HAS NEVER MATTERED. In ordinary decode every row past L-1 is zero (the
cache is seeded once and the kernel appends one row per step), so the leak is a
small fixed pull that no gate resolves.

WHY IT MATTERS FOR DFLASH. In a batch-B speculative VERIFY pass those rows are
NOT zero: the block's own later tokens write them, in the same dispatch. Slot
j's distribution then depends on slots j+1.., which is the one property the
pass exists to have. Measured at batch 8 with DECODE_HIDDEN_TAPS, holding slot
0 fixed and changing only the tail: slot 0's hidden state differs from LAYER 1
onward at every L % 8 != 0 and is bit-identical at every L % 8 == 0, and by the
36th layer max|d| reaches 208. Over 29 consecutive positions the tail moved
slot 0's argmax at 2 of them (7%).

    python3 dflash_causal_probe.py                       # batch 1, shipping
    python3 dflash_causal_probe.py --prefix taps_b8_L --batch 8
"""

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE))

MASK_TOKEN_ID = 151669


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--stack", default="6080")
    ap.add_argument("--max-L", type=int, default=48)
    ap.add_argument("--lo", type=int, default=6)
    ap.add_argument("--hi", type=int, default=26)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--split",
        action="store_true",
        help="also poison K alone and V alone, which is what localises it",
    )
    args = ap.parse_args()

    import gc

    import numpy as np

    import dflash_draft_decoder as DD
    import qwen3_4b_q4nx_inference as INF
    import qwen3_4b_q4nx_weights as gw

    model = args.model or INF.MODEL_DEFAULT
    B = args.batch
    prompt = list(INF.PARIS_PROMPT)

    qm = gw.Q4nxModel(model)
    Kc, Vc, lg = gw.forward_prompt(qm, prompt)
    first = int(lg[-1].argmax())
    del qm, lg
    gc.collect()

    env = {"DECODE_STACK": args.stack} if B > 1 else None
    dec = INF.FusedDecoder(
        model=model,
        max_L=args.max_L,
        batch=B,
        template_prefix=args.prefix,
        env_extra=env,
    )
    n = dec.UNI_DEC
    rng = np.random.default_rng(0)
    Z = np.zeros((n, 8, dec.DK_TOT_A), np.float32)

    def tile(A, m):
        return np.concatenate([A] * (m // A.shape[1] + 1), axis=1)[:, :m]

    print(
        f"[causal probe] batch {B}, template {dec.gen.prefix if hasattr(dec.gen,'prefix') else args.prefix}, "
        f"ATTN_MAXL {dec.ATTN_MAXL}\n"
        f"  poisoning KV rows L..L+7, which the mask must exclude\n"
        f"    L  R=ceil(L/8)*8   max|dlogit|      corr   argmax"
    )
    bad = 0
    for L in range(args.lo, min(args.hi, dec.maxL - B) + 1):
        # A FULLY SEEDED prefix, so the host mirror of the KV cache is complete.
        # Walking there with dispatches instead would leave the mirror's rows
        # P.. zero, and the poison write re-uploads the whole region -- wiping
        # the device-appended rows and measuring that instead.
        dec.seed_kv(tile(Kc, L - 1), tile(Vc, L - 1), L - 1)
        toks = [first] * B
        a = np.asarray(dec.dispatch(toks if B > 1 else first, L - 1), np.float32)
        a0 = a[0] if a.ndim > 1 else a
        cases = (
            (("K only", 1, 0), ("V only", 0, 1), ("both", 1, 1))
            if args.split
            else (("both", 1, 1),)
        )
        cols = []
        for name, kk, vv in cases:
            g = (rng.standard_normal((n, 8, dec.DK_TOT_A)) * 2.0).astype(np.float32)
            DD.append_context_kv(dec, g if kk else Z, g if vv else Z, L)
            b = np.asarray(dec.dispatch(toks if B > 1 else first, L - 1), np.float32)
            b0 = b[0] if b.ndim > 1 else b
            DD.append_context_kv(dec, Z, Z, L)
            e = float(np.abs(a0 - b0).max())
            cols.append((name, e, float(np.corrcoef(a0, b0)[0, 1]), int(b0.argmax())))
        R = -(-L // 8) * 8
        e = cols[-1][1]
        bad += e != 0.0
        extra = (
            "  " + "  ".join(f"{nm} {v:.4f}" for nm, v, _, _ in cols[:-1])
            if args.split
            else ""
        )
        print(
            f"  {L:3d} {R:8d}    {e:11.5f}  {cols[-1][2]:.6f}   "
            f"{int(a0.argmax())} {cols[-1][3]}{extra}"
        )

    print(
        "\n"
        + (
            "PASS -- the engine reads only rows 0..L-1"
            if not bad
            else f"LEAK at {bad} of the tested L (exact only at L % 8 == 0)"
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
