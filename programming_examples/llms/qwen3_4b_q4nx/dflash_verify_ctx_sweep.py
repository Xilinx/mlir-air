#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The batch-8 verify pass against batch 1, AS A FUNCTION OF CONTEXT LENGTH.

`dflash_verify_gate.py` runs this comparison at ONE context length -- the
5-token Paris prompt -- and passes 8/8. That was the only regime that worked:
at P=5 the whole block fits in a single 16-key attention block, and every token
that spanned two blocks came back wrong (corr 0.08-0.5, 1 of 8 argmaxes right at
P=96) while batch 1 at the same context decoded coherently. The cause was a
shared-L1 lock scope, one loop level too high -- see
docs/DFlashFeasibility.md section 3.11. Fixed:

    P     blocks  agree/8   corr(slot 0)   worst-slot corr   (worst, before)
    8       1       6/8       0.98693         0.94432          0.97027
   16       2       8/8       0.98565         0.95742          0.30692
   20       2       8/8       0.98802         0.95238          0.38601
   96       7       8/8       0.98857         0.96504          0.07635

Keep running it swept rather than at one P: a single-point gate is what let the
original defect through, and it only ever appeared past a 16-key context.

--corr defaults to 0.95, which is tight. What is left is the ordinary
batch-vs-batch-1 disagreement -- the two take different reduction orders, and in
bf16 that moves an argmax wherever the top two logits are close -- and it runs
0.93-0.99 at every context, including the single-block ones. Read the whole row,
not the pass/fail.

    python3 dflash_verify_ctx_sweep.py --prompts prompts_gsm8k.json
    python3 dflash_verify_ctx_sweep.py --prompts prompts_gsm8k.json --per-token
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import os

os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", help="JSON list of id lists; uses the first")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-L", type=int, default=512)
    ap.add_argument("--stack", default="6080")
    ap.add_argument("--prefix", default="taps_b8_L")
    ap.add_argument("--taps", action="store_true", default=True)
    ap.add_argument("--lens", default="5,8,12,16,20,24,32,48,64,96")
    ap.add_argument("--corr", type=float, default=0.95)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--per-token",
        action="store_true",
        help="one row per token: its own ceil(L_t/16) against the block count "
        "the shim actually pushes, and its correlation. The min-over-tokens "
        "summary cannot separate 'the tokens that skip a block are wrong' "
        "from 'every token is wrong'.",
    )
    args = ap.parse_args()

    import gc

    import numpy as np

    import qwen3_4b_q4nx_inference as INF
    import qwen3_4b_q4nx_weights as gw

    model = args.model or INF.MODEL_DEFAULT
    B = args.batch
    if args.prompts:
        full = json.loads(Path(args.prompts).read_text())[0]
    else:
        full = list(INF.PARIS_PROMPT) * 40
    Ps = [int(x) for x in args.lens.split(",") if int(x) <= len(full)]

    # Every prefill BEFORE either decoder exists: the dequantized model plus two
    # multi-GiB weight BOs is enough to die during allocation with no traceback.
    qm = gw.Q4nxModel(model)
    pf = {}
    for P in Ps:
        Kc, Vc, lg = gw.forward_prompt(qm, full[:P])
        pf[P] = (Kc, Vc, int(lg[-1].argmax()))
    del qm
    gc.collect()

    d1 = INF.FusedDecoder(model=model, max_L=max(Ps) + B + 1)
    env = {"DECODE_STACK": args.stack, "DECODE_MASK_BIDIR": "0"}
    if args.taps:
        env["DECODE_HIDDEN_TAPS"] = "1"
    dB = INF.FusedDecoder(
        model=model,
        max_L=args.max_L,
        batch=B,
        template_prefix=args.prefix,
        env_extra=env,
    )

    print(
        f"\n  batch {B} vs {B} sequential batch-1 steps, same KV seed, same tokens\n"
        f"  one 16-key attention block covers P + B <= 16\n"
        f"    P   blocks  agree/{B}  corr(slot0)  worst corr   rel(slot0)"
    )
    bad = 0
    for P in Ps:
        Kc, Vc, first = pf[P]
        d1.seed_kv(Kc, Vc, P)
        toks, ref = [first], []
        for t in range(B):
            y = np.asarray(d1.dispatch(toks[t], P + t), np.float32)
            ref.append(y)
            toks.append(int(y.argmax()))
        dB.seed_kv(Kc, Vc, P)
        got = np.asarray(dB.dispatch(toks[:B], P), np.float32)
        ag = sum(int(got[t].argmax()) == int(ref[t].argmax()) for t in range(B))
        cs = [float(np.corrcoef(got[t], ref[t])[0, 1]) for t in range(B)]
        r0 = float(
            np.sqrt(((got[0] - ref[0]) ** 2).mean())
            / max(np.sqrt((ref[0] ** 2).mean()), 1e-9)
        )
        # What the shim/memtile/cores all loop, uniformly for the whole block.
        push = (P + 15 + B - 1) // 16
        ok = min(cs) >= args.corr
        bad += not ok
        print(
            f"  {P:4d}  {push:5d}    {ag}/{B}     {cs[0]:8.5f}   {min(cs):8.5f}   "
            f"{r0:.3e}" + ("" if ok else "   <-- FAIL")
        )
        if args.per_token:
            for t in range(B):
                Lt = P + t
                own = (Lt + 15) // 16  # = ceil(Lt/16), what THIS token walks
                print(
                    f"        t={t} L={Lt:3d}  own={own} push={push} "
                    f"skip={push - own}  corr={cs[t]:8.5f}"
                    + ("" if cs[t] >= args.corr else "  <-- bad")
                )
    print("\n" + ("PASS" if not bad else f"FAIL at {bad} of {len(Ps)} context lengths"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
