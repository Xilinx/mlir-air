#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The batch-8 verify pass against batch 1, AS A FUNCTION OF CONTEXT LENGTH.

`dflash_verify_gate.py` runs this comparison at ONE context length -- the
5-token Paris prompt -- and passes 8/8. That turns out to be the only regime
that works: at P=5 the whole block fits in a single 16-key attention block
(P + B <= 16), and the batch-8 pass is only correct there.

    P     agree/8   corr(slot 0)   worst-slot corr
    5       8/8       0.99559          0.97126
    8       7/8       0.99478          0.99061
   12       4/8       0.99504          0.22384
   16       5/8       0.30692          0.30692
   20       1/8       0.48705          0.38601
   96       1/8       0.32474          0.07635

Batch 1 at the same P is fine -- it decodes coherent text at P=96 across seven
attention blocks -- so this is batch>1 AND rounds>1, not either alone.

That matters beyond DFlash: a batch-8 verify pass that only works below a
16-token context cannot verify anything, and every acceptance number measured
through it is measuring the target's failure rather than the drafter's quality.

    python3 dflash_verify_ctx_sweep.py --prompts prompts_gsm8k.json
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
        nblk = -(-(P + B) // 16)
        ok = min(cs) >= args.corr
        bad += not ok
        print(
            f"  {P:4d}  {nblk:5d}    {ag}/{B}     {cs[0]:8.5f}   {min(cs):8.5f}   "
            f"{r0:.3e}" + ("" if ok else "   <-- FAIL")
        )
    print("\n" + ("PASS" if not bad else f"FAIL at {bad} of {len(Ps)} context lengths"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
