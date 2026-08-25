#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What the batch-8 verify COSTS, in accepted tokens.

`batch_dispatch_check.py` reports how far the batched logits are from the
sequential ones. That is the right question for "is the engine correct" and the
wrong one for "should we ship it": a speculative decoder does not consume
logits, it consumes ACCEPT/REJECT decisions, and a logit difference that never
moves one is free. This counts the decisions.

THE DRAFT IS SELF-DRAFTED, AND THAT IS THE WHOLE TRICK. Each block is drafted
greedily by the BATCH-1 engine, so by construction a batch-1 verifier would
accept every token of it -- the draft IS its own argmax. The lossless ceiling is
therefore exactly B, with no reference run needed and no draft-model quality
mixed in. Whatever the batch-8 verifier rejects is the projection kernel
difference and nothing else:

    batch 1 uses the v1 GEMV, batch 8 the q4k mmul, 1.599% apart at the
    projection output (proj_qmm_gate.py) while both sit ~1% from exact fp32.
    Neither is wrong. They are different approximations, and this measures
    what choosing between them costs downstream.

Greedy acceptance, the standard prefix rule: the verify pass's logits at block
position t predict position t+1, so draft token t+1 is accepted iff the batched
argmax there matches the batch-1 argmax. The accepted prefix ends at the first
mismatch -- a later agreement does not count, because a real decoder has already
rolled back.

    python3 spec_accept.py                       # batch 8, 15 blocks
    python3 spec_accept.py --batch 4 --blocks 20

Needs the same template pairs batch_dispatch_check.py does. Not a pass/fail
gate: it reports a distribution and a mean accepted length, which is the number
the block-size work in DFlashFeasibility.md section 5f consumes.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-l", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--prompt-ids", default="128000,791,6864,315,9822,374")
    ap.add_argument(
        "--blocks",
        type=int,
        default=0,
        help="0 = as many as the decode window holds",
    )
    ap.add_argument(
        "--drift",
        action="store_true",
        help="let each engine carry its OWN KV forward instead of re-seeding "
        "both from a common history each block. That measures standalone "
        "divergence of the two engines, NOT the per-block verify loss -- the "
        "projection difference accumulates in K and V and swamps everything "
        "within two blocks.",
    )
    ap.add_argument("--ref-prefix", default="decode_b1_L")
    args = ap.parse_args()

    import numpy as np
    import llama32_1b_q4nx_inference as inf

    kv_path = HERE / "_spec_accept_kv.npz"
    ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    inf.run_prefill(ids, args.seq_len, kv_path)
    pf = np.load(kv_path)
    fk, fv = pf["k"].astype(np.float32), pf["v"].astype(np.float32)
    first = int(pf["first"])
    P0 = fk.shape[1]

    B = args.batch
    # The batch-1 reference is the tighter window: it must hold P + t, while the
    # batched one derives ATTN_MAXL = L + B - 1 and has B-1 positions of slack.
    room = (args.max_l - P0) // B
    nblk = min(args.blocks, room) if args.blocks else room
    if nblk < 1:
        sys.exit(f"no room for a block: P0={P0}, B={B}, window {args.max_l}")

    print(f"\nspeculative acceptance  [llama-3.2-1b, batch {B}, {nblk} blocks]")
    print(f"  draft: greedy self-draft by the batch-1 engine (ceiling = {B})")
    print(
        "  history: "
        + (
            "each engine its OWN (divergence test)"
            if args.drift
            else "re-seeded identical per block (verify-loss test)"
        )
    )

    ref = inf.FusedDecoder(P0 + B, batch=1, template_prefix=args.ref_prefix)
    dec = inf.FusedDecoder(P0 + B, batch=B)

    # ---- the committed trajectory, from the batch-1 engine ----
    ref.seed_kv(fk, fv, P0)
    traj, nxt = [], first
    for t in range(nblk * B):
        traj.append(nxt)
        nxt = int(np.asarray(ref.dispatch(nxt, P0 + t), np.float32).argmax())

    if not args.drift:
        # ONE prefill over prompt + the whole trajectory gives K/V at every
        # position, so each block can start both engines from BYTE-IDENTICAL
        # history. Without this the two engines carry their own caches and the
        # ~1.6% projection difference accumulates in K and V -- which swamps
        # the per-block question within two blocks and is what --drift shows.
        # A speculative decoder has ONE authoritative KV, so matched history is
        # the condition the verify actually runs under.
        long_kv = HERE / "_spec_accept_long.npz"
        inf.run_prefill(ids + traj, args.seq_len, long_kv)
        lp = np.load(long_kv)
        lk, lv = lp["k"].astype(np.float32), lp["v"].astype(np.float32)

    accepted, rows = [], []
    for b in range(nblk):
        P = P0 + b * B
        toks = traj[b * B : (b + 1) * B]
        if args.drift:
            if b == 0:
                ref.seed_kv(fk, fv, P0)
                dec.seed_kv(fk, fv, P0)
        else:
            ref.seed_kv(lk[:, :P], lv[:, :P], P)
            dec.seed_kv(lk[:, :P], lv[:, :P], P)

        # ---- B sequential dispatches: the reference decisions ----
        want = [
            int(np.asarray(ref.dispatch(toks[t], P + t), np.float32).argmax())
            for t in range(B)
        ]
        # ---- one batched verify over the same block ----
        got = np.asarray(dec.dispatch(toks, P), np.float32)
        gm = [int(got[t].argmax()) for t in range(B)]

        # Draft token t+1 is want[t]; the bonus token is want[B-1]. A prefix of
        # length a means the first a of those B predictions matched.
        a = 0
        while a < B and gm[a] == want[a]:
            a += 1
        accepted.append(a)
        rows.append((b, P, a, want, gm))

    ref.close()
    dec.close()

    print(f"\n  {'blk':>4} {'pos':>5} {'accepted':>9}   first mismatch")
    for b, P, a, want, gm in rows:
        note = "-" if a == B else f"t{a}: batched {gm[a]} vs {want[a]}"
        print(f"  {b:>4} {P:>5} {a:>6}/{B}   {note}")

    acc = np.array(accepted)
    full = int((acc == B).sum())
    print(
        f"\n  mean accepted {acc.mean():.2f} of {B}"
        f"   ({acc.mean() / B * 100:.1f}% of the lossless ceiling)\n"
        f"  full blocks   {full}/{len(acc)}"
        f"   min {acc.min()}  max {acc.max()}"
    )
    if full == len(acc):
        print(
            "\n  EVERY block accepted in full. The kernel difference never moved\n"
            "  a greedy decision, so on this trajectory the batched verify is\n"
            "  lossless in the only sense a decoder can observe."
        )
    else:
        lost = B - acc.mean()
        print(
            f"\n  The kernel difference costs {lost:.2f} tokens per block of {B}.\n"
            f"  Speedup is proportional to accepted length, so that is a "
            f"{lost / B * 100:.1f}%\n  haircut on whatever the block-size "
            "analysis predicts."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
