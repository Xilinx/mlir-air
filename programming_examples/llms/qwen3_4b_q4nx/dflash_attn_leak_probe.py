#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""WHERE the attention leak past L comes from: the WEIGHT the phantom rows carry.

`dflash_causal_probe.py` establishes the behaviour -- a decode step at context
length L depends on KV rows L..ceil(L/8)*8-1, on the V side only, and not at
all on their K. Two mechanisms survive that result and they predict different
weights:

  (a) the softmax INCLUDES the phantom keys: they are in the numerator and the
      denominator, and the output is a convex combination over ceil(L/8)*8 rows.
  (b) `attn_fv` pairs VALID scores with the wrong V rows: the denominator is
      the correct sum over L keys, and some of that weight lands on phantom V.

THE CONSTRUCTION. Seed every position of every layer with the SAME key k0, so
every score is equal and the softmax is uniform whatever q is -- the attention
output is then the plain MEAN of the attended V rows, and nothing about the
hidden state matters. Build ONE layer (`UNI_DEC_OVERRIDE=1`) with
`DECODE_ACC_STOP=2`, which makes the layer output `x + o_proj(attn)` -- no FFN,
so the readback is LINEAR in the attention output. The X buffer already carries
it across the shim, so there is no new tap to route (the o-gather memtile tap,
`DECODE_PROBE=2`, does not route in this configuration -- the dispatch times
out).

Run twice with the phantom rows' V at 1 and at 0. Everything except the
attention output cancels:

    X[1] - X[0] = o_proj(f * 1)  =  f * (column sums of o_proj)

so f, the fraction of the output's weight sitting on phantom rows, is one least
squares fit against a vector the host already has. The token's own key at L-1
carries an unknown weight; call its ratio to a seeded key rho. With
R = ceil(L/8)*8,

    (a)  f = (R-L) / (R-1+rho)     LINEAR in L over a fixed R, zero at L=R
    (b)  f = (R-L) / (L-1+rho)     hyperbolic

Sweeping L across one R window separates them on shape alone, without knowing
rho -- and f(R) == 0 exactly is the built-in control that the construction
works at all.

WHAT IT MEASURED, AND WHY BOTH (a) AND (b) ARE DEAD:

    f                            -0.0063, and CONSTANT over L = 9..34
    at L = 16, 24, 32            +0.000000 exactly  <- three controls
    vs the phantom V VALUE       linear (same f at V=1 and V=16)
    vs the phantom row COUNT     no dependence at all
    one row set vs all five      the IDENTICAL difference vector, cos 1.0000
    rows at or past ceil(L/8)*8  exactly 0

Both (a) and (b) predict a per-row additive weight, and the contribution does
not depend on the row count -- so neither is what is happening. `--structure`
is the sweep that shows it, and the write was verified by reading the KV cache
back (only the intended row is set). With the K/V asymmetry on top, what is
left is a DESCRIPTOR asymmetry in the KV readback for a partial last group,
not kernel arithmetic: next instrument is `shim_volume.py` / `shim_schedule.py`.

The leaked weight is ~0.6% of the attention output, not the ~25% a real extra
key would carry -- which is why ordinary decode looks fine (corr 0.997) and why
it still compounds to an argmax flip at 7% of positions across 36 layers.

    cd ../../fused_decode && ./_build_accstop.sh     # accstop_b1_L63/64
    python3 dflash_attn_leak_probe.py                # the f(L) sweep
    python3 dflash_attn_leak_probe.py --structure --lo 19 --hi 19
"""

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE))

_ENV = {
    "UNI_DEC_OVERRIDE": "1",
    "DECODE_ACC_STOP": "2",
    "DECODE_NO_LM_WAVES": "1",
    "DECODE_PROBE": "0",
    "DECODE_HIDDEN_TAPS": "0",
    "DECODE_MASK_BIDIR": "0",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prefix", default="accstop_b1_L")
    ap.add_argument("--max-L", type=int, default=48)
    ap.add_argument("--lo", type=int, default=17)
    ap.add_argument("--hi", type=int, default=24)
    ap.add_argument(
        "--vph",
        type=float,
        default=16.0,
        help="phantom V level. X is bf16, so f ~ 6e-03 against a V of 1 sits at "
        "the rounding floor (61%% of the difference lands off o_proj(ones)); "
        "16 buys four bits of it back and f is divided out again.",
    )
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--at", type=int, default=19)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    import numpy as np

    import dflash_draft_decoder as DD
    import qwen3_4b_q4nx_inference as INF
    import qwen3_4b_q4nx_weights as gw

    model = args.model or INF.MODEL_DEFAULT

    # o_proj of the all-ones activation, from the host weights. The device runs
    # the q4k-requantized o_proj, so this direction is right to ~1e-2 -- which
    # is nothing against two hypotheses that differ by 25%.
    qm = gw.Q4nxModel(model)
    o_w = np.asarray(qm.layer_weights(0)["o"], np.float32)  # [DQ, D]
    dirv = o_w.sum(axis=0)  # o_proj(ones)
    del qm, o_w
    import gc

    gc.collect()

    dec = INF.FusedDecoder(
        model=model, max_L=args.max_L, template_prefix=args.prefix, env_extra=_ENV
    )
    assert dec.UNI_DEC == 1, f"needs UNI_DEC_OVERRIDE=1, got {dec.UNI_DEC}"
    n, DKT, K = dec.UNI_DEC, dec.DK_TOT_A, dec.K
    FROM = dec.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    assert dirv.size == K, (dirv.size, K)

    def read_x():
        dec.x_bo.sync(FROM, K * 2, 0)
        return np.frombuffer(dec.x_bo.map(), dtype=dec.bf16, count=K).astype(np.float32)

    rng = np.random.default_rng(7)
    k0 = (rng.standard_normal(DKT) * 0.1).astype(np.float32)  # ONE key, everywhere
    tok = int(INF.PARIS_FIRST)
    den = float(dirv @ dirv)

    print(
        "\n  Uniform softmax by construction (every seeded key is k0), one layer,\n"
        "  layer output = x + o_proj(attn). f is fit against o_proj(ones).\n"
        "     L    R      f        (R-L)/(R-1+rho) is LINEAR in L; (R-L)/(L-1+rho) is not"
    )
    rows = []
    for L in range(args.lo, min(args.hi, dec.maxL) + 1):
        fk = np.broadcast_to(k0, (n, L - 1, DKT)).copy()
        fv = np.zeros((n, L - 1, DKT), np.float32)  # v0 = 0; it cancels regardless
        pk = np.broadcast_to(k0, (n, 8, DKT)).copy()
        xs = []
        for vph in (args.vph, 0.0):
            dec.seed_kv(fk, fv, L - 1)
            dec.dispatch(tok, L - 1)  # appends the token's own key at L-1
            DD.append_context_kv(dec, pk, np.full((n, 8, DKT), vph, np.float32), L)
            dec.dispatch(tok, L - 1)
            xs.append(read_x())
        d = (xs[0] - xs[1]) / args.vph
        f = float(dirv @ d) / den
        # how much of the difference the fit explains -- if the difference is
        # not along o_proj(ones) the construction is not doing what it claims
        resid = float(np.linalg.norm(d - f * dirv) / max(np.linalg.norm(d), 1e-30))
        R = -(-L // 8) * 8
        rows.append((L, R, f))
        print(f"    {L:3d}  {R:3d}   {f:+.6f}   (off-direction residual {resid:.3f})")

    if args.structure:
        # WHERE the phantom V lands. `f` is constant in the phantom ROW COUNT,
        # so the contribution is not per-row; and most of the difference is not
        # along o_proj(ones), so it is not uniform across the output either.
        # Sweep one kv head at a time, then one row at a time, and read off
        # which slices carry it.
        L = args.at
        fk = np.broadcast_to(k0, (n, L - 1, DKT)).copy()
        fv = np.zeros((n, L - 1, DKT), np.float32)
        pk = np.broadcast_to(k0, (n, 8, DKT)).copy()
        HD, NH = 128, DKT // 128

        def diff(pv):
            xs = []
            for scale in (1.0, 0.0):
                dec.seed_kv(fk, fv, L - 1)
                dec.dispatch(tok, L - 1)
                DD.append_context_kv(dec, pk, pv * scale, L)
                dec.dispatch(tok, L - 1)
                xs.append(read_x())
            return (xs[0] - xs[1]) / args.vph

        base = np.linalg.norm(diff(np.full((n, 8, DKT), args.vph, np.float32)))
        print(f"\n  structure at L={L} (R={-(-L // 8) * 8}), ||d|| all-on = {base:.5f}")
        print("    per KV HEAD of the phantom rows:")
        for h in range(NH):
            pv = np.zeros((n, 8, DKT), np.float32)
            pv[:, :, h * HD : (h + 1) * HD] = args.vph
            print(f"      head {h}: ||d|| {np.linalg.norm(diff(pv)):.5f}")
        print("    per phantom ROW (all heads):")
        for r in range(8):
            pv = np.zeros((n, 8, DKT), np.float32)
            pv[:, r, :] = args.vph
            print(
                f"      row L+{r} (pos {L + r}): ||d|| {np.linalg.norm(diff(pv)):.5f}"
            )
        return 0

    win = [(L, R, f) for L, R, f in rows if R == rows[0][1] and L < R]
    if len(win) >= 3:
        R = win[0][1]

        def fit(pred):
            best = None
            for rho in np.arange(0.0, 80.0, 0.005):
                e = sum((f - pred(L, R, rho)) ** 2 for L, _, f in win)
                if best is None or e < best[1]:
                    best = (float(rho), float(e))
            return best

        a_rho, a_err = fit(lambda L, R, r: (R - L) / (R - 1 + r))
        b_rho, b_err = fit(lambda L, R, r: (R - L) / (L - 1 + r))
        print(
            f"\n  over the R={R} window ({len(win)} points):\n"
            f"    (a) softmax includes the phantom keys : rho {a_rho:6.2f}  "
            f"residual {a_err:.3e}\n"
            f"    (b) attn_fv pairs valid scores wrongly: rho {b_rho:6.2f}  "
            f"residual {b_err:.3e}\n"
            f"  -> {'(a)' if a_err < b_err else '(b)'} by "
            f"{max(a_err, b_err) / max(min(a_err, b_err), 1e-30):.1f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
