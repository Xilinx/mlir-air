#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does llama-3.2-1B's gate projection leave getActivationBf16's LUT domain?

THIS IS A MEASUREMENT, AND IT IS THE ONE THE BATCH-8 BISECT STOPPED AT. That
bisect cleared every stage of a decode layer except the silu (see
docs/DFlashFeasibility.md, "The batched engine and the real model"): with the
silu in, batch 8 disagrees with batch 1 by 4.0-4.8x at cos 0.07-0.24; with it
replaced by `up - gate`, the same two builds agree to cos 0.99. The proposed
mechanism is that the real gate-up runs off the end of the 64-entry LUT, where
a fraction of a percent of input -- which is all the q4k mmul and the v1 GEMV
differ by -- moves a value between bins that are not near each other. That
mechanism was INFERRED. This measures its premise.

`batch_equiv.py` already demonstrated the mechanism once, on synthetic weights:
at its original fill the gate-up had rms 3.7, ~5% of elements were out of range,
and two builds whose gate-up outputs agreed to 2% produced layer outputs 101%
apart -- collapsing to 9.6% when the silu was removed. That is the same
signature the real model now shows. The comment that survived that episode says
"a trained model does not do this". This asks whether that is true.

WHY THE HIDDEN STATE COMES FROM THE DEVICE. The gate-up input is
`rmsnorm(x + oproj)`, and rmsnorm throws the magnitude away -- so the answer
depends on the DIRECTION of a real hidden state and on nothing else about it.
A random direction would answer the wrong question: trained hidden states are
strongly anisotropic, and it is exactly the outlier dimensions that drive a
projection output large. So the state is read back from a real dispatch. The
`acc2` build (DECODE_ACC_STOP=2) is the right source because its layer output
IS `x + oproj` and the bisect measured it at cos 0.999 against batch 1 -- it is
the last point in the layer where the two engines are known to agree.

    python3 silu_range.py                   # batch 1, layer 0
    python3 silu_range.py --batch 8
    python3 silu_range.py --layers 0,7,15   # sweep depth

Needs the acc2 template pair beside the model:

    cd ../../fused_decode
    ./build_template.sh 1 128 && ./build_template.sh 1 127   # -> acc2_* after
    GLU_ROW_PROBE= DECODE_ACC_STOP=2 ...                     # (see the doc)

The exit code is NOT a gate -- this reports a distribution. Read the numbers.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# getActivationBf16, read off kernels/lut_based_ops.h: A_FUNC == A_SILU gives
# step_bits = -2, and bias = 32 with LUT_elems = 64. aie::linear_approx indexes
# with (x << -step_bits) + bias, so bin i covers x = (i - 32) / 4 and the table
# spans [-8, +7.75] in steps of 0.25. The 64th bin starts at 7.75, so the last
# representable input is just under 8.
LUT_LO, LUT_HI, LUT_STEP, LUT_N = -8.0, 8.0, 0.25, 64


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-l", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--prompt-ids", default="128000,791,6864,315,9822,374")
    ap.add_argument(
        "--layers",
        default="0",
        help="comma-separated layer indices whose gate/up weights to apply. "
        "The hidden state is layer 0's post-attention residual either way -- a "
        "deeper layer's weights on it is an approximation, but the question is "
        "whether the WEIGHTS can reach out of range and layer 0 alone would not "
        "answer that.",
    )
    ap.add_argument(
        "--identify",
        action="store_true",
        help="also run the FULL layer and identify which GLU formula the device "
        "is evaluating (needs the one_* template pair)",
    )
    ap.add_argument(
        "--full-prefix",
        default="one_b{B}_L",
        help="template family whose layer output is the whole layer",
    )
    ap.add_argument("--min-occ", type=int, default=200)
    ap.add_argument("--probe2-prefix", default="gp2_b{B}_L")
    ap.add_argument("--probe4-prefix", default="gp4_b{B}_L")
    ap.add_argument("--probe5-prefix", default="gp5_b{B}_L")
    ap.add_argument(
        "--acc-prefix",
        default="acc2_b{B}_L",
        help="template family whose layer output is x + oproj",
    )
    args = ap.parse_args()

    import numpy as np
    import llama32_1b_q4nx_inference as inf

    layers = [int(s) for s in args.layers.split(",") if s.strip()]

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def silu_lut(x):
        """getActivationBf16 emulated: 64 bins of 0.25 from -8, a linear fit per
        bin, and bf16 coefficients because the table is bf16. The out-of-range
        policy is the one thing here that is a GUESS -- clamping to the end bins
        is assumed. Everything the real model produces is in range anyway (which
        is the finding), so the guess does not carry the conclusion."""
        from ml_dtypes import bfloat16

        edge = LUT_LO + LUT_STEP * np.arange(LUT_N + 1)
        lo, hi = edge[:-1], edge[1:]
        # secant fit per bin, matching what a linear_approx table is built from
        a = (silu(hi) - silu(lo)) / LUT_STEP
        b = silu(lo) - a * lo
        a = np.asarray(a, bfloat16).astype(np.float32)
        b = np.asarray(b, bfloat16).astype(np.float32)
        idx = np.clip(np.floor((x - LUT_LO) / LUT_STEP).astype(np.int64), 0, LUT_N - 1)
        xb = np.asarray(np.asarray(x, bfloat16), np.float32)
        return np.asarray(a[idx] * xb + b[idx], bfloat16).astype(np.float32)

    def rel(a, b):
        b = np.asarray(b, np.float64)
        a = np.asarray(a, np.float64)
        d = float(np.sqrt(np.mean(b**2)))
        return float(np.sqrt(np.mean((a - b) ** 2))) / d if d else float("inf")

    # ---- a real hidden state, from a real prefill ----
    kv_path = HERE / "_batch_check_kv.npz"
    ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    inf.run_prefill(ids, args.seq_len, kv_path)
    pf = np.load(kv_path)
    fk, fv = pf["k"].astype(np.float32), pf["v"].astype(np.float32)
    first = int(pf["first"])
    P = fk.shape[1]

    rng = np.random.default_rng(0)
    toks = [first]
    while len(toks) < args.batch:
        t = int(rng.integers(1000, 100000))
        if t not in toks:
            toks.append(t)

    def post_attn(batch, tokens, positions_from, prefix=None):
        """x + oproj for `batch` tokens, read off the device."""
        pfx = prefix or args.acc_prefix.format(B=batch)
        dec = inf.FusedDecoder(P + args.batch, batch=batch, template_prefix=pfx)
        dec.seed_kv(fk, fv, P)
        out = []
        if batch > 1:
            dec.dispatch(tokens, positions_from)
        else:
            # one dispatch per token, each at its OWN position -- the same
            # reference the bisect used
            pass
        FROM = dec.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        if batch > 1:
            dec.x_bo.sync(FROM)
            out = [
                np.frombuffer(dec.x_bo.map(), dtype=dec.bf16, count=dec.K * batch)
                .astype(np.float32)
                .reshape(batch, dec.K)
                .copy()
            ]
        else:
            # Seeded ONCE, deliberately. Each dispatch appends its own K/V, so
            # token t attends to tokens 0..t-1 of the block exactly as the
            # batched run does. Re-seeding per token would zero those positions
            # and make the REFERENCE the wrong one -- which is not a subtle
            # failure: token 0 stays right and the rest degrade with t.
            for t, tok in enumerate(tokens):
                dec.dispatch(tok, positions_from + t)
                dec.x_bo.sync(FROM)
                out.append(
                    np.frombuffer(dec.x_bo.map(), dtype=dec.bf16, count=dec.K)
                    .astype(np.float32)
                    .copy()[None, :]
                )
        emb = np.stack([np.asarray(dec.embed[t], np.float32) for t in tokens])
        dec.close()
        del dec
        return np.concatenate(out, 0), emb

    print(f"\ngate-up  [llama-3.2-1b, batch {args.batch} vs 1, P={P}]")
    h, emb = post_attn(args.batch, toks, P)
    h1 = post_attn(1, toks, P)[0] if args.batch > 1 else h
    print(
        f"  x+oproj  batch {args.batch}: rms {np.sqrt((h**2).mean()):.4f} "
        f"max|.| {np.abs(h).max():.3f}"
    )
    if args.batch > 1:
        print(f"  x+oproj  batch 1 vs batch {args.batch}: {rel(h, h1)*100:.2f}%")
        # Is x_bo actually the layer OUTPUT? If the device never wrote back, h
        # is still the embedding that was uploaded -- which would look like a
        # real hidden state and quietly invalidate everything below.
        print(
            f"  x+oproj vs the EMBEDDING that went in: {rel(h, emb)*100:.1f}% "
            f"(a few % here would mean the readback is the input, not the output)"
        )
        # And where the 31% sits: a permutation fault puts a token's own
        # reference off the diagonal.
        print("\n  per-token, batch 8 row vs batch 1 reference (own on diagonal):")
        print("        " + "".join(f"{r:>8d}" for r in range(args.batch)))
        for t in range(args.batch):
            print(
                f"    t{t}  "
                + "".join(f"{rel(h[t], h1[r])*100:>7.1f}%" for r in range(args.batch))
            )

    # ---- the real weights, on the host ----
    import llama32_1b_q4nx_weights as W
    import llama32_1b_q4nx_prefill as PF

    model = W.Q4nxModel(PF.MODEL_DEFAULT)

    def rmsnorm(x, w, eps=1e-5):
        return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * w

    print(
        f"\n  RANGE -- is the gate anywhere near the edge of the table?\n"
        f"  {'layer':>5}  {'gate rms':>9}  {'gate max':>9}  "
        f"{'|g|>=8':>9}  {'up rms':>8}"
    )
    worst = 0.0
    cache = {}
    for k in layers:
        wq = model.layer_weights(k)
        _, w_ph2 = model.layer_rms(k)
        wp = np.asarray(w_ph2, np.float32)
        wg = np.asarray(wq["gate"], np.float32)
        wu = np.asarray(wq["up"], np.float32)
        g = rmsnorm(h, wp) @ wg
        u = rmsnorm(h, wp) @ wu
        cache[k] = (wp, wg, wu, g, u)
        frac8 = float((np.abs(g) >= 8.0).mean())
        worst = max(worst, frac8)
        print(
            f"  {k:>5}  {np.sqrt((g**2).mean()):>9.3f}  {np.abs(g).max():>9.2f}  "
            f"{frac8*100:>8.2f}%  {np.sqrt((u**2).mean()):>8.3f}"
        )
    print(
        f"  the table spans [{LUT_LO:g}, {LUT_HI:g}) in {LUT_N} bins of "
        f"{LUT_STEP:g}."
    )

    # ---- the decisive part: does the LUT amplify, on THESE values? ----
    # Feed both engines' real gate-ups through exact silu and through the LUT.
    # If the LUT is the amplifier, its column blows up where exact silu's does
    # not. If both columns are small, the silu is exonerated and the 4.4x has
    # to come from somewhere the bisect did not separate.
    if args.batch > 1:
        print(
            f"\n  AMPLIFICATION -- batch {args.batch} vs batch 1, on the same "
            f"weights\n"
            f"  {'layer':>5}  {'gate in':>9}  {'exact out':>10}  {'LUT out':>9}  "
            f"{'LUT gain':>9}"
        )
        for k in layers:
            wp, wg, wu, g, u = cache[k]
            g1 = rmsnorm(h1, wp) @ wg
            u1 = rmsnorm(h1, wp) @ wu
            d_in = rel(g, g1)
            d_ex = rel(silu(g) * u, silu(g1) * u1)
            d_lut = rel(silu_lut(g) * u, silu_lut(g1) * u1)
            print(
                f"  {k:>5}  {d_in*100:>8.2f}%  {d_ex*100:>9.2f}%  "
                f"{d_lut*100:>8.2f}%  {d_lut/d_ex if d_ex else float('nan'):>8.1f}x"
            )
        print(
            "\n  'gate in' is how far apart the two engines' gate-ups are before\n"
            "  the nonlinearity; 'exact out' and 'LUT out' are how far apart the\n"
            "  GLU outputs are after it. A LUT gain near 1 means the silu passes\n"
            "  the difference through and is NOT what turns it into 4x."
        )

    # And what the LUT costs on its own, which is a property of the shipping
    # engine and not of batching at all.
    print(f"\n  LUT ERROR vs exact silu (batch {args.batch} inputs):")
    for k in layers:
        _, _, _, g, u = cache[k]
        print(
            f"    layer {k:>2}: silu {rel(silu_lut(g), silu(g))*100:>7.2f}%   "
            f"GLU out {rel(silu_lut(g)*u, silu(g)*u)*100:>7.2f}%"
        )

    # ---- IDENTIFY what the device actually computes ----
    # Everything above says the layer output should be ~3% apart, and the device
    # says 4.4x. So the device is not evaluating the formula this script models.
    # Rather than guess which one it IS, run the whole layer on the host under
    # each candidate and see which reproduces the measured output. `full` is the
    # same one-wave build with the accumulator left alone.
    if not args.identify:
        return 0
    fpfx = args.full_prefix.format(B=args.batch)
    print(f"\n  IDENTIFY -- full layer 0 from {fpfx}, against host candidates")
    dev, _ = post_attn(args.batch, toks, P, prefix=fpfx)
    dev1 = post_attn(1, toks, P, prefix=args.full_prefix.format(B=1))[0]
    print(
        f"    device batch {args.batch} vs batch 1: {rel(dev, dev1)*100:.1f}%  "
        f"rms ratio {np.sqrt((dev**2).mean())/np.sqrt((dev1**2).mean()):.2f}"
    )
    wp, wg, wu, g, u = cache[layers[0]]
    wd = np.asarray(model.layer_weights(layers[0])["down"], np.float32)
    S = silu_lut
    cands = {
        "silu(gate)*up   (correct)": S(g) * u,
        "silu(up)*gate   (halves swapped)": S(u) * g,
        "gate*up         (no silu)": g * u,
        "silu(gate)*gate (up lost)": S(g) * g,
        "silu(up)*up     (gate lost)": S(u) * u,
        "up - gate       (the probe)": u - g,
        "silu(gate)      (up = 1)": S(g),
        "up              (silu skipped)": u,
    }
    # A GLU formula fault is not the only shape available. The batched down
    # projection accumulates, and an accumulator that is not cleared between
    # tokens gives every token the SUM of the block's contributions -- which for
    # 8 partially-correlated rows lands somewhere between sqrt(8) and 8 times
    # too large, and 4.23 is inside that.
    correct = S(g) * u
    cands["SUM of all tokens' glu (acc not cleared)"] = np.broadcast_to(
        correct.sum(0), correct.shape
    )
    cands["CUMSUM over the block (acc leaks fwd)"] = np.cumsum(correct, 0)
    # THE CONTROL. The same host model against the batch-1 device output, where
    # the device is the shipping engine and is known good. If this does not
    # match, the host model is what is wrong and the table below means nothing.
    g1 = rmsnorm(h1, wp) @ wg
    u1 = rmsnorm(h1, wp) @ wu
    ctl = h1 + (S(g1) * u1) @ wd
    print(
        f"    CONTROL  host model vs BATCH-1 device: {rel(ctl, dev1)*100:.1f}%  "
        f"rms ratio {np.sqrt((ctl**2).mean())/np.sqrt((dev1**2).mean()):.2f}"
    )
    print(f"    {'candidate':<34} {'vs device':>10} {'rms ratio':>10}")
    for nm, gl in cands.items():
        pred = h + gl @ wd
        r = np.sqrt((pred**2).mean()) / np.sqrt((dev**2).mean())
        print(f"    {nm:<34} {rel(pred, dev)*100:>9.1f}% {r:>10.2f}")
    # Per-token, for the correct formula. A per-token error that GROWS with t
    # is an accumulator leaking forward; one that is flat is a whole-block fault.
    pc = h + correct @ wd
    print(
        "    per-token, correct formula vs device: "
        + " ".join(f"t{t}:{rel(pc[t], dev[t])*100:.0f}%" for t in range(args.batch))
    )
    print(
        "    per-token device/host rms:            "
        + " ".join(
            f"t{t}:{np.sqrt((dev[t]**2).mean())/np.sqrt((pc[t]**2).mean()):.1f}x"
            for t in range(args.batch)
        )
    )
    # Direction vs magnitude. Subtract the residual (which is known correct) and
    # ask whether what the device added to it POINTS the right way. A high
    # cosine with a scale far from 1 is an accumulation-count bug -- the right
    # sum taken too many times -- and is a different repair from a wrong sum.
    dg = dev - h  # what the device's GLU+down branch contributed
    hg = correct @ wd  # what it should have
    cos = float((dg * hg).sum() / (np.sqrt((dg**2).sum()) * np.sqrt((hg**2).sum())))
    k = float((dg * hg).sum() / (hg**2).sum())  # least-squares scale
    resid = dg - k * hg
    print(
        f"\n    device_glu_branch vs host_glu_branch:  cos {cos:.4f}  "
        f"best-fit scale {k:.2f}\n"
        f"    residual after removing {k:.2f}x: "
        f"{np.sqrt((resid**2).mean())/np.sqrt((dg**2).mean())*100:.1f}% of it"
    )
    # Same decomposition per token: is the scale the same for every row?
    print(
        "    per-token scale: "
        + " ".join(
            f"t{t}:{float((dg[t]*hg[t]).sum()/(hg[t]**2).sum()):.2f}"
            for t in range(args.batch)
        )
    )
    print(
        "\n    The candidate that matches the device is what the batched GLU\n"
        "    computes. If NONE match, the fault is upstream of the GLU formula."
    )

    # ---- the probe builds, against the SAME validated host model ----
    # The bisect read GLU_ROW_PROBE=2 as "the plumbing is clean, so it is the
    # silu". But probe 2 computes `up - gate`, whose rms is ~13x the real GLU
    # output, so it can only clear the plumbing for a signal that large. Check
    # it the way the full layer was just checked -- against a host prediction,
    # not against another device build.
    probes = {}
    for pf, name, gl in (
        (
            args.probe2_prefix,
            "GLU_ROW_PROBE=2  y = up - gate      (13x too big)",
            u - g,
        ),
        (args.probe4_prefix, "GLU_ROW_PROBE=4  y = gate*up        (no LUT)", g * u),
        (args.probe5_prefix, "GLU_ROW_PROBE=5  y = silu(gate)     (LUT alone)", S(g)),
    ):
        if not pf:
            continue
        try:
            pv, _ = post_attn(args.batch, toks, P, prefix=pf.format(B=args.batch))
        except Exception as e:  # template not built -- say so and move on
            print(f"\n    {name}: no template ({type(e).__name__})")
            continue
        pred = h + gl @ wd
        pdg, phg = pv - h, gl @ wd
        pcos = float(
            (pdg * phg).sum() / (np.sqrt((pdg**2).sum()) * np.sqrt((phg**2).sum()))
        )
        print(
            f"\n    {name}\n"
            f"      vs host prediction: {rel(pred, pv)*100:.1f}%   "
            f"rms ratio {np.sqrt((pv**2).mean())/np.sqrt((pred**2).mean()):.2f}   "
            f"branch cos {pcos:.4f}"
        )
        probes[pf] = pv

    # ---- read the device's silu curve back out ----
    # Probe 5's GLU output is f(gate) for a SCALAR f, so even though only
    # `f(gate) @ wd` ever crosses the shim, f is recoverable: bin the gate into
    # the LUT's own 64 bins and each bin's value is one unknown against 2048
    # equations per token. 64 unknowns, 16384 equations -- heavily
    # overdetermined, and it needs no new build and no device probe tap.
    pv5 = probes.get(args.probe5_prefix)
    if pv5 is None:
        return 0
    edge = LUT_LO + LUT_STEP * np.arange(LUT_N + 1)
    ctr = (edge[:-1] + edge[1:]) / 2
    idx = np.clip(np.floor((g - LUT_LO) / LUT_STEP).astype(np.int64), 0, LUT_N - 1)

    def recover(target):
        """Least-squares f over the 64 bins, from `target` = f(gate) @ wd."""
        rows, rhs = [], []
        for t in range(args.batch):
            ind = np.zeros((g.shape[1], LUT_N), np.float32)
            ind[np.arange(g.shape[1]), idx[t]] = 1.0
            rows.append((ind.T @ wd).T)  # [D, LUT_N]
            rhs.append(target[t])
        A = np.concatenate(rows, 0).astype(np.float64)
        y = np.concatenate(rhs, 0).astype(np.float64)
        keep = A.std(0) > 0
        c = np.full(LUT_N, np.nan)
        c[keep] = np.linalg.lstsq(A[:, keep], y, rcond=None)[0]
        return c

    # CONTROL: recover from a host target whose f is known exactly. If this does
    # not come back as silu, the recovery is what is broken, not the device.
    ctl_c = recover(np.stack([S(g[t]) @ wd for t in range(args.batch)]))
    dev_c = recover(pv5 - h)
    occ = np.bincount(idx.ravel(), minlength=LUT_N)
    print("\n    RECOVERED silu curve on the batched GLU core")
    print(f"    {'x':>7} {'n':>7} {'true':>9} {'DEVICE':>9} {'control':>9}")
    for b in range(LUT_N):
        if occ[b] < args.min_occ or not np.isfinite(dev_c[b]):
            continue
        print(
            f"    {ctr[b]:>7.2f} {occ[b]:>7d} {silu(ctr[b]):>9.3f} "
            f"{dev_c[b]:>9.3f} {ctl_c[b]:>9.3f}"
        )
    ok = np.isfinite(dev_c) & (occ > args.min_occ)
    # Is it an affine map? One (slope, intercept) fitting the whole curve means
    # the table is handing back the SAME entry for every input -- the index is
    # not moving -- which is a different fault from a corrupted table.
    A2 = np.stack([ctr[ok], np.ones(int(ok.sum()))], 1)
    (sl, ic), *_ = np.linalg.lstsq(A2, dev_c[ok], rcond=None)
    print(
        f"\n    best affine fit to the DEVICE curve: {sl:.3f}*x + {ic:.3f}"
        f"   (residual {rel(A2 @ [sl, ic], dev_c[ok])*100:.1f}%)"
        f"\n    true silu over the same bins is close to 0.5*x + 0.0"
    )
    print(
        f"\n    control vs true silu: {rel(ctl_c[ok], silu(ctr[ok]))*100:.1f}%"
        f"   (validates the recovery)\n"
        f"    DEVICE  vs true silu: {rel(dev_c[ok], silu(ctr[ok]))*100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
