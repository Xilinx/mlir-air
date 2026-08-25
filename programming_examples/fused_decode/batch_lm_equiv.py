#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""THE device gate for the BATCHED LM HEAD. Token t's logits must equal batch 1's.

batch_equiv.py gates the decode engine and reads layer outputs. It cannot see
the LM head at all: the vocab waves write Y, not X, and at batch 1 they are a
different arm of a different core. This is the same question one arm over --

    one batch-B lm-head wave, B distinct embeddings in
      ==
    B batch-1 lm-head waves, one embedding each

-- and unlike the decode gate it is an EXACT statement of intent rather than an
approximate one, because the LM head has no position in it. No attention, no
rope, no KV: token t's logits are a pure function of token t's row. So B rows in
one wave must give the B answers B separate waves give, and any difference is
the batching.

HOW THE HEAD IS ISOLATED, given that it sits behind sixteen decode layers whose
output it reads. Not by trimming the wave range -- UNI_WAVE_LO=UNI_DEC builds
fine and then times out on device at batch 1 as well as batch 8, so a vocab wave
does not run standalone in this design and using it would gate the harness
rather than the head. Instead both templates are built with DECODE_ACC_STOP=1,
which drops the two residual ADDS while leaving every transfer in place: the
sequence is the shipping one, all sixteen layers run, and each one's output is
its input. So X still holds exactly the B rows the host wrote when the lm head
reads it, and the logits are a function of those rows alone.

    DECODE_NO_LM_WAVES=0 DECODE_ACC_STOP=1 ./build_template.sh 8 1   # -> lm_b8_L1
    DECODE_NO_LM_WAVES=0 DECODE_ACC_STOP=1 ./build_template.sh 1 1   # -> lm_b1_L1
    python3 batch_lm_equiv.py --batch 8 --prefix lm

DISTINCT ROWS, not B copies of one. The decode gate feeds one row B times
because there the tokens are supposed to differ by POSITION; here they are
supposed to differ by INPUT, and B copies of one row would pass on an engine
that broadcast token 0 to every output slot -- which is precisely the wiring
error a new drain path introduces.

The fills follow batch_equiv.py's, for its reasons (real q4k blocks, bf16 of a
bounded float rather than raw int16). Two things differ, both because that gate
never reaches the lm head:

  the FINAL NORM  lives in the last K elements of the rms BO, and that buffer is
                  a different SIZE at batch 1 and batch 8, so one seeded draw
                  over the whole buffer would give the two builds different norm
                  weights. It is drawn on its own.
  the WEIGHT BO   is sized for ALL UNI_DEC layers plus the head, not for one
                  layer. The lm-head waves read their weights at UNI_DEC*W_LAYER
                  -- 304 M elements in -- so a BO sized the way the decode gate
                  sizes it (one layer + the head, 101 M) has the head reading
                  past its end. That does not fault and it does not hang: the
                  dispatch reports COMPLETED and every logit comes back exactly
                  zero, which is a very convincing-looking wrong answer.

Exit code is the gate: 0 equivalent.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from batch_equiv import bf16, compare, geom, template, weight_bo  # noqa: E402


def rms_bo(g, seed):
    """The norm BO. Zero except the final norm, which both builds must share.

    The vocab wave reads exactly two things out of here: rmsW at the final-norm
    slot, and a dummy rmsW2 at offset 0 that exists only to keep the shared
    packet group hole-free. Nothing else is touched, so nothing else needs a
    value -- and giving the rest zeros is what keeps the two builds' buffers
    comparable when they are not the same length.
    """
    buf = np.zeros(g["rms_size"], np.int16)
    # The LAST K elements, which is where the builder puts model.norm.weight
    # (_final_norm_off = the rms slabs + the whole rope region, and the rope
    # region is the part that scales with the batch).
    nrm = bf16(np.random.default_rng(seed).uniform(0.8, 1.2, size=g["k1"]))
    buf[-g["k1"] :] = nrm
    return buf


def logit_view(y, g, batch):
    """[batch][UNI_LM*voc_chunk] out of the Y buffer.

    A wave writes B tokens' chunk of the vocab, token-major inside the wave:
    wave w, token t at decode_y + w*B*voc_chunk + t*voc_chunk. So a token's
    logits are UNI_LM strided slices, and the stride is the thing worth
    checking -- a batched drain that got it wrong would interleave two tokens'
    vocabularies and still fill the buffer.
    """
    base, vc = g["decode_y"], g["voc_chunk"]
    nw = g["voc_n"] // (vc * batch)
    return np.stack(
        [
            np.concatenate(
                [
                    y[base + (w * batch + t) * vc : base + (w * batch + t + 1) * vc]
                    for w in range(nw)
                ]
            )
            for t in range(batch)
        ]
    )


def dispatch(xclbin, insts, g, rows, wfill, seed, xrt, wait_ms=120000):
    """One dispatch. Returns the lm head's logits, [batch][UNI_LM*voc_chunk].

    `wfill` is passed in rather than built here because it is 773 MB on
    llama-3.2-1b and identical in every dispatch this gate makes.
    """
    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kn = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
    kern = xrt.kernel(ctx, kn.get_name())

    ib = np.fromfile(str(insts), dtype=np.uint32)
    i_bo = xrt.bo(dev, ib.nbytes, xrt.bo.cacheable, kern.group_id(1))
    i_bo.write(ib, 0)
    i_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    names = ("x", "w", "r", "y", "kv")
    sizes = (g["k"], g["w_elems"], g["rms_size"], g["ny"], g["kv_elems"])
    bos = {}
    for i, (name, n) in enumerate(zip(names, sizes), start=3):
        bos[name] = xrt.bo(dev, n * 2, xrt.bo.host_only, kern.group_id(i))

    fills = {
        "w": wfill,
        "r": rms_bo(g, seed + 2),
        "kv": np.zeros(g["kv_elems"], np.int16),
        "y": np.zeros(g["ny"], np.int16),
        "x": np.concatenate(rows)[: g["k"]],
    }
    for name, buf in fills.items():
        bos[name].write(buf, 0)
        bos[name].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    st = kern(3, i_bo, ib.size, bos["x"], bos["w"], bos["r"], bos["y"], bos["kv"]).wait(
        wait_ms
    )
    bos["y"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    y = np.frombuffer(bos["y"].map(), dtype=np.int16, count=g["ny"]).copy()
    if not str(st).endswith("COMPLETED"):
        # WHERE it stopped, from the only progress signal a hung dispatch
        # leaves. The vocab region fills token-major, so a prefix that landed
        # names the token the drain got to.
        lg = logit_view(y, g, len(rows))
        got = " ".join(
            f"t{t}:{int(np.count_nonzero(lg[t]))}/{lg.shape[1]}"
            for t in range(len(rows))
        )
        print(f"  TIMEOUT. logits written per token: {got}")
        bos.clear()
        del i_bo, kern, ctx, xb, dev
        raise RuntimeError(f"dispatch state={st} ({Path(xclbin).name})")
    bos.clear()
    del i_bo, kern, ctx, xb, dev
    return logit_view(y, g, len(rows))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=1, help="the templates' baked L")
    ap.add_argument("--prefix", default="lm")
    ap.add_argument("--seed", type=int, default=0)
    # 5e-2, and the same 5e-2 batch_equiv uses, for the same reason: the
    # batch-1 template runs the v1 GEMV and the batched one the q4k mmul, which
    # proj_qmm_gate.py measures 1.7% apart at the projection output. The lm head
    # IS one projection, so a per-token 1-2% is the kernel difference and
    # nothing else. What a WIRING error looks like is the matrix below.
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--wait", type=int, default=120000)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="dispatch the batched build only and report that it completed. "
        "Answers 'does it hang', which is what a new drain path fails at first.",
    )
    args = ap.parse_args()

    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("pyxrt not importable: this gate needs the NPU")

    # Sized for the WHOLE sequence: n_layers=1 is right for the decode gate
    # and silently wrong here (see the header). One cheap import to learn
    # UNI_DEC, then the real geometry.
    nl = geom(args.model, args.vocab_chunk_i2, args.L, 1)["uni_dec"]
    gN = geom(args.model, args.vocab_chunk_i2, args.L, args.batch, nl)
    g1 = geom(args.model, args.vocab_chunk_i2, args.L, 1, nl)
    # One token's row, for the rms fill and the X slicing. g["k"] is the whole
    # X buffer (B rows at batch B), so the row width comes from the batch-1 one.
    for g in (gN, g1):
        g["k1"] = g1["k"]
    rng = np.random.default_rng(args.seed)
    rows = [bf16(rng.uniform(-1.0, 1.0, size=g1["k"])) for _ in range(args.batch)]

    bn, bni = template(args.prefix, args.batch, args.L)
    b1, b1i = template(args.prefix, 1, args.L)
    for f in (bn, b1):
        if not f.exists():
            sys.exit(
                f"{f.name} not found. Build both lm-head templates from THIS tree:\n"
                f"    DECODE_NO_LM_WAVES=0 UNI_WAVE_LO=<UNI_DEC> "
                f"UNI_WAVE_HI=<UNI_DEC+1> ./build_template.sh <batch> {args.L}\n"
                f"and rename decode.* to {args.prefix}_b<batch>_L{args.L}.*"
            )

    print(f"\nlm-head batch equivalence  [{args.model}, batch {args.batch}]")
    print(f"  weight BO {gN['w_elems'] * 2 / 2**20:.0f} MiB ({nl} layers + head)")
    wfill = weight_bo(gN["w_elems"], args.seed + 1)
    yN = dispatch(bn, bni, gN, rows, wfill, args.seed, xrt, args.wait)
    print(f"  batch {args.batch}: dispatch COMPLETED, {yN.shape} logits back")
    uniq = np.unique(yN).size
    print(f"  {uniq} distinct values" + ("" if uniq > 8 else "   <-- FLAT"))
    if uniq <= 8:
        print(
            "\n  The output carries no information, so nothing below would mean\n"
            "  anything. Check the weight fill before reading any comparison."
        )
        return 1
    if args.smoke:
        print("\n  --smoke says the wire does not hang and nothing more.")
        return 0

    # A batched head that broadcast one token to every slot would pass every
    # per-token comparison below if the reference happened to agree with it, so
    # ask the cheap question first: are the rows even different?
    same = [
        t for t in range(1, args.batch) if np.array_equal(yN[t], yN[0])
    ]  # bit-equal
    if same:
        print(f"  tokens {same} are BIT-IDENTICAL to token 0   <-- FLAT ACROSS TOKENS")

    refs = [
        dispatch(b1, b1i, g1, [rows[t]], wfill, args.seed, xrt, args.wait)[0]
        for t in range(args.batch)
    ]

    bad = []
    for t in range(args.batch):
        n, rel, nnf = compare(yN[t], refs[t])
        ok = rel <= args.tol and not nnf
        if not ok:
            bad.append(t)
        print(
            f"  token {t}: {n} of {refs[t].size} bytes differ, rms rel {rel:.2e}"
            + (f", {nnf} NON-FINITE" if nnf else "")
            + ("" if ok else "   <-- OVER TOL")
        )

    # THE GATE, and the per-token numbers above are only its diagonal. A
    # tolerance on its own cannot separate "the batched kernel rounds
    # differently" from "token 3 got token 5's row" -- both land a few percent
    # off when the rows are drawn from one distribution. This can: every
    # token's logits must be MUCH closer to its own row's reference than to any
    # other row's. Two different rows through the same weights give an O(1)
    # difference; the same row through two kernels gives 1-2%.
    print("\n  each token vs every reference (own reference on the diagonal):")
    cross = np.array(
        [
            [compare(yN[t], refs[r])[1] for r in range(args.batch)]
            for t in range(args.batch)
        ]
    )
    hdr = "".join(f"{r:>9d}" for r in range(args.batch))
    print(f"        {hdr}")
    for t in range(args.batch):
        row = "".join(f"{cross[t, r]:>9.2e}" for r in range(args.batch))
        print(f"    t{t}  {row}")
    mismatched = [t for t in range(args.batch) if int(np.argmin(cross[t])) != t]
    margin = min(
        cross[t, r] / cross[t, t]
        for t in range(args.batch)
        for r in range(args.batch)
        if r != t and cross[t, t]
    )

    if bad or same or mismatched:
        if mismatched:
            print(f"\n  tokens {mismatched} are closest to ANOTHER token's reference.")
        print(
            f"\n  NOT EQUIVALENT: tokens {sorted(set(bad) | set(same) | set(mismatched))}."
            " The batched lm head is\n"
            "  not computing what B separate lm-head waves compute."
        )
        return 1
    print(
        f"\n  every token's logits match a batch-1 wave on ITS OWN row (tol "
        f"{args.tol}), and are {margin:.0f}x closer to that reference than to "
        f"any other -- GATE PASS"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
