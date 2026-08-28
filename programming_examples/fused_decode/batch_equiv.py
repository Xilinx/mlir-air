#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""THE device gate for batching. Batch-8 token t must equal batch-1 at position P+t.

WHAT CHANGED FROM THE FIRST VERSION OF THIS FILE, and why it matters. The first
version asserted that B IDENTICAL tokens give B IDENTICAL rows. That is false,
and the reason is the whole point of the batch: a block of B tokens occupies B
CONSECUTIVE positions, so token t attends to t more keys than token 0 and
rotates by a different RoPE angle. Identical inputs are supposed to give
DIFFERENT outputs. An "all rows equal" gate would have passed on an engine that
gave every token position P's context -- exactly the failure batch_attn_mask.py
was written to warn about.

THE PROPERTY THAT IS ACTUALLY TRUE, and that DFlash rests on:

    one batch-B dispatch at position P
      ==
    B batch-1 dispatches at positions P, P+1, ... P+B-1, same X each time

Both append the same K/V at the same positions and both mask token t to keys
0..P+t. They differ only in WHEN the appends happen -- the block does all B up
front -- and a key past a token's own L is masked, so that cannot show. If this
holds, speculative verify is lossless; if it does not, the batch is scoring
tokens against a context they should not see, and nothing downstream would say
so.

    --tokens 0        token 0 only: needs ONE batch-1 template, at the same L
                      as the batched build. Already covers the X feed, the
                      batched mmul, both egress gathers, the rms chunk
                      regeneration and residual accumulate, the glu row loop,
                      the QKV L2 transpose, rope, the q broadcast, the first
                      attention pass, the o gather and the layer-out drain.
    --tokens all      every token: needs a batch-1 template per position
                      (decode_b1_L{P+1+t}), because a non-DYNSEQ template bakes
                      L. What the extra templates buy is the ONE thing token 0
                      cannot see: that token t gets a DIFFERENT and correct
                      answer rather than a copy of token 0's.

WHAT IS READ BACK. Nothing -- the decode drains its layer output IN PLACE into
the X buffer (arg0), which is what makes layer chaining work. So the gate writes
X, dispatches, and reads X.

SYNTHETIC EVERYTHING, DELIBERATELY. This is a DATAFLOW gate: it asks whether
the engine moves the right bytes to the right places, which is where this design
has failed before and where the failures are silent. Numerics have their own
device gates (q4k_mm_gate.py, proj_qmm_gate.py) and they pass.

BUT NOT RANDOM BYTES, AND NOT AN ILL-CONDITIONED FILL EITHER. Three traps, each
of which produced a confident wrong answer before it was found:

  random bytes in a BO      the device returned 0x7F81 in every element -- one
                            uniform NaN -- because random bytes are random bf16
                            SCALES. A gate whose output is constant passes on
                            anything. The weights are REAL q4k blocks now.
  raw int16 for X           half the int16 range is a bf16 exponent of 0xF0 and
                            up; 0xFF80 alone is -inf. Every output came back
                            non-finite, from the INPUT.
  `min` drawn independently  of `scale`. In a real q4k block `min` is the block
                            minimum, so w is CENTRED; drawn independently it is
                            a perturbation on a mean of 7.5*scale and every dot
                            product becomes a cancelling sum. A 1.65% kernel
                            difference then reads as 772% at the layer output.

The reporting is built around the same lesson: a difference is read as a shape
(offset or scale, permuted or not, which tokens, how it moves with _WSCALE)
before it is read as a number. See _WSCALE and shape_of_difference.

The norms are near 1, X is order 1, and the KV cache starts at zero -- zero
rather than random because the two builds pad the cache to different lengths, so
"the same random bytes" would not be the same cache. The gate checks the output
is not constant before it compares anything.

    python3 batch_equiv.py --batch 8 --L 128
    python3 batch_equiv.py --batch 8 --L 128 --tokens all

Exit code is the gate: 0 equivalent.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Multiplier on the q4k scale, so the CONDITIONING of the test can be swept
# rather than argued about. This is the difference between a gate that measures
# the engine and one that measures floating point, and the sweep is the evidence
# that told the two apart [measured, llama-3.2-1b batch 8 vs batch 1 at L 1]:
#
#   scale   layer output   rope K
#   1.0        49%          1.1%
#   0.5        14%          1.1%
#   0.25       3.6%         1.1%
#   0.1        0.67%        1.2%
#
# Read the two columns against each other. K is produced by the projection and
# nothing else, and it does not move: 1.1% is the GEMV-to-mmul kernel swap,
# which proj_qmm_gate.py measures independently at 1.65%. The LAYER OUTPUT moves
# by two orders of magnitude over the same sweep, because silu sits between them
# and its relative error blows up near its own zero -- so the layer-output number
# is a function of how hard the nonlinearity is being driven, not of the wiring.
# A wiring fault would move BOTH columns and would not care about the scale.
#
# 0.1 puts the layer output below the K floor, which is as sensitive as this
# comparison can be: at that point the batched engine agrees with the batch-1
# engine more closely than the kernel swap alone allows.
_WSCALE = float(os.environ.get("BATCH_EQUIV_WSCALE", "0.1"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llms" / "bench"))


def geom(model, vocab_i2, L, batch, n_layers=1):
    """BO element counts, read off the builder rather than restated.

    decode_geometry.geometry() imports fused_decode.py under a given env and
    reads its sizes back. The batch and the context length are passed the same
    way the build passed them -- through the environment -- so a mismatch
    between what was built and what is dispatched is not expressible here.
    """
    import decode_geometry as dg

    old = {k: os.environ.get(k) for k in ("DECODE_BATCH", "DECODE_GOLDEN_L")}
    os.environ["DECODE_BATCH"] = str(batch)
    os.environ["DECODE_GOLDEN_L"] = str(L)
    try:
        return dg.geometry(model, vocab_i2, L, n_layers=n_layers)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bf16(x):
    """float array -> its int16 bit pattern, which is what a BO holds."""
    from ml_dtypes import bfloat16

    return np.asarray(x, dtype=np.float32).astype(bfloat16).view(np.int16)


def weight_bo(n_elems, seed, n_distinct=64):
    """A weight BO of REAL q4k blocks, not random bytes.

    Random bytes are random bf16 SCALES, which is how the first version of this
    gate got a NaN in every output element. These are packed the way the
    kernel's dequant expects (w = q*scale + min) with scale and min small enough
    that a 2048-wide dot product stays in range.

    n_distinct blocks are packed and then tiled to fill the buffer: packing all
    39680 of them takes minutes and buys nothing, but ONE would make every
    row-block of the output identical and hide a whole class of permutation
    error. 64 is enough that neighbours differ.

    AND `min` IS -7.5*scale, NOT AN INDEPENDENT RANDOM, which is the difference
    between measuring the engine and measuring floating point. In a real q4k
    block `min` is the block minimum, so w = q*scale + min is CENTRED on zero.
    Drawn independently it is a small perturbation on a mean of 7.5*scale, and
    every dot product against a zero-mean activation becomes a sum of ~2048
    terms of magnitude 55 cancelling down to a result of magnitude 2.
    A 1.65% difference between two kernels -- which is what proj_qmm_gate.py
    measures between the GEMV and the batched mmul, and is expected -- then
    lands as a 20% difference in the result, compounds through each projection,
    and reaches the layer output as 772x. That is not a batching fault, and the
    first version of this fill spent a day looking like one.
    """
    import proj_qmm_pack as pk

    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_distinct):
        q = rng.integers(0, 16, size=(pk.ROW_BLOCK, pk.COL_BLOCK), dtype=np.uint8)
        # Scaled so the GATE-UP output stays inside silu's LUT. getActivationBf16
        # is a 64-bin linear approximation over about [-8, 8) with a truncating
        # out-of-range policy, so a gate value past that comes back as whatever
        # the clamped bin holds -- and which bin a near-8 value lands in flips on
        # a fraction of a percent of input. With the first scale here the gate-up
        # output had an rms of 3.7 and roughly 5% of 8192 elements per token were
        # outside the range, so the layer output was CHAOTIC: 101% apart between
        # two builds whose gate-up outputs agreed to 2%. Removing the silu
        # collapsed that to 9.6%, which is how it was found.
        #
        # A trained model does not do this -- the shipping engine uses the same
        # LUT and passes `make verify` -- so an out-of-range gate is a property
        # of the fill and nothing else.
        scale = _WSCALE * rng.uniform(0.0008, 0.004, size=(pk.ROW_BLOCK, pk.N_GROUPS))
        # Centred, with a little jitter so the mins are not a function of the
        # scales alone.
        mn = -7.5 * scale + rng.uniform(-0.2, 0.2, size=scale.shape) * scale
        blocks.append(pk.pack_q4k_block(q, scale, mn))
    slab = np.concatenate(blocks)
    reps = -(-n_elems // slab.size)
    return np.tile(slab, reps)[:n_elems]


def rms_bo(g, batch, seed):
    """The norm/LUT buffer, with EVERY token's rope LUT set to the same words.

    The LUT is per position, and the batched build carries B of them. Giving
    them all the same value is what makes token t's rotation comparable to a
    batch-1 dispatch at position P+t -- that batch-1 build has exactly one LUT,
    and it has to be this one.
    """
    rng = np.random.default_rng(seed)
    buf = bf16(rng.uniform(0.8, 1.2, size=g["rms_size"])).copy()
    lut_off, lut_len = g["rms_lut_off"], g["rope_w_len"]
    # cos/sin, so in [-1, 1]. The SAME block for every position: that is what
    # makes token t comparable to a batch-1 dispatch, whose single LUT has to be
    # this one. It does mean the gate cannot see a LUT INDEX error that lands on
    # another slot -- which is what --tokens all is for, since the causal L
    # still differs per token.
    lut = bf16(np.random.default_rng(seed + 77).uniform(-1.0, 1.0, size=lut_len))
    for t in range(batch):
        buf[lut_off + t * lut_len : lut_off + (t + 1) * lut_len] = lut
    return buf


def compare(got, want):
    """(bytes that differ, rms relative difference) between two layer outputs.

    NOT a byte compare, and the reason is a property of the design rather than
    a concession. The batch-1 template runs the v1 GEMV projection; the batched
    one runs the q4k mmul. Those are different kernels computing the same
    product, in different accumulation orders -- proj_qmm_gate.py already
    measured the batched one at 1.4x the GEMV's error and 1.7% rms apart at the
    PROJECTION OUTPUT, on device, off the same weights. Bit-equality was never
    available here, so a byte gate reports a difference the design has already
    accounted for and says nothing about whether the WIRING is right.

    What the wiring gets wrong looks different: a swapped stride, a token
    reading another token's row or the wrong position's context moves a large
    fraction of the row by O(1), not a few percent. So the number to read is the
    rms relative difference, and the byte count is kept only because it is the
    cheaper thing to eyeball.
    """

    def f32(v):
        return (v.view(np.int16).astype(np.uint16).astype(np.uint32) << 16).view(
            np.float32
        )

    fa, fb = f32(got).astype(np.float64), f32(want).astype(np.float64)
    # Non-finite is its own answer, not a bad rms. Random weights over many
    # layers overflow bf16 on their own, and a run that did is not evidence
    # about the wiring either way -- see --waves.
    ok = np.isfinite(fa) & np.isfinite(fb)
    nnf = int(fa.size - ok.sum())
    if not ok.any():
        return int((got != want).sum()), float("inf"), nnf
    den = float(np.sqrt(np.mean(fb[ok] ** 2)))
    num = float(np.sqrt(np.mean((fa[ok] - fb[ok]) ** 2)))
    return int((got != want).sum()), (num / den if den else num), nnf


def as_f32(v):
    """bf16 bit patterns -> float32. A BO holds the bit patterns."""
    return (v.view(np.int16).astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def kv_stage(gN, g1, kvN, kv1, batch, tol):
    """Compare the K and V the rope core appended, and gate on it.

    Returns False if anything here is wrong. A tap that costs nothing.

    Both runs already write the DDR KV cache and the gate already reads it
    back, so this needs no extra channel, no extra shim task and no rebuild --
    which matters, because the memtile taps DECODE_PROBE offers cannot be
    placed on the o-gather without deadlocking the loop close.

    What it covers is everything up to and including rope: the rms core's
    chunk regeneration, the chunk-major @xnorm feed, the tile-blocked X
    broadcast, the batched mmul, both egress gathers, the id-demux, the QKV L2
    transpose and rope itself. If token 0's K and V match batch 1's, the first
    half of the layer is right and a difference in the layer output is
    attention, the o gather, the GLU or the residual accumulate.

    Region-major layout: group g's K region starts at g*stride and its V region
    at (NGRP+g)*stride; token t sits at t*width inside. Batch 1 at the same L
    writes position L-1, which is token 0's slot.
    """
    w, stride, ngrp, L = gN["kv_region"]
    if (w, stride, ngrp, L) != tuple(g1["kv_region"]):
        print(f"    kv: geometry differs {gN['kv_region']} vs {g1['kv_region']}")
        return False
    ok = True
    base = (L - 1) * w
    print("    rope KV taps  (batched token 0 vs batch 1, same position):")
    for name, r0 in (("K", 0), ("V", ngrp)):
        for g in range(ngrp):
            off = (r0 + g) * stride + base
            a, b = kvN[off : off + w], kv1[off : off + w]
            n, rel, nnf = compare(a, b)
            # A few wild elements and a uniform tilt are different faults, and
            # the rms cannot tell them apart. `close` is the share that agrees
            # to 5%, which is where a rounding difference lives.
            fa, fb = as_f32(a).astype(np.float64), as_f32(b).astype(np.float64)
            den = np.where(np.abs(fb) > 0, np.abs(fb), 1.0)
            close = int(np.sum(np.abs(fa - fb) / den <= 0.05))
            # The VALUES right and the ORDER wrong is a descriptor; the values
            # wrong is arithmetic. Nothing else tells those apart when the
            # elements all sit within a few percent of each other, which is
            # exactly when an rms difference looks like a scale.
            perm = "PERMUTED" if np.array_equal(np.sort(fa), np.sort(fb)) else ""
            nz = fb != 0
            r = fa[nz] / fb[nz] if nz.any() else np.zeros(1)
            med = float(np.median(r))
            tight = int(np.sum(np.isclose(r, med, rtol=1e-2)))
            perm += f" ratio {med:.4g} ({tight}/{int(nz.sum())} within 1%)"
            print(
                f"      {name} group {g}: {n} of {w} bytes differ, rms rel {rel:.2e}"
                f", {close} of {w} within 5% {perm}"
                + (f", {nnf} NON-FINITE" if nnf else "")
                + ("" if rel <= tol and not nnf else "   <-- WRONG BEFORE ROPE")
            )
            ok = ok and rel <= tol and not nnf
            # Every token in this gate gets the SAME x and the SAME rope LUT, and
            # neither K nor V depends on position beyond the LUT -- so all B
            # tokens' K (and V) have to be bit-identical to each other. They come
            # out of ONE mmul over one A operand, so anything else is the batch
            # mixing tokens, which no comparison against batch 1 would separate
            # from arithmetic.
            same = sum(
                1
                for t in range(1, batch)
                if np.array_equal(
                    kvN[off + t * w : off + (t + 1) * w], kvN[off : off + w]
                )
            )
            if same != batch - 1:
                bad = [
                    t
                    for t in range(1, batch)
                    if not np.array_equal(
                        kvN[off + t * w : off + (t + 1) * w], kvN[off : off + w]
                    )
                ]
                # Zeros vs wrong values: a partially-written region and a
                # mis-computed one look the same in an rms and are different
                # faults. The cache starts at zero, so a zero is "never arrived".
                det = ", ".join(
                    f"t{t} rms rel "
                    f"{compare(kvN[off + t * w:off + (t + 1) * w], kvN[off:off + w])[1]:.2e}"
                    f" ({int(np.sum(kvN[off + t * w:off + (t + 1) * w] == 0))}/{w} still zero)"
                    for t in bad
                )
                print(
                    f"        only {same} of {batch - 1} other tokens match "
                    f"token 0 -- identical inputs, so the batch is MIXING "
                    f"TOKENS: {det}"
                )
                ok = False
            if g == 0:
                print("        ref " + " ".join(f"{v:10.4g}" for v in fb[:8]))
                print("        got " + " ".join(f"{v:10.4g}" for v in fa[:8]))
                for t in range(1, batch):
                    z = np.nonzero(kvN[off + t * w : off + (t + 1) * w] == 0)[0]
                    if z.size:
                        brk = np.nonzero(np.diff(z) != 1)[0]
                        runs = np.split(z, brk + 1)
                        print(
                            f"         t{t} zero runs: "
                            + ", ".join(f"[{r[0]}..{r[-1]}]" for r in runs[:6])
                        )
                    r = as_f32(kvN[off + t * w : off + (t + 1) * w]).astype(np.float64)
                    if not np.array_equal(r, fa):
                        # WHERE it stops matching says whether the transfer was
                        # cut short or landed somewhere else.
                        d = np.nonzero(r != fa)[0]
                        print(
                            f"         t{t} "
                            + " ".join(f"{v:10.4g}" for v in r[:8])
                            + f"   first differs at {int(d[0])}"
                        )
    return ok


def probe_stages(gN, g1, probeN, probe1, t, tol):
    """Compare the DECODE_PROBE taps stage by stage, batched token t vs batch 1.

    This is what turns "the layer output is wrong" into "wrong from HERE on".
    The layer output is the only thing that crosses the shim, so without the
    taps a numeric divergence anywhere in the layer looks the same at the end.
    Build both templates with DECODE_PROBE set and the taps say which stage
    first disagrees:

      Q  rope's q, after the whole block has landed in the q memtile
      O  the gathered attention output
      D  the GLU output, on its way to the down projection

    Nothing prints when the taps are off, which is the default.
    """
    taps = gN.get("probe") or {}
    if not taps:
        return
    print("    stage taps  (batched token vs batch 1):")
    first = True
    for k in ("Q", "O", "D"):
        if k not in taps or k not in (g1.get("probe") or {}):
            continue
        offN, ln = taps[k]
        off1, l1 = g1["probe"][k]
        if ln != l1:
            print(f"      {k}: per-token length {ln} vs {l1} -- not comparable")
            continue
        a = probeN[offN + t * ln : offN + (t + 1) * ln]
        b = probe1[off1 : off1 + l1]
        n, rel, nnf = compare(a, b)
        over = not (rel <= tol) or nnf
        note = "   <-- FIRST STAGE TO DIVERGE" if over and first else ""
        first = first and not over
        print(
            f"      {k}: {n} of {ln} bytes differ, rms rel {rel:.2e}"
            + (f", {nnf} NON-FINITE" if nnf else "")
            + note
        )


def shape_of_difference(yN, y1, batch, K):
    """What KIND of wrong, printed. `rms rel 771` on its own localises nothing.

    Each line separates a family of wiring fault that the others cannot:

      per-token rms      one row far larger than the rest is an accumulate
                         running too many times; all rows equal is a token
                         index that never reaches the data
      sorted-equal       the same VALUES in a different order is a gather or
                         a stride, not arithmetic
      scale              a constant ratio to the reference is a count -- a
                         refeed, a residual added B times, a doubled round
      leading run        how far in the two agree, which is where the first
                         descriptor diverges
    """
    a = as_f32(yN).astype(np.float64)
    b = as_f32(y1).astype(np.float64)
    rows = [a[t * K : (t + 1) * K] for t in range(batch)]
    print("\n  shape of the difference")
    print(f"    reference rms {np.sqrt(np.mean(b**2)):.4g}")
    print(
        "    batched rms per token  "
        + " ".join(f"{np.sqrt(np.mean(r**2)):.3g}" for r in rows)
    )
    same = sum(1 for r in rows[1:] if np.array_equal(r, rows[0]))
    print(f"    tokens identical to token 0: {same} of {batch - 1}")
    perm = np.array_equal(np.sort(rows[0]), np.sort(b))
    print(f"    token 0 is a PERMUTATION of the reference: {perm}")
    nz = b != 0
    if nz.any():
        ratio = rows[0][nz] / b[nz]
        print(
            f"    token 0 / reference: median {np.median(ratio):.4g}, "
            f"{np.sum(np.isclose(ratio, np.median(ratio), rtol=1e-2))} of "
            f"{nz.sum()} within 1% of it"
        )
    d = np.nonzero(yN[:K] != y1)[0]
    print(f"    they agree over the first {int(d[0]) if d.size else K} elements")
    print("    first 8   ref " + " ".join(f"{v:11.4g}" for v in b[:8]))
    print("              got " + " ".join(f"{v:11.4g}" for v in rows[0][:8]))
    # A row that repeats with the period of an egress round, a chunk or a
    # row-block is a descriptor that re-read the same window; the rms per
    # period says which one.
    for name, p in (("PAYLOAD 512", 512), ("chunk 512", 512), ("row-block 32", 32)):
        if p < K and p != 512 or name.startswith("PAYLOAD"):
            seg = rows[0][: K - K % p].reshape(-1, p)
            eq = int(sum(1 for r in seg[1:] if np.array_equal(r, seg[0])))
            print(f"    token 0 repeats every {name}: {eq} of {len(seg) - 1}")


def dispatch(xclbin, insts, g, batch, x_row, seed, xrt, wait_ms=60000, scalar=None):
    """One dispatch. Returns the X buffer afterwards -- the layer output.

    Raw int16, not floats: this compares BYTES. A bit-level difference a float
    compare would round away is exactly the kind of layout error the gate is
    for.

    `insts` is a path to a frozen insts.bin, or an already-assembled uint32
    array -- a DYNSEQ build has no frozen stream, it emits a TXN builder the
    host calls per dispatch. `scalar` is the trailing context-length argument
    such a build's kernel signature carries (decode_dynseq.dispatch_args).
    """
    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kn = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
    kern = xrt.kernel(ctx, kn.get_name())

    ib = (
        np.asarray(insts, dtype=np.uint32)
        if isinstance(insts, np.ndarray)
        else np.fromfile(str(insts), dtype=np.uint32)
    )
    i_bo = xrt.bo(dev, ib.nbytes, xrt.bo.cacheable, kern.group_id(1))
    i_bo.write(ib, 0)
    i_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # decode_geometry's own key names: k (X), w_elems, rms_size, ny (Y),
    # kv_elems -- not restated, read straight off the builder.
    names = ("x", "w", "r", "y", "kv")
    sizes = (g["k"], g["w_elems"], g["rms_size"], g["ny"], g["kv_elems"])
    bos = {}
    for i, (name, n) in enumerate(zip(names, sizes), start=3):
        bos[name] = xrt.bo(dev, n * 2, xrt.bo.host_only, kern.group_id(i))

    fills = {
        # Identical bytes in both dispatches, so any difference in the output is
        # attributable to the batching and nothing else.
        "w": weight_bo(g["w_elems"], seed + 1),
        "r": rms_bo(g, batch, seed + 2),
        "kv": np.zeros(g["kv_elems"], np.int16),
        "y": np.zeros(g["ny"], np.int16),
        # The SAME row, B times: the block is B copies of one token, so any
        # difference between rows comes from the POSITION, which is the thing
        # under test.
        "x": np.tile(x_row, batch)[: g["k"]],
    }
    for name, buf in fills.items():
        bos[name].write(buf, 0)
        bos[name].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _t0 = time.perf_counter()
    _extra = [] if scalar is None else [int(scalar)]
    st = kern(
        3, i_bo, ib.size, bos["x"], bos["w"], bos["r"], bos["y"], bos["kv"], *_extra
    ).wait(wait_ms)
    _el = time.perf_counter() - _t0
    for b in ("x", "y", "kv"):
        bos[b].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(bos["x"].map(), dtype=np.int16, count=g["k"]).copy()
    yout = np.frombuffer(bos["y"].map(), dtype=np.int16, count=g["ny"]).copy()
    kvout = np.frombuffer(bos["kv"].map(), dtype=np.int16, count=g["kv_elems"]).copy()
    if not str(st).endswith("COMPLETED"):
        # HOW FAR DID IT GET. A timeout says nothing about where; the output
        # buffers do. The layer output is written IN PLACE over X, the KV cache
        # is written by the rope append, and Y takes the host drains -- so
        # whether each one moved brackets the hang between the phases that write
        # them. Cheap, and the only progress signal a hung dispatch leaves.
        def moved(name, n, ref):
            got = np.frombuffer(bos[name].map(), dtype=np.int16, count=n)
            ix = np.nonzero(got != ref[:n])[0]
            if not ix.size:
                return f"{name}: nothing written"
            # Contiguous runs, not just a count: WHICH regions moved is what
            # localises the stall. The KV cache is region-major, so a run tells
            # you which token and which of K / V got there.
            brk = np.nonzero(np.diff(ix) != 1)[0]
            runs = np.split(ix, brk + 1)
            shown = ", ".join(f"[{r[0]}..{r[-1]}]" for r in runs[:8])
            more = f" (+{len(runs) - 8} more)" if len(runs) > 8 else ""
            return f"{name}: {ix.size} of {n} in {len(runs)} runs: {shown}{more}"

        print(f"  TIMEOUT after {_el:.1f}s. what the device managed to write:")
        for name, n in (("x", g["k"]), ("y", g["ny"]), ("kv", g["kv_elems"])):
            print("    " + moved(name, n, fills[name]))
        bos.clear()
        del i_bo, kern, ctx, xb, dev
        raise RuntimeError(f"dispatch state={st} ({Path(xclbin).name})")
    # pyxrt's objects have to go before the context and the device or the
    # interpreter segfaults at exit -- the shipping driver keeps an explicit
    # release order for the same reason.
    bos.clear()
    del i_bo, kern, ctx, xb, dev
    return out, yout, kvout


def template(prefix, batch, L):
    xb = HERE / f"{prefix}_b{batch}_L{L}.xclbin"
    return xb, xb.with_suffix("").with_suffix(".insts.bin")


# One DYNSEQ build serves every context length, so the gate stops needing a
# template per position -- which is what made `--tokens all` cost a build per
# token and kept the mask arithmetic untestable on device. The stream comes from
# the compiler-emitted TXN builder instead of a frozen insts.bin.
_TXN_CACHE = {}


def dynseq_insts(prefix, batch, build_L, arg):
    """The instruction stream for one dispatch of a DYNSEQ build.

    `build_L` names the template (its compile-time ATTN_MAXL); `arg` is the
    runtime context length the stream is assembled for -- and, under
    DECODE_MASK_MODE_RTP, the mask mode riding in its bit 30. The builder is
    cached because constructing it parses and JITs the emitted header, which is
    slow enough to dominate a multi-position sweep.
    """
    hdr = HERE / f"{prefix}_b{batch}_L{build_L}.txn.h"
    if not hdr.exists():
        sys.exit(
            f"{hdr.name} not found -- a DYNSEQ build emits a TXN builder, not an\n"
            f"insts.bin. Build it with\n"
            f"    DECODE_DYNSEQ=1 ./build_template.sh {batch} {build_L}"
        )
    key = str(hdr)
    if key not in _TXN_CACHE:
        air_py = str(HERE.parent.parent / "python")
        if air_py not in sys.path:
            sys.path.insert(0, air_py)
        from air.backend.txn_builder import TxnBuilder

        b = TxnBuilder(key)
        names = b.function_names
        if len(names) != 1:
            sys.exit(f"{hdr.name} declares {len(names)} builders; expected 1")
        _TXN_CACHE[key] = (b, names[0])
    b, name = _TXN_CACHE[key]
    return b(name, int(arg))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=128, help="token 0's context length")
    ap.add_argument("--prefix", default="decode")
    ap.add_argument(
        "--build-L",
        type=int,
        default=None,
        help="the L the DYNSEQ templates were BUILT at, if it differs from the "
        "L being dispatched. One dynseq build serves every runtime L up to its "
        "compile-time ATTN_MAXL, and --bidir dispatches the reference at "
        "L+B-1 -- so both templates have to be built high enough to cover that, "
        "e.g. --L 128 --batch 8 needs ATTN_MAXL >= 135, i.e. --build-L 144.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument(
        "--tokens",
        default="0",
        choices=["0", "all"],
        help="'0' needs one batch-1 template; 'all' needs one per position",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=5e-2,
        help="rms relative difference a token may show against its batch-1 "
        "reference. Not zero, and see compare(): the two builds run DIFFERENT "
        "projection kernels. What this is sized against is a wiring fault, "
        "which moves a row by O(1), not by a few percent.",
    )
    ap.add_argument(
        "--wait",
        type=int,
        default=60000,
        help="ms to wait for a dispatch. Raise it to tell a HANG from a "
        "slow run: a batched dispatch is B times the attention of a "
        "batch-1 one, and the default was chosen for batch 1.",
    )
    ap.add_argument(
        "--dynseq",
        action="store_true",
        help="the templates were built with DECODE_DYNSEQ=1: assemble each "
        "dispatch's stream from the emitted TXN builder instead of reading a "
        "frozen insts.bin. ONE build then serves every position, so --tokens "
        "all no longer needs a batch-1 template per token.",
    )
    ap.add_argument(
        "--bidir",
        action="store_true",
        help="check the BIDIRECTIONAL mask (DFlash's draft pass): every token "
        "of the block attends to the whole block, so compare EVERY token "
        "against a batch-1 dispatch at context L+B-1 -- which is what they "
        "should all now see. Use with a DECODE_MASK_BIDIR=1 build (mode baked "
        "in), or add --dynseq for a DECODE_MASK_MODE_RTP=1 build (mode set per "
        "dispatch in the RTP-L's bit 30).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="dispatch the batched build only and report that it completed. "
        "No reference, no comparison -- answers 'does it hang', which is "
        "the first thing a new wire fails at.",
    )
    args = ap.parse_args()
    # Under --dynseq the template NAME carries the build's ATTN_MAXL, while the
    # dispatch carries the runtime L. They are the same number unless --bidir
    # pushes a reference past the build's window.
    build_L = args.build_L if args.build_L is not None else args.L

    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("pyxrt not importable: this gate needs the NPU")

    # A DYNSEQ build's BOs are sized by its compile-time ATTN_MAXL, not by the L
    # being dispatched -- that is the whole point of it. Sizing them from the
    # runtime L instead hangs the dispatch with nothing written.
    geom_L = build_L if args.dynseq else args.L
    gN = geom(args.model, args.vocab_chunk_i2, geom_L, args.batch, args.n_layers)
    g1 = geom(args.model, args.vocab_chunk_i2, geom_L, 1, args.n_layers)
    rng = np.random.default_rng(args.seed)
    # bf16 of a bounded float, NOT raw int16. A BO holds bit patterns, and half
    # the int16 range is a bf16 exponent of 0xF0 and up -- 0xFF80 alone is -inf.
    # The first version of this line filled X from rng.integers and every output
    # came back non-finite, from the INPUT rather than from anything the engine
    # did. Same trap as the random-weight NaN this file already documents, one
    # buffer over.
    row = bf16(rng.uniform(-1.0, 1.0, size=g1["k"]))

    bn, bni = template(args.prefix, args.batch, build_L)
    if not bn.exists():
        sys.exit(
            f"{bn.name} not found. Build it from THIS tree with\n"
            f"    DECODE_BATCH={args.batch} DECODE_GOLDEN_L={args.L} "
            f"make compile-decode\n"
            f"and rename decode.* to {bn.stem}.*"
        )

    # fused_decode.py's MASK_MODE_BIT. Set in the RTP-L, it tells the attention
    # cores to give every token of the block the same context (the DFlash draft
    # pass) instead of the causal staircase (the verify pass).
    # Only meaningful with --dynseq; a DECODE_MASK_BIDIR build has the mode
    # folded into the core and ignores it.
    MASK_MODE_BIT = 1 << 30

    mode = "bidirectional" if args.bidir else "causal"
    print(
        f"\nbatch equivalence  [{args.model}, batch {args.batch}, L {args.L}"
        + (f", {mode}]" if (args.dynseq or args.bidir) else "]")
    )
    scalarN = None
    if args.dynseq:
        scalarN = args.L | (MASK_MODE_BIT if args.bidir else 0)
        bni = dynseq_insts(args.prefix, args.batch, build_L, scalarN)
    yN, probeN, kvN = dispatch(
        bn, bni, gN, args.batch, row, args.seed, xrt, args.wait, scalarN
    )
    print(f"  batch {args.batch}: dispatch COMPLETED, {yN.size} elements back")
    # A constant output makes every comparison below trivially true. This is not
    # hypothetical: random-byte weights produced 0x7F81 -- one NaN -- in every
    # element, and the gate would have "passed".
    uniq = np.unique(yN).size
    print(
        f"  {uniq} distinct values in the output" + ("" if uniq > 8 else "  <-- FLAT")
    )
    if uniq <= 8:
        print(
            "\n  The output carries no information, so nothing below would mean\n"
            "  anything. Check the weight fill (real q4k blocks, bounded scales)\n"
            "  before reading any comparison."
        )
        return 1
    if args.smoke:
        print(
            "\n  --smoke says the wire does not hang and nothing more. It does\n"
            "  NOT say the answers are right; run without it for that."
        )
        return 0

    K = g1["k"]
    positions = range(args.batch if (args.tokens == "all" or args.bidir) else 1)
    bad, missing, kv_ok = [], [], True
    _G1 = {geom_L: g1}  # reference geometry per context length; geom() is slow
    for t in positions:
        # Causal: token t is at position L+t-1 and sees L+t keys. Bidirectional:
        # every token sees the whole block, so every one of them should match the
        # SAME batch-1 dispatch -- the one at L+B-1. That single reference for all
        # B tokens is the check; a build that quietly kept the staircase would
        # match at t=B-1 and nowhere else.
        ref_L = (args.L + args.batch - 1) if args.bidir else (args.L + t)
        # The reference's geometry is a function of ITS context length, not the
        # batched build's: geom() sizes the KV BO from L. Feeding a template
        # built at L+B-1 the BOs for L hangs the dispatch with nothing written,
        # which reads exactly like a device fault and is not one.
        gk = geom_L if args.dynseq else ref_L
        if gk not in _G1:
            _G1[gk] = geom(args.model, args.vocab_chunk_i2, gk, 1, args.n_layers)
        g1t = _G1[gk]
        if args.dynseq:
            b1, _ = template(args.prefix, 1, build_L)
            b1i, s1 = dynseq_insts(args.prefix, 1, build_L, ref_L), ref_L
        else:
            b1, b1i = template(args.prefix, 1, ref_L)
            s1 = None
        if not b1.exists():
            missing.append(b1.name)
            continue
        y1, probe1, kv1 = dispatch(b1, b1i, g1t, 1, row, args.seed, xrt, args.wait, s1)
        got = yN[t * K : (t + 1) * K]
        n, rel, nnf = compare(got, y1)
        if not (rel <= args.tol) or nnf:
            bad.append((t, n, int(np.argmax(got != y1)), rel, nnf))
        print(
            f"  token {t} (L {ref_L}): {n} of {K} bytes differ, "
            f"rms rel {rel:.2e}"
            + (f", {nnf} NON-FINITE" if nnf else "")
            + ("" if rel <= args.tol and not nnf else f"   <-- FAIL")
        )
        if t == 0:
            kv_ok = kv_stage(gN, g1t, kvN, kv1, args.batch, args.tol)
        probe_stages(gN, g1t, probeN, probe1, t, args.tol)

    if missing:
        print(
            "\n  missing batch-1 references: "
            + ", ".join(missing)
            + "\n  Each position needs its own template -- a non-DYNSEQ build\n"
            "  bakes L. Build them with DECODE_GOLDEN_L=<L> and rename."
        )
        return 1
    if bad:
        t, n, first, rel, nnf = bad[0]
        print(
            f"\n  first mismatch: token {t}, element {first}, "
            f"{n} bytes differ, rms rel {rel:.2e}"
        )
        if nnf:
            print(
                f"  {nnf} elements are NOT FINITE. Read that first: random\n"
                "  weights over many layers overflow bf16 with no help from the\n"
                "  batching, and the comparison below means nothing until they\n"
                "  are gone. Rebuild both templates with UNI_WAVE_HI=1 and pass\n"
                "  --prefix for the one-wave pair."
            )
            return 1
        shape_of_difference(yN, y1, args.batch, K)
        print(
            "  Read WHICH tokens differ before anything else:\n"
            "    token 0 too        -> the batched PATH: the @xnorm chunk feed,\n"
            "                          the tile-blocked broadcast, a gather, or\n"
            "                          the rms chunk regeneration\n"
            "    only t > 0         -> the batched POSITION: the KV append slot,\n"
            "                          the rope LUT index, or the causal L\n"
            "    every t > 0 equal\n"
            "    to token 0         -> the position is not reaching the token at\n"
            "                          all; every one got position P"
        )
        return 1
    if not kv_ok:
        print(
            "\n  The layer outputs agree, and the KV taps do not. Read the taps:\n"
            "  they see the first half of the layer directly, so they catch what\n"
            "  a layer-output comparison averages away."
        )
        return 1
    n = len(list(positions))
    print(
        f"\n  {n} token{'s' if n > 1 else ''} agree with batch 1 at their own "
        f"position to within --tol {args.tol:.0e}, and every token's K and V "
        f"match -- GATE PASS"
    )
    if args.tokens == "0":
        print(
            "  Token 0 only. That covers the whole batched data path but NOT\n"
            "  that tokens 1.. get their own context; --tokens all does."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
