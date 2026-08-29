#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The folded pre-pass's waves, on device, against the CPU reference.

Both families, and they are the same shape: 25 fc sub-waves and 20 context-K/V
sub-waves, all i2=1 j2=5 dest="rms". Two things come out of this that nothing
before it could give:

  correctness  does the extra-wave machinery (arm, weight feed, X through the
               rms core, proj scalars, egress) actually compute W.tap?
  BANDWIDTH    what GB/s does an extra wave reach? Every number in the fold's
               case -- the ~3.4 ms, the ~250 ms step, the 0.91x -- assumes an
               extra wave streams weights at the verify pass's ~11 GB/s. Ten
               short waves of 2-4 MB are not the same workload as one 2.26 GB
               stream, and if per-wave overhead dominates, the fold is not
               worth its complexity. See docs/DFlashFeasibility.md section 3.13.

WHAT COMES BACK IS NOT W.tap. The rms core's regen multiplies by the norm
weight and by a per-row scale, and the residual pass adds the result into the
same buffer the X arrived in, so the wave's output slot holds

    tap + W . (tap / rms(tap))

The norm weight is fed as ONES (that is what makes the forwarding need no
kernel change), and rms(tap) is a per-row scalar the host can compute because
the host wrote the tap. So the correction is exact:

    W . tap  =  (readback - tap) * rms(tap)

    python3 dflash_fc_wave_gate.py                  # correctness + timing
    python3 dflash_fc_wave_gate.py --reps 20        # more timing samples
"""

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "fused_decode"))

import numpy as np

import dflash_prepass_waves as P
from proj_qmm_pack import BLOCK_BF16


def _load_target_fd(batch, L, no_lm="0"):
    """fused_decode at the TARGET's geometry, with the fc wave configured.

    The same module the template was built from, loaded with the same
    environment, so every derived extent here (X slots, the ones offset, the
    extra BO length) is the builder's own arithmetic rather than a copy of it.
    """
    import importlib.util
    import json
    import os

    fd_draft = P._load_draft_fd()
    waves, _ = P.wave_specs(fd_draft)
    # EVERY wave, always -- `w_off` and the extra BO's extent are absolute, so a
    # build that omits a family would give the ones behind it different offsets
    # and this gate would be checking a different table from the shipping one.
    # Which waves actually DISPATCH is the launch range's business.

    for k in list(os.environ):
        if k.startswith("DECODE_"):
            os.environ.pop(k, None)
    os.environ.update(
        DECODE_MODEL="qwen3-4b",
        VOCAB_CHUNK_I2="30",
        LM_HEAD="0",
        NLAYERS="1",
        UNIFIED="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L=str(L),
        DECODE_NO_LM_WAVES=no_lm,
        DECODE_STACK="6080",
        DECODE_BATCH=str(batch),
        W_DUAL_CHAN="1",
        FUSED_DECODE_EMIT_ONLY="1",
        DECODE_EXTRA_WAVES=json.dumps([w.as_config() for w in waves]),
    )
    import re
    import subprocess

    fdpy = _HERE.parent.parent / "fused_decode" / "fused_decode.py"
    spec = importlib.util.spec_from_file_location("fd_target_fc", str(fdpy))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The ABI the template was built with, read off the EMITTED signature rather
    # than recomputed here. The emit only runs under __main__, so it takes a
    # subprocess -- worth the ~10 s: a BO sized by a second, drifting copy of the
    # builder's arithmetic is the failure that dispatches COMPLETED and writes
    # nothing (build_template.sh's header records the same class of bug).
    out = subprocess.run(
        [sys.executable, str(fdpy)],
        cwd=str(fdpy.parent),
        env={**os.environ, "FUSED_DECODE_EMIT_ONLY": "1"},
        capture_output=True,
        text=True,
    ).stdout
    m = re.search(r"func\.func @q4nx_decode\(([^)]*)\)", out)
    if not m:
        raise RuntimeError("could not find the emitted ABI; did the build fail?")
    abi = [int(x) for x in re.findall(r"memref<(\d+)xbf16>", m.group(1))]
    return mod, fd_draft, waves, abi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prefix", default="decode_b8_L154")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=154)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--corr", type=float, default=0.99)
    ap.add_argument("--timeout", type=int, default=60000)
    ap.add_argument("--no-lm-waves", default="0", dest="no_lm_waves")
    # A SECOND INSTRUCTION STREAM AGAINST THE SAME XCLBIN. UNI_WAVE_LO/HI are
    # build-time and they restrict only the launch loop -- "keeps ABI/CDO fixed
    # at UNI_DEC/UNI_LM", fused_decode.py says so where they are read -- so a
    # narrower range recompiles to the same device and a shorter insts.bin.
    # That is what lets the pre-pass's two halves sit in different DISPATCHES
    # without a second PDI, which is the whole point of folding it at all.
    ap.add_argument("--insts", default=None, help="insts.bin to dispatch instead")
    ap.add_argument("--only", default="", help="report only groups with this prefix")
    args = ap.parse_args()

    import ml_dtypes
    import pyxrt

    bf16 = ml_dtypes.bfloat16
    fd, fd_draft, waves, abi = _load_target_fd(args.batch, args.L, args.no_lm_waves)
    if len(abi) != 6:
        raise RuntimeError(f"expected 6 BOs (x,w,rms,y,kvc,extra), got {abi}")
    B, K = fd.BATCH, fd.K
    xslots = sorted({w.x_slot for w in waves})
    # Every sub-wave contributes to exactly one PROJECTION: fc, or one drafter
    # layer's context K/V. Which one is the only thing that differs between the
    # two families -- the shape, the X width, the arm and the readback are
    # identical, which is the whole point of giving the K/V waves dest="rms".
    groups = {}
    for k, w in enumerate(waves):
        groups.setdefault(w.group, []).append((k, w))
    print(
        f"[waves] {len(waves)} sub-waves in {len(groups)} projections, "
        f"X slots {xslots}, M={waves[0].m} K={waves[0].k}  "
        f"X_SLOTS={fd.X_SLOTS}  extra BO={fd.EXTRA_W_ELEMS}"
    )

    # ---- the extra BO, straight out of the shipped cache --------------------
    npz = np.load(_HERE / "_draft_q4nx_w2ch.npz")
    g_sub = P.geom_for(waves[0].m, waves[0].k, fd_draft)
    blob = np.concatenate(
        [P.fc_extra_bo(fd_draft, npz), P.ctxkv_extra_bo(fd_draft, npz)]
    )
    assert blob.size == fd.EXTRA_W_ELEMS, (blob.size, fd.EXTRA_W_ELEMS)
    n_sub = g_sub.n_blocks * BLOCK_BF16

    # ---- the X rows, and the CPU reference for each projection --------------
    rng = np.random.default_rng(0)
    tap = {sl: rng.normal(0, 1, (B, K)).astype(bf16) for sl in xslots}
    tapf = {sl: tap[sl].astype(np.float32) for sl in xslots}
    rms = {
        sl: np.sqrt((tapf[sl] * tapf[sl]).mean(-1, keepdims=True) + 1e-6)
        for sl in xslots
    }
    # fc(concat(h_0..h_4)) = sum_s W_s . h_s, and sub-wave (s, t) is rows
    # [t*M, (t+1)*M) of W_s -- so it contributes output band t of tap s and
    # nothing else. A context-K/V sub-wave is the same statement with one X and
    # `out_band` counted from the K/V window's first row rather than the slab's.
    # Build every reference from the SAME bytes the device holds.
    M = waves[0].m
    nband = {g: max(w.out_band for _, w in ws) + 1 for g, ws in groups.items()}
    ref = {g: np.zeros((B, nband[g] * M), np.float32) for g in groups}
    for g, ws in groups.items():
        for k, w in ws:
            Wk = P.dequant_cascade(blob[k * n_sub : (k + 1) * n_sub], w.m, w.k, g_sub)
            ref[g][:, w.out_band * M : (w.out_band + 1) * M] += tapf[w.x_slot] @ Wk.T

    # ---- device ------------------------------------------------------------
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(
        str(_HERE.parent.parent / "fused_decode" / f"{args.prefix}.xclbin")
    )
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    insts = np.fromfile(
        args.insts
        or (_HERE.parent.parent / "fused_decode" / f"{args.prefix}.insts.bin"),
        dtype=np.uint8,
    )
    TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, (
        pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    )
    bos = [
        pyxrt.bo(dev, int(n) * 2, pyxrt.bo.host_only, kern.group_id(3 + i))
        for i, n in enumerate(abi)
    ]
    x_bo, e_bo, r_bo = bos[0], bos[5], bos[2]
    ib = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    ib.write(insts, 0)
    ib.sync(TO)
    assert abi[0] == fd.X_SLOTS * B * K, (abi[0], fd.X_SLOTS * B * K)

    x_host = np.zeros(fd.X_SLOTS * B * K, bf16)
    for sl in xslots:
        x_host[sl * B * K : (sl + 1) * B * K] = tap[sl].reshape(-1)
    x_bo.write(x_host.view(np.int16), 0)
    # ONES on the norm weight: rms_chunk is already the strided gather this
    # wants, so w == 1 leaves only the per-row scale the host divides back out.
    r_host = np.zeros(r_bo.size() // 2, bf16)
    r_host[fd.RMS_ONES_OFF : fd.RMS_ONES_OFF + K] = bf16(1.0)
    r_bo.write(r_host.view(np.int16), 0)
    e_bo.write(np.asarray(blob).view(np.int16), 0)
    for b in (x_bo, r_bo, e_bo):
        b.sync(TO)

    ms, st = [], None
    for _ in range(args.reps):
        t0 = time.perf_counter()
        st = kern(3, ib, insts.size, *bos).wait(args.timeout)
        ms.append((time.perf_counter() - t0) * 1e3)
        if not str(st).endswith("COMPLETED"):
            break
    x_bo.sync(FROM)
    xall = np.frombuffer(x_bo.map(), dtype=bf16, count=fd.X_SLOTS * B * K)
    if not str(st).endswith("COMPLETED"):
        print(f"  DISPATCH {st}")
        for sl in range(fd.X_SLOTS):
            v = xall[sl * B * K : (sl + 1) * B * K].astype(np.float32)
            if np.count_nonzero(v):
                print(f"  X slot {sl:3d}: absmax {np.abs(v).max():.4g}")
        return 2

    # W.h = (readback - h) * rms(h), band by band: the norm weight is ones, so
    # the only thing left on the X the projection saw is the per-row 1/rms, and
    # the residual added the result into the row the tap arrived in.
    #
    # AND THE BAND IS ALWAYS 0. residual1 lands egress round r at column band r
    # of every token row, and an i2=1 wave has exactly round 0 -- the rms core
    # does not know WHICH output rows the wave computed, only that one round
    # arrived. So every sub-wave deposits into columns [0, M) of its own slot,
    # and it is `out_band` that says where those M values belong in the answer.
    got = {g: np.zeros_like(r) for g, r in ref.items()}
    src = slice(0, M)
    for g, ws in groups.items():
        for k, w in ws:
            sl = fd.EXTRA_OUT_SLOT[k]
            out = xall[sl * B * K : (sl + 1) * B * K].astype(np.float32).reshape(B, K)
            got[g][:, w.out_band * M : (w.out_band + 1) * M] += (
                out[:, src] - tapf[w.x_slot][:, src]
            ) * rms[w.x_slot]

    def cos(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    mb = fd.EXTRA_W_ELEMS * 2 / 1e6
    med = float(np.median(ms))
    print(f"  dispatch  median {med:.3f} ms over {len(ms)}  (min {min(ms):.3f})")
    print(f"  weights   {mb:.1f} MB over {len(waves)} waves")
    worst = 1.0
    for g in sorted(groups):
        if not g.startswith(args.only):
            continue
        rel = float(np.abs(got[g] - ref[g]).max() / max(np.abs(ref[g]).max(), 1e-9))
        c = cos(got[g], ref[g])
        worst = min(worst, c)
        print(
            f"  {g:<9} {len(groups[g]):2d} waves  rel {rel:.3e}  cos {c:.6f}"
            f"{'' if c >= args.corr else '   <-- FAIL'}"
        )
    ok = worst >= args.corr
    print()
    print("PASS" if ok else f"FAIL (worst cos {worst:.6f} < {args.corr})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
