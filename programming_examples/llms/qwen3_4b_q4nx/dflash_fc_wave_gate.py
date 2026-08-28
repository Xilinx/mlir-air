#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ONE fc wave of the folded pre-pass, on device, against the CPU reference.

This is the step that decides whether folding the pre-pass into the target's
program is worth doing at all, and it deliberately runs the fc wave ALONE --
`UNI_WAVE_LO/HI` restrict the launch loop to the extra wave, so the dispatch
contains no decode layer, no attention and no LM head. Two things come out of
it that nothing before it could give:

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
same buffer the X arrived in, so X slot 0 holds

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


def _load_target_fd(batch, L):
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
    fc = [w for w in waves if w.name.startswith("fc")][:1]

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
        DECODE_NO_LM_WAVES="0",
        DECODE_STACK="6080",
        DECODE_BATCH=str(batch),
        W_DUAL_CHAN="1",
        FUSED_DECODE_EMIT_ONLY="1",
        DECODE_EXTRA_WAVES=json.dumps([w.as_config() for w in fc]),
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
    return mod, fd_draft, fc[0], abi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prefix", default="decode_b8_L130")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--L", type=int, default=130)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--corr", type=float, default=0.99)
    args = ap.parse_args()

    import pyxrt

    fd, fd_draft, wave, abi = _load_target_fd(args.batch, args.L)
    if len(abi) != 6:
        raise RuntimeError(f"expected 6 BOs (x,w,rms,y,kvc,extra), got {abi}")
    B, K = fd.BATCH, fd.K
    print(
        f"[fc wave] {wave.name}: M={wave.m} K={wave.k} I2={wave.i2} J2={wave.j2} "
        f"x_slot={wave.x_slot}  X_SLOTS={fd.X_SLOTS}  extra BO={fd.EXTRA_W_ELEMS}"
    )

    # ---- the slab this wave reads, straight out of the shipped cache --------
    npz = np.load(_HERE / "_draft_q4nx_w2ch.npz")
    g_slab = P.geom_for(wave.m, wave.k, fd_draft)
    reord = np.asarray(npz["W_fc"]).reshape(-1, BLOCK_BF16)[P.fc_slab_perm(fd_draft)]
    slab = reord[: g_slab.n_blocks].reshape(-1)
    assert slab.size == fd.EXTRA_W_ELEMS, (slab.size, fd.EXTRA_W_ELEMS)

    # ---- the tap, and the CPU reference for what the wave should compute ----
    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(0)
    tap_b = rng.normal(0, 1, (B, K)).astype(bf16)
    tap_f = tap_b.astype(np.float32)  # exactly what the device will see
    W = P.dequant_cascade(slab, wave.m, wave.k, g_slab)  # what the device holds
    ref = tap_f @ W.T  # [B, M]

    # ---- device ------------------------------------------------------------
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(
        str(_HERE.parent.parent / "fused_decode" / f"{args.prefix}.xclbin")
    )
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    insts = np.fromfile(
        _HERE.parent.parent / "fused_decode" / f"{args.prefix}.insts.bin",
        dtype=np.uint8,
    )

    TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, (
        pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    )

    def mk(n_elems, grp):
        return pyxrt.bo(dev, int(n_elems) * 2, pyxrt.bo.host_only, kern.group_id(grp))

    ib = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    ib.write(insts, 0)
    ib.sync(TO)

    x_bo, w_bo, r_bo, y_bo, kvc, e_bo = (mk(n, 3 + i) for i, n in enumerate(abi))
    assert abi[0] == fd.X_SLOTS * B * K, (abi[0], fd.X_SLOTS * B * K)
    assert abi[5] == fd.EXTRA_W_ELEMS, (abi[5], fd.EXTRA_W_ELEMS)

    # X slot `x_slot` holds the tap; the ones run makes the regen a copy.
    x_host = np.zeros(fd.X_SLOTS * B * K, bf16)
    x_host[wave.x_slot * B * K : (wave.x_slot + 1) * B * K] = tap_b.reshape(-1)
    x_bo.write(x_host.view(np.int16), 0)
    x_bo.sync(TO)

    r_host = np.zeros(r_bo.size() // 2, bf16)
    r_host[fd.RMS_ONES_OFF : fd.RMS_ONES_OFF + K] = bf16(1.0)
    r_bo.write(r_host.view(np.int16), 0)
    r_bo.sync(TO)

    e_bo.write(np.asarray(slab).view(np.int16), 0)
    e_bo.sync(TO)

    ms = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        st = kern(3, ib, insts.size, x_bo, w_bo, r_bo, y_bo, kvc, e_bo).wait(60000)
        ms.append((time.perf_counter() - t0) * 1e3)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"fc wave dispatch state={st}")

    x_bo.sync(FROM)
    got = np.frombuffer(x_bo.map(), dtype=bf16, count=B * K).astype(np.float32)
    got = got.reshape(B, K)

    # (readback - tap) * rms(tap): the norm weight is ones, so the only thing
    # left on the X the projection saw is the per-row 1/rms the host can undo.
    r = np.sqrt((tap_f * tap_f).mean(-1, keepdims=True) + 1e-6)
    fixed = (got - tap_f) * r

    def cos(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    rel = float(np.abs(fixed - ref).max() / max(np.abs(ref).max(), 1e-9))
    c = cos(fixed, ref)
    mb = fd.EXTRA_W_ELEMS * 2 / 1e6
    med = float(np.median(ms))
    print(f"  dispatch  median {med:.3f} ms over {args.reps}  (min {min(ms):.3f})")
    print(f"  weights   {mb:.1f} MB  ->  {mb / med:.2f} GB/s")
    print(f"  W.tap     rel {rel:.3e}  cos {c:.6f}")
    ok = c >= args.corr
    print("\n" + ("PASS" if ok else f"FAIL (cos {c:.6f} < {args.corr})"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
