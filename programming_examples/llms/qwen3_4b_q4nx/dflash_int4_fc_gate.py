#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the int4-AWQ `fc` (+ hidden_norm), split along K.

The quantized form of the tap fusion that dflash_fc_gate.py runs in bf16:
65.5 MB of weight becomes 17.2 MB, which is the point -- fc is otherwise a 21%
surcharge on a draft pass whose 5 Q4 decode layers are ~315 MB.

Two GEMM launches rather than one because the int4 GEMM is only correct when
tile_k_l2 == K and K=12800 does not fit L2; then an add and a norm, because a
herd tile takes at most two incoming L3 streams. Both constraints, and how they
were measured, are in dflash_int4_fc_builder.py.

THE REFERENCE IS THE DEQUANTIZED WEIGHT, NOT THE ORIGINAL. Comparing an int4
GEMM against full-precision fc measures the quantization, which
dflash_int4.self_check already reports (5.5e-02 against a 1.1e-01 step) and
which is not what this gate is for. Comparing against `awq_dequantize(...)`
isolates the ENGINE: whether the packing, the tile layout and the dequant in
the kernel agree with the packer. Both numbers are printed so neither hides the
other.

WHAT THIS GATE IS BUILT TO CATCH beyond "is the arithmetic right":

  a chunk/weight-block swap    the GEMMs are the same shape and differ only in
                               which (A_i, W_i) pair they are given, so a
                               mis-wired arg map produces a plausible result. A
                               cross-pair distance matrix is printed so
                               "partial 1 got block 0" cannot read as a pass.
  a dropped partial            the tail is checked against the DEVICE's own
                               partials as well as end to end, so an add that
                               silently passed one operand through shows up on
                               its own rather than being absorbed by the norm's
                               scale-invariance.

    python3 dflash_int4_fc_gate.py
    python3 dflash_int4_fc_gate.py --synthetic     # random weights, no checkpoint
    python3 dflash_int4_fc_gate.py --compile-only
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

M = 32  # padded context rows (tile_m 16 x herd_m 1 x 2 outer tiles)
D = 2560
FC_IN = 12800


def _rel(a, b):
    import numpy as np

    return np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b**2).mean()), 1e-9)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=None, help="K chunks (default 2)")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="use matmul_int4_packed's OWN random weight construction at this "
        "shape instead of the real fc. Isolates the quantizer from the "
        "shape/tile choice: if this passes and the real fc does not, the "
        "quantizer is wrong; if both fail, the shape or tiles are.",
    )
    ap.add_argument("--no-norm", action="store_true", help="GEMMs only, no add/norm")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_int4 as I
    import dflash_int4_fc_builder as FB
    import dflash_sumnorm

    I.paths()

    # NOT compile_mv_int4_bf16: that builds the GEMV (-DDIM_K, no -DDIM_N).
    # The GEMV object links, loads, runs, and returns NaNs.
    I.compile_int4_gemm_kernel()

    P = args.chunks or FB.N_CHUNKS
    KC = FC_IN // P
    with_norm = not args.no_norm
    module = FB.build_int4_fc_module(M, P, with_norm=with_norm)
    print(
        f"[int4 fc] module: {str(module).count('air.launch')} launches "
        f"({P} GEMM K={KC}" + (f" + {P-1} add + norm)" if with_norm else ")")
    )

    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="dflash_int4_fc",
        runtime_loop_tiling_sizes=[2, 2],
        # Without this the identical module runs to completion and returns
        # rel ~0.85 instead of 7.4e-03. No crash, no diagnostic.
        stack_size=16384,
    )
    compiled = backend.compile(module)
    if args.compile_only:
        backend.unload()
        print("[int4 fc] compile-only done")
        return 0

    rng = np.random.default_rng(0)
    if args.synthetic:
        rs = np.random.default_rng(42)
        Ws, qs, ss, zs = [], [], [], []
        for i in range(P):
            unp = rs.integers(0, 16, size=(D, KC), dtype=np.uint8)
            q = (unp[:, 0::2] | (unp[:, 1::2] << 4)).astype(np.uint8)
            s = rs.uniform(0.005, 0.02, size=(KC // I.GS, D)).astype(bfloat16)
            z = rs.integers(7, 9, size=(KC // I.GS, D), dtype=np.uint8)
            Ws.append(I.awq_dequantize(q, s, z))
            qs.append(q)
            ss.append(s)
            zs.append(z)
        hn_w = rs.normal(1.0, 0.1, D).astype(bfloat16)
    else:
        from qwen3_4b_draft_weights import DraftWeights

        dw = DraftWeights()
        Ws = FB.split_fc_weight(np.asarray(dw.fc()), P)
        qs, ss, zs = zip(*(I.awq_quantize(W) for W in Ws))
        qs, ss, zs = list(qs), list(ss), list(zs)
        hn_w = np.asarray(dw.hidden_norm(), bfloat16)

    Bs = [
        np.ascontiguousarray(I.pack_for_device(qs[i], ss[i], zs[i], M, KC, D))
        for i in range(P)
    ]
    W_dq = [I.awq_dequantize(qs[i], ss[i], zs[i]) for i in range(P)]

    taps = np.zeros((M, FC_IN), bfloat16)
    taps[: args.ctx] = rng.normal(0, 1, (args.ctx, FC_IN)).astype(bfloat16)
    As = FB.split_taps(taps, P)

    ins = list(As) + Bs + [np.zeros((M, D), bfloat16) for _ in range(P)]
    if with_norm:
        ins += [np.zeros((M, D), bfloat16) for _ in range(P - 1)]  # fold scratch
        ins += [hn_w, np.zeros((M, D), bfloat16)]

    import filelock

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        res = fn(*ins)
    backend.unload()

    parts = [
        np.asarray(res[2 * P + i]).reshape(M, D).astype(np.float32) for i in range(P)
    ]
    ref_dq = [np.asarray(As[i], np.float32) @ W_dq[i].T for i in range(P)]
    ref_fp = [
        np.asarray(As[i], np.float32) @ np.asarray(Ws[i], np.float32).T
        for i in range(P)
    ]

    bad = 0
    print(
        f"\n[int4 fc] ctx {args.ctx} of {M} rows, weight "
        f"{sum(W.size for W in Ws)*2/1e6:.1f} MB bf16 -> "
        f"{sum(B.size for B in Bs)/1e6:.1f} MB packed int4"
    )
    for i in range(P):
        e_eng = _rel(parts[i][: args.ctx], ref_dq[i][: args.ctx])
        e_tot = _rel(parts[i][: args.ctx], ref_fp[i][: args.ctx])
        spill = int((np.abs(parts[i][args.ctx :]) > 1e-3).sum())
        ok = e_eng <= args.tol and spill == 0
        bad += not ok
        print(
            f"  partial {i}: vs dequantized weight {e_eng:.3e}  "
            f"(vs full precision {e_tot:.3e}), padded spill {spill}"
            + ("" if ok else "   <-- FAIL")
        )

    if P > 1:
        print("\n  cross-pair distance (row = device partial, col = reference):")
        print("         " + "".join(f"{c:>10}" for c in range(P)))
        for i in range(P):
            row = [_rel(parts[i][: args.ctx], ref_dq[c][: args.ctx]) for c in range(P)]
            best = int(np.argmin(row))
            bad += best != i
            print(
                f"    P{i:<5}"
                + "".join(f"{v:>10.2e}" for v in row)
                + ("" if best == i else f"   <-- closest to W{best}")
            )

    if with_norm:
        out = np.asarray(res[3 * P + (P - 1) + 1]).reshape(M, D).astype(np.float32)
        # Against the DEVICE's own partials: isolates the add/norm tail from the
        # GEMMs it consumes.
        ref_tail = dflash_sumnorm.reference(parts, hn_w)
        ref_end = dflash_sumnorm.reference(ref_dq, hn_w)
        e_tail = _rel(out[: args.ctx], ref_tail[: args.ctx])
        e_e2e = _rel(out[: args.ctx], ref_end[: args.ctx])
        spill = int((np.abs(out[args.ctx :]) > 1e-3).sum())
        ok = e_tail <= args.tol and e_e2e <= args.tol and spill == 0
        bad += not ok
        print(
            f"\n  add+norm: vs device partials {e_tail:.3e}, "
            f"end-to-end {e_e2e:.3e}, padded spill {spill}"
            + ("" if ok else "   <-- FAIL")
        )

    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
