#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the DFlash drafter's tap fusion: hidden_norm(fc(taps)).

The first piece of the DRAFTER (as opposed to the target) to run on NPU2. Two
air.launch ops in one func -- a 16x12800x2560 GEMM then an RMSNorm -- built by
dflash_fc_builder.py and compiled through XRTBackend.

WHAT IT IS CHECKED AGAINST, and why that matters more than the tolerance: the
REAL `fc.weight` and `hidden_norm.weight` out of z-lab/Qwen3-4B-DFlash-b16, not
random fill. A GEMM with a transposed or mis-strided weight still correlates
well on random data if the shapes happen to line up; against the real matrix,
whose rows are not exchangeable, a layout error shows up immediately.

The reference is f32 numpy, so the residual is bf16 rounding through a
K=12800 reduction plus the engine's own accumulation order -- the same class of
difference batch_equiv.py's 5e-2 is sized for, and for the same reason.

    python3 dflash_fc_gate.py
    python3 dflash_fc_gate.py --compile-only
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ctx", type=int, default=8, help="real context rows used")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_fc_builder as B
    from qwen3_4b_draft_weights import DraftWeights

    # The GEMM herd links an external microkernel object by name. Without it the
    # aiecc pipeline fails at the Peano link/opt stage with no message naming the
    # missing symbol -- it just reports "pipeline failed".
    B._paths()
    from shared.infra.external_kernels import compile_gemm_mm

    compile_gemm_mm(
        tile_m=B.TILE_M,
        tile_n=B.TILE_N,
        tile_k_l1=B.TILE_K_L1,
        sym_suffix="_m32",
        out_name="mm_m32.o",
    )

    module = B.build_fc_module()
    print(f"[fc gate] module: {str(module).count('air.launch')} launches")

    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        # ELF, not xclbin: this is a MULTI-LAUNCH module, and the multi-launch
        # path is the ELF one (it is what emits load_pdi between launches --
        # AIRRtToNpuPass.cpp:1244 gates that on outputElf). On the xclbin path
        # the two launches' instruction streams collide, which aiecc reports as
        # `edge 'air.insts.bin' produced duplicate output path` and then a bare
        # "pipeline failed" several stages later.
        output_format=os.environ.get("DFLASH_FC_FMT", "elf"),
        instance_name="dflash_fc",
        runtime_loop_tiling_sizes=[2, 2],
    )
    compiled = backend.compile(module)
    if args.compile_only:
        backend.unload()
        print("[fc gate] compile-only done")
        return 0

    dw = DraftWeights()
    fc_w = np.asarray(dw.fc_bf16())  # [2560, 12800]
    hn_w = np.asarray(dw.hidden_norm(), bfloat16)  # [2560]

    # Real taps are target hidden states, order 1 and roughly zero-mean; the
    # scale matters because RMSNorm divides it straight back out, so a wrong one
    # would hide a fc error rather than expose it.
    rng = np.random.default_rng(0)
    taps = np.zeros((B.CTX_PAD, B.FC_IN), bfloat16)
    taps[: args.ctx] = rng.normal(0, 1, (args.ctx, B.FC_IN)).astype(bfloat16)

    # The GEMM wants B as [K, N]; fc.weight is [out, in]. Transposed once here.
    fc_wT = np.ascontiguousarray(fc_w.T)

    args_in_order = [
        taps,
        fc_wT,
        np.zeros((B.CTX_PAD, B.D), bfloat16),
        hn_w,
        np.zeros((B.CTX_PAD, B.D), bfloat16),
    ]

    import filelock

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        results = fn(*args_in_order)
    backend.unload()

    got = np.asarray(results[4]).reshape(B.CTX_PAD, B.D).astype(np.float32)
    ref = B.reference(taps, fc_w, hn_w)

    bad = 0
    print(
        f"\n[fc gate] real fc.weight {fc_w.shape}, hidden_norm {hn_w.shape}, "
        f"ctx {args.ctx} of {B.CTX_PAD} rows"
    )
    for t in range(args.ctx):
        g, r = got[t], ref[t]
        rel = np.sqrt(((g - r) ** 2).mean()) / max(np.sqrt((r**2).mean()), 1e-9)
        corr = np.corrcoef(g, r)[0, 1]
        ok = rel <= args.tol
        bad += not ok
        print(
            f"  row {t}: rms rel {rel:.3e}, corr {corr:.6f}"
            + ("" if ok else "   <-- FAIL")
        )

    # The padded rows are fed zeros, so their fc output is zero and the norm of
    # a zero row is zero -- a row that comes back non-zero means the GEMM wrote
    # outside the rows it was given.
    tail = got[args.ctx :]
    spill = int((np.abs(tail) > 1e-3).sum())
    print(
        f"  padded rows {args.ctx}..{B.CTX_PAD-1}: {spill} non-zero elements"
        + ("" if spill == 0 else "   <-- SPILL")
    )
    bad += spill != 0

    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
