#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the DFlash drafter's context K/V projection.

Five air.launch ops in one func -- one 32x2560x2048 GEMM per drafter layer,
each producing [k_ctx | v_ctx] for that layer from the SAME target_hidden.
Checked against the real per-layer k_proj/v_proj out of
z-lab/Qwen3-4B-DFlash-b16.

TWO THINGS THIS GATE IS BUILT TO CATCH, neither of which a correlation on
random data would:

  the wrong layer's weights   every launch takes the same input and differs
                              ONLY in which weight it is given, so a mis-wired
                              arg map produces a perfectly plausible result
                              that belongs to another layer. Each launch is
                              compared against ITS OWN layer's reference, and
                              the cross-layer distances are reported so
                              "layer 3 got layer 1's weights" cannot read as a
                              pass.
  k and v swapped             they are the same shape and both plausible. The
                              halves are compared separately.

    python3 dflash_ctxkv_gate.py
    python3 dflash_ctxkv_gate.py --compile-only
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _rel(a, b):
    import numpy as np

    return np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b**2).mean()), 1e-9)


def _run_split(args, B, DraftWeights, module, np, bfloat16):
    """Separate K/V GEMMs + k_norm. K is checked BOTH before and after the norm.

    Checking the pre-norm K matters: k_norm is scale-invariant in its input, so
    a K that is wrong by a per-row scale factor -- exactly what a mis-strided
    reshape produces -- comes out of the norm looking correct.
    """
    import os
    import tempfile

    import filelock
    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="dflash_ctxkv_split",
        runtime_loop_tiling_sizes=[2, 2],
    )
    compiled = backend.compile(module)
    if args.compile_only:
        backend.unload()
        print("[ctxkv gate] compile-only done")
        return 0

    dw = DraftWeights()
    N, C, KVD, HD = B.N_LAYERS, B.CTX_PAD, B.KV_DIM, B.HEAD_DIM
    rows = C * B.N_KV_HEADS
    kw = [
        np.ascontiguousarray(
            np.asarray(dw.bf16(f"layers.{L}.self_attn.k_proj.weight")).T
        )
        for L in range(N)
    ]
    vw = [
        np.ascontiguousarray(
            np.asarray(dw.bf16(f"layers.{L}.self_attn.v_proj.weight")).T
        )
        for L in range(N)
    ]
    # k_norm is per-head and shared by every layer; assert that rather than
    # assume it, since packing the wrong one is silent.
    kn = [np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight")) for L in range(N)]
    # k_norm is per-head, and it is NOT shared between layers -- checked, not
    # assumed. The first version of this module passed layer 0's weight to all
    # five and every layer still gated clean, because the reference made the
    # same assumption. Comparing the checkpoint tensors is what caught it.
    shared = all(np.array_equal(kn[0], k) for k in kn[1:])
    print(
        f"  k_norm shared across layers: {shared} "
        f"(each layer is given its own either way)"
    )

    rng = np.random.default_rng(0)
    th = np.zeros((C, B.D), bfloat16)
    th[: args.ctx] = rng.normal(0, 1, (args.ctx, B.D)).astype(bfloat16)

    ins = [th]
    for L in range(N):
        ins += [
            kw[L],
            np.zeros((C, KVD), bfloat16),
            vw[L],
            np.zeros((C, KVD), bfloat16),
        ]
    for L in range(N):
        ins.append(np.asarray(kn[L], bfloat16))
        ins.append(np.zeros((rows, HD), bfloat16))

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        res = fn(*ins)
    backend.unload()

    bad = 0
    nb = 1 + 4 * N
    print(f"\n[ctxkv split] ctx {args.ctx} of {C} rows, {N} layers")
    for L in range(N):
        k_raw = np.asarray(res[1 + 4 * L + 1]).reshape(C, KVD).astype(np.float32)
        v_ctx = np.asarray(res[1 + 4 * L + 3]).reshape(C, KVD).astype(np.float32)
        k_out = np.asarray(res[nb + 2 * L + 1]).reshape(rows, HD).astype(np.float32)

        rk_ref = B.reference(th, kw[L])
        rv_ref = B.reference(th, vw[L])
        # k_norm over each (position, head) row of 128, the same view the
        # prelude's reshape produces.
        x = rk_ref.reshape(rows, HD)
        var = (x**2).mean(-1, keepdims=True)
        kn_ref = (x / np.sqrt(var + 1e-6)) * np.asarray(kn[L], np.float32)

        e_k = _rel(k_raw[: args.ctx], rk_ref[: args.ctx])
        e_v = _rel(v_ctx[: args.ctx], rv_ref[: args.ctx])
        e_n = _rel(k_out[: args.ctx * 8], kn_ref[: args.ctx * 8])
        ok = max(e_k, e_v, e_n) <= args.tol
        bad += not ok
        print(
            f"  layer {L}: k_raw {e_k:.3e}, v {e_v:.3e}, k_norm {e_n:.3e}"
            + ("" if ok else "   <-- FAIL")
        )

    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument(
        "--split",
        action="store_true",
        help="gate the SPLIT module: separate K and V GEMMs plus k_norm, 15 "
        "launches. The fused form (5 launches) cannot feed k_norm, whose input "
        "must be a contiguous [ctx*8, 128].",
    )
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_ctxkv_builder as B
    from qwen3_4b_draft_weights import DraftWeights

    B._paths()
    from shared.infra.external_kernels import compile_gemm_mm

    compile_gemm_mm(
        tile_m=B.TILE_M,
        tile_n=B.TILE_N,
        tile_k_l1=B.TILE_K_L1,
        sym_suffix="_m32",
        out_name="mm_m32.o",
    )

    module = B.build_ctxkv_split_module() if args.split else B.build_ctxkv_module()
    print(
        f"[ctxkv gate] module: {str(module).count('air.launch')} launches"
        + (" (split + k_norm)" if args.split else " (fused k|v)")
    )
    if args.split:
        return _run_split(args, B, DraftWeights, module, np, bfloat16)

    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="elf",  # multi-launch is the ELF path (see dflash_fc_gate)
        instance_name="dflash_ctxkv",
        runtime_loop_tiling_sizes=[2, 2],
    )
    compiled = backend.compile(module)
    if args.compile_only:
        backend.unload()
        print("[ctxkv gate] compile-only done")
        return 0

    dw = DraftWeights()
    ws = [B.layer_kv_weight(dw, L) for L in range(B.N_LAYERS)]

    rng = np.random.default_rng(0)
    th = np.zeros((B.CTX_PAD, B.D), bfloat16)
    th[: args.ctx] = rng.normal(0, 1, (args.ctx, B.D)).astype(bfloat16)

    args_in_order = [th]
    for L in range(B.N_LAYERS):
        args_in_order.append(ws[L])
        args_in_order.append(np.zeros((B.CTX_PAD, B.KV2), bfloat16))

    import filelock

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        results = fn(*args_in_order)
    backend.unload()

    refs = [B.reference(th, ws[L]) for L in range(B.N_LAYERS)]
    bad = 0

    def rel(a, b):
        return np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b**2).mean()), 1e-9)

    print(
        f"\n[ctxkv gate] ctx {args.ctx} of {B.CTX_PAD} rows, "
        f"{B.N_LAYERS} layers, k=[0:{B.KV_DIM}] v=[{B.KV_DIM}:{B.KV2}]"
    )
    got = []
    for L in range(B.N_LAYERS):
        g = np.asarray(results[2 + 2 * L]).reshape(B.CTX_PAD, B.KV2).astype(np.float32)
        got.append(g)
        r = refs[L]
        rk = rel(g[: args.ctx, : B.KV_DIM], r[: args.ctx, : B.KV_DIM])
        rv = rel(g[: args.ctx, B.KV_DIM :], r[: args.ctx, B.KV_DIM :])
        ok = rk <= args.tol and rv <= args.tol
        bad += not ok
        spill = int((np.abs(g[args.ctx :]) > 1e-3).sum())
        bad += spill != 0
        print(
            f"  layer {L}: k rel {rk:.3e}, v rel {rv:.3e}, "
            f"padded-row spill {spill}" + ("" if ok and not spill else "   <-- FAIL")
        )

    # Cross-layer: each output must be closest to its OWN layer's reference.
    print("\n  cross-layer distance (row = device output, col = reference):")
    hdr = "".join(f"{c:>10}" for c in range(B.N_LAYERS))
    print(f"    {'':<6}{hdr}")
    for L in range(B.N_LAYERS):
        row = [rel(got[L][: args.ctx], refs[c][: args.ctx]) for c in range(B.N_LAYERS)]
        best = int(np.argmin(row))
        bad += best != L
        print(
            f"    L{L:<5}"
            + "".join(f"{v:>10.2e}" for v in row)
            + ("" if best == L else f"   <-- closest to L{best}")
        )

    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
