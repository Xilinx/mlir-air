#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the DFlash drafter's context K/V in int4-AWQ.

Twenty air.launch ops in one func -- per drafter layer an int4 K GEMM, an int4 V
GEMM, a bf16 k_norm and RoPE on K -- against the real per-layer
k_proj/v_proj/k_norm out of z-lab/Qwen3-4B-DFlash-b16. The bf16 form (no RoPE)
is dflash_ctxkv_gate.py --split.

THE REFERENCE IS THE DEQUANTIZED WEIGHT for the GEMM checks: comparing int4
against full precision measures the quantizer, not the engine.

FOUR THINGS THIS GATE IS BUILT TO CATCH, none of which a correlation on random
data would:

  the wrong layer's weights   every launch takes the SAME input and differs only
                              in which weight it is given, so a mis-wired arg map
                              produces a plausible result belonging to another
                              layer. Cross-layer distances are printed.
  k and v swapped             same shape, both plausible. Checked separately.
  a wrong k_norm weight       k_norm is per head and NOT shared between layers.
                              An earlier bf16 build passed layer 0's weight to
                              all five and every layer still gated clean because
                              the reference made the same assumption, so the
                              checkpoint tensors are compared directly here.

  RoPE that never rotates    every row is also compared against the UNROTATED
                             k_norm output, so a LUT stuck at position 0, or one
                             whose row offset ignores the 8 KV heads per
                             position, shows up as an unrotated distance near
                             zero instead of the ~0.7 it should be.

  Pre-norm K is checked as well as post-norm, because k_norm is scale-invariant
  in its input: a K wrong by a per-row scale -- what a mis-strided reshape
  produces -- comes out of the norm looking correct.

    python3 dflash_ctxkv_int4_gate.py
    python3 dflash_ctxkv_int4_gate.py --no-rope --compile-only
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


def rope_ref(x, positions, theta=1000000.0):
    """Half-split NEOX RoPE on [rows, 128] where row r is (positions[r//8], r%8).

    y[:64] = x1*cos - x2*sin ; y[64:] = x1*sin + x2*cos -- rope.cc's form, which
    is NOT HF's interleaved layout.
    """
    import numpy as np

    rows, hd = x.shape
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.outer(np.asarray(positions, np.float64), inv)
    cos = np.repeat(np.cos(ang), 8, axis=0)[:rows].astype(np.float32)
    sin = np.repeat(np.sin(ang), 8, axis=0)[:rows].astype(np.float32)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument(
        "--start",
        type=int,
        default=137,
        help="absolute position of the FIRST context row. The drafter sees "
        "position_ids[start - ctx : start + block] (model.py:246), so the "
        "context rows carry real sequence positions, not 0..ctx-1. A "
        "non-zero default is deliberate: at start=0 a wrong LUT row offset "
        "would be invisible for row 0.",
    )
    ap.add_argument("--no-rope", action="store_true", help="stop after k_norm")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_int4 as I
    import dflash_ctxkv_int4_builder as B
    from qwen3_4b_draft_weights import DraftWeights

    I.paths()
    I.compile_int4_gemm_kernel()
    from shared.infra.external_kernels import compile_rope

    compile_rope()

    module = B.build_ctxkv_int4_module(with_rope=not args.no_rope)
    print(f"[ctxkv int4] module: {str(module).count('air.launch')} launches")

    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="dflash_ctxkv_int4",
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=16384,  # without it the module runs and returns rel ~0.85
    )
    compiled = backend.compile(module)
    if args.compile_only:
        backend.unload()
        print("[ctxkv int4] compile-only done")
        return 0

    dw = DraftWeights()
    N, C, KVD, HD = B.N_LAYERS, B.CTX_PAD, B.KV_DIM, B.HEAD_DIM
    rows = C * B.N_KV_HEADS

    kq, vq, kdq, vdq, kpk, vpk = [], [], [], [], [], []
    for L in range(N):
        kw, vw = B.layer_kv_weights(dw, L)
        for w, q_, dq_, pk_ in ((kw, kq, kdq, kpk), (vw, vq, vdq, vpk)):
            q, s, z = I.awq_quantize(w)
            q_.append((q, s, z))
            dq_.append(I.awq_dequantize(q, s, z))
            pk_.append(np.ascontiguousarray(I.pack_for_device(q, s, z, C, B.D, KVD)))

    kn = [np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight")) for L in range(N)]
    shared = all(np.array_equal(kn[0], k) for k in kn[1:])
    print(f"  k_norm shared across layers: {shared} (each layer is given its own)")

    rng = np.random.default_rng(0)
    th = np.zeros((C, B.D), bfloat16)
    th[: args.ctx] = rng.normal(0, 1, (args.ctx, B.D)).astype(bfloat16)

    ins = [th]
    for L in range(N):
        ins += [
            kpk[L],
            np.zeros((C, KVD), bfloat16),
            vpk[L],
            np.zeros((C, KVD), bfloat16),
        ]
    for L in range(N):
        ins += [np.asarray(kn[L], bfloat16), np.zeros((rows, HD), bfloat16)]
    positions = np.arange(args.start, args.start + C)
    if not args.no_rope:
        ins.append(B.rope_lut(positions))
        ins += [np.zeros((rows, HD), bfloat16) for _ in range(N)]

    import filelock

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        res = fn(*ins)
    backend.unload()

    thf = np.asarray(th, np.float32)
    bad = 0
    nb = 1 + 4 * N
    got_k = []
    packed_mb = sum(p.size for p in kpk + vpk) / 1e6
    print(
        f"\n[ctxkv int4] ctx {args.ctx} of {C} rows, {N} layers, "
        f"weight {2 * N * B.D * KVD * 2 / 1e6:.1f} MB bf16 -> {packed_mb:.1f} MB int4"
    )
    for L in range(N):
        k_raw = np.asarray(res[1 + 4 * L + 1]).reshape(C, KVD).astype(np.float32)
        v_ctx = np.asarray(res[1 + 4 * L + 3]).reshape(C, KVD).astype(np.float32)
        k_out = np.asarray(res[nb + 2 * L + 1]).reshape(rows, HD).astype(np.float32)
        got_k.append(k_raw)

        rk = thf @ kdq[L].T
        rv = thf @ vdq[L].T
        x = rk.reshape(rows, HD)
        kn_ref = (x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)) * np.asarray(
            kn[L], np.float32
        )

        e_k = _rel(k_raw[: args.ctx], rk[: args.ctx])
        e_v = _rel(v_ctx[: args.ctx], rv[: args.ctx])
        e_n = _rel(k_out[: args.ctx * 8], kn_ref[: args.ctx * 8])
        spill = int((np.abs(k_raw[args.ctx :]) > 1e-3).sum()) + int(
            (np.abs(v_ctx[args.ctx :]) > 1e-3).sum()
        )
        line = f"  layer {L}: k_raw {e_k:.3e}, v {e_v:.3e}, k_norm {e_n:.3e}"
        worst = max(e_k, e_v, e_n)
        if not args.no_rope:
            rb = nb + 2 * N
            k_rop = np.asarray(res[rb + 1 + L]).reshape(rows, HD).astype(np.float32)
            e_r = _rel(
                k_rop[: args.ctx * 8], rope_ref(kn_ref, positions)[: args.ctx * 8]
            )
            # Same rows without the rotation: RoPE at position 0 is the
            # identity, and a LUT that never varies with position would still
            # match at every row if this were not checked separately.
            e_id = _rel(k_rop[: args.ctx * 8], kn_ref[: args.ctx * 8])
            line += f", rope {e_r:.3e} (unrotated {e_id:.3e})"
            worst = max(worst, e_r)
        ok = worst <= args.tol and spill == 0
        bad += not ok
        print(line + f", spill {spill}" + ("" if ok else "   <-- FAIL"))

    print("\n  cross-layer distance on K (row = device, col = reference):")
    print("         " + "".join(f"{c:>10}" for c in range(N)))
    for L in range(N):
        row = [
            _rel(got_k[L][: args.ctx], (thf @ kdq[c].T)[: args.ctx]) for c in range(N)
        ]
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
