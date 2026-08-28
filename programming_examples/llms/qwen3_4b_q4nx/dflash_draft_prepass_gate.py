#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the DFlash drafter's non-decode half, all 24 launches.

fc + hidden_norm feeding the five layers' context K/V + k_norm + RoPE in one
func, against the real z-lab/Qwen3-4B-DFlash-b16 weights. The two halves have
their own gates (dflash_int4_fc_gate.py, dflash_ctxkv_int4_gate.py); what is new
here is the WIRING, so that is what this checks hardest:

  fc's output really is what K/V reads    `target_hidden` is no longer a host
                                          buffer -- it is an intermediate that
                                          nothing outside the device sees. Every
                                          K/V reference is therefore computed
                                          from the DEVICE's `target_hidden`, not
                                          from a host recomputation, and the
                                          end-to-end number is reported beside
                                          it. If the K/V launches were reading a
                                          stale or wrong buffer, the first number
                                          would move and the second would not
                                          explain it.
  layer wiring survives the merge         arg numbers shift by fc's block, so
                                          the cross-layer distance matrix is
                                          re-run here rather than trusted from
                                          the standalone gate.
  the padded rows stay zero               ctx is 8 of 32; anything else is spill.

    python3 dflash_draft_prepass_gate.py
    python3 dflash_draft_prepass_gate.py --compile-only
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument(
        "--start",
        type=int,
        default=137,
        help="absolute position of the first context row (see the ctxkv gate)",
    )
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_ctxkv_int4_builder as CK
    import dflash_draft_prepass as PP
    import dflash_int4 as I
    import dflash_int4_fc_builder as FC
    import dflash_sumnorm
    from dflash_ctxkv_int4_gate import rope_ref
    from qwen3_4b_draft_weights import DraftWeights

    I.paths()
    I.compile_int4_gemm_kernel()
    from shared.infra.external_kernels import compile_rope

    compile_rope()

    module = PP.build_prepass_module()
    lay = PP.prepass_arg_layout()
    print(
        f"[prepass] module: {str(module).count('air.launch')} launches, "
        f"{lay['n_args']} args"
    )

    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="dflash_draft_prepass",
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=16384,
    )
    compiled = backend.compile(module)
    if args.compile_only:
        backend.unload()
        print("[prepass] compile-only done")
        return 0

    dw = DraftWeights()
    N, C, KVD, HD = PP.N_LAYERS, PP.CTX_PAD, CK.KV_DIM, CK.HEAD_DIM
    P, KC = PP.N_CHUNKS, FC.FC_IN // PP.N_CHUNKS
    rows = C * CK.N_KV_HEADS

    fcw = FC.split_fc_weight(np.asarray(dw.fc()), P)
    fc_pk, fc_dq = [], []
    for W in fcw:
        q, s, z = I.awq_quantize(W)
        fc_pk.append(np.ascontiguousarray(I.pack_for_device(q, s, z, C, KC, PP.D)))
        fc_dq.append(I.awq_dequantize(q, s, z))
    hn_w = np.asarray(dw.hidden_norm(), bfloat16)

    kpk, vpk, kdq, vdq = [], [], [], []
    for L in range(N):
        kw, vw = CK.layer_kv_weights(dw, L)
        for w, pk_, dq_ in ((kw, kpk, kdq), (vw, vpk, vdq)):
            q, s, z = I.awq_quantize(w)
            pk_.append(np.ascontiguousarray(I.pack_for_device(q, s, z, C, PP.D, KVD)))
            dq_.append(I.awq_dequantize(q, s, z))
    kn = [np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight")) for L in range(N)]

    rng = np.random.default_rng(0)
    taps = np.zeros((C, FC.FC_IN), bfloat16)
    taps[: args.ctx] = rng.normal(0, 1, (args.ctx, FC.FC_IN)).astype(bfloat16)
    As = FC.split_taps(taps, P)
    positions = np.arange(args.start, args.start + C)

    ins = [None] * lay["n_args"]
    for i, a in enumerate(lay["taps"]):
        ins[a] = As[i]
    for i, a in enumerate(lay["fc_w"]):
        ins[a] = fc_pk[i]
    for a in lay["fc_partial"] + lay["fc_fold"]:
        ins[a] = np.zeros((C, PP.D), bfloat16)
    ins[lay["hn_w"]] = hn_w
    ins[lay["target_hidden"]] = np.zeros((C, PP.D), bfloat16)
    for L in range(N):
        ins[lay["k_w"][L]] = kpk[L]
        ins[lay["v_w"][L]] = vpk[L]
        ins[lay["k_raw"][L]] = np.zeros((C, KVD), bfloat16)
        ins[lay["v_ctx"][L]] = np.zeros((C, KVD), bfloat16)
        ins[lay["k_norm_w"][L]] = np.asarray(kn[L], bfloat16)
        ins[lay["k_nrm"][L]] = np.zeros((rows, HD), bfloat16)
        ins[lay["k_ctx"][L]] = np.zeros((rows, HD), bfloat16)
    ins[lay["rope_lut"]] = CK.rope_lut(positions)
    assert all(x is not None for x in ins), [i for i, x in enumerate(ins) if x is None]

    import filelock

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        res = fn(*ins)
    backend.unload()

    def out(a, shape):
        return np.asarray(res[a]).reshape(shape).astype(np.float32)

    bad = 0
    ctx = args.ctx

    # --- fc half -----------------------------------------------------------
    th_dev = out(lay["target_hidden"], (C, PP.D))
    fc_ref = dflash_sumnorm.reference(
        [np.asarray(As[i], np.float32) @ fc_dq[i].T for i in range(P)], hn_w
    )
    e_fc = _rel(th_dev[:ctx], fc_ref[:ctx])
    sp = int((np.abs(th_dev[ctx:]) > 1e-3).sum())
    ok = e_fc <= args.tol and sp == 0
    bad += not ok
    print(f"\n[prepass] ctx {ctx} of {C} rows, {N} layers, positions {positions[0]}..")
    print(
        f"  target_hidden: {e_fc:.3e}, padded spill {sp}" + ("" if ok else "  <-- FAIL")
    )

    # --- K/V half, referenced against the DEVICE's target_hidden -----------
    got_k = []
    for L in range(N):
        k_raw = out(lay["k_raw"][L], (C, KVD))
        v_ctx = out(lay["v_ctx"][L], (C, KVD))
        k_rop = out(lay["k_ctx"][L], (rows, HD))
        got_k.append(k_raw)

        rk = th_dev @ kdq[L].T  # device target_hidden -> isolates the wiring
        rv = th_dev @ vdq[L].T
        x = rk.reshape(rows, HD)
        kn_ref = (x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)) * np.asarray(
            kn[L], np.float32
        )
        rp_ref = rope_ref(kn_ref, positions)

        # end to end: the host's own fc reference all the way through, so the
        # error accumulated across all four stages is visible too.
        x2 = (fc_ref @ kdq[L].T).reshape(rows, HD)
        e2e = rope_ref(
            (x2 / np.sqrt((x2**2).mean(-1, keepdims=True) + 1e-6))
            * np.asarray(kn[L], np.float32),
            positions,
        )

        e_k = _rel(k_raw[:ctx], rk[:ctx])
        e_v = _rel(v_ctx[:ctx], rv[:ctx])
        e_r = _rel(k_rop[: ctx * 8], rp_ref[: ctx * 8])
        e_e = _rel(k_rop[: ctx * 8], e2e[: ctx * 8])
        sp = int((np.abs(k_raw[ctx:]) > 1e-3).sum()) + int(
            (np.abs(v_ctx[ctx:]) > 1e-3).sum()
        )
        ok = max(e_k, e_v, e_r) <= args.tol and sp == 0
        bad += not ok
        print(
            f"  layer {L}: k {e_k:.3e}, v {e_v:.3e}, k_ctx {e_r:.3e} "
            f"(end to end {e_e:.3e}), spill {sp}" + ("" if ok else "  <-- FAIL")
        )

    print("\n  cross-layer distance on K (row = device, col = reference):")
    print("         " + "".join(f"{c:>10}" for c in range(N)))
    for L in range(N):
        row = [_rel(got_k[L][:ctx], (th_dev @ kdq[c].T)[:ctx]) for c in range(N)]
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
