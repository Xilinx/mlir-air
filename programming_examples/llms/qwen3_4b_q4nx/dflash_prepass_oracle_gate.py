#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The device pre-pass against the REAL drafter's own KV rows.

`dflash_draft_prepass_gate.py` checks the 24-launch pre-pass against a numpy
chain built from the same weights: it proves the engine, and it cannot prove
that the chain is the drafter's. This one runs the pre-pass on taps recorded
from `dflash_draft_oracle.py` and compares its output against what
z-lab/Qwen3-4B-DFlash-b16 ITSELF put in its KV cache for the same block --
K after `k_norm` and RoPE, V raw.

So this is the gate for everything the numpy chain assumed rather than showed:
the tap concatenation order, that `fc` sees them raw, that `hidden_norm` comes
after `fc` and before `k/v_proj`, that `k_norm` precedes RoPE, that the
positions are absolute, and the head layout inside a 1024-wide row.

THREE NUMBERS, AND THEY MEAN DIFFERENT THINGS. `int4` is the whole device chain
against the oracle, so it carries the AWQ quantization -- `dflash_int4`'s
round-trip is 5.5e-02 against a 1.1e-01 step, and the measured distance here is
that size. `dq` is the same device output against a numpy chain over the
DEQUANTIZED weights fed the DEVICE's own `target_hidden`, which isolates the
engine and sits near 1e-02. `cos` is the one to read for structure: at ~0.99+
the vectors are the same thing scaled, which quantization does and a wrong
concatenation order, a swapped norm and a missing rotation do not.

A structural error moves all three. Quantization moves only `int4`.

    python3 dflash_draft_oracle.py            # once, in its own process (torch)
    python3 dflash_prepass_oracle_gate.py
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


def _cos(a, b):
    import numpy as np

    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--oracle", default="dflash_draft_oracle.npz")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tol-dq", type=float, default=5e-2, help="engine, vs dequantized")
    ap.add_argument(
        "--tol-int4", type=float, default=0.25, help="whole chain vs oracle"
    )
    ap.add_argument("--cos", type=float, default=0.99)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_ctxkv_int4_builder as CK
    import dflash_draft_prepass as PP
    import dflash_int4 as I
    import dflash_int4_fc_builder as FC
    import dflash_sumnorm
    from qwen3_4b_draft_weights import DraftWeights

    o = np.load(args.oracle)
    ctx_pos = o["ctx_positions"]
    ctx = len(ctx_pos)
    taps_o = o["taps"]  # [ctx, 12800]
    print(
        f"[prepass oracle] block {int(o['block'])}, start {int(o['start'])}, "
        f"ctx {ctx} at positions {ctx_pos.tolist()}"
    )

    I.paths()
    I.compile_int4_gemm_kernel()
    from shared.infra.external_kernels import compile_rope

    compile_rope()

    module = PP.build_prepass_module()
    lay = PP.prepass_arg_layout()

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
        print("[prepass oracle] compile-only done")
        return 0

    dw = DraftWeights()
    N, C, KVD, HD = PP.N_LAYERS, PP.CTX_PAD, CK.KV_DIM, CK.HEAD_DIM
    P, KC = PP.N_CHUNKS, FC.FC_IN // PP.N_CHUNKS
    rows = C * CK.N_KV_HEADS
    assert ctx <= C, f"oracle ctx {ctx} exceeds CTX_PAD {C}"

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

    # The taps go in as the drafter saw them; the padded rows stay zero, and the
    # LUT carries the oracle's ABSOLUTE positions (block 0's are 0..P-1, which
    # is exactly the case where a relative-position bug would be invisible --
    # every later block's are not).
    taps = np.zeros((C, FC.FC_IN), bfloat16)
    taps[:ctx] = np.asarray(taps_o, bfloat16)
    As = FC.split_taps(taps, P)
    positions = np.zeros(C, np.int64)
    positions[:ctx] = ctx_pos

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

    import filelock

    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        res = fn(*ins)
    backend.unload()

    def out(a, shape):
        return np.asarray(res[a]).reshape(shape).astype(np.float32)

    bad = 0
    th_dev = out(lay["target_hidden"], (C, PP.D))[:ctx]
    th_o = np.asarray(o["target_hidden"], np.float32)[:ctx]
    th_dq = dflash_sumnorm.reference(
        [np.asarray(As[i], np.float32)[:ctx] @ fc_dq[i].T for i in range(P)], hn_w
    )
    e_i4, e_dq, c = _rel(th_dev, th_o), _rel(th_dev, th_dq), _cos(th_dev, th_o)
    ok = e_i4 <= args.tol_int4 and e_dq <= args.tol_dq and c >= args.cos
    bad += not ok
    print(
        f"\n  target_hidden: int4 vs oracle {e_i4:.3e}, engine vs dq {e_dq:.3e}, "
        f"cos {c:.6f}" + ("" if ok else "   <-- FAIL")
    )

    from dflash_ctxkv_int4_gate import rope_ref

    for L in range(N):
        k_dev = out(lay["k_ctx"][L], (rows, HD))[: ctx * 8]
        v_dev = out(lay["v_ctx"][L], (C, KVD))[:ctx]
        k_o = np.asarray(o["k_ctx"][L], np.float32).reshape(ctx * 8, HD)
        v_o = np.asarray(o["v_ctx"][L], np.float32)

        # Engine reference: the DEVICE's own target_hidden through the
        # dequantized weights. Using the oracle's would fold fc's quantization
        # into every layer's number a second time -- measured, that turns a
        # 7e-03 engine error into 7e-02 and hides what it is supposed to show.
        rk = th_dev @ kdq[L].T
        x = rk.reshape(ctx * 8, HD)
        k_dq = rope_ref(
            (x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6))
            * np.asarray(kn[L], np.float32),
            ctx_pos,
        )
        v_dq = th_dev @ vdq[L].T

        ek_i4, ek_dq, ck = _rel(k_dev, k_o), _rel(k_dev, k_dq), _cos(k_dev, k_o)
        ev_i4, ev_dq, cv = _rel(v_dev, v_o), _rel(v_dev, v_dq), _cos(v_dev, v_o)
        ok = (
            max(ek_i4, ev_i4) <= args.tol_int4
            and max(ek_dq, ev_dq) <= args.tol_dq
            and min(ck, cv) >= args.cos
        )
        bad += not ok
        print(
            f"  layer {L}: k int4 {ek_i4:.3e} dq {ek_dq:.3e} cos {ck:.6f} | "
            f"v int4 {ev_i4:.3e} dq {ev_dq:.3e} cos {cv:.6f}"
            + ("" if ok else "   <-- FAIL")
        )

    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
