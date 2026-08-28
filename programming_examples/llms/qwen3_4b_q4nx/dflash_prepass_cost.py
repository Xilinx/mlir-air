# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Where the DFlash pre-pass's time goes -- the third PDI, and the largest
# single line in a speculative step after the verify pass itself.
#
# Three measurements, each answering one candidate explanation:
#
#   --split   host vs device. The invoker in air/backend/xrt.py allocates a
#             fresh BO per argument on EVERY call, copies the host array into
#             it twice, and syncs every one of them back afterwards -- and the
#             pre-pass passes all 30.9 MB of its weights as arguments. That
#             looks like the answer and is not: the device is 68 of the 81 ms.
#
#   --scale   bytes vs launches, by varying the drafter layer count. Each
#             layer adds 2 int4 GEMMs (2560x1024) and 2 launches; fc is fixed
#             at 2 launches. A straight line through the origin against weight
#             BYTES, with 8 / 16 / 24 launches on it, says the 24 launches
#             cost nothing measurable and fusing them would gain nothing.
#
#   --herd    whether the 0.46 GB/s is just the herd width. HERD_N maps to
#             ROWS, and NPU2 has four compute rows, so HERD_N=4 already
#             saturates the one column this GEMM uses; 8 and 16 do not place.
#             Going wider means more COLUMNS (HERD_M > 1), which the L2
#             A-stage assert in dflash_int4 blocks at fc's K.
#
# Measured on qwen3-4b, NPU2, block 8 (docs/DFlashFeasibility.md section 3.13):
#
#     full run()   81.1 ms      device only  68.1 ms      ELF reload  36.5 ms
#     layers 1/3/5 -> 42.4 / 55.1 / 66.8 ms over 19.9 / 25.4 / 30.9 MB
#                  -> 0.46-0.47 GB/s at every point, intercept ~0
#
# 0.46 GB/s against the ~11 GB/s the batch-8 verify pass achieves on the same
# silicon. The pre-pass is not slow because of how it is called or how many
# launches it has; it is slow because it streams 30.9 MB of constant weights
# through a generic int4 GEMM on four cores, once per block, to produce at
# most eight rows.
#
#     python3 dflash_prepass_cost.py --split --scale --herd

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE))


def _weights(dw, I, CK, FC, C, D, P):
    """AWQ-quantize and pack once; every phase below reuses these."""
    import numpy as np
    from ml_dtypes import bfloat16

    KC = FC.FC_IN // P
    fc_pk = [
        np.ascontiguousarray(I.pack_for_device(*I.awq_quantize(W), C, KC, D))
        for W in FC.split_fc_weight(np.asarray(dw.fc()), P)
    ]
    kpk, vpk, kn = [], [], []
    for L in range(CK.N_LAYERS):
        kw, vw = CK.layer_kv_weights(dw, L)
        kpk.append(
            np.ascontiguousarray(
                I.pack_for_device(*I.awq_quantize(kw), C, D, CK.KV_DIM)
            )
        )
        vpk.append(
            np.ascontiguousarray(
                I.pack_for_device(*I.awq_quantize(vw), C, D, CK.KV_DIM)
            )
        )
        kn.append(np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight"), bfloat16))
    return fc_pk, kpk, vpk, kn


def _prepass_args(lay, n_layers, fc_pk, kpk, vpk, kn, hn_w, taps, CK, FC, C, D, P):
    import numpy as np
    from ml_dtypes import bfloat16

    KVD, HD, NKV = CK.KV_DIM, CK.HEAD_DIM, CK.N_KV_HEADS
    ins = [None] * lay["n_args"]
    for i, a in enumerate(lay["taps"]):
        ins[a] = FC.split_taps(taps, P)[i]
    for i, a in enumerate(lay["fc_w"]):
        ins[a] = fc_pk[i]
    for a in lay["fc_partial"] + lay["fc_fold"]:
        ins[a] = np.zeros((C, D), bfloat16)
    ins[lay["hn_w"]] = hn_w
    ins[lay["target_hidden"]] = np.zeros((C, D), bfloat16)
    for L in range(n_layers):
        ins[lay["k_w"][L]] = kpk[L]
        ins[lay["v_w"][L]] = vpk[L]
        ins[lay["k_raw"][L]] = np.zeros((C, KVD), bfloat16)
        ins[lay["v_ctx"][L]] = np.zeros((C, KVD), bfloat16)
        ins[lay["k_norm_w"][L]] = kn[L]
        ins[lay["k_nrm"][L]] = np.zeros((C * NKV, HD), bfloat16)
        ins[lay["k_ctx"][L]] = np.zeros((C * NKV, HD), bfloat16)
    ins[lay["rope_lut"]] = CK.rope_lut(np.zeros(C, np.int64))
    return ins


def phase_split(args):
    """Host BO traffic vs device time, through the real PrepassRunner."""
    import numpy as np

    import dflash_int4_fc_builder as FC
    from dflash_prepass_runner import PrepassRunner

    r = PrepassRunner()
    n = args.rows
    taps = np.zeros((n, FC.FC_IN), np.float32)
    pos = list(range(96, 96 + n))

    r.run(taps, pos)  # warm: first call pays page faults
    r.t_load = r.t_run = 0.0
    r.n_run = 0
    for _ in range(args.iters):
        r.run(taps, pos)
    print(f"  full run()  : {r.t_run / r.n_run * 1e3:7.1f} ms  BO alloc+up+kernel+down")
    print(
        f"  ELF reload  : {r.t_load / r.n_run * 1e3:7.1f} ms  per block, not shipping"
    )

    r.backend.n_warmup_iters, r.backend.n_perf_iters = 3, args.iters
    r.run(taps, pos)
    print(f"  DEVICE only : {r.backend.last_latency_us / 1e3:7.1f} ms  kernel + wait")

    lay = r.lay
    wa = set(lay["fc_w"]) | set(lay["k_w"]) | set(lay["v_w"])
    ins = _prepass_args(
        lay,
        r.N,
        r.fc_pk,
        r.kpk,
        r.vpk,
        r.kn,
        r.hn_w,
        np.zeros((r.C, FC.FC_IN), r.bf16),
        r.CK,
        FC,
        r.C,
        r.D,
        r.P,
    )
    tot = sum(a.size * a.itemsize for a in ins)
    wgt = sum(ins[a].size * ins[a].itemsize for a in wa)
    print(
        f"  {lay['n_args']} args, {tot / 1e6:.1f} MB per dispatch each way; "
        f"{wgt / 1e6:.1f} MB ({100 * wgt / tot:.0f}%) is CONSTANT weights"
    )


def phase_scale(args):
    """Bytes or launches? Vary the drafter layer count."""
    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_ctxkv_int4_builder as CK
    import dflash_draft_prepass as PP
    import dflash_int4 as I
    import dflash_int4_fc_builder as FC
    from air.backend.xrt import XRTBackend
    from qwen3_4b_draft_weights import DraftWeights

    I.paths()
    I.compile_int4_gemm_kernel()
    from shared.infra.external_kernels import compile_rope

    compile_rope()
    dw = DraftWeights()
    C, D, P = PP.CTX_PAD, PP.D, PP.N_CHUNKS
    fc_pk, kpk, vpk, kn = _weights(dw, I, CK, FC, C, D, P)
    hn_w = np.asarray(dw.hidden_norm(), bfloat16)
    taps = np.zeros((C, FC.FC_IN), bfloat16)

    print(
        f"  {'layers':>6} {'launches':>8} {'weight MB':>10} {'device ms':>10} {'GB/s':>7}"
    )
    for NL in (1, 3, 5):
        lay = PP.prepass_arg_layout(n_layers=NL)
        mod = PP.build_prepass_module(n_layers=NL)
        nl = str(mod).count("air.launch")
        be = XRTBackend(
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="dflash_draft_prepass",
            runtime_loop_tiling_sizes=[2, 2],
            stack_size=16384,
        )
        be.n_warmup_iters, be.n_perf_iters = 3, args.iters
        fn = be.load(be.compile(mod))
        ins = _prepass_args(lay, NL, fc_pk, kpk, vpk, kn, hn_w, taps, CK, FC, C, D, P)
        fn(*ins)
        ms = be.last_latency_us / 1e3
        wb = sum(
            ins[a].nbytes
            for a in list(lay["fc_w"]) + list(lay["k_w"]) + list(lay["v_w"])
        )
        print(f"  {NL:>6} {nl:>8} {wb / 1e6:>10.2f} {ms:>10.1f} {wb / 1e6 / ms:>7.2f}")
        be.unload()


def phase_herd(args):
    """Is 0.46 GB/s just the herd width? The context-K/V half at 4 / 8 / 16.

    K=2560 here, so the L2 stage has headroom fc's K=6400 does not, and
    pack_for_device's granularity is (tile_n, tile_k_l1) -- independent of
    herd_n -- so one set of packed weights serves every width.
    """
    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_ctxkv_int4_builder as CK
    import dflash_int4 as I
    import dflash_int4_fc_builder as FC
    from air.backend.xrt import XRTBackend
    from qwen3_4b_draft_weights import DraftWeights

    I.paths()
    I.compile_int4_gemm_kernel()
    from shared.infra.external_kernels import compile_rope

    compile_rope()
    dw = DraftWeights()
    C, NL, D = CK.CTX_PAD, CK.N_LAYERS, FC.D
    KVD, HD, NKV = CK.KV_DIM, CK.HEAD_DIM, CK.N_KV_HEADS
    _, kpk, vpk, kn = _weights(dw, I, CK, FC, C, D, FC.N_CHUNKS)

    print(
        f"  {'herd_n':>6} {'rows':>5} {'weight MB':>10} {'device ms':>10} {'GB/s':>7}"
    )
    for HN in (4, 8, 16):
        I.HERD_N = HN
        try:
            be = XRTBackend(
                omit_while_true_loop=False,
                output_format="elf",
                instance_name="dflash_ctxkv_int4",
                runtime_loop_tiling_sizes=[2, 2],
                stack_size=16384,
            )
            be.n_warmup_iters, be.n_perf_iters = 3, args.iters
            fn = be.load(be.compile(CK.build_ctxkv_int4_module()))
        except Exception:
            # NPU2 has four compute rows and HERD_N is the ROW extent, so
            # anything past 4 fails placement ('aie.tile' row index must be
            # less than the number of rows). Going wider is a COLUMN change.
            print(f"  {HN:>6} {HN:>5}   does not place -- NPU2 has 4 compute rows")
            continue
        ins = [np.zeros((C, D), bfloat16)]
        for L in range(NL):
            ins += [
                kpk[L],
                np.zeros((C, KVD), bfloat16),
                vpk[L],
                np.zeros((C, KVD), bfloat16),
            ]
        for L in range(NL):
            ins += [kn[L], np.zeros((C * NKV, HD), bfloat16)]
        ins += [CK.rope_lut(np.zeros(C, np.int64))]
        ins += [np.zeros((C * NKV, HD), bfloat16) for _ in range(NL)]
        fn(*ins)
        ms = be.last_latency_us / 1e3
        wb = sum(a.nbytes for a in kpk + vpk)
        print(f"  {HN:>6} {HN:>5} {wb / 1e6:>10.2f} {ms:>10.1f} {wb / 1e6 / ms:>7.2f}")
        be.unload()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", action="store_true", help="host vs device")
    ap.add_argument("--scale", action="store_true", help="bytes vs launches")
    ap.add_argument("--herd", action="store_true", help="herd width")
    ap.add_argument("--rows", type=int, default=8, help="context rows for --split")
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    if not (args.split or args.scale or args.herd):
        args.split = args.scale = args.herd = True

    # One phase per process would be cleaner, but each pays a multi-minute AWQ
    # quantization of the same weights, so they share one.
    for name, fn in (
        ("split", phase_split),
        ("scale", phase_scale),
        ("herd", phase_herd),
    ):
        if getattr(args, name):
            print(f"\n[{name}]")
            fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
