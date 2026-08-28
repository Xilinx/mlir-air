# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The DFlash drafter's tap fusion, as a two-launch AIR module:
#
#     taps [CTX, 12800]  --fc-->  [CTX, 2560]  --hidden_norm-->  target_hidden
#
# WHY THIS IS ITS OWN MODULE AND NOT A FUSED_DECODE PHASE. Everything about
# fused_decode's projection machinery is table-driven off I2P/J2P/DEST, so `fc`
# (I2P=5, J2P=25 at this geometry) looked like a fifth table entry. It is not:
# `FULL4` (fused_decode.py:1180) is `NPH == 4 and ...` and gates the whole fused
# four-phase structure, including the RMS_BAND_STREAM level 3 path that
# docs/DFlashFeasibility.md section 3.2 depends on. NPH=5 silently selects a
# different, far less exercised design. So fc gets its own launch -- which is
# also the shape the multi-launch route wants.
#
# WHY GEMM AND NOT GEMV. fc is applied to every context row with the SAME
# 2560x12800 weight. A GEMV streams that weight per row: at CTX=8 that is 8 x
# 65 MB per draft call, which is more traffic than the drafter's entire 5 layers.
# A GEMM with M=CTX streams it once. That is the whole reason for the padding
# below -- the arithmetic on the padded rows is free next to the weight stream.
#
# THE SHAPE IS THIN, AND THAT CONSTRAINS THE HERD. `_build_gemm_module` requires
# M % (tile_m * herd_m) == 0, and the registry's shapes are prefill-sized (M >=
# 64) where the default 8x4 herd is fine. At M=16 an 8-row herd needs M >= 128,
# so this uses a 1x4 herd. tile_m is NOT free: gemm_method_spec('drain')
# forces 32, and mm_aie2p.cc additionally static-asserts DIM_M % (2*r) == 0,
# which rules out 8. M must then be a multiple of 32, hence CTX_PAD. The registry has no entry for
# 16x12800x2560 -- these tiles are chosen to satisfy the divisibility rules
# (12800 = 128*100, 2560 = 64*40) AND the 64 KB L1 ceiling -- the first
# attempt used the registry's 320x128 staging, which is 80 KB and does not fit.
# They are a correctness
# choice rather than a tuned one.

CTX_PAD = 32  # padded context rows; the real ctx is <= block size (8)
FC_IN = 12800
D = 2560

TILE_M = 32  # forced by gemm_method_spec('drain')
HERD_M = 1
HERD_N = 4
TILE_K_L2 = 128  # 12800 = 128 * 100
TILE_K_L1 = 32  # 128 = 32 * 4
TILE_N = 64  # 2560 = 64 * 40; tile_k_l2*tile_n*2 = 16 KB of L1


def _paths():
    import os
    import sys
    from pathlib import Path

    pe = Path(__file__).resolve().parent.parent.parent
    for p in (pe / "llms", pe, pe / "weighted_rms_norm"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return pe


def build_fc_module(ctx_pad=CTX_PAD, with_norm=True):
    """Two air.launch ops in one func: fc GEMM, then hidden_norm.

    Args (bf16 throughout):
        %arg0  taps    [ctx_pad, 12800]
        %arg1  fc_wT   [12800, 2560]     fc.weight TRANSPOSED (see below)
        %arg2  fused   [ctx_pad, 2560]   fc output, and the norm's input
        %arg3  hn_w    [2560]            hidden_norm.weight
        %arg4  out     [ctx_pad, 2560]   target_hidden
    """
    _paths()
    from ml_dtypes import bfloat16

    from shared.builders.gemm_builder import _build_gemm_module, gemm_method_spec
    from shared.infra.stitching import (
        FuncArg,
        KernelSlice,
        stitch_elf,
        _wrap_ir_in_launch,
    )
    from weighted_rms_norm import build_module as build_rms

    _spec = gemm_method_spec("drain")
    kw = dict(_spec["build_kwargs"])
    # The GEMM slice calls into the external microkernel and its epilogue by
    # name. stitch_elf prefixes every symbol a slice defines, so these have to
    # be declared extern or the prefixed call has no callee.
    _sfx = _spec["sym_suffix"]
    _gemm_externs = {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + _sfx,
        "@zero_f32_mn" + _sfx,
        "@f32_to_bf16_mn" + _sfx,
    }
    gemm_ir = str(
        _build_gemm_module(
            ctx_pad,
            FC_IN,
            D,
            TILE_M,
            TILE_K_L2,
            TILE_K_L1,
            TILE_N,
            herd_m=HERD_M,
            herd_n=HERD_N,
            **kw,
        )
    )
    base_args = [
        FuncArg("%arg0", f"memref<{ctx_pad}x{FC_IN}xbf16>"),
        FuncArg("%arg1", f"memref<{FC_IN}x{D}xbf16>"),
        FuncArg("%arg2", f"memref<{ctx_pad}x{D}xbf16>"),
    ]
    # The GEMM's arg order is (A, B, C) with B laid out [K, N]. `fc.weight` is
    # stored [out, in] = [2560, 12800], so the HOST transposes it once at load
    # -- 65 MB moved once per process, against moving it per dispatch.
    slices = [KernelSlice(gemm_ir, "fc", {0: 0, 1: 1, 2: 2}, extern_syms=_gemm_externs)]

    if with_norm:
        rms_ir = _wrap_ir_in_launch(str(build_rms(ctx_pad, D, bfloat16, 16, herd_x=1)))
        base_args += [
            FuncArg("%arg3", f"memref<{D}xbf16>"),
            FuncArg("%arg4", f"memref<{ctx_pad}x{D}xbf16>"),
        ]
        slices.append(
            KernelSlice(
                rms_ir,
                "hn",
                {0: 2, 1: 3, 2: 4},
                extern_syms={"@zero_vectorized_bf16"},
            )
        )

    return stitch_elf("dflash_fc", base_args, slices)


def reference(taps, fc_w, hn_w, eps=1e-6):
    """numpy hidden_norm(fc(taps)) -- the same arithmetic in f32."""
    import numpy as np

    x = np.asarray(taps, np.float32) @ np.asarray(fc_w, np.float32).T
    var = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * np.asarray(hn_w, np.float32)


if __name__ == "__main__":
    import sys

    m = build_fc_module()
    txt = str(m)
    n_launch = txt.count("air.launch")
    print(f"[fc] {len(txt.splitlines())} lines, {n_launch} air.launch ops, parsed OK")
    print(f"[fc] M={CTX_PAD} K={FC_IN} N={D}, herd {HERD_M}x{HERD_N}, tile_m={TILE_M}")
    sys.exit(0 if n_launch >= 2 else 1)
