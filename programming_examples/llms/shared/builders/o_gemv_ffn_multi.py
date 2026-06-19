# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""o_gemv_ffn — three-launch multi-launch ELF for the LLAMA decode block.

Three sub-launches stitched into one ELF, where the post-attention
residual is routed through a row-0 subview of a packed 2D arg so a single
NPU-computed value feeds two downstream consumers without a host copy:

  Stage 1 (matvec_2tile_add):  res1 = wo @ attn_out + x_residual
                               written into arg6[0]
  Stage 2 (matvec_swiglu_rms): swiglu = silu(gate @ rms_norm(arg6)) * up
                               with gate/up interleaved into arg7
                               and rms_norm reading row 0 = res1,
                                                row 1 = ffn_norm_w
  Stage 3 (matvec_2tile_add):  output = wdown @ swiglu + res1
                               re-reading res1 from arg6[0]

Requires mlir-aie with N-D rank-reducing subview support in
`traceSubviewToBlockArgument`; without it, the row-0 subview on arg6
is rejected at `aie.dma_bd` lowering.

15-arg ABI (matches the baseline single-op-per-launch o_gemv_ffn so the
caller can pass dead args as zero placeholders):

    arg0:  memref<emb x emb xbf16>           wo                STATIC
    arg1:  memref<emb xbf16>                  attn_out          INPUT
    arg2:  memref<emb xbf16>                  (dead)
    arg3:  memref<emb xbf16>                  x_residual        INPUT
    arg4:  memref<emb xbf16>                  (dead — was res1 bus)
    arg5:  memref<emb xbf16>                  (dead — was ffn_norm_w; now in arg6[1])
    arg6:  memref<2 x emb xbf16>              packed RMS input  STATIC (row 1 = ffn_norm_w)
    arg7:  memref<2*hidden x emb xbf16>       interleaved gate/up  STATIC
    arg8:  memref<hidden xbf16>               (dead)
    arg9:  memref<hidden x emb xbf16>         (dead — folded into arg7)
    arg10: memref<hidden xbf16>               (dead)
    arg11: memref<hidden xbf16>               swiglu            INTERMEDIATE
    arg12: memref<emb x hidden xbf16>         wdown             STATIC
    arg13: memref<emb xbf16>                  (dead)
    arg14: memref<emb xbf16>                  output            OUTPUT
"""

import argparse
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "matrix_vector_multiplication",
        "bf16_cascade",
    ),
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "decode_ffn_swiglu"),
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matvec_2tile_add import build_module as build_2tile_add
from matvec_swiglu_rms import build_module as build_swiglu_rms
from shared.infra.stitching import (
    stitch_elf,
    KernelSlice,
    FuncArg,
)
from air.ir import Module, Context
from air.backend.xrt import XRTBackend

# Stage-2 cascade params validated at emb=2048, hidden=8192.
_STAGE2_TILE_M = 32
_STAGE2_M_INPUT = 4
_STAGE2_HERD_COLS = 8
_STAGE2_N_CASCADE = 4

# Symbols defined in mv_bf16.o, shared by stages 1 and 3. Excluded from
# per-launch prefix renaming so both call sites resolve the same symbol.
_EXTERNS = {
    "@zero_vectorized_bf16",
    "@matvec_vectorized_bf16",
    "@partial_plus_r_bf16",
}


def build_o_gemv_ffn_module(emb_dim=2048, hidden_dim=8192):
    """Build the three-launch o_gemv_ffn module.

    All three stages share core columns, so they are sequenced inside one
    ELF (each as its own aie.device). Stage 1's output and Stage 3's
    residual both read/write a row-0 subview of arg6, eliminating any
    standalone L3 buffer for the post-attention residual.
    """
    stage1 = build_2tile_add(emb_dim, emb_dim, m=8, k=512, n_cores=8)
    stage2 = build_swiglu_rms(
        2 * hidden_dim,
        emb_dim,
        _STAGE2_TILE_M,
        _STAGE2_M_INPUT,
        _STAGE2_HERD_COLS,
        _STAGE2_N_CASCADE,
        bfloat16,
        bfloat16,
    )
    stage3 = build_2tile_add(emb_dim, hidden_dim, m=8, k=512, n_cores=8)

    # Stage 1 — matvec_2tile_add local (A=0, B=1, R=2, D=3):
    #   wo (arg0) @ attn_out (arg1) + x_residual (arg3)  →  arg6[0]
    # Stage 2 — matvec_swiglu_rms local (A_interleaved=0, packed_rms=1, D=2):
    #   w_gateup (arg7), packed (arg6 native 2D), swiglu (arg11)
    # Stage 3 — matvec_2tile_add local (A=0, B=1, R=2, D=3):
    #   wdown (arg12) @ swiglu (arg11) + arg6[0]  →  output (arg14)
    # The post-attention residual is routed through a row-0 subview of arg6 (the
    # packed RMSNorm-input buffer), declared in the func-body prelude and aliased
    # into stage1's R operand and stage3's R operand.
    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<2x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{2 * hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{emb_dim}x{hidden_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{emb_dim}xbf16>"),
    ]
    prelude = (
        f"    %arg6_row0_strided = memref.subview %arg6[0, 0] [1, {emb_dim}] [1, 1]\n"
        f"        : memref<2x{emb_dim}xbf16> to memref<{emb_dim}xbf16, strided<[1]>>\n"
        f"    %arg6_row0 = memref.cast %arg6_row0_strided\n"
        f"        : memref<{emb_dim}xbf16, strided<[1]>> to memref<{emb_dim}xbf16>"
    )
    slices = [
        KernelSlice(str(stage1), "s1", {0: 0, 1: 1, 2: 3},
                    arg_aliases={3: "%arg6_row0"}, extern_syms=_EXTERNS),
        KernelSlice(str(stage2), "s2", {0: 7, 1: 6, 2: 11}, extern_syms=_EXTERNS),
        KernelSlice(str(stage3), "s3", {0: 12, 1: 11, 3: 14},
                    arg_aliases={2: "%arg6_row0"}, extern_syms=_EXTERNS),
    ]
    # args 2,4,5,8,9,10,13 are dead-ABI placeholders: present so this ELF's
    # signature matches the int4 o_gemv_ffn variant, but not wired to any launch.
    return stitch_elf(
        "o_gemv_ffn",
        base_args,
        slices,
        prelude=prelude,
        allow_unreferenced_args={2, 4, 5, 8, 9, 10, 13},
    )


def o_gemv_ffn_reference(
    wo, attn_out, x_residual, ffn_norm_w, wgate, wup, wdown, eps=1e-5
):
    """CPU F32 reference for the 3-launch o_gemv_ffn pipeline."""
    res1 = wo.astype(np.float32) @ attn_out.astype(np.float32) + x_residual.astype(
        np.float32
    )
    rstd = 1.0 / np.sqrt((res1 * res1).mean() + eps)
    normed = (res1 * rstd) * ffn_norm_w.astype(np.float32)
    normed_bf16 = normed.astype(bfloat16).astype(np.float32)
    gate = wgate.astype(np.float32) @ normed_bf16
    up = wup.astype(np.float32) @ normed_bf16
    swiglu = (gate * 0.5 * (np.tanh(gate / 2.0) + 1.0)) * up
    swiglu_bf16 = swiglu.astype(bfloat16).astype(np.float32)
    output = (wdown.astype(np.float32) @ swiglu_bf16 + res1).astype(bfloat16)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="o_gemv_ffn_multi.py",
        description="3-launch o_gemv_ffn (matvec_2tile_add + matvec_swiglu_rms "
        "+ matvec_2tile_add) with arg6[0]-subview-routed residual.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"], default="elf"
    )
    args = parser.parse_args()

    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    print(f"O GEMV + FFN 3-launch: emb_dim={emb_dim}, hidden_dim={hidden_dim}")

    module = build_o_gemv_ffn_module(emb_dim, hidden_dim)
    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="o_gemv_ffn",
            use_lock_race_condition_fix=False,
        )
        backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    np.random.seed(42)
    wo = (np.random.randn(emb_dim, emb_dim) * 0.02).astype(bfloat16)
    attn_out = np.random.randn(emb_dim).astype(bfloat16)
    x_residual = np.random.randn(emb_dim).astype(bfloat16)
    ffn_norm_w = (np.random.randn(emb_dim) * 0.1 + 1.0).astype(bfloat16)
    gate = (np.random.randn(hidden_dim, emb_dim) * 0.02).astype(bfloat16)
    up = (np.random.randn(hidden_dim, emb_dim) * 0.02).astype(bfloat16)
    w_gateup = np.empty((2 * hidden_dim, emb_dim), dtype=bfloat16)
    w_gateup[0::2] = gate
    w_gateup[1::2] = up
    wdown = (np.random.randn(emb_dim, hidden_dim) * 0.01).astype(bfloat16)
    packed = np.empty((2, emb_dim), dtype=bfloat16)
    packed[0] = 0.0
    packed[1] = ffn_norm_w
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)

    expected = o_gemv_ffn_reference(
        wo, attn_out, x_residual, ffn_norm_w, gate, up, wdown
    )

    # ABI placeholders for dead args.
    z_emb = np.zeros(emb_dim, dtype=bfloat16)
    z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
    z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)

    from air.backend.xrt_runner import XRTRunner

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="o_gemv_ffn",
        use_lock_race_condition_fix=False,
    )
    sys.exit(
        runner.run_test(
            module,
            inputs=[
                wo,
                attn_out,
                z_emb,
                x_residual,
                z_emb,
                z_emb,
                packed,
                w_gateup,
                z_hidden,
                z_hidden_emb,
                z_hidden,
                swiglu_buf,
                wdown,
                z_emb,
            ],
            expected_outputs=[expected],
            rtol=0.1,
            atol=2.0,
            min_correlation=0.99,
        )
    )
