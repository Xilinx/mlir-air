# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Backbone-specific rms_gemms_rope variant: Q+K+V fused into ONE GEMM.

4 launches instead of shared/builders/rms_gemms_rope_multi.py's 6 (RMSNorm,
QKV-fused-GEMM, RoPE-Q-from-wide, RoPE-K-from-wide). V needs no RoPE, so it is
read directly from the fused GEMM's wide output buffer host-side -- no launch
for it at all. RMSNorm is unchanged, reused as-is.

New isolated file: does not touch shared/builders/rms_gemms_rope_multi.py or
shared/infra/stitching.py, so llama32_1b/smollm2_1_7b/qwen builds are
untouched. Tile choice (tile_n=80) matches the REAL registry tiles Q/K/V
already use unfused (both resolve to tile_n=80 there too -- unlike the GateUp
case, this is not a repeat of that tile-mismatch mistake).
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(_HERE.parent.parent.parent)
]

from ml_dtypes import bfloat16
from air_examples.llms.shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
)
from rope_from_wide import build_rope_from_wide


def build_rms_gemms_rope_module_fused_qkv(
    seq_len,
    emb_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    herd_m=4,
    herd_n=4,
    qkv_tile_n=80,
    print_kernels=False,
    b_stationary=False,
    bfp16=None,
):
    """RMSNorm + fused QKV GEMM + RoPE-Q-from-wide + RoPE-K-from-wide.

    Base args:
      %arg0 x_in     (seq_len, emb_dim)                    input
      %arg1 norm_w   (emb_dim,)
      %arg2 normed   (seq_len, emb_dim)                    RMSNorm output
      %arg3 w_qkv    (emb_dim, emb_dim+2*kv_dim)            [Wq|Wk|Wv] concatenated
      %arg4 qkv      (seq_len, emb_dim+2*kv_dim)            fused GEMM output
      %arg5 lut_q    (seq_len*n_heads*head_dim,)
      %arg6 q_roped  (seq_len, emb_dim)
      %arg7 lut_k    (seq_len*n_kv_heads*head_dim,)
      %arg8 k_roped  (seq_len, kv_dim)

    V has no RoPE -- read it directly from the returned qkv buffer's columns
    [emb_dim+kv_dim : emb_dim+2*kv_dim] host-side, no launch needed.

    bfp16=(tile_n, tile_k_l2, tile_k_l1): the QKV GEMM takes bfp16ebs8 weights
    (gemm_bfp16.py); %arg3 is then the packed w_qkv (pack_b_bfp16ebs8).
    """
    from air_examples.llms.shared.builders.gemm_builder import _build_gemm_module
    from air_examples.llms.shared.infra.external_kernels import compile_gemm_mm
    from air_examples.weighted_rms_norm.weighted_rms_norm import (
        build_module as build_rms,
    )

    qkv_n = emb_dim + 2 * kv_dim  # 960 + 320 + 320 = 1600

    print("  [1/4] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    w_qkv_type = f"memref<{emb_dim}x{qkv_n}xbf16>"
    if bfp16:
        from gemm_bfp16 import (
            bfp16_extern_syms,
            bfp16_weight_type,
            build_gemm_bfp16,
            compile_mm_bfp16,
        )

        tn, tk2, tk1 = bfp16
        print(f"  [2/4] QKV GEMM, fused, bfp16 weights (tile_n {tn}, K {tk2}x{tk1})...")
        compile_mm_bfp16(32, tn, tk1, "_qkvb", "mm_qkvb.o")
        qkv_ir = str(
            build_gemm_bfp16(
                seq_len,
                emb_dim,
                qkv_n,
                32,
                tk2,
                tk1,
                tn,
                herd_m,
                herd_n,
                "_qkvb",
                "mm_qkvb.o",
            )
        )
        qkv_extern_syms = bfp16_extern_syms("_qkvb")
        w_qkv_type = bfp16_weight_type(emb_dim, qkv_n, tn, tk1)
    else:
        print("  [2/4] QKV GEMM, fused (drain)...")
        compile_gemm_mm(
            tile_m=32,
            tile_n=qkv_tile_n,
            tile_k_l1=32,
            sym_suffix="_qkv",
            out_name="mm_qkv.o",
        )
        qkv_ir = str(
            _build_gemm_module(
                seq_len,
                emb_dim,
                qkv_n,
                32,
                emb_dim,
                32,
                qkv_tile_n,
                herd_m,
                herd_n,
                external_bf16_out=True,
                sym_suffix="_qkv",
                link_with_name="mm_qkv.o",
                b_stationary=b_stationary,
            )
        )
        qkv_extern_syms = {
            "@matmul_bf16",
            "@op_has_no_registered_library_name_qkv",
            "@zero_f32_mn_qkv",
            "@f32_to_bf16_mn_qkv",
        }

    print(f"  [3/4] RoPE Q from wide (cols 0:{emb_dim} of {qkv_n})...")
    rope_q_ir = str(
        build_rope_from_wide(seq_len, qkv_n, 0, n_heads, head_dim, bfloat16, herd_x=8)
    )

    print(f"  [4/4] RoPE K from wide (cols {emb_dim}:{emb_dim+kv_dim} of {qkv_n})...")
    rope_k_ir = str(
        build_rope_from_wide(
            seq_len, qkv_n, emb_dim, n_kv_heads, head_dim, bfloat16, herd_x=8
        )
    )

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("QKV GEMM", qkv_ir),
            ("RoPE Q wide", rope_q_ir),
            ("RoPE K wide", rope_k_ir),
        ]:
            print(
                f"\n{'='*60}\n  Sub-kernel: {name} ({len(ir.splitlines())} lines)\n{'='*60}"
            )
            print(ir)

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", w_qkv_type),
        FuncArg("%arg4", f"memref<{seq_len}x{qkv_n}xbf16>"),
        FuncArg("%arg5", f"memref<{seq_len*n_heads*head_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{seq_len*n_kv_heads*head_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{kv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(qkv_ir, "qkv", {0: 2, 1: 3, 2: 4}, extern_syms=qkv_extern_syms),
        KernelSlice(rope_q_ir, "rq", {0: 4, 1: 5, 2: 6}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 4, 1: 7, 2: 8}, extern_syms={"@rope"}),
    ]

    module = stitch_elf("rms_gemms_rope_fused_qkv", base_args, slices)
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module
