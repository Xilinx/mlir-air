# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LM Head GEMV multi-launch — 8-partition GEMV in one ELF for decode.

Partitions the large vocab projection into 8 GEMVs of M=16384, K=2048 each,
stitched as 8 air.launch ops in one ELF. Single-token decode version of the
prefill LM Head (which uses GEMM with M=seq_len).

17 func args: 1 shared input (1D) + 8 weights (2D) + 8 outputs (1D).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "matrix_vector_multiplication", "bf16"
    ),
)

from llama_kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _fix_launch_func_args,
    _rename_all_with_externs,
)

_EXTERN_FUNCS = {"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"}


def build_lm_head_gemv_module(
    emb_dim=2048,
    n_partitions=8,
    n_part=16384,
    tile_m=8,
    m_input=4,
    herd_m=8,
):
    """Build multi-launch LM Head GEMV: n_partitions GEMV launches in one func.

    Each partition: GEMV with M=n_part (output dim), K=emb_dim (input dim).
    All partitions share the same input vector.

    Returns:
        Module with func @lm_head_gemv and (1 + 2*n_partitions) memref args:
            %arg0: x (emb_dim,) — shared input vector (1D)
            %arg(1+2*p): weight_p (n_part, emb_dim) — partition weight (2D)
            %arg(2+2*p): output_p (n_part,) — partition output (1D)
    """
    from matvec import build_module as build_gemv
    from ml_dtypes import bfloat16

    print(
        f"  Building {n_partitions}-partition LM Head GEMV (M_part={n_part}, K={emb_dim})..."
    )
    gemv_ir = str(
        build_gemv(n_part, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    # Extract private function declarations (shared across all partitions)
    privates = _extract_private_funcs(gemv_ir)
    privates_str = "\n  ".join(p.strip() for p in privates)

    # Stitch 8 copies with different arg mappings
    # GEMV func has 3 args: {0: weight (MxK), 1: input (K,), 2: output (M,)}
    # Combined func: arg0=shared_input, arg(1+2p)=weight_p, arg(2+2p)=output_p
    # Per-partition mapping: {0: 1+2*p, 1: 0, 2: 2+2*p}
    bodies, maps_all = [], []
    for p in range(n_partitions):
        prefix = f"p{p}"
        body = _extract_between_func_and_return(gemv_ir)
        maps = _extract_affine_maps(gemv_ir)
        body = _rename_all_with_externs(body, prefix, _EXTERN_FUNCS)
        maps = [_rename_all_with_externs(m, prefix, _EXTERN_FUNCS) for m in maps]
        body = _fix_launch_func_args(body, prefix, {0: 1 + 2 * p, 1: 0, 2: 2 + 2 * p})
        bodies.append(body)
        maps_all.extend(maps)

    # Build func signature: 1 shared input + 8 (weight, output) pairs
    arg_lines = [f"    %arg0: memref<{emb_dim}xbf16>"]
    for p in range(n_partitions):
        arg_lines.append(f"    %arg{1+2*p}: memref<{n_part}x{emb_dim}xbf16>")
        arg_lines.append(f"    %arg{2+2*p}: memref<{n_part}xbf16>")

    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @lm_head_gemv(
{(',' + chr(10)).join(arg_lines)}
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(
            f"  Module: {len(combined.splitlines())} lines, "
            f"{1+2*n_partitions} args, {n_partitions} launches, parsed OK"
        )
        return module
