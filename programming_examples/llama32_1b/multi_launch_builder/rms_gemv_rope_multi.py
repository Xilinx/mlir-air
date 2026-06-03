# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMV + RoPE Q+K -- 6-launch multi-launch ELF for decode.

Merges the decode attention front-half into a single ELF:
  L1: RMSNorm    [1,1]  x * norm_w -> normed        (M=1, N=2048)
  L2: Q GEMV     [8,1]  wq @ normed -> q             (M=2048, K=2048)
  L3: K GEMV     [8,1]  wk @ normed -> k             (M=512, K=2048)
  L4: V GEMV     [8,1]  wv @ normed -> v             (M=512, K=2048)
  L5: RoPE Q     [1,1]  q * lut_q -> q_roped         (N=32, dim=64)
  L6: RoPE K     [1,1]  k * lut_k -> k_roped         (N=8, dim=64)

All shared buffers are 1D (matching GEMV/RoPE expectations). RMSNorm
operates at M=1 for decode, so its 2D (1, emb_dim) I/O is equivalent to
1D (emb_dim). A custom wrapper builds the RMSNorm launch with 1D func
args and expand_shape/collapse_shape conversions inside the launch body.

13 func args (6 launches):
    %arg0:  x_in        memref<2048xbf16>         RMSNorm input (1D decode)
    %arg1:  norm_w      memref<2048xbf16>         RMSNorm weight
    %arg2:  normed      memref<2048xbf16>         RMSNorm output / GEMV input
    %arg3:  wq          memref<2048x2048xbf16>    Q weight (transposed)
    %arg4:  q           memref<2048xbf16>         Q output / RoPE Q input
    %arg5:  wk          memref<512x2048xbf16>     K weight (transposed)
    %arg6:  k           memref<512xbf16>          K output / RoPE K input
    %arg7:  wv          memref<512x2048xbf16>     V weight (transposed)
    %arg8:  v           memref<512xbf16>          V output (final)
    %arg9:  lut_q       memref<2048xbf16>         RoPE Q LUT (32*64)
    %arg10: lut_k       memref<512xbf16>          RoPE K LUT (8*64)
    %arg11: q_roped     memref<2048xbf16>         RoPE Q output (final)
    %arg12: k_roped     memref<512xbf16>          RoPE K output (final)

Usage:
    python3 rms_gemv_rope_multi.py -p           # print combined MLIR
    python3 rms_gemv_rope_multi.py              # compile + run + validate
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "matrix_vector_multiplication", "bf16"
    ),
)

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from llama_kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
    _rename_all_with_externs,
)

range_ = for_

EPS = 1e-5


# ---------------------------------------------------------------------------
# 1D RMSNorm wrapper (accepts 1D args, converts to 2D inside launch)
# ---------------------------------------------------------------------------


@module_builder
def _build_rms_1d(n, np_dtype, vector_size=16):
    """Build RMSNorm for M=1 with 1D func args (decode-friendly).

    The standard weighted_rms_norm builds with 2D (M, N) I/O memrefs.
    For decode (M=1), the GEMV expects 1D (N,) input. This wrapper:
      - Uses 1D memref<N x bf16> func args
      - Inside air.launch: expand_shape 1D -> (1, N) for the herd body
      - The herd body is the standard M=1 single-tile RMSNorm

    Func signature: (x_1d: [N], weight: [N], out_1d: [N])
    """
    from air.dialects.memref import expand_shape as memref_expand_shape

    xrt_dtype = type_mapper(np_dtype)
    assert n % vector_size == 0

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    # L3 types
    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    l3_2d_ty = MemRefType.get([1, n], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([n], xrt_dtype, memory_space=l1_mem_space)
    l1VecTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    @FuncOp.from_py_func(l3_1d_ty, l3_1d_ty, l3_1d_ty)
    def rms_norm_1d(x_1d, weight, out_1d):
        @launch(operands=[x_1d, weight, out_1d])
        def rms_launch(l_x_1d, l_weight, l_out_1d):
            # Expand 1D -> 2D for RMSNorm herd (which uses 2D DMA offsets)
            l_x_2d = memref_expand_shape(l3_2d_ty, l_x_1d, [[0, 1]], [], [1, n])
            l_out_2d = memref_expand_shape(l3_2d_ty, l_out_1d, [[0, 1]], [], [1, n])

            @segment(name="rms_seg", operands=[l_x_2d, l_weight, l_out_2d])
            def rms_seg(s_x_2d, s_weight, s_out_2d):
                @herd(
                    name="rms_herd",
                    sizes=[1, 1],
                    operands=[s_x_2d, s_weight, s_out_2d],
                )
                def rms_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
                    l1_row = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])
                    l1_weight_buf = AllocOp(l1RowTy, [], [])
                    l1_acc = AllocOp(l1VecTy, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    n_f = arith.ConstantOp(xrt_dtype, float(n))
                    eps_f = arith.ConstantOp(xrt_dtype, EPS)

                    v_zero = BroadcastOp(vecTy, cst0)

                    # DMA weight to L1
                    dma_memcpy_nd(l1_weight_buf, l3_weight)

                    # M=1: single row, no loop needed
                    row = arith.ConstantOp.create_index(0)

                    # DMA: load single row from 2D L3 to L1
                    dma_memcpy_nd(
                        l1_row,
                        l3_in,
                        src_offsets=[row, 0],
                        src_sizes=[1, n],
                        src_strides=[n, 1],
                    )

                    # Step 1: Vectorized sum of x^2
                    transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                    for j in range_(0, n, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_sq = arith.mulf(v_x, v_x)
                        # Break mulf->addf chain via store/load
                        transfer_write(None, v_sq, sub_tmp, [c0], identity_map, [True])
                        v_sq_rd = transfer_read(
                            vecTy, sub_tmp, [c0], identity_map, cst0, [True]
                        )
                        v_acc = transfer_read(
                            vecTy, l1_acc, [c0], identity_map, cst0, [True]
                        )
                        v_sum = arith.addf(v_acc, v_sq_rd)
                        transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                        yield_([])

                    # Horizontal reduce
                    v_final = transfer_read(
                        vecTy, l1_acc, [c0], identity_map, cst0, [True]
                    )
                    total_sum = vector_reduction(xrt_dtype, "add", v_final)
                    rms = arith.divf(total_sum, n_f)

                    # Step 2: rstd = rsqrt(rms + eps) in f32
                    f32 = F32Type.get()
                    rms_eps = arith.addf(rms, eps_f)
                    rms_eps_f32 = arith.extf(f32, rms_eps)
                    rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                    rstd = arith.truncf(xrt_dtype, rstd_f32)

                    # Step 3: y = x * rstd * weight
                    v_rstd = BroadcastOp(vecTy, rstd)
                    for j in range_(0, n, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_w = subview(l1_weight_buf.result, [j], [vector_size], [1])
                        sub_out = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_w = transfer_read(
                            vecTy, sub_w, [c0], identity_map, cst0, [True]
                        )
                        v_normed = arith.mulf(v_x, v_rstd)
                        v_weighted = arith.mulf(v_normed, v_w)
                        transfer_write(
                            None,
                            v_weighted,
                            sub_out,
                            [c0],
                            identity_map,
                            [True],
                        )
                        yield_([])

                    # DMA: write result row back to 2D L3
                    dma_memcpy_nd(
                        l3_out,
                        l1_out,
                        dst_offsets=[row, 0],
                        dst_sizes=[1, n],
                        dst_strides=[n, 1],
                    )

                    DeallocOp(l1_row)
                    DeallocOp(l1_out)
                    DeallocOp(l1_weight_buf)
                    DeallocOp(l1_acc)


# ---------------------------------------------------------------------------
# 1D RoPE launch builder (accepts 1D args, herd processes rows of head_dim)
# ---------------------------------------------------------------------------


@module_builder
def _build_rope_1d(n_rows, embed_dim, np_dtype, herd_x=1):
    """Build a RoPE launch with 1D func args (for decode GEMV compatibility).

    Func signature:
      (in_1d: [total], lut_1d: [total], out_1d: [total])

    The herd processes n_rows rows of embed_dim elements each.

    Args:
        n_rows:    Number of RoPE rows (n_heads for Q, n_kv_heads for K)
        embed_dim: RoPE column width per row (head_dim=64)
        herd_x:    Number of tiles for row-parallel
    """
    xrt_dtype = type_mapper(np_dtype)
    total = n_rows * embed_dim
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert embed_dim % 16 == 0
    assert n_rows % total_tiles == 0

    l3_1d_ty = MemRefType.get([total], xrt_dtype)
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get(
        shape=[embed_dim], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    rope_func = FuncOp(
        "rope", ([l1RowTy, l1RowTy, l1RowTy, T.i32()], []), visibility="private"
    )
    rope_func.attributes["link_with"] = StringAttr.get("rope.o")
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    rows_per_tile = n_rows // total_tiles

    # Affine map: row_offset = (local_row + tile_id * rows_per_tile) * embed_dim
    row_offset_map = AffineMap.get(
        0,
        3,  # s0=local_row, s1=_tx, s2=_ty
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),  # local_row
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),  # _tx
                                AffineConstantExpr.get(herd_y),
                            ),
                            AffineSymbolExpr.get(2),  # _ty
                        ),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(embed_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_1d_ty, l3_1d_ty, l3_1d_ty)
    def rope_1d(arg0_in, arg1_lut, arg2_out):
        @launch(operands=[arg0_in, arg1_lut, arg2_out])
        def rope_launch(l_in, l_lut, l_out):
            @segment(name="rope_seg", operands=[l_in, l_lut, l_out])
            def rope_seg(s_in, s_lut, s_out):
                @herd(
                    name="rope_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_lut, s_out],
                )
                def rope_body(_tx, _ty, _sx, _sy, h_in, h_lut, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_lut = AllocOp(l1RowTy, [], [])
                    l1_out_buf = AllocOp(l1RowTy, [], [])

                    dim_i32 = ConstantOp(T.i32(), embed_dim)

                    for local_row in range_(rows_per_tile):
                        row_offset = affine_apply(row_offset_map, [local_row, _tx, _ty])

                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[row_offset],
                            src_sizes=[embed_dim],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_lut,
                            h_lut,
                            src_offsets=[row_offset],
                            src_sizes=[embed_dim],
                            src_strides=[1],
                        )

                        CallOp(rope_func, [l1_in, l1_lut, l1_out_buf, dim_i32])

                        dma_memcpy_nd(
                            h_out,
                            l1_out_buf,
                            dst_offsets=[row_offset],
                            dst_sizes=[embed_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out_buf)

                rope_body.attributes["link_with"] = StringAttr.get("rope.o")


# External kernel function names shared across all sub-kernels
_EXTERN_FUNCS = {
    "@zero_vectorized_bf16",  # RMSNorm (if used)
    "@matvec_vectorized_bf16_bf16",  # GEMV
    "@linalg_fill_bf16",  # GEMV
    "@rope",  # RoPE
}


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_rms_gemv_rope_module(
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    # GEMV tile config
    tile_m=8,
    m_input=4,
    herd_m=8,
    # RoPE config
    rope_herd_x=1,
    print_kernels=False,
):
    """Build 6-launch module: RMSNorm + Q/K/V GEMVs + RoPE Q + RoPE K.

    All shared buffers are 1D memrefs (decode: M=1 tokens).

    Returns:
        Module with func @rms_gemv_rope and 13 memref args:
            %arg0:  x_in        (emb_dim,)             RMSNorm input
            %arg1:  norm_w      (emb_dim,)             RMSNorm weight
            %arg2:  normed      (emb_dim,)             RMSNorm output
            %arg3:  wq          (emb_dim, emb_dim)     Q weight
            %arg4:  q           (emb_dim,)             Q GEMV output
            %arg5:  wk          (kv_dim, emb_dim)      K weight
            %arg6:  k           (kv_dim,)              K GEMV output
            %arg7:  wv          (kv_dim, emb_dim)      V weight
            %arg8:  v           (kv_dim,)              V GEMV output
            %arg9:  lut_q       (emb_dim,)             RoPE Q LUT
            %arg10: lut_k       (kv_dim,)              RoPE K LUT
            %arg11: q_roped     (emb_dim,)             RoPE Q output
            %arg12: k_roped     (kv_dim,)              RoPE K output
    """
    from matvec import build_module as build_gemv

    q_total = n_heads * head_dim  # = emb_dim = 2048
    k_total = n_kv_heads * head_dim  # = kv_dim = 512

    assert q_total == emb_dim
    assert k_total == kv_dim

    # ---- Build sub-kernels ----

    # 1. RMSNorm at M=1 with 1D I/O (custom wrapper)
    print("  [1/6] RMSNorm (decode, 1D wrapper)...")
    rms_ir = str(_build_rms_1d(emb_dim, bfloat16, 16))

    # 2-4. Q/K/V GEMVs (already produce air.launch with 1D I/O)
    print("  [2/6] Q GEMV...")
    q_ir = str(
        build_gemv(emb_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print("  [3/6] K GEMV...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print("  [4/6] V GEMV...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    # 5-6. RoPE Q/K (1D in/out, launch+segment wrapper)
    # Decode: Q has n_heads=32 rows of head_dim=64, K has n_kv_heads=8 rows
    print(
        f"  [5/6] RoPE Q (n_rows={n_heads}, embed_dim={head_dim}, "
        f"herd_x={rope_herd_x})..."
    )
    rope_q_ir = str(_build_rope_1d(n_heads, head_dim, bfloat16, rope_herd_x))

    print(
        f"  [6/6] RoPE K (n_rows={n_kv_heads}, embed_dim={head_dim}, "
        f"herd_x={rope_herd_x})..."
    )
    rope_k_ir = str(_build_rope_1d(n_kv_heads, head_dim, bfloat16, rope_herd_x))

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q GEMV", q_ir),
            ("K GEMV", k_ir),
            ("V GEMV", v_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Arg mapping (combined func arg indices):
    #   RMSNorm:  {0->0, 1->1, 2->2}       (x_in, norm_w, normed)
    #   Q GEMV:   {0->3, 1->2, 2->4}       (wq, normed, q)
    #   K GEMV:   {0->5, 1->2, 2->6}       (wk, normed, k)
    #   V GEMV:   {0->7, 1->2, 2->8}       (wv, normed, v)
    #   RoPE Q:   {0->4, 1->9, 2->11}      (q, lut_q, q_roped)
    #   RoPE K:   {0->6, 1->10, 2->12}     (k, lut_k, k_roped)

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "r", {0: 0, 1: 1, 2: 2}),
        (q_ir, "q", {0: 3, 1: 2, 2: 4}),
        (k_ir, "k", {0: 5, 1: 2, 2: 6}),
        (v_ir, "v", {0: 7, 1: 2, 2: 8}),
        (rope_q_ir, "rq", {0: 4, 1: 9, 2: 11}),
        (rope_k_ir, "rk", {0: 6, 1: 10, 2: 12}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all_with_externs(body, prefix, _EXTERN_FUNCS)
        maps = [_rename_all_with_externs(m, prefix, _EXTERN_FUNCS) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Collect private func declarations from all sub-kernels
    all_privates = set()
    for ir in [q_ir, rope_q_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(all_privates)

    # Assemble (13 func args, 6 launches)
    combined = (
        "\n".join(maps_all)
        + f"""
module {{
  {privates_str}
  func.func @rms_gemv_rope(
    %arg0: memref<{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}xbf16>,
    %arg5: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg6: memref<{kv_dim}xbf16>,
    %arg7: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg8: memref<{kv_dim}xbf16>,
    %arg9: memref<{q_total}xbf16>,
    %arg10: memref<{k_total}xbf16>,
    %arg11: memref<{q_total}xbf16>,
    %arg12: memref<{k_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
{bodies[3]}
{bodies[4]}
{bodies[5]}
    return
  }}
}}
"""
    )

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(
            f"  Module: {len(combined.splitlines())} lines, "
            f"13 args, 6 launches, parsed OK"
        )
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def _rms_norm_ref(x_1d, weight, eps=1e-5):
    """CPU RMSNorm reference for 1D input (M=1 decode)."""
    x_f32 = x_1d.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_flat, lut_flat, head_dim):
    """CPU RoPE reference for flat 1D arrays."""
    x = x_flat.astype(np.float32).reshape(-1, head_dim)
    lut = lut_flat.astype(np.float32).reshape(-1, head_dim)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * lut[:, 0::2] - x[:, 1::2] * lut[:, 1::2]
    out[:, 1::2] = x[:, 0::2] * lut[:, 1::2] + x[:, 1::2] * lut[:, 0::2]
    return out.astype(bfloat16).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EMB_DIM = 2048
    KV_DIM = 512
    N_HEADS = 32
    N_KV_HEADS = 8
    HEAD_DIM = 64

    parser = argparse.ArgumentParser(
        description="RMSNorm + QKV GEMV + RoPE QK multi-launch decode test"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print combined MLIR and exit",
    )
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    args = parser.parse_args()

    print(
        f"RMS+GEMV+RoPE Multi-Launch (decode): emb={EMB_DIM}, "
        f"kv={KV_DIM}, heads={N_HEADS}/{N_KV_HEADS}, dk={HEAD_DIM}"
    )

    module = build_rms_gemv_rope_module(
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rms_gemv_rope",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run: build test data, run, verify ----
    np.random.seed(42)

    # Inputs
    x_in = np.random.uniform(-1.0, 1.0, (EMB_DIM,)).astype(bfloat16)
    norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    wq = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    wk = np.random.uniform(-0.1, 0.1, (KV_DIM, EMB_DIM)).astype(bfloat16)
    wv = np.random.uniform(-0.1, 0.1, (KV_DIM, EMB_DIM)).astype(bfloat16)

    # RoPE LUTs (decode: single position, one row per head)
    from rope_lut.rope_lut import generate_lut

    # For decode, LUT is just one position: (1, head_dim) repeated per head
    # But the LUT shape must match the total: n_heads * head_dim = emb_dim
    # Use position 0 for simplicity in test
    base_lut_row = generate_lut(1, HEAD_DIM, bfloat16)  # (1, 64)
    lut_q = np.tile(base_lut_row, (N_HEADS, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(base_lut_row, (N_KV_HEADS, 1)).flatten().astype(bfloat16)

    # CPU reference
    print("Computing CPU reference...")
    normed_ref = _rms_norm_ref(x_in, norm_w)
    q_ref = np.dot(wq.astype(np.float32), normed_ref.astype(np.float32)).astype(
        bfloat16
    )
    k_ref = np.dot(wk.astype(np.float32), normed_ref.astype(np.float32)).astype(
        bfloat16
    )
    v_ref = np.dot(wv.astype(np.float32), normed_ref.astype(np.float32)).astype(
        bfloat16
    )

    # Apply RoPE
    q_roped_ref = _rope_ref(q_ref, lut_q, HEAD_DIM)
    k_roped_ref = _rope_ref(k_ref, lut_k, HEAD_DIM)

    # Output buffers (zeroed)
    normed_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    q_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    k_buf = np.zeros(KV_DIM, dtype=bfloat16)
    v_buf = np.zeros(KV_DIM, dtype=bfloat16)

    # Func signature: 13 args
    # (x_in, norm_w, normed, wq, q, wk, k, wv, v, lut_q, lut_k, q_roped, k_roped)
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_gemv_rope",
    )

    # XRTRunner: inputs = first N args, expected_outputs = last M args.
    # Last 2 (arg11=q_roped, arg12=k_roped) are outputs.
    # First 11 (arg0-arg10) are inputs (including zeroed intermediates).
    exit(
        runner.run_test(
            module,
            inputs=[
                x_in,  # arg0
                norm_w,  # arg1
                normed_buf,  # arg2 (intermediate, zeroed)
                wq,  # arg3
                q_buf,  # arg4 (intermediate, zeroed)
                wk,  # arg5
                k_buf,  # arg6 (intermediate, zeroed)
                wv,  # arg7
                v_buf,  # arg8 (V output, also an intermediate)
                lut_q,  # arg9
                lut_k,  # arg10
            ],
            expected_outputs=[
                q_roped_ref,  # arg11
                k_roped_ref,  # arg12
            ],
            rtol=0.2,
            atol=0.5,
            min_correlation=0.99,
        )
    )
