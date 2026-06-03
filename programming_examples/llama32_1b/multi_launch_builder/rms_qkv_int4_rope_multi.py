# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + int4-AWQ QKV GEMV + RoPE Q+K -- 6-launch decode ELF.

AWQ-quantized sibling of rms_gemv_rope_multi.py. Q/K/V projections use
the int4 packed GEMV from matrix_vector_multiplication/int4_awq/, while
RMSNorm and RoPE stay bf16.

  L1: RMSNorm        [1,1]  x * norm_w   -> normed       (M=1, N=2048)
  L2: Q  int4 GEMV   [8,1]  wq_p @ normed -> q            (M=2048, K=2048)
  L3: K  int4 GEMV   [8,1]  wk_p @ normed -> k            (M=512,  K=2048)
  L4: V  int4 GEMV   [8,1]  wv_p @ normed -> v            (M=512,  K=2048)
  L5: RoPE Q         [1,1]  q * lut_q     -> q_roped      (N=32, dim=64)
  L6: RoPE K         [1,1]  k * lut_k     -> k_roped      (N=8,  dim=64)

13 func args (same count as bf16 sibling; arg3/5/7 retyped):
    %arg0:  x_in        memref<2048xbf16>            RMSNorm input
    %arg1:  norm_w      memref<2048xbf16>            RMSNorm weight
    %arg2:  normed      memref<2048xbf16>            RMSNorm out / GEMV in
    %arg3:  wq_packed   memref<TQ x TILE x i8>       Q packed (Q|S|Z)
    %arg4:  q           memref<2048xbf16>            Q out / RoPE Q in
    %arg5:  wk_packed   memref<TKV x TILE x i8>      K packed
    %arg6:  k           memref<512xbf16>             K out / RoPE K in
    %arg7:  wv_packed   memref<TKV x TILE x i8>      V packed
    %arg8:  v           memref<512xbf16>             V out (final)
    %arg9:  lut_q       memref<2048xbf16>            RoPE Q LUT
    %arg10: lut_k       memref<512xbf16>             RoPE K LUT
    %arg11: q_roped     memref<2048xbf16>            RoPE Q out (final)
    %arg12: k_roped     memref<512xbf16>             RoPE K out (final)

Usage:
    python3 rms_qkv_int4_rope_multi.py -p           # print combined MLIR
    python3 rms_qkv_int4_rope_multi.py              # compile + run + validate
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
        os.path.dirname(__file__),
        "..",
        "..",
        "matrix_vector_multiplication",
        "int4_awq",
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
    _rename_all_with_externs,
)
from llama_kernel_builder.external_kernels import compile_mv_int4_bf16, compile_rope

range_ = for_

EPS = 1e-5


def _extract_air_channels(mlir_text):
    """Pick up top-level `air.channel @name ...` declarations.

    The int4 GEMV builder declares its L3/L2/L1 channels explicitly; the
    bf16 GEMV builder relied on air-dma-to-channel to materialize them
    later. Stitching's _extract_between_func_and_return drops everything
    outside the func body, so we re-extract these lines and re-emit them
    (after prefix renaming) at module top in the combined module.
    """
    return [l for l in mlir_text.split("\n") if l.lstrip().startswith("air.channel @")]


# ---------------------------------------------------------------------------
# 1D RMSNorm wrapper — identical to bf16 sibling (M=1 decode form)
# ---------------------------------------------------------------------------


@module_builder
def _build_rms_1d(n, np_dtype, vector_size=16):
    from air.dialects.memref import expand_shape as memref_expand_shape

    xrt_dtype = type_mapper(np_dtype)
    assert n % vector_size == 0

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    l3_2d_ty = MemRefType.get([1, n], xrt_dtype)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([n], xrt_dtype, memory_space=l1_mem_space)
    l1VecTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    @FuncOp.from_py_func(l3_1d_ty, l3_1d_ty, l3_1d_ty)
    def rms_norm_1d(x_1d, weight, out_1d):
        @launch(operands=[x_1d, weight, out_1d])
        def rms_launch(l_x_1d, l_weight, l_out_1d):
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

                    dma_memcpy_nd(l1_weight_buf, l3_weight)

                    row = arith.ConstantOp.create_index(0)

                    dma_memcpy_nd(
                        l1_row,
                        l3_in,
                        src_offsets=[row, 0],
                        src_sizes=[1, n],
                        src_strides=[n, 1],
                    )

                    transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                    for j in range_(0, n, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_sq = arith.mulf(v_x, v_x)
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

                    v_final = transfer_read(
                        vecTy, l1_acc, [c0], identity_map, cst0, [True]
                    )
                    total_sum = vector_reduction(xrt_dtype, "add", v_final)
                    rms = arith.divf(total_sum, n_f)

                    f32 = F32Type.get()
                    rms_eps = arith.addf(rms, eps_f)
                    rms_eps_f32 = arith.extf(f32, rms_eps)
                    rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                    rstd = arith.truncf(xrt_dtype, rstd_f32)

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
# 1D RoPE launch builder — identical to bf16 sibling
# ---------------------------------------------------------------------------


@module_builder
def _build_rope_1d(n_rows, embed_dim, np_dtype, herd_x=1):
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

    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(herd_y),
                            ),
                            AffineSymbolExpr.get(2),
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


# External kernel function names preserved across stitching
_EXTERN_FUNCS = {
    "@matvec_int4_bf16_packed",  # int4 GEMV micro-kernel
    "@zero_vectorized_bf16",  # int4 GEMV zero-init
    "@rope",  # RoPE
}


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _packed_dims(M, K, GS, M_TILE, K_CHUNK, N_CORES, M_PER_LAUNCH):
    """Compute (total_tiles, tile_bytes) for int4 packed BO."""
    n_gpc = K_CHUNK // GS
    tile_bytes = M_TILE * (K_CHUNK // 2) + n_gpc * M_TILE * 2 + n_gpc * M_TILE
    M_per_core_per_launch = M_PER_LAUNCH // N_CORES
    M_div = M_per_core_per_launch // M_TILE
    K_div = K // K_CHUNK
    N_LAUNCHES = M // M_PER_LAUNCH
    total_tiles = N_LAUNCHES * N_CORES * M_div * K_div
    return total_tiles, tile_bytes


def build_rms_qkv_int4_rope_module(
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    # int4 GEMV config — all three Q/K/V share these
    gs=128,
    m_tile=8,
    k_chunk=2048,
    n_cores=8,
    rope_herd_x=1,
    print_kernels=False,
):
    """Build 6-launch module: RMSNorm + int4 Q/K/V GEMVs + RoPE Q + RoPE K."""
    from matvec_int4_packed import build_module as build_int4_gemv

    q_total = n_heads * head_dim
    k_total = n_kv_heads * head_dim
    assert q_total == emb_dim
    assert k_total == kv_dim
    assert k_chunk == emb_dim, "K_CHUNK must equal emb_dim for single-chunk GEMV"

    tq, tile_bytes = _packed_dims(
        emb_dim, emb_dim, gs, m_tile, k_chunk, n_cores, emb_dim
    )
    tkv, tile_bytes_kv = _packed_dims(
        kv_dim, emb_dim, gs, m_tile, k_chunk, n_cores, kv_dim
    )
    assert (
        tile_bytes == tile_bytes_kv
    ), "Q and K/V must share tile_bytes (same K/GS/M_TILE)"

    print("  [1/6] RMSNorm (decode, 1D wrapper)...")
    rms_ir = str(_build_rms_1d(emb_dim, bfloat16, 16))

    print(f"  [2/6] Q int4 GEMV (M={emb_dim}, K={emb_dim})...")
    q_ir = str(
        build_int4_gemv(
            emb_dim, emb_dim, GS=gs, M_TILE=m_tile, K_CHUNK=k_chunk, N_CORES=n_cores
        )
    )

    print(f"  [3/6] K int4 GEMV (M={kv_dim}, K={emb_dim})...")
    k_ir = str(
        build_int4_gemv(
            kv_dim, emb_dim, GS=gs, M_TILE=m_tile, K_CHUNK=k_chunk, N_CORES=n_cores
        )
    )

    print(f"  [4/6] V int4 GEMV (M={kv_dim}, K={emb_dim})...")
    v_ir = str(
        build_int4_gemv(
            kv_dim, emb_dim, GS=gs, M_TILE=m_tile, K_CHUNK=k_chunk, N_CORES=n_cores
        )
    )

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
            ("Q int4 GEMV", q_ir),
            ("K int4 GEMV", k_ir),
            ("V int4 GEMV", v_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # Arg map (combined func indices):
    #   RMSNorm:  {0->0, 1->1, 2->2}     (x_in, norm_w, normed)
    #   Q GEMV:   {0->3, 1->2, 2->4}     (wq_packed, normed, q)
    #   K GEMV:   {0->5, 1->2, 2->6}     (wk_packed, normed, k)
    #   V GEMV:   {0->7, 1->2, 2->8}     (wv_packed, normed, v)
    #   RoPE Q:   {0->4, 1->9, 2->11}    (q, lut_q, q_roped)
    #   RoPE K:   {0->6, 1->10, 2->12}   (k, lut_k, k_roped)
    bodies, maps_all, channels_all = [], [], []
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
        channels = _extract_air_channels(ir)
        body = _rename_all_with_externs(body, prefix, _EXTERN_FUNCS)
        maps = [_rename_all_with_externs(m, prefix, _EXTERN_FUNCS) for m in maps]
        channels = [
            _rename_all_with_externs(c, prefix, _EXTERN_FUNCS) for c in channels
        ]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)
        channels_all.extend(channels)

    # Private func declarations: one set from int4 GEMV (matvec + zero),
    # one from RoPE.
    all_privates = set()
    for ir in [q_ir, rope_q_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(all_privates)

    channels_str = "\n  ".join(channels_all)

    combined = "\n".join(maps_all) + f"""
module {{
  {channels_str}
  {privates_str}
  func.func @rms_qkv_int4_rope(
    %arg0: memref<{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{tq}x{tile_bytes}xi8>,
    %arg4: memref<{emb_dim}xbf16>,
    %arg5: memref<{tkv}x{tile_bytes}xi8>,
    %arg6: memref<{kv_dim}xbf16>,
    %arg7: memref<{tkv}x{tile_bytes}xi8>,
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
    x_f32 = x_1d.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_flat, lut_flat, head_dim):
    """Half-split RoPE matching HuggingFace Llama and rope_halfsplit.cc.

    LUT row layout: [cos_0..cos_{half-1}, sin_0..sin_{half-1}].
    Rotation:
        out[i]        = x[i]*cos[i] - x[i+half]*sin[i]
        out[i+half]   = x[i]*sin[i] + x[i+half]*cos[i]
    """
    half = head_dim // 2
    x = x_flat.astype(np.float32).reshape(-1, head_dim)
    lut = lut_flat.astype(np.float32).reshape(-1, head_dim)
    cos_v = lut[:, :half]
    sin_v = lut[:, half:]
    x1 = x[:, :half]
    x2 = x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * cos_v - x2 * sin_v
    out[:, half:] = x1 * sin_v + x2 * cos_v
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
    GS = 128
    M_TILE = 8
    K_CHUNK = 2048
    N_CORES = 8

    parser = argparse.ArgumentParser(
        description="RMSNorm + int4 QKV GEMV + RoPE QK multi-launch decode test"
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
        f"RMS+int4 QKV+RoPE Multi-Launch (decode): emb={EMB_DIM}, "
        f"kv={KV_DIM}, heads={N_HEADS}/{N_KV_HEADS}, dk={HEAD_DIM}, "
        f"gs={GS}"
    )

    module = build_rms_qkv_int4_rope_module(
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        gs=GS,
        m_tile=M_TILE,
        k_chunk=K_CHUNK,
        n_cores=N_CORES,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Ensure external .o files are in CWD where aiecc will look for them.
    compile_mv_int4_bf16(m_tile=M_TILE, k_chunk=K_CHUNK, gs=GS)
    compile_rope()

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rms_qkv_int4_rope",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # --- compile-and-run ---
    from matvec_int4_packed import pack_inputs, cpu_reference as int4_gemv_ref

    np.random.seed(42)

    # bf16 inputs
    x_in = np.random.uniform(-1.0, 1.0, (EMB_DIM,)).astype(bfloat16)
    norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)

    # Synthetic AWQ-style quantized weights (matches int4_awq test recipe)
    def gen_awq_weights(m, k, gs):
        a_q_unp = np.random.randint(0, 16, size=(m, k), dtype=np.uint8)
        a_q = (a_q_unp[:, 0::2] | (a_q_unp[:, 1::2] << 4)).astype(np.uint8)
        n_groups = k // gs
        a_s = np.random.uniform(0.005, 0.02, size=(n_groups, m)).astype(bfloat16)
        a_z = np.random.randint(7, 9, size=(n_groups, m), dtype=np.uint8)
        return a_q, a_s, a_z

    wq_q, wq_s, wq_z = gen_awq_weights(EMB_DIM, EMB_DIM, GS)
    wk_q, wk_s, wk_z = gen_awq_weights(KV_DIM, EMB_DIM, GS)
    wv_q, wv_s, wv_z = gen_awq_weights(KV_DIM, EMB_DIM, GS)

    # Pack each weight's Q+S+Z into a contiguous L3 slab (one launch each).
    wq_packed = pack_inputs(
        wq_q, wq_s, wq_z, EMB_DIM, EMB_DIM, GS, M_TILE, K_CHUNK, N_CORES, EMB_DIM
    )
    wk_packed = pack_inputs(
        wk_q, wk_s, wk_z, KV_DIM, EMB_DIM, GS, M_TILE, K_CHUNK, N_CORES, KV_DIM
    )
    wv_packed = pack_inputs(
        wv_q, wv_s, wv_z, KV_DIM, EMB_DIM, GS, M_TILE, K_CHUNK, N_CORES, KV_DIM
    )

    # RoPE LUTs (decode: single position, replicated per head). Use the
    # half-split [cos..., sin...] layout that matches rope_halfsplit.cc and
    # HuggingFace Llama's rotate_half. AWQ inference flows through the same
    # RoPE path as bf16 (quantization only affects the linear projections).
    from llama32_1b_weights import generate_rope_lut, LlamaConfig

    cfg = LlamaConfig()
    cfg.head_dim = HEAD_DIM
    base_lut_row = generate_rope_lut(cfg, seq_len=1, dtype=bfloat16)[0]  # (HEAD_DIM,)
    lut_q = np.tile(base_lut_row, (N_HEADS, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(base_lut_row, (N_KV_HEADS, 1)).flatten().astype(bfloat16)

    # CPU reference: RMS -> dequant int4 GEMV -> RoPE
    print("Computing CPU reference...")
    normed_ref = _rms_norm_ref(x_in, norm_w)
    q_ref = int4_gemv_ref(wq_q, wq_s, wq_z, normed_ref)
    k_ref = int4_gemv_ref(wk_q, wk_s, wk_z, normed_ref)
    v_ref = int4_gemv_ref(wv_q, wv_s, wv_z, normed_ref)
    q_roped_ref = _rope_ref(q_ref, lut_q, HEAD_DIM)
    k_roped_ref = _rope_ref(k_ref, lut_k, HEAD_DIM)

    # Intermediate buffers
    normed_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    q_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    k_buf = np.zeros(KV_DIM, dtype=bfloat16)
    v_buf = np.zeros(KV_DIM, dtype=bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_qkv_int4_rope",
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )

    exit(
        runner.run_test(
            module,
            inputs=[
                x_in,  # arg0
                norm_w,  # arg1
                normed_buf,  # arg2 (intermediate)
                wq_packed,  # arg3
                q_buf,  # arg4 (intermediate)
                wk_packed,  # arg5
                k_buf,  # arg6 (intermediate)
                wv_packed,  # arg7
                v_buf,  # arg8 (V output)
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
