# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O bfp16 GEMM + Residual Add + FFN (bfp16 Gate/Up/Down + SwiGLU) — 8-launch ELF.

bfp16 prefill sibling of o_ffn_int4_multi.py. Per-layer weight slots
arg1 / arg7 / arg9 / arg12 carry bfp16ebs8-packed uint8 BOs instead of
int4 Q+S+Z; arg slots and launch shape are unchanged.

Usage:
    from o_ffn_bfp16_multi import build_o_ffn_bfp16_module
    module = build_o_ffn_bfp16_module(seq_len, emb_dim, hidden_dim)
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Shared scaffolding (llama_kernel_builder.stitching, ffn_swiglu builder,
# weighted_rms_norm, rope_lut) currently lives under the bf16 example.
# Cross-link until those move to a shared location.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "llama32_1b"),
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "matrix_multiplication",
        "bf16_x_bfp16",
    ),
)

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
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


# ---------------------------------------------------------------------------
# 2D→2D eltwise add — verbatim from bf16 sibling.
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_2d(rows, cols, np_dtype, vector_size=16, herd_x=8, herd_y=1):
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    n = rows * cols
    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    total_tiles = herd_x * herd_y
    chunk_size = n // total_tiles
    tile_n = cols
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_2d_ty)
    def eltwise_add_2d(arg0_2d, arg1_2d, arg2_2d):
        @launch(operands=[arg0_2d, arg1_2d, arg2_2d])
        def add_launch(l_a, l_b, l_out):
            a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
            b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out, [[0, 1]])

            @segment(name="add_seg", operands=[a_flat, b_flat, out_flat])
            def add_seg(s_a, s_b, s_out):
                offset_map = AffineMap.get(
                    0,
                    3,
                    [
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
                                AffineConstantExpr.get(chunk_size),
                            ),
                        )
                    ],
                )

                @herd(
                    name="add_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_a, s_b, s_out],
                )
                def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                    l1_a = AllocOp(l1_ty, [], [])
                    l1_b = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    for loop_iv in range_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        for j in range_(0, tile_n, vector_size):
                            sub_a = subview(l1_a.result, [j], [vector_size], [1])
                            sub_b = subview(l1_b.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_a = transfer_read(
                                vec_ty, sub_a, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vec_ty, sub_b, [c0], identity_map, cst0, [True]
                            )
                            v_sum = arith.addf(v_a, v_b)
                            transfer_write(
                                None, v_sum, sub_out, [c0], identity_map, [True]
                            )
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_out)


# ---------------------------------------------------------------------------
# 2D→1D eltwise add — same as bf16 sibling (FFN final add).
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_1d(rows, cols, np_dtype, vector_size=16, total_tiles=8):
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    n = rows * cols
    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    chunk_size = n // total_tiles
    tile_n = cols
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_1d_ty)
    def eltwise_add(a_2d, b_2d, out_1d):
        @launch(operands=[a_2d, b_2d, out_1d])
        def add_launch(l_a, l_b, l_out):
            a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
            b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])

            @segment(name="add_seg", operands=[a_flat, b_flat, l_out])
            def add_seg(s_a, s_b, s_out):
                offset_map = AffineMap.get(
                    0,
                    3,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(1),
                                    ),
                                    AffineSymbolExpr.get(2),
                                ),
                                AffineConstantExpr.get(chunk_size),
                            ),
                        )
                    ],
                )

                @herd(
                    name="add_herd", sizes=[total_tiles, 1], operands=[s_a, s_b, s_out]
                )
                def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                    l1_a = AllocOp(l1_ty, [], [])
                    l1_b = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    for loop_iv in range_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        for j in range_(0, tile_n, vector_size):
                            sub_a = subview(l1_a.result, [j], [vector_size], [1])
                            sub_b = subview(l1_b.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_a = transfer_read(
                                vec_ty, sub_a, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vec_ty, sub_b, [c0], identity_map, cst0, [True]
                            )
                            v_sum = arith.addf(v_a, v_b)
                            transfer_write(
                                None, v_sum, sub_out, [c0], identity_map, [True]
                            )
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_out)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_o_ffn_bfp16_module(
    seq_len=2048,
    emb_dim=2048,
    hidden_dim=8192,
    # Shared bfp16 tile config — used by ALL 4 GEMMs (O / Gate / Up / Down).
    # Uniform tile_m=tile_n required so the shared private kernel decls
    # (@matmul_*, @zero_*, @f32_to_bf16_mn) don't collide across GEMMs.
    # 32x32 fits the memtile bank budget with 4 GEMMs sharing segment scope.
    tile_m=32,
    tile_n=32,
    tile_k_l1=128,
    o_herd_m=8,
    o_herd_n=4,
    gate_herd_m=8,
    gate_herd_n=4,
    # Down K=8192: tile_k_l2=tile_k_l1=128 keeps L2 A within memtile budget;
    # the L1 C f32 accumulator persists at segment scope across all K-l2 stages.
    down_tile_k_l2=128,
    down_herd_m=8,
    down_herd_n=4,
    # SwiGLU config
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    print_kernels=False,
):
    """Build 8-launch module: bfp16 O + Res Add + FFN RMSNorm + bfp16 Gate/Up +
    SwiGLU + bfp16 Down + FFN Add.

    Same 15-arg combined func ABI as the int4/bf16 siblings; the 4
    O/Gate/Up/Down weights are 3D u8 BOs carrying bfp16ebs8 byte tiles:

        %arg0:  attn_out     (seq_len, emb_dim)              bf16
        %arg1:  wo_packed    (N_o/n, K_o/k, tile_bytes)      u8
        %arg2:  proj         (seq_len, emb_dim)              bf16
        %arg3:  x_residual   (seq_len, emb_dim)              bf16
        %arg4:  res1         (seq_len, emb_dim)              bf16
        %arg5:  ffn_norm_w   (emb_dim,)                       bf16
        %arg6:  normed2      (seq_len, emb_dim)              bf16
        %arg7:  w_gate_packed (N_h/n, K_e/k, tile_bytes)     u8
        %arg8:  gate         (seq_len, hidden_dim)           bf16
        %arg9:  w_up_packed  (N_h/n, K_e/k, tile_bytes)      u8
        %arg10: up           (seq_len, hidden_dim)           bf16
        %arg11: swiglu       (seq_len, hidden_dim)           bf16
        %arg12: w_down_packed (N_e/n', K_h/k, tile_bytes')   u8
        %arg13: down         (seq_len, emb_dim)              bf16
        %arg14: output       (seq_len*emb_dim,)              bf16 (1D)
    """
    from llama32_1b_int4.bfp16_gemm_builder import (
        _build_bfp16_gemm_module as build_bfp16_gemm,
    )
    from matmul_bf16_x_bfp16 import bfp_tile_bytes
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from llama_kernel_builder.ffn_swiglu.silu_and_mul import (
        build_module_2d as build_swiglu,
    )

    n_total = seq_len * emb_dim

    # Packed-BO shapes — uniform tile_n across all 4 weight BOs.
    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    n_o_div = emb_dim // tile_n
    n_h_div = hidden_dim // tile_n
    n_e_div = emb_dim // tile_n
    k_e_div = emb_dim // tile_k_l1
    k_h_div = hidden_dim // tile_k_l1

    # ---- Build sub-kernels ----

    print("  [1/8] O bfp16 GEMM (M=K=N=emb_dim)...")
    o_ir = str(
        build_bfp16_gemm(
            seq_len,
            emb_dim,
            emb_dim,
            tile_m,
            emb_dim,
            tile_k_l1,
            tile_n,
            o_herd_m,
            o_herd_n,
        )
    )

    print("  [2/8] Residual Add (2D -> 2D)...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    print("  [3/8] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    print("  [4/8] Gate bfp16 GEMM (K=emb_dim, N=hidden_dim)...")
    gate_ir = str(
        build_bfp16_gemm(
            seq_len,
            emb_dim,
            hidden_dim,
            tile_m,
            emb_dim,
            tile_k_l1,
            tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    print("  [5/8] Up bfp16 GEMM (K=emb_dim, N=hidden_dim)...")
    up_ir = str(
        build_bfp16_gemm(
            seq_len,
            emb_dim,
            hidden_dim,
            tile_m,
            emb_dim,
            tile_k_l1,
            tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    print("  [6/8] SwiGLU...")
    swiglu_ir = _wrap_ir_in_launch(
        str(
            build_swiglu(
                seq_len,
                hidden_dim,
                swiglu_tile_n,
                bfloat16,
                swiglu_herd_x,
                swiglu_herd_y,
            )
        )
    )

    print(
        f"  [7/8] Down bfp16 GEMM (K=hidden_dim={hidden_dim}, N=emb_dim, "
        f"tile_k_l2={down_tile_k_l2} -> {hidden_dim // down_tile_k_l2} K-l2 stages)..."
    )
    down_ir = str(
        build_bfp16_gemm(
            seq_len,
            hidden_dim,
            emb_dim,
            tile_m,
            down_tile_k_l2,
            tile_k_l1,
            tile_n,
            down_herd_m,
            down_herd_n,
        )
    )

    print("  [8/8] FFN Add (2D -> 1D)...")
    ffn_add_ir = str(_build_add_2d_to_1d(seq_len, emb_dim, bfloat16))

    if print_kernels:
        for name, ir in [
            ("O bfp16 GEMM", o_ir),
            ("Res Add", res_add_ir),
            ("FFN RMSNorm", rms_ir),
            ("Gate bfp16 GEMM", gate_ir),
            ("Up bfp16 GEMM", up_ir),
            ("SwiGLU", swiglu_ir),
            ("Down bfp16 GEMM", down_ir),
            ("FFN Add", ffn_add_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Arg mapping:
    #   L1  O bfp16:      {0:0, 1:1, 2:2}        (attn_out, wo_packed, proj)
    #   L2  Res Add:      {0:2, 1:3, 2:4}        (proj, x_residual, res1[2D])
    #   L3  FFN RMSNorm:  {0:4, 1:5, 2:6}        (res1, ffn_norm_w, normed2)
    #   L4  Gate bfp16:   {0:6, 1:7, 2:8}        (normed2, w_gate_packed, gate)
    #   L5  Up bfp16:     {0:6, 1:9, 2:10}       (normed2, w_up_packed, up)
    #   L6  SwiGLU:       {0:8, 1:10, 2:11}      (gate, up, swiglu)
    #   L7  Down bfp16:   {0:11, 1:12, 2:13}     (swiglu, w_down_packed, down)
    #   L8  FFN Add:      {0:13, 1:4, 2:14}      (down, res1[2D], output[1D])

    _EXTERN_FUNCS = {
        "@matmul_bf16_x_bfp16_packed_f32",
        "@zero_vectorized_f32_mn",
        "@f32_to_bf16_mn",
        "@zero_vectorized_bf16",
        "@silu_and_mul_bf16",
    }

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (o_ir, "og", {0: 0, 1: 1, 2: 2}),
        (res_add_ir, "ra", {0: 2, 1: 3, 2: 4}),
        (rms_ir, "rm", {0: 4, 1: 5, 2: 6}),
        (gate_ir, "gg", {0: 6, 1: 7, 2: 8}),
        (up_ir, "ug", {0: 6, 1: 9, 2: 10}),
        (swiglu_ir, "sw", {0: 8, 1: 10, 2: 11}),
        (down_ir, "dg", {0: 11, 1: 12, 2: 13}),
        (ffn_add_ir, "fa", {0: 13, 1: 4, 2: 14}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all_with_externs(body, prefix, _EXTERN_FUNCS)
        maps = [_rename_all_with_externs(m, prefix, _EXTERN_FUNCS) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Collect private decls: bfp16 GEMM's three (matmul + zero_f32 + f32_to_bf16)
    # and SwiGLU's @silu_and_mul_bf16.
    all_privates = set()
    for ir in [o_ir, swiglu_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(sorted(all_privates))

    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @o_ffn_bfp16(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{n_o_div}x{k_e_div}x{tile_bytes}xi8>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{seq_len}x{emb_dim}xbf16>,
    %arg4: memref<{seq_len}x{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<{seq_len}x{emb_dim}xbf16>,
    %arg7: memref<{n_h_div}x{k_e_div}x{tile_bytes}xi8>,
    %arg8: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg9: memref<{n_h_div}x{k_e_div}x{tile_bytes}xi8>,
    %arg10: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg11: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg12: memref<{n_e_div}x{k_h_div}x{tile_bytes}xi8>,
    %arg13: memref<{seq_len}x{emb_dim}xbf16>,
    %arg14: memref<{n_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
{bodies[3]}
{bodies[4]}
{bodies[5]}
{bodies[6]}
{bodies[7]}
    return
  }}
}}
"""

    with Context() as ctx:
        try:
            module = Module.parse(combined, ctx)
        except Exception:
            with open("/tmp/debug_o_ffn_bfp16.mlir", "w") as f:
                f.write(combined)
            print("  PARSE ERROR: dumped to /tmp/debug_o_ffn_bfp16.mlir")
            raise
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="O bfp16 GEMM + Res Add + FFN (bfp16 Gate/Up/Down) multi-launch"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--print-kernels", action="store_true")
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

    SEQ_LEN, EMB_DIM, HIDDEN_DIM = 2048, 2048, 8192
    TILE_N = 32  # must match build_o_ffn_bfp16_module's tile_n default
    TILE_K_L1 = 128

    print(
        f"O+FFN bfp16 Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, "
        f"hidden={HIDDEN_DIM}"
    )
    module = build_o_ffn_bfp16_module(
        seq_len=SEQ_LEN,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
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
            instance_name="o_ffn_bfp16",
            stack_size=2048,
            runtime_loop_tiling_sizes=[2, 2],
        )
        backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run ----
    from matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
        cpu_reference_from_bfp_packed,
    )

    np.random.seed(42)

    attn_out = np.random.uniform(-1, 1, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    x_residual = np.random.uniform(-1, 1, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    ffn_norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)

    def _w(K, N):
        return (np.random.randn(K, N) * (1.0 / np.sqrt(K))).astype(bfloat16)

    Wo = _w(EMB_DIM, EMB_DIM)
    Wgate = _w(EMB_DIM, HIDDEN_DIM)
    Wup = _w(EMB_DIM, HIDDEN_DIM)
    Wdown = _w(HIDDEN_DIM, EMB_DIM)
    wo_packed = pack_b_bfp16ebs8(Wo, TILE_N, TILE_K_L1)
    wgate_packed = pack_b_bfp16ebs8(Wgate, TILE_N, TILE_K_L1)
    wup_packed = pack_b_bfp16ebs8(Wup, TILE_N, TILE_K_L1)
    wdown_packed = pack_b_bfp16ebs8(Wdown, TILE_N, TILE_K_L1)

    print("Computing CPU reference...")
    proj_ref = cpu_reference_from_bfp_packed(
        wo_packed, attn_out, SEQ_LEN, EMB_DIM, EMB_DIM, TILE_N, TILE_K_L1
    )
    res1_ref = (proj_ref.astype(np.float32) + x_residual.astype(np.float32)).astype(
        bfloat16
    )
    r1 = res1_ref.astype(np.float32)
    rms = np.sqrt(np.mean(r1**2, axis=-1, keepdims=True) + 1e-5)
    normed2_ref = (r1 / rms * ffn_norm_w.astype(np.float32)).astype(bfloat16)
    gate_ref = cpu_reference_from_bfp_packed(
        wgate_packed, normed2_ref, SEQ_LEN, EMB_DIM, HIDDEN_DIM, TILE_N, TILE_K_L1
    )
    up_ref = cpu_reference_from_bfp_packed(
        wup_packed, normed2_ref, SEQ_LEN, EMB_DIM, HIDDEN_DIM, TILE_N, TILE_K_L1
    )
    g = gate_ref.astype(np.float32)
    swiglu_ref = (g / (1 + np.exp(-g)) * up_ref.astype(np.float32)).astype(bfloat16)
    down_ref = cpu_reference_from_bfp_packed(
        wdown_packed, swiglu_ref, SEQ_LEN, HIDDEN_DIM, EMB_DIM, TILE_N, TILE_K_L1
    )
    output_ref = (down_ref.astype(np.float32) + res1_ref.astype(np.float32)).astype(
        bfloat16
    )

    proj_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    res1_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    normed2_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    gate_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    up_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    swiglu_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    down_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="o_ffn_bfp16",
        stack_size=2048,
        runtime_loop_tiling_sizes=[2, 2],
    )

    exit(
        runner.run_test(
            module,
            inputs=[
                attn_out,
                wo_packed,
                proj_buf,
                x_residual,
                res1_buf,
                ffn_norm_w,
                normed2_buf,
                wgate_packed,
                gate_buf,
                wup_packed,
                up_buf,
                swiglu_buf,
                wdown_packed,
                down_buf,
            ],
            expected_outputs=[output_ref.flatten()],
            rtol=0.5,
            atol=10.0,
            min_correlation=0.99,
            max_mismatch_percentage=0.5,
        )
    )
