# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O Projection + Residual Add + FFN — 8-launch multi-launch ELF.

Merges o_proj_add (2 launches) + ffn_full (6 launches) into a single
AIR function with 8 sequential air.launch operations:
  1. O GEMM          [8,4]   attn_out x wo -> proj
  2. Residual Add    [8,1]   proj + x_residual -> res1 (2D, collapse inside)
  3. FFN RMSNorm     [8,1]   res1 x ffn_norm_w -> normed2
  4. Gate GEMM       [8,4]   normed2 x w_gate -> gate
  5. Up GEMM         [8,4]   normed2 x w_up -> up
  6. SwiGLU          [8,1]   SiLU(gate) x up -> swiglu
  7. Down GEMM       [8,4]   swiglu x w_down -> down
  8. FFN Add         [8,1]   down + res1 -> output (1D)

15 func args (8 launches). The res1 buffer (arg4) is shared between the
Residual Add output and FFN input — 2D canonical type with collapse_shape
inside both add launches.

Usage:
    python3 o_ffn_multi.py -p           # print combined MLIR
    python3 o_ffn_multi.py              # compile + run + validate
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

from kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _rename_all,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
)

range_ = for_


# ---------------------------------------------------------------------------
# 2D→2D eltwise add (all 3 args 2D, collapse inside launch)
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_2d(rows, cols, np_dtype, vector_size=16, herd_x=8, herd_y=1):
    """Eltwise add: all 3 args are 2D memrefs, collapsed to 1D inside launch.

    Unlike the standard _build_add_2d (2D inputs, 1D output), this version
    keeps the OUTPUT as 2D too, so subsequent launches can read it as 2D
    without expand_shape. The collapse_shape inside the launch gives a 1D
    view for the DMA writes — same bytes in DDR, just different type.
    """
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
# Module builder
# ---------------------------------------------------------------------------


def build_o_ffn_module(
    seq_len=2048,
    emb_dim=2048,
    hidden_dim=8192,
    # O GEMM tile config
    o_tile_m=64,
    o_tile_k_l2=256,
    o_tile_k_l1=32,
    o_tile_n=64,
    o_herd_m=8,
    o_herd_n=4,
    # Gate/Up GEMM tile config
    gate_tile_m=64,
    gate_tile_k_l2=64,
    gate_tile_k_l1=32,
    gate_tile_n=128,
    gate_herd_m=8,
    gate_herd_n=4,
    # Down GEMM tile config
    down_tile_m=64,
    down_tile_k_l2=256,
    down_tile_k_l1=32,
    down_tile_n=64,
    down_herd_m=8,
    down_herd_n=4,
    # SwiGLU config
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    print_kernels=False,
):
    """Build 8-launch module: O GEMM + Res Add + FFN RMSNorm + Gate/Up + SwiGLU + Down + FFN Add.

    Returns:
        Module with func @o_ffn and 15 memref args:
            %arg0:  attn_out     (seq_len, emb_dim)
            %arg1:  wo           (emb_dim, emb_dim)
            %arg2:  proj         (seq_len, emb_dim)       intermediate
            %arg3:  x_residual   (seq_len, emb_dim)       skip connection
            %arg4:  res1         (seq_len, emb_dim)       shared (2D, collapse in adds)
            %arg5:  ffn_norm_w   (emb_dim,)
            %arg6:  normed2      (seq_len, emb_dim)       intermediate
            %arg7:  w_gate       (emb_dim, hidden_dim)
            %arg8:  gate         (seq_len, hidden_dim)    intermediate
            %arg9:  w_up         (emb_dim, hidden_dim)
            %arg10: up           (seq_len, hidden_dim)    intermediate
            %arg11: swiglu       (seq_len, hidden_dim)    intermediate
            %arg12: w_down       (hidden_dim, emb_dim)
            %arg13: down         (seq_len, emb_dim)       intermediate
            %arg14: output       (n_total,)               1D final output
    """
    from kernel_builder.gemm_builder import _build_gemm_module
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    n_total = seq_len * emb_dim

    # ---- Build sub-kernels ----

    # L1: O GEMM
    print("  [1/8] O GEMM...")
    o_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            o_tile_m,
            o_tile_k_l2,
            o_tile_k_l1,
            o_tile_n,
            o_herd_m,
            o_herd_n,
        )
    )

    # L2: Residual Add (2D → 2D, all args 2D, collapse inside)
    print("  [2/8] Residual Add (2D -> 2D)...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    # L3: FFN RMSNorm (bare herd → wrap)
    print("  [3/8] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # L4: Gate GEMM
    print("  [4/8] Gate GEMM...")
    gate_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    # L5: Up GEMM
    print("  [5/8] Up GEMM...")
    up_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    # L6: SwiGLU (bare herd → wrap)
    print("  [6/8] SwiGLU...")
    from kernel_builder.ffn_swiglu.silu_and_mul import (
        build_module_2d as build_swiglu,
    )

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

    # L7: Down GEMM
    print("  [7/8] Down GEMM...")
    down_ir = str(
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            down_tile_m,
            down_tile_k_l2,
            down_tile_k_l1,
            down_tile_n,
            down_herd_m,
            down_herd_n,
        )
    )

    # L8: FFN Add (2D inputs, 1D output — same as ffn_full's add)
    print("  [8/8] FFN Add (2D -> 1D)...")

    # Build the 2D→1D add (same pattern as ffn_full's eltwise_add)
    @module_builder
    def _build_add_2d_to_1d():
        from air.dialects.memref import collapse_shape as memref_collapse_shape

        xrt_dtype = type_mapper(bfloat16)
        l3_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        l3_1d_ty = MemRefType.get([n_total], xrt_dtype)
        total_tiles = 8
        chunk_size = n_total // total_tiles
        tile_n = emb_dim
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
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

                    @herd(name="add_herd", sizes=[8, 1], operands=[s_a, s_b, s_out])
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
                            for j in range_(0, tile_n, 16):
                                sub_a = subview(l1_a.result, [j], [16], [1])
                                sub_b = subview(l1_b.result, [j], [16], [1])
                                sub_out = subview(l1_out.result, [j], [16], [1])
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

    ffn_add_ir = str(_build_add_2d_to_1d())

    if print_kernels:
        for name, ir in [
            ("O GEMM", o_ir),
            ("Res Add", res_add_ir),
            ("FFN RMSNorm", rms_ir),
            ("Gate GEMM", gate_ir),
            ("Up GEMM", up_ir),
            ("SwiGLU", swiglu_ir),
            ("Down GEMM", down_ir),
            ("FFN Add", ffn_add_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Arg mapping:
    #   L1  O GEMM:       {0:0, 1:1, 2:2}        (attn_out, wo, proj)
    #   L2  Res Add:      {0:2, 1:3, 2:4}         (proj, x_residual, res1[2D])
    #   L3  FFN RMSNorm:  {0:4, 1:5, 2:6}         (res1, ffn_norm_w, normed2)
    #   L4  Gate GEMM:    {0:6, 1:7, 2:8}          (normed2, w_gate, gate)
    #   L5  Up GEMM:      {0:6, 1:9, 2:10}         (normed2, w_up, up)
    #   L6  SwiGLU:       {0:8, 1:10, 2:11}        (gate, up, swiglu)
    #   L7  Down GEMM:    {0:11, 1:12, 2:13}       (swiglu, w_down, down)
    #   L8  FFN Add:      {0:13, 1:4, 2:14}        (down, res1[2D], output[1D])

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
        body = _rename_all(body, prefix)
        maps = [_rename_all(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Collect private func declarations (SwiGLU has @silu_and_mul_bf16)
    all_privates = set()
    for ir in [swiglu_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(sorted(all_privates))

    # Assemble (15 func args, 8 launches)
    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @o_ffn(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{seq_len}x{emb_dim}xbf16>,
    %arg4: memref<{seq_len}x{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<{seq_len}x{emb_dim}xbf16>,
    %arg7: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg8: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg9: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg10: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg11: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg12: memref<{hidden_dim}x{emb_dim}xbf16>,
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
            with open("/tmp/debug_o_ffn.mlir", "w") as f:
                f.write(combined)
            print("  PARSE ERROR: dumped to /tmp/debug_o_ffn.mlir")
            raise
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O GEMM + Res Add + FFN multi-launch")
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

    print(f"O+FFN Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, hidden={HIDDEN_DIM}")
    module = build_o_ffn_module(
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
            instance_name="o_ffn",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run ----
    np.random.seed(42)
    N_TOTAL = SEQ_LEN * EMB_DIM

    attn_out = np.random.uniform(-1, 1, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    wo = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    x_residual = np.random.uniform(-1, 1, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    ffn_norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    w_gate = np.random.uniform(-0.1, 0.1, (EMB_DIM, HIDDEN_DIM)).astype(bfloat16)
    w_up = np.random.uniform(-0.1, 0.1, (EMB_DIM, HIDDEN_DIM)).astype(bfloat16)
    w_down = np.random.uniform(-0.1, 0.1, (HIDDEN_DIM, EMB_DIM)).astype(bfloat16)

    # CPU reference
    print("Computing CPU reference...")
    proj_ref = (attn_out.astype(np.float32) @ wo.astype(np.float32)).astype(bfloat16)
    res1_ref = (proj_ref.astype(np.float32) + x_residual.astype(np.float32)).astype(
        bfloat16
    )

    # RMSNorm
    r1 = res1_ref.astype(np.float32)
    rms = np.sqrt(np.mean(r1**2, axis=-1, keepdims=True) + 1e-5)
    normed2_ref = (r1 / rms * ffn_norm_w.astype(np.float32)).astype(bfloat16)

    gate_ref = (normed2_ref.astype(np.float32) @ w_gate.astype(np.float32)).astype(
        bfloat16
    )
    up_ref = (normed2_ref.astype(np.float32) @ w_up.astype(np.float32)).astype(bfloat16)

    # SiLU(gate) * up
    g = gate_ref.astype(np.float32)
    swiglu_ref = (g / (1 + np.exp(-g)) * up_ref.astype(np.float32)).astype(bfloat16)

    down_ref = (swiglu_ref.astype(np.float32) @ w_down.astype(np.float32)).astype(
        bfloat16
    )
    output_ref = (down_ref.astype(np.float32) + res1_ref.astype(np.float32)).astype(
        bfloat16
    )

    # Buffers
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
        instance_name="o_ffn",
    )

    # 14 inputs (args 0-13) + 1 expected output (arg 14, 1D)
    exit(
        runner.run_test(
            module,
            inputs=[
                attn_out,
                wo,
                proj_buf,
                x_residual,
                res1_buf,
                ffn_norm_w,
                normed2_buf,
                w_gate,
                gate_buf,
                w_up,
                up_buf,
                swiglu_buf,
                w_down,
                down_buf,
            ],
            expected_outputs=[output_ref.flatten()],
            rtol=0.5,
            atol=10.0,
            min_correlation=0.99,
        )
    )
