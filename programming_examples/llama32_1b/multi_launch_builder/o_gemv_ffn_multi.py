#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O GEMV + FFN — 8-launch multi-launch ELF for decode.

Merges the entire post-attention + FFN pipeline into a single ELF:
  L1: O GEMV       [8,1]  wo x attn_out -> proj          (M=2048, K=2048)
  L2: Eltwise Add  [8,1]  proj + x_residual -> res1      (N=2048)
  L3: RMSNorm      [1,1]  res1 x ffn_norm_w -> normed2   (M=1, N=2048)
  L4: Gate GEMV    [8,1]  wgate x normed2 -> gate         (M=8192, K=2048)
  L5: Up GEMV      [8,1]  wup x normed2 -> up             (M=8192, K=2048)
  L6: SiLU x mul   [8,1]  SiLU(gate) x up -> swiglu      (N=8192)
  L7: Down GEMV    [8,1]  wdown x swiglu -> down          (M=2048, K=8192)
  L8: Eltwise Add  [8,1]  down + res1 -> output           (N=2048)

func @o_gemv_ffn(
    %arg0:  memref<2048x2048xbf16>,   # wo
    %arg1:  memref<2048xbf16>,         # attn_out
    %arg2:  memref<2048xbf16>,         # proj
    %arg3:  memref<2048xbf16>,         # x_residual
    %arg4:  memref<2048xbf16>,         # res1
    %arg5:  memref<2048xbf16>,         # ffn_norm_w
    %arg6:  memref<2048xbf16>,         # normed2
    %arg7:  memref<8192x2048xbf16>,   # wgate
    %arg8:  memref<8192xbf16>,         # gate
    %arg9:  memref<8192x2048xbf16>,   # wup
    %arg10: memref<8192xbf16>,         # up
    %arg11: memref<8192xbf16>,         # swiglu
    %arg12: memref<2048x8192xbf16>,   # wdown
    %arg13: memref<2048xbf16>,         # down
    %arg14: memref<2048xbf16>,         # output
)
"""

import argparse
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "matrix_vector_multiplication", "bf16"
    ),
)

from llama32_1b.kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _rename_all,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
    _rename_all_with_externs,
)
from air.backend.xrt import XRTBackend

# ---------------------------------------------------------------------------
# 1D RMSNorm builder for decode (M=1)
# ---------------------------------------------------------------------------


def _build_rms_1d_ir(emb_dim, vector_size=16):
    """Build a 1D RMSNorm that accepts memref<Nxbf16> args.

    Standard weighted_rms_norm produces 2D func args (memref<1xNxbf16>).
    For the merged decode module, all activations are 1D. This builder
    creates a func with 1D args and uses memref.expand_shape inside the
    air.launch to convert 1D -> 2D before passing to the RMSNorm body.

    Returns the MLIR text (string) of the wrapped module.
    """
    from air.ir import (
        Context,
        Module,
        MemRefType,
        VectorType,
        IntegerAttr,
        AffineMap,
        AffineMapAttr,
        F32Type,
    )
    from air.dialects.air import (
        module_builder,
        MemorySpace,
        launch,
        segment,
        herd,
        dma_memcpy_nd,
    )
    from air.dialects import arith, math as math_dialect
    from air.dialects.memref import (
        AllocOp,
        DeallocOp,
        subview,
        expand_shape as memref_expand_shape,
    )
    from air.dialects.vector import (
        transfer_read,
        transfer_write,
        BroadcastOp,
        reduction as vector_reduction,
    )
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_, yield_
    from air.backend.xrt_runner import type_mapper

    n = emb_dim

    @module_builder
    def _build():
        from air.dialects.air import T

        xrt_dtype = type_mapper(bfloat16)
        N = n
        EPS = 1e-5

        vecTy_g = VectorType.get([vector_size], xrt_dtype)
        identity_map_g = AffineMapAttr.get(AffineMap.get_identity(1))

        # L3 types: 1D for func args, 2D for internal RMSNorm
        l3_1d_ty = MemRefType.get([N], xrt_dtype)
        l3_2d_ty = MemRefType.get([1, N], xrt_dtype)
        l3_weight_ty = MemRefType.get([N], xrt_dtype)

        # L1 types
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_row_ty = MemRefType.get([N], xrt_dtype, memory_space=l1_space)
        l1_vec_ty = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_space)

        @FuncOp.from_py_func(l3_1d_ty, l3_weight_ty, l3_1d_ty)
        def weighted_rms_norm_1d(arg0, arg1, arg2):
            @launch(operands=[arg0, arg1, arg2])
            def rms_launch(l_in, l_weight, l_out):
                # expand_shape: 1D memref<N> -> 2D memref<1xN>
                in_2d = memref_expand_shape(l3_2d_ty, l_in, [[0, 1]], [], [1, n])
                out_2d = memref_expand_shape(l3_2d_ty, l_out, [[0, 1]], [], [1, n])

                @segment(name="rms_seg", operands=[in_2d, l_weight, out_2d])
                def rms_seg(s_in, s_weight, s_out):
                    @herd(
                        name="herd_0",
                        sizes=[1, 1],
                        operands=[s_in, s_weight, s_out],
                    )
                    def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
                        l1_row = AllocOp(l1_row_ty, [], [])
                        l1_out = AllocOp(l1_row_ty, [], [])
                        l1_weight = AllocOp(l1_row_ty, [], [])
                        l1_acc = AllocOp(l1_vec_ty, [], [])

                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        n_f = arith.ConstantOp(xrt_dtype, float(N))
                        eps_f = arith.ConstantOp(xrt_dtype, EPS)

                        v_zero = BroadcastOp(vecTy_g, cst0)

                        # DMA weight to L1
                        dma_memcpy_nd(l1_weight, l3_weight)

                        # M=1: single row, no loop needed
                        # DMA: load row 0
                        dma_memcpy_nd(
                            l1_row,
                            l3_in,
                            src_offsets=[0, 0],
                            src_sizes=[1, N],
                            src_strides=[N, 1],
                        )

                        # Step 1: sum of x^2
                        transfer_write(
                            None,
                            v_zero,
                            l1_acc,
                            [c0],
                            identity_map_g,
                            [True],
                        )
                        for j in for_(0, N, vector_size):
                            sub_row = subview(l1_row.result, [j], [vector_size], [1])
                            sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vecTy_g,
                                sub_row,
                                [c0],
                                identity_map_g,
                                cst0,
                                [True],
                            )
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(
                                None,
                                v_sq,
                                sub_tmp,
                                [c0],
                                identity_map_g,
                                [True],
                            )
                            v_sq_rd = transfer_read(
                                vecTy_g,
                                sub_tmp,
                                [c0],
                                identity_map_g,
                                cst0,
                                [True],
                            )
                            v_acc = transfer_read(
                                vecTy_g,
                                l1_acc,
                                [c0],
                                identity_map_g,
                                cst0,
                                [True],
                            )
                            v_sum = arith.addf(v_acc, v_sq_rd)
                            transfer_write(
                                None,
                                v_sum,
                                l1_acc,
                                [c0],
                                identity_map_g,
                                [True],
                            )
                            yield_([])

                        # Horizontal reduce
                        v_final = transfer_read(
                            vecTy_g,
                            l1_acc,
                            [c0],
                            identity_map_g,
                            cst0,
                            [True],
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
                        v_rstd = BroadcastOp(vecTy_g, rstd)
                        for j in for_(0, N, vector_size):
                            sub_row = subview(l1_row.result, [j], [vector_size], [1])
                            sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vecTy_g,
                                sub_row,
                                [c0],
                                identity_map_g,
                                cst0,
                                [True],
                            )
                            v_w = transfer_read(
                                vecTy_g,
                                sub_w,
                                [c0],
                                identity_map_g,
                                cst0,
                                [True],
                            )
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(
                                None,
                                v_weighted,
                                sub_out,
                                [c0],
                                identity_map_g,
                                [True],
                            )
                            yield_([])

                        # DMA: write result row
                        dma_memcpy_nd(
                            l3_out,
                            l1_out,
                            dst_offsets=[0, 0],
                            dst_sizes=[1, N],
                            dst_strides=[N, 1],
                        )

                        DeallocOp(l1_row)
                        DeallocOp(l1_out)
                        DeallocOp(l1_weight)
                        DeallocOp(l1_acc)

    return str(_build())


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_o_gemv_ffn_module(
    emb_dim=2048,
    hidden_dim=8192,
    tile_m=8,
    m_input=4,
    down_tile_m=2,
    down_m_input=1,
    herd_m=8,
):
    """Build 8-launch O GEMV + FFN decode pipeline in one ELF.

    Combines: O GEMV + Add + RMSNorm + Gate GEMV + Up GEMV + SiLU*mul
              + Down GEMV + Add

    K=2048 GEMVs use tile_m=8, m_input=4 (original optimal params).
    K=8192 Down GEMV uses tile_m=2, m_input=1 (smaller tiles for large K).
    The external func type mismatch is resolved by renaming the Down GEMV's
    @matvec to @dg_matvec_vectorized_bf16_bf16 with separate link_with.
    """
    from matvec import build_module as build_gemv
    from eltwise_add.eltwise_add import build_module as build_add
    from llama32_1b.kernel_builder.ffn_swiglu.silu_and_mul import (
        build_module as build_silu,
    )

    # ------- L1: O GEMV (M=2048, K=2048) -------
    print("  [1/8] O GEMV...")
    o_gemv_ir = str(
        build_gemv(emb_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    # ------- L2: Eltwise Add (N=2048, herd=[8,1]) -------
    print("  [2/8] Eltwise Add (post-attn residual)...")
    add1_ir = _wrap_ir_in_launch(
        str(
            build_add(
                emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1
            )
        )
    )

    # ------- L3: RMSNorm (M=1, N=2048, herd=[1,1]) — custom 1D wrapper -------
    print("  [3/8] RMSNorm (1D decode)...")
    rms_ir = _build_rms_1d_ir(emb_dim, vector_size=16)

    # ------- L4: Gate GEMV (M=8192, K=2048) -------
    print("  [4/8] Gate GEMV...")
    gate_ir = str(
        build_gemv(hidden_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    # ------- L5: Up GEMV (M=8192, K=2048) -------
    print("  [5/8] Up GEMV...")
    up_ir = str(
        build_gemv(hidden_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    # ------- L6: SiLU x mul (N=8192, herd=[8,1]) -------
    print("  [6/8] SiLU x mul...")
    silu_ir = _wrap_ir_in_launch(
        str(build_silu(hidden_dim, hidden_dim // 8, bfloat16, herd_x=8, herd_y=1))
    )

    # ------- L7: Down GEMV (M=2048, K=8192) — smaller tiles, renamed extern func -------
    print("  [7/8] Down GEMV...")
    down_ir = str(
        build_gemv(
            emb_dim, hidden_dim, down_tile_m, down_m_input, herd_m, bfloat16, bfloat16
        )
    )

    # ------- L8: Eltwise Add (N=2048, herd=[8,1]) -------
    print("  [8/8] Eltwise Add (FFN residual)...")
    add2_ir = _wrap_ir_in_launch(
        str(
            build_add(
                emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1
            )
        )
    )

    # -----------------------------------------------------------------------
    # Stitch all 8 launches into a single func
    # -----------------------------------------------------------------------
    # Arg mapping: each sub-kernel has 3 func args (0, 1, 2).
    # Map to combined func args (0..14).
    stitch_specs = [
        (o_gemv_ir, "og", {0: 0, 1: 1, 2: 2}),  # wo, attn_out, proj
        (add1_ir, "a1", {0: 2, 1: 3, 2: 4}),  # proj, x_residual, res1
        (rms_ir, "rm", {0: 4, 1: 5, 2: 6}),  # res1, ffn_norm_w, normed2
        (gate_ir, "gg", {0: 7, 1: 6, 2: 8}),  # wgate, normed2, gate
        (up_ir, "ug", {0: 9, 1: 6, 2: 10}),  # wup, normed2, up
        (silu_ir, "sw", {0: 8, 1: 10, 2: 11}),  # gate, up, swiglu
        (down_ir, "dg", {0: 12, 1: 11, 2: 13}),  # wdown, swiglu, down
        (add2_ir, "a2", {0: 13, 1: 4, 2: 14}),  # down, res1, output
    ]

    # Down GEMV (K=8192) has different @matvec signature than K=2048 GEMVs.
    # Solution: rename Down GEMV's external functions and link with a separate .o
    # compiled with -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16
    _EXTERN_K2048 = {
        "@matvec_vectorized_bf16_bf16",
        "@linalg_fill_bf16",
        "@silu_and_mul_bf16",
    }
    # Down GEMV: matvec/linalg_fill NOT preserved → get renamed with "dg" prefix
    _EXTERN_DOWN = {"@silu_and_mul_bf16"}

    bodies, maps_all = [], []
    for ir, prefix, arg_map in stitch_specs:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        externs = _EXTERN_DOWN if prefix == "dg" else _EXTERN_K2048
        body = _rename_all_with_externs(body, prefix, externs)
        maps = [_rename_all_with_externs(m, prefix, externs) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        # Down GEMV: also change link_with in the herd body
        if prefix == "dg":
            body = body.replace('link_with = "mv.o"', 'link_with = "mv_k8192.o"')
        bodies.append(body)
        maps_all.extend(maps)

    # Collect private func declarations
    k2048_privates = _extract_private_funcs(o_gemv_ir)
    silu_privates = _extract_private_funcs(silu_ir)

    # Down GEMV: rename private declarations AND change link_with to "mv_k8192.o"
    down_privates = _extract_private_funcs(down_ir)
    down_privates_renamed = []
    for p in down_privates:
        p_renamed = _rename_all_with_externs(p, "dg", _EXTERN_DOWN)
        # Change link_with from "mv.o" to "mv_k8192.o"
        p_renamed = p_renamed.replace('link_with = "mv.o"', 'link_with = "mv_k8192.o"')
        down_privates_renamed.append(p_renamed.strip())

    seen_funcs = set()
    all_privates = []
    for p in k2048_privates + down_privates_renamed + silu_privates:
        fname = re.search(r"@(\w+)", p)
        if fname and fname.group(1) not in seen_funcs:
            seen_funcs.add(fname.group(1))
            all_privates.append(p.strip())

    combined = "\n".join(maps_all) + f"""
module {{
  {chr(10).join('  ' + p for p in all_privates)}
  func.func @o_gemv_ffn(
    %arg0: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<{emb_dim}xbf16>,
    %arg7: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg8: memref<{hidden_dim}xbf16>,
    %arg9: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg10: memref<{hidden_dim}xbf16>,
    %arg11: memref<{hidden_dim}xbf16>,
    %arg12: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg13: memref<{emb_dim}xbf16>,
    %arg14: memref<{emb_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, 15 args, 8 launches")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def o_gemv_ffn_reference(
    wo, attn_out, x_residual, ffn_norm_w, wgate, wup, wdown, eps=1e-5
):
    """CPU F32 reference for the full O GEMV + FFN decode pipeline.

    All vectors are 1D (decode: single token).

    Returns:
        output: (emb_dim,) = res1 + down_proj(SwiGLU(gate, up))
        where res1 = proj + x_residual
    """
    # O projection
    proj = wo.astype(np.float32) @ attn_out.astype(np.float32)

    # Residual add
    res1 = proj + x_residual.astype(np.float32)

    # RMSNorm
    w_f32 = ffn_norm_w.astype(np.float32)
    rms = np.sqrt(np.mean(res1 * res1) + eps)
    normed2 = (res1 / rms) * w_f32

    # Gate + Up
    gate = wgate.astype(np.float32) @ normed2
    up = wup.astype(np.float32) @ normed2

    # SiLU x mul
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    swiglu = (gate * sigmoid) * up

    # Down projection
    down = wdown.astype(np.float32) @ swiglu

    # Final residual add
    output = res1 + down
    return output.astype(bfloat16)


# ---------------------------------------------------------------------------
# Main (standalone test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="O GEMV + FFN 8-launch multi-launch decode test"
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

    print(
        f"O GEMV + FFN Multi-Launch (decode): emb_dim={emb_dim}, "
        f"hidden_dim={hidden_dim}"
    )

    module = build_o_gemv_ffn_module(emb_dim, hidden_dim)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong="all",
            runtime_loop_tiling_sizes=[16, 16],
            output_format=args.output_format,
            instance_name="o_gemv_ffn",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # Test data
    np.random.seed(42)
    wo = (np.random.randn(emb_dim, emb_dim) * 0.02).astype(bfloat16)
    attn_out = np.random.randn(emb_dim).astype(bfloat16)
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = np.random.randn(emb_dim).astype(bfloat16)
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    ffn_norm_w = (np.random.randn(emb_dim) * 0.1 + 1.0).astype(bfloat16)
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    wgate = (np.random.randn(hidden_dim, emb_dim) * 0.02).astype(bfloat16)
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    wup = (np.random.randn(hidden_dim, emb_dim) * 0.02).astype(bfloat16)
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    wdown = (np.random.randn(emb_dim, hidden_dim) * 0.01).astype(bfloat16)
    down_buf = np.zeros(emb_dim, dtype=bfloat16)

    # CPU reference
    output_ref = o_gemv_ffn_reference(
        wo, attn_out, x_residual, ffn_norm_w, wgate, wup, wdown
    )

    # Run on NPU
    from air.backend.xrt_runner import XRTRunner

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong="all",
        output_format="elf",
        instance_name="o_gemv_ffn",
        runtime_loop_tiling_sizes=[16, 16],
        use_lock_race_condition_fix=False,
    )
    sys.exit(
        runner.run_test(
            module,
            inputs=[
                wo,  # arg0
                attn_out,  # arg1
                proj_buf,  # arg2
                x_residual,  # arg3
                res1_buf,  # arg4
                ffn_norm_w,  # arg5
                normed2_buf,  # arg6
                wgate,  # arg7
                gate_buf,  # arg8
                wup,  # arg9
                up_buf,  # arg10
                swiglu_buf,  # arg11
                wdown,  # arg12
                down_buf,  # arg13
            ],
            expected_outputs=[output_ref],
            rtol=0.5,
            atol=10.0,
            min_correlation=0.99,
        )
    )
