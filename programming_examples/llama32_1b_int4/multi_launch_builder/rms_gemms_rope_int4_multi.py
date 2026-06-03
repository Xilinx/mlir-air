# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + int4-AWQ QKV GEMMs + RoPE Q+K — 6-launch multi-launch ELF.

int4-AWQ prefill sibling of rms_gemms_rope_multi.py. Same 6-launch shape,
same combined func arg slots (13 args), same RMSNorm + RoPE sub-modules.
The 3 Q/K/V GEMMs swap from bf16 _build_gemm_module to the int4-AWQ
matmul_int4_packed.build_module (mmul + dequant-to-L1, f32 accumulator).

Per-layer weight slots arg3 / arg5 / arg7 become 3D uint8 packed BOs
(Q+S+Z per tile) instead of 2D bf16 weight matrices; everything else
unchanged.

Usage:
    python3 rms_gemms_rope_int4_multi.py -p
    python3 rms_gemms_rope_int4_multi.py --compile-mode compile-only
    python3 rms_gemms_rope_int4_multi.py            # compile + run + validate
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Shared scaffolding (llama_kernel_builder.stitching, ffn_swiglu builder,
# weighted_rms_norm, rope_lut) currently lives under the bf16 example.
# Cross-link until those move to a shared location.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "llama32_1b"),
)
# int4 GEMM builder + packer live in the standalone matrix_multiplication example.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "matrix_multiplication",
        "int4_awq",
    ),
)

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from llama_kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _rename_all,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
    _rename_all_with_externs,
)

range_ = for_


# ---------------------------------------------------------------------------
# 2D RoPE launch builder — verbatim from the bf16 sibling so the stitched
# arg-mapping stays identical.
# ---------------------------------------------------------------------------


@module_builder
def _build_rope_2d(outer_rows, outer_cols, embed_dim, np_dtype, herd_x):
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    total = outer_rows * outer_cols
    rope_rows = total // embed_dim
    herd_y = 1
    total_tiles = herd_x * herd_y

    assert embed_dim % 16 == 0, "embed_dim must be divisible by 16"
    assert total % embed_dim == 0
    assert rope_rows % total_tiles == 0

    l3_2d_ty = MemRefType.get([outer_rows, outer_cols], xrt_dtype)
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

    rows_per_tile = rope_rows // total_tiles

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

    @FuncOp.from_py_func(l3_2d_ty, l3_1d_ty, l3_2d_ty)
    def rope_2d(arg0_2d, arg1_lut, arg2_2d):
        @launch(operands=[arg0_2d, arg1_lut, arg2_2d])
        def rope_launch(l_in_2d, l_lut, l_out_2d):
            in_flat = memref_collapse_shape(l3_1d_ty, l_in_2d, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out_2d, [[0, 1]])

            @segment(name="rope_seg", operands=[in_flat, l_lut, out_flat])
            def rope_seg(s_in, s_lut, s_out):
                @herd(
                    name="rope_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_lut, s_out],
                )
                def rope_body(_tx, _ty, _sx, _sy, h_in, h_lut, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_lut = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])

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

                        CallOp(rope_func, [l1_in, l1_lut, l1_out, dim_i32])

                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row_offset],
                            dst_sizes=[embed_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out)

                rope_body.attributes["link_with"] = StringAttr.get("rope.o")


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_rms_gemms_rope_int4_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    # int4 GEMM tile config — same for all three Q/K/V (proven Q-proj config).
    gs=128,
    tile_m=16,
    tile_n=16,
    tile_k_l1=128,
    tile_k_l2=None,  # defaults to k = emb_dim (single segment-K iter)
    herd_m=8,
    herd_n=4,
    # RoPE config
    rope_herd_x=8,
    print_kernels=False,
):
    """Build 6-launch module: RMSNorm + int4 Q/K/V GEMMs + RoPE Q + RoPE K.

    Combined func @rms_gemms_rope_int4 with 13 args (mirrors bf16 sibling,
    with arg3/5/7 swapped to packed-uint8 weight BOs):
        arg0:  x_in          (seq_len, emb_dim)              bf16
        arg1:  norm_w        (emb_dim,)                       bf16
        arg2:  normed        (seq_len, emb_dim)               bf16
        arg3:  wq_packed     (Nq_div, K_div, tile_bytes)      uint8
        arg4:  q             (seq_len, emb_dim)               bf16
        arg5:  wk_packed     (Nkv_div, K_div, tile_bytes)     uint8
        arg6:  k             (seq_len, kv_dim)                bf16
        arg7:  wv_packed     (Nkv_div, K_div, tile_bytes)     uint8
        arg8:  v             (seq_len, kv_dim)                bf16
        arg9:  lut_q         (n_heads*seq_len*head_dim,)      bf16
        arg10: lut_k         (n_kv_heads*seq_len*head_dim,)   bf16
        arg11: q_roped       (seq_len, emb_dim)               bf16
        arg12: k_roped       (seq_len, kv_dim)                bf16
    """
    from llama32_1b_int4.gemm_builder import _build_int4_gemm_module as build_int4_gemm
    from matmul_int4_packed import packed_tile_bytes
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    if tile_k_l2 is None:
        tile_k_l2 = emb_dim  # single segment-K iter; matches Q-proj config

    q_total = seq_len * emb_dim
    k_total = seq_len * kv_dim

    # Packed-BO 3D shape per Q/K/V tile dimension.
    _, _, _, tile_bytes_q = packed_tile_bytes(tile_n, tile_k_l1, gs)
    _, _, _, tile_bytes_kv = packed_tile_bytes(tile_n, tile_k_l1, gs)
    assert tile_bytes_q == tile_bytes_kv  # same tile_n, tile_k_l1, gs
    nq_div = emb_dim // tile_n
    nkv_div = kv_dim // tile_n
    k_div = emb_dim // tile_k_l1

    # ---- Build sub-kernels ----

    print("  [1/6] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    print(f"  [2/6] Q int4 GEMM (M={seq_len}, K={emb_dim}, N={emb_dim})...")
    q_ir = str(
        build_int4_gemm(
            seq_len,
            emb_dim,
            emb_dim,
            gs,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    print(f"  [3/6] K int4 GEMM (M={seq_len}, K={emb_dim}, N={kv_dim})...")
    k_ir = str(
        build_int4_gemm(
            seq_len,
            emb_dim,
            kv_dim,
            gs,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    print(f"  [4/6] V int4 GEMM (M={seq_len}, K={emb_dim}, N={kv_dim})...")
    v_ir = str(
        build_int4_gemm(
            seq_len,
            emb_dim,
            kv_dim,
            gs,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    print(
        f"  [5/6] RoPE Q (outer={seq_len}x{emb_dim}, embed_dim={head_dim}, "
        f"herd_x={rope_herd_x})..."
    )
    rope_q_ir = str(_build_rope_2d(seq_len, emb_dim, head_dim, bfloat16, rope_herd_x))

    print(
        f"  [6/6] RoPE K (outer={seq_len}x{kv_dim}, embed_dim={head_dim}, "
        f"herd_x={rope_herd_x})..."
    )
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q int4 GEMM", q_ir),
            ("K int4 GEMM", k_ir),
            ("V int4 GEMM", v_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Same arg mapping as bf16 sibling (int4 GEMM keeps (A, B, C) arg order):
    #   RMSNorm:    {0->0, 1->1, 2->2}    (x_in, norm_w, normed)
    #   Q int4:     {0->2, 1->3, 2->4}    (normed, wq_packed, q)
    #   K int4:     {0->2, 1->5, 2->6}    (normed, wk_packed, k)
    #   V int4:     {0->2, 1->7, 2->8}    (normed, wv_packed, v)
    #   RoPE Q:     {0->4, 1->9, 2->11}   (q[2D], lut_q[1D], q_roped[2D])
    #   RoPE K:     {0->6, 1->10, 2->12}  (k[2D], lut_k[1D], k_roped[2D])

    _EXTERN_FUNCS = {
        "@matmul_int4_bf16_packed_f32",
        "@zero_vectorized_f32_mn",
        "@f32_to_bf16_mn",
        "@zero_vectorized_bf16",
        "@rope",
    }

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "r", {0: 0, 1: 1, 2: 2}),
        (q_ir, "q", {0: 2, 1: 3, 2: 4}),
        (k_ir, "k", {0: 2, 1: 5, 2: 6}),
        (v_ir, "v", {0: 2, 1: 7, 2: 8}),
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

    # Private func decls: RMSNorm's zero kernel, RoPE's @rope decl, and the
    # int4 GEMM's three private funcs (matmul/zero_f32/f32_to_bf16).
    all_privates = set()
    for ir in [rms_ir, rope_q_ir, q_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(sorted(all_privates))

    combined = (
        "\n".join(maps_all)
        + f"""
module {{
  {privates_str}
  func.func @rms_gemms_rope_int4(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{nq_div}x{k_div}x{tile_bytes_q}xi8>,
    %arg4: memref<{seq_len}x{emb_dim}xbf16>,
    %arg5: memref<{nkv_div}x{k_div}x{tile_bytes_kv}xi8>,
    %arg6: memref<{seq_len}x{kv_dim}xbf16>,
    %arg7: memref<{nkv_div}x{k_div}x{tile_bytes_kv}xi8>,
    %arg8: memref<{seq_len}x{kv_dim}xbf16>,
    %arg9: memref<{q_total}xbf16>,
    %arg10: memref<{k_total}xbf16>,
    %arg11: memref<{seq_len}x{emb_dim}xbf16>,
    %arg12: memref<{seq_len}x{kv_dim}xbf16>
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
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def _rms_norm_ref(x, weight, eps=1e-5):
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_2d, lut_2d):
    """Half-split RoPE matching the rope_halfsplit.cc kernel (= rope.o)
    and HuggingFace LlamaRotaryEmbedding:
      out[i]      = x[i]*cos[i] - x[i+half]*sin[i]
      out[i+half] = x[i+half]*cos[i] + x[i]*sin[i]
    LUT layout is concatenated [cos_0..cos_{half-1}, sin_0..sin_{half-1}]."""
    x = x_2d.astype(np.float32)
    lut = lut_2d.astype(np.float32)
    half = x.shape[1] // 2
    cos = lut[:, :half]
    sin = lut[:, half:]
    x1 = x[:, :half]
    x2 = x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * cos - x2 * sin
    out[:, half:] = x2 * cos + x1 * sin
    return out.astype(bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEQ_LEN = 2048
    EMB_DIM = 2048
    KV_DIM = 512
    N_HEADS = 32
    N_KV_HEADS = 8
    HEAD_DIM = 64
    GS = 128

    parser = argparse.ArgumentParser(
        description="RMSNorm + int4-AWQ QKV GEMMs + RoPE QK multi-launch test"
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
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    args = parser.parse_args()

    print(
        f"RMS+int4 QKV+RoPE Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, "
        f"kv={KV_DIM}, heads={N_HEADS}/{N_KV_HEADS}, dk={HEAD_DIM}, gs={GS}"
    )

    module = build_rms_gemms_rope_int4_module(
        seq_len=SEQ_LEN,
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        gs=GS,
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
            instance_name="rms_gemms_rope_int4",
            stack_size=16384,
        )
        backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run ----
    from matmul_int4_packed import pack_inputs, cpu_reference as int4_matmul_ref

    np.random.seed(42)

    # bf16 inputs
    x_in = np.random.uniform(-1.0, 1.0, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)

    # int4-AWQ weights for Q / K / V — random nibbles, plausible scales/zeros.
    def _random_int4_weight(N, K, gs):
        W_q_unp = np.random.randint(0, 16, size=(N, K), dtype=np.uint8)
        W_q = (W_q_unp[:, 0::2] | (W_q_unp[:, 1::2] << 4)).astype(np.uint8)
        n_groups = K // gs
        W_s = np.random.uniform(0.005, 0.02, size=(n_groups, N)).astype(bfloat16)
        W_z = np.random.randint(7, 9, size=(n_groups, N), dtype=np.uint8)
        return W_q, W_s, W_z

    Wq_q, Wq_s, Wq_z = _random_int4_weight(EMB_DIM, EMB_DIM, GS)
    Wk_q, Wk_s, Wk_z = _random_int4_weight(KV_DIM, EMB_DIM, GS)
    Wv_q, Wv_s, Wv_z = _random_int4_weight(KV_DIM, EMB_DIM, GS)

    # Pack per-(n_outer, k_outer) tiles, NPU-ready layout.
    wq_packed = pack_inputs(Wq_q, Wq_s, Wq_z, SEQ_LEN, EMB_DIM, EMB_DIM, GS, 16, 128)
    wk_packed = pack_inputs(Wk_q, Wk_s, Wk_z, SEQ_LEN, EMB_DIM, KV_DIM, GS, 16, 128)
    wv_packed = pack_inputs(Wv_q, Wv_s, Wv_z, SEQ_LEN, EMB_DIM, KV_DIM, GS, 16, 128)

    # RoPE LUTs. Use the half-split layout (concatenated [cos..., sin...])
    # produced by `llama32_1b_weights.generate_rope_lut` — same producer
    # the verifier / inference paths use, and the layout `rope.o`
    # (rope_halfsplit.cc) expects. Per-head expansion repeats the
    # (seq_len, head_dim) base LUT N_HEADS / N_KV_HEADS times.
    from llama32_1b_weights import LlamaConfig, generate_rope_lut

    base_lut = generate_rope_lut(LlamaConfig(), seq_len=SEQ_LEN)
    lut_q = np.repeat(base_lut, N_HEADS, axis=0)
    lut_k = np.repeat(base_lut, N_KV_HEADS, axis=0)

    # CPU reference: bf16 RMSNorm then int4 dequant-matmul then bf16 RoPE.
    print("Computing CPU reference...")
    normed_ref = _rms_norm_ref(x_in, norm_w)
    q_ref = int4_matmul_ref(Wq_q, Wq_s, Wq_z, normed_ref)
    k_ref = int4_matmul_ref(Wk_q, Wk_s, Wk_z, normed_ref)
    v_ref = int4_matmul_ref(Wv_q, Wv_s, Wv_z, normed_ref)

    q_flat = q_ref.reshape(SEQ_LEN, N_HEADS, HEAD_DIM).reshape(
        SEQ_LEN * N_HEADS, HEAD_DIM
    )
    q_roped_ref = _rope_ref(q_flat, lut_q.reshape(-1, HEAD_DIM)).reshape(
        SEQ_LEN, EMB_DIM
    )
    k_flat = k_ref.reshape(SEQ_LEN, N_KV_HEADS, HEAD_DIM).reshape(
        SEQ_LEN * N_KV_HEADS, HEAD_DIM
    )
    k_roped_ref = _rope_ref(k_flat, lut_k.reshape(-1, HEAD_DIM)).reshape(
        SEQ_LEN, KV_DIM
    )

    normed_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    q_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    v_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    q_roped_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_roped_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_gemms_rope_int4",
        stack_size=16384,
    )

    exit(
        runner.run_test(
            module,
            inputs=[
                x_in,
                norm_w,
                normed_buf,
                wq_packed,
                q_buf,
                wk_packed,
                k_buf,
                wv_packed,
                v_buf,
                lut_q.flatten(),
                lut_k.flatten(),
            ],
            expected_outputs=[
                q_roped_ref,
                k_roped_ref,
            ],
            rtol=0.2,
            atol=0.5,
            min_correlation=0.99,
            max_mismatch_percentage=0.5,
        )
    )
