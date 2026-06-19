# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMMs + RoPE Q+K — 6-launch multi-launch ELF.

Merges rms_attn_gemms (4 launches) + rope_qk (2 launches) into a single
AIR function with 6 sequential air.launch operations:
  1. RMSNorm      [8,1]   x_in x norm_w -> normed
  2. Q GEMM       [8,4]   normed x wq -> q
  3. K GEMM       [8,4]   normed x wk -> k
  4. V GEMM       [8,4]   normed x wv -> v
  5. RoPE Q       [8,1]   q(2D->1D) x lut_q -> q_roped(1D->2D)
  6. RoPE K       [8,1]   k(2D->1D) x lut_k -> k_roped(1D->2D)

13 func args (6 launches). Q/K GEMM outputs are 2D memrefs shared with
RoPE launches that use memref.collapse_shape inside the launch body.

Usage:
    python3 rms_gemms_rope_multi.py -p           # print combined MLIR
    python3 rms_gemms_rope_multi.py              # compile + run + validate
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
# 2D RoPE launch builder (accepts 2D in/out, collapses to 1D inside launch)
# ---------------------------------------------------------------------------


@module_builder
def _build_rope_2d(outer_rows, outer_cols, embed_dim, np_dtype, herd_x):
    """Build a RoPE launch with 2D in/out args (for GEMM type compatibility).

    The outer 2D shape (outer_rows, outer_cols) matches the GEMM output type.
    Inside the launch, collapse_shape flattens to 1D, and the RoPE herd
    processes the flat array with embed_dim-wide rows.

    Func signature:
      (in_2d: [outer_rows, outer_cols], lut_1d: [total], out_2d: [outer_rows, outer_cols])

    Args:
        outer_rows: 2D func arg rows (e.g. seq_len=2048)
        outer_cols: 2D func arg cols (e.g. emb_dim=2048 or kv_dim=512)
        embed_dim:  RoPE column width per row (head_dim=64)
        herd_x:     Number of tiles for row-parallel
    """
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    total = outer_rows * outer_cols
    rope_rows = total // embed_dim  # actual RoPE rows (n_heads * seq_len)
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


def build_rms_gemms_rope_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    herd_m=8,
    herd_n=4,
    # RoPE config
    rope_herd_x=8,
    print_kernels=False,
):
    """Build 6-launch module: RMSNorm + Q/K/V GEMMs + RoPE Q + RoPE K.

    Returns:
        Module with func @rms_gemms_rope and 13 memref args:
            %arg0:  x_in        (seq_len, emb_dim)       input
            %arg1:  norm_w      (emb_dim,)               RMSNorm weight
            %arg2:  normed      (seq_len, emb_dim)       RMSNorm output
            %arg3:  wq          (emb_dim, emb_dim)       Q weight
            %arg4:  q           (seq_len, emb_dim)       Q GEMM output (2D)
            %arg5:  wk          (emb_dim, kv_dim)        K weight
            %arg6:  k           (seq_len, kv_dim)        K GEMM output (2D)
            %arg7:  wv          (emb_dim, kv_dim)        V weight
            %arg8:  v           (seq_len, kv_dim)        V GEMM output
            %arg9:  lut_q       (q_total,)               RoPE Q LUT (1D)
            %arg10: lut_k       (k_total,)               RoPE K LUT (1D)
            %arg11: q_roped     (seq_len, emb_dim)       RoPE Q output (2D)
            %arg12: k_roped     (seq_len, kv_dim)        RoPE K output (2D)
    """
    from block_builder.gemm_builder import _build_gemm_module, gemm_registry_config
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    # Per-GEMM config from the kernel_registry JSON (single source of truth): method
    # (fused-cast vs drain) AND all tiles are looked up per shape, never hardcoded.
    # Q (large) resolves to fused-cast (_m64), K/V (small) to drain (_m32); both
    # co-link in one ELF (distinct symbols + mm_*.o; each air.launch reconfigures
    # L1/L2 so launch buffers don't accumulate). Adapts automatically to other models.
    q_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")

    def _gemm_kw_and_tiles(spec):
        return (
            dict(spec["build_kwargs"]),
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
        )

    q_total = seq_len * emb_dim  # = n_heads * seq_len * head_dim
    k_total = seq_len * kv_dim  # = n_kv_heads * seq_len * head_dim

    # RoPE rows: the LUT has one row per (position, head) pair in seq-first order
    # Q: n_heads * seq_len rows of head_dim
    # K: n_kv_heads * seq_len rows of head_dim
    rope_q_rows = n_heads * seq_len  # 65536
    rope_k_rows = n_kv_heads * seq_len  # 16384

    # ---- Build sub-kernels ----

    # 1. RMSNorm (bare herd → wrap in launch+segment)
    print("  [1/6] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # 2-4. Q/K/V GEMMs — method + ALL tiles come from the registry spec per shape.
    _q_kw, _q_tm, _q_k2, _q_k1, _q_tn = _gemm_kw_and_tiles(q_spec)
    _k_kw, _k_tm, _k_k2, _k_k1, _k_tn = _gemm_kw_and_tiles(k_spec)
    _v_kw, _v_tm, _v_k2, _v_k1, _v_tn = _gemm_kw_and_tiles(v_spec)
    _qm = q_spec["method"]
    _km = k_spec["method"]
    _vm = v_spec["method"]
    print(f"  [2/6] Q GEMM ({_qm})...")
    q_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            _q_tm,
            _q_k2,
            _q_k1,
            _q_tn,
            herd_m,
            herd_n,
            **_q_kw,
        )
    )
    print(f"  [3/6] K GEMM ({_km})...")
    k_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            _k_tm,
            _k_k2,
            _k_k1,
            _k_tn,
            herd_m,
            herd_n,
            **_k_kw,
        )
    )
    print(f"  [4/6] V GEMM ({_vm})...")
    v_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            _v_tm,
            _v_k2,
            _v_k1,
            _v_tn,
            herd_m,
            herd_n,
            **_v_kw,
        )
    )

    # 5-6. RoPE Q/K (2D in/out with collapse_shape inside launch)
    # Outer 2D shape matches GEMM output type; inner processing uses head_dim
    print(
        f"  [5/6] RoPE Q (outer={seq_len}x{emb_dim}, embed_dim={head_dim}, herd_x={rope_herd_x})..."
    )
    rope_q_ir = str(_build_rope_2d(seq_len, emb_dim, head_dim, bfloat16, rope_herd_x))

    print(
        f"  [6/6] RoPE K (outer={seq_len}x{kv_dim}, embed_dim={head_dim}, herd_x={rope_herd_x})..."
    )
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q GEMM", q_ir),
            ("K GEMM", k_ir),
            ("V GEMM", v_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Arg mapping (combined func arg indices). Base bf16 args 0..12; when a GEMM is
    # fused-cast it gets ONE extra f32 C-scratch arg appended at 13+ and its arg_map
    # gains slot 2->scratch, 3->bf16-out (drain stays {0:in,1:w,2:out}).
    #   RMSNorm:  {0->0, 1->1, 2->2}       (x_in, norm_w, normed)
    #   Q GEMM:   normed=2, wq=3, q=4   (+ q_f32 scratch if fused)
    #   K GEMM:   normed=2, wk=5, k=6   (+ k_f32 scratch if fused)
    #   V GEMM:   normed=2, wv=7, v=8   (+ v_f32 scratch if fused)
    #   RoPE Q:   {0->4, 1->9, 2->11}      (q[2D], lut_q[1D], q_roped[2D])
    #   RoPE K:   {0->6, 1->10, 2->12}     (k[2D], lut_k[1D], k_roped[2D])

    # Combined extern_funcs: GEMM + RoPE + RMSNorm externals. Each fused/drain GEMM
    # uses suffixed mm.o symbols (_m64 / _m32) so both variants co-link in one ELF.
    _EXTERN_FUNCS = {"@matmul_bf16", "@zero_vectorized_bf16", "@rope"}
    for sp in (q_spec, k_spec, v_spec):
        sfx = sp["sym_suffix"]
        _EXTERN_FUNCS |= {
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    # Build the GEMM arg_maps + f32-scratch func args. f32-scratch args start at 13.
    _scratch_args = []  # MLIR type strings appended to the func signature
    _next_scratch = 13

    def _gemm_arg_map(spec, in_idx, w_idx, out_idx, out_rows, out_cols):
        nonlocal _next_scratch
        if spec["needs_f32_scratch"]:
            sc = _next_scratch
            _next_scratch += 1
            _scratch_args.append(f"    ,%arg{sc}: memref<{out_rows}x{out_cols}xf32>")
            # fused-cast: {0:in, 1:w, 2:Cf32-scratch, 3:bf16-out}
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        # drain: {0:in, 1:w, 2:bf16-out}
        return {0: in_idx, 1: w_idx, 2: out_idx}

    q_map = _gemm_arg_map(q_spec, 2, 3, 4, seq_len, emb_dim)
    k_map = _gemm_arg_map(k_spec, 2, 5, 6, seq_len, kv_dim)
    v_map = _gemm_arg_map(v_spec, 2, 7, 8, seq_len, kv_dim)

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "r", {0: 0, 1: 1, 2: 2}),
        (q_ir, "q", q_map),
        (k_ir, "k", k_map),
        (v_ir, "v", v_map),
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

    # Collect private func declarations (RoPE has @rope, RMSNorm has
    # @zero_vectorized_bf16). Each external GEMM variant adds its suffixed mm.o
    # symbols; q_ir covers _m64 (fused Q), k_ir covers _m32 (drain K/V).
    all_privates = set()
    for ir in [rms_ir, rope_q_ir, q_ir, k_ir, v_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(sorted(all_privates))

    _scratch_sig = ("\n" + "\n".join(_scratch_args)) if _scratch_args else ""

    # Assemble (13 base func args + N f32 scratch, 6+ launches)
    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @rms_gemms_rope(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg4: memref<{seq_len}x{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg6: memref<{seq_len}x{kv_dim}xbf16>,
    %arg7: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg8: memref<{seq_len}x{kv_dim}xbf16>,
    %arg9: memref<{q_total}xbf16>,
    %arg10: memref<{k_total}xbf16>,
    %arg11: memref<{seq_len}x{emb_dim}xbf16>,
    %arg12: memref<{seq_len}x{kv_dim}xbf16>{_scratch_sig}
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
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def _rms_norm_ref(x, weight, eps=1e-5):
    """CPU RMSNorm reference."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_2d, lut_2d):
    """CPU RoPE reference."""
    x = x_2d.astype(np.float32)
    lut = lut_2d.astype(np.float32)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * lut[:, 0::2] - x[:, 1::2] * lut[:, 1::2]
    out[:, 1::2] = x[:, 0::2] * lut[:, 1::2] + x[:, 1::2] * lut[:, 0::2]
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

    parser = argparse.ArgumentParser(
        description="RMSNorm + QKV GEMMs + RoPE QK multi-launch test"
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
        f"RMS+QKV+RoPE Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, "
        f"kv={KV_DIM}, heads={N_HEADS}/{N_KV_HEADS}, dk={HEAD_DIM}"
    )

    module = build_rms_gemms_rope_module(
        seq_len=SEQ_LEN,
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
            instance_name="rms_gemms_rope",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run: build test data, run, verify ----
    np.random.seed(42)

    # Inputs
    x_in = np.random.uniform(-1.0, 1.0, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    wq = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    wk = np.random.uniform(-0.1, 0.1, (EMB_DIM, KV_DIM)).astype(bfloat16)
    wv = np.random.uniform(-0.1, 0.1, (EMB_DIM, KV_DIM)).astype(bfloat16)

    # RoPE LUTs (seq-first: repeated per head)
    from rope_lut.rope_lut import generate_lut

    base_lut = generate_lut(SEQ_LEN, HEAD_DIM, bfloat16)  # (SEQ_LEN, HEAD_DIM)
    lut_q = np.repeat(base_lut, N_HEADS, axis=0)  # (N_HEADS*SEQ_LEN, HEAD_DIM)
    lut_k = np.repeat(base_lut, N_KV_HEADS, axis=0)  # (N_KV_HEADS*SEQ_LEN, HEAD_DIM)

    # CPU reference
    print("Computing CPU reference...")
    normed_ref = _rms_norm_ref(x_in, norm_w)
    q_ref = (normed_ref.astype(np.float32) @ wq.astype(np.float32)).astype(bfloat16)
    k_ref = (normed_ref.astype(np.float32) @ wk.astype(np.float32)).astype(bfloat16)
    v_ref = (normed_ref.astype(np.float32) @ wv.astype(np.float32)).astype(bfloat16)

    # Apply RoPE to Q and K in seq-first layout
    q_2d = q_ref.reshape(SEQ_LEN, N_HEADS, HEAD_DIM)  # (seq, heads, dk)
    q_flat = q_2d.reshape(SEQ_LEN * N_HEADS, HEAD_DIM)  # seq-first order
    q_roped_ref = _rope_ref(q_flat, lut_q.reshape(-1, HEAD_DIM))
    q_roped_ref = q_roped_ref.reshape(SEQ_LEN, EMB_DIM)  # back to (seq, emb)

    k_2d = k_ref.reshape(SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    k_flat = k_2d.reshape(SEQ_LEN * N_KV_HEADS, HEAD_DIM)
    k_roped_ref = _rope_ref(k_flat, lut_k.reshape(-1, HEAD_DIM))
    k_roped_ref = k_roped_ref.reshape(SEQ_LEN, KV_DIM)

    # Output buffers (zeroed)
    normed_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    q_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    v_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    q_roped_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_roped_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)

    # Func signature: 13 args
    # (x_in, norm_w, normed, wq, q, wk, k, wv, v, lut_q, lut_k, q_roped, k_roped)
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_gemms_rope",
    )

    # XRTRunner convention: inputs = first N func args, expected_outputs = last M args.
    # Func has 13 args total. Last 2 (arg11=q_roped, arg12=k_roped) are outputs.
    # First 11 (arg0-arg10) are inputs (including zeroed intermediate buffers).
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
                v_buf,  # arg8 (intermediate, zeroed)
                lut_q.flatten(),  # arg9
                lut_k.flatten(),  # arg10
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
