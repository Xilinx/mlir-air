# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMMs + per-head QK-norm + RoPE Q+K — 8-launch prefill ELF.

This is the Qwen3 generalization of `rms_gemms_rope_multi.build_rms_gemms_rope_module`.
Qwen3 applies a per-head RMSNorm (QK-norm) to Q and K AFTER the projection
GEMM and BEFORE RoPE. RoPE's nonlinearity-free rotation does NOT commute past
the (nonlinear) QK-norm, so the QK-norm must sit physically between the GEMM
and RoPE. We express it with the existing `weighted_rms_norm` row-wise RMSNorm
kernel: q[seq, n_heads*head_dim] viewed as [seq*n_heads, head_dim] rows gives a
row-wise RMSNorm over head_dim with weight q_norm (head_dim,) broadcast across
all rows — exactly per-head QK-norm. No new C kernel needed.

8 launches (vs the 6 of rms_gemms_rope):
  1. RMSNorm      x_in x norm_w -> normed
  2. Q GEMM       normed x wq -> q          (seq, q_dim)
  3. K GEMM       normed x wk -> k          (seq, kv_dim)
  4. V GEMM       normed x wv -> v          (seq, kv_dim)
  5. QK-norm Q    q  x q_norm -> q_n        (per-head RMSNorm, head_dim)  <-- NEW
  6. QK-norm K    k  x k_norm -> k_n        (per-head RMSNorm, head_dim)  <-- NEW
  7. RoPE Q       q_n(2D->1D) x lut_q -> q_roped(1D->2D)
  8. RoPE K       k_n(2D->1D) x lut_k -> k_roped(1D->2D)

The QK-norm slices reshape the 2D Q/K GEMM-output buffers (seq, *_dim) into
[seq*heads, head_dim] views via a collapse_shape -> expand_shape prelude and
route in/out operands onto those SSA values with `arg_aliases`, so no extra
func arg is needed for the reshaped view. QK-norm writes to dedicated output
buffers (q_n, k_n) that RoPE then consumes, keeping the data flow explicit
(not in-place).

The QK-norm RMSNorm kernel reads `weighted_rms_norm.EPS`; Qwen3 uses eps=1e-6,
so we temporarily override that module global during the QK-norm build (same
pattern the decode `rms_gemv_rope_multi.EPS` override uses).
"""

import numpy as np
from ml_dtypes import bfloat16


# ---------------------------------------------------------------------------
# Per-head RMSNorm (QK-norm) with 2D in/out args (collapse to 1D inside launch,
# process head_dim-wide rows). Modeled on rms_gemms_rope_multi._build_rope_2d so
# the L1 DMA reads a collapse_shape of a block argument — the allowed AIE
# dma_bd chain (subview/cast/collapse of a block arg). expand_shape on a func
# arg is REJECTED by the AIE backend, which is why we cannot just reshape the
# 2D GEMM-output buffer to [rows, head_dim] and feed weighted_rms_norm.
#
# Math mirrors weighted_rms_norm: sum(x^2) accumulated in f32, rstd = rsqrt(
# mean + eps) in f32, epilogue y = x * rstd * weight in bf16 vectors. eps is a
# build-time arg (Qwen3 = 1e-6).
# ---------------------------------------------------------------------------


from air.ir import (
    MemRefType, IntegerAttr, AffineMap, AffineExpr, AffineSymbolExpr,
    AffineConstantExpr, AffineMapAttr, VectorType, F32Type,
)
from air.dialects.air import (
    module_builder, launch, segment, herd, dma_memcpy_nd, MemorySpace, T,
)
from air.dialects.affine import apply as affine_apply
from air.dialects import arith, math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.vector import (
    transfer_read, transfer_write, BroadcastOp, reduction as vector_reduction,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_ as range_, yield_
from air.backend.xrt_runner import type_mapper


@module_builder
def _build_qknorm_2d(outer_rows, outer_cols, head_dim, np_dtype, eps, herd_x, vector_size=16):
    """Build a per-head RMSNorm launch with 2D in/out args.

    The outer 2D shape (outer_rows=seq_len, outer_cols=q_dim or kv_dim) matches
    the GEMM output type. Inside the launch the buffers are collapse_shape'd to
    1D and the herd processes total/head_dim rows of head_dim each, RMSNorm-ing
    each row with the shared weight (head_dim,).

    Func signature:
      (in_2d: [outer_rows, outer_cols], weight_1d: [head_dim], out_2d: [outer_rows, outer_cols])
    """
    xrt_dtype = type_mapper(np_dtype)
    total = outer_rows * outer_cols
    rope_rows = total // head_dim  # n_heads * seq_len
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert head_dim % vector_size == 0
    assert total % head_dim == 0
    assert rope_rows % total_tiles == 0
    rows_per_tile = rope_rows // total_tiles

    f32 = F32Type.get()
    vecTy = VectorType.get([vector_size], xrt_dtype)
    vecTyF32 = VectorType.get([vector_size], f32)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_2d_ty = MemRefType.get([outer_rows, outer_cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([total], xrt_dtype)
    l3_w_ty = MemRefType.get([head_dim], xrt_dtype)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_mem_space)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1_mem_space)
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    # row_offset = (local_row + tile_id * rows_per_tile) * head_dim
    row_offset_map = AffineMap.get(
        0, 3,
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
                AffineConstantExpr.get(head_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_2d_ty, l3_w_ty, l3_2d_ty)
    def qknorm_2d(arg0_2d, arg1_w, arg2_2d):
        @launch(operands=[arg0_2d, arg1_w, arg2_2d])
        def qkn_launch(l_in_2d, l_w, l_out_2d):
            in_flat = memref_collapse_shape(l3_1d_ty, l_in_2d, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out_2d, [[0, 1]])

            @segment(name="qkn_seg", operands=[in_flat, l_w, out_flat])
            def qkn_seg(s_in, s_w, s_out):
                @herd(name="qkn_herd", sizes=[herd_x, herd_y],
                      operands=[s_in, s_w, s_out])
                def qkn_body(_tx, _ty, _sx, _sy, h_in, h_w, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])
                    l1_w = AllocOp(l1RowTy, [], [])
                    l1_acc = AllocOp(l1VecTyF32, [], [])
                    l1_sq = AllocOp(l1SqTy, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    cst0_f32 = arith.ConstantOp(f32, 0.0)
                    n_f = arith.ConstantOp(f32, float(head_dim))
                    eps_f = arith.ConstantOp(f32, eps)
                    v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

                    # weight DMA once per tile (broadcast across rows).
                    dma_memcpy_nd(l1_w, h_w, src_offsets=[0],
                                  src_sizes=[head_dim], src_strides=[1])

                    for local_row in range_(rows_per_tile):
                        row_off = affine_apply(row_offset_map, [local_row, _tx, _ty])
                        dma_memcpy_nd(l1_in, h_in, src_offsets=[row_off],
                                      src_sizes=[head_dim], src_strides=[1])

                        # sum of x^2 in f32.
                        transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            v_x = transfer_read(vecTy, sub_in, [c0], identity_map, cst0, [True])
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(None, v_sq, l1_sq, [c0], identity_map, [True])
                            v_sq_rd = transfer_read(vecTy, l1_sq, [c0], identity_map, cst0, [True])
                            v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                            v_acc = transfer_read(vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True])
                            v_sum = arith.addf(v_acc, v_sq_f32)
                            transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                            yield_([])

                        v_final = transfer_read(vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True])
                        total_sum = vector_reduction(f32, "add", v_final)
                        rms = arith.divf(total_sum, n_f)
                        rms_eps = arith.addf(rms, eps_f)
                        rstd_f32 = math_dialect.rsqrt(rms_eps)
                        rstd = arith.truncf(xrt_dtype, rstd_f32)
                        v_rstd = BroadcastOp(vecTy, rstd)

                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            sub_w = subview(l1_w.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(vecTy, sub_in, [c0], identity_map, cst0, [True])
                            v_w = transfer_read(vecTy, sub_w, [c0], identity_map, cst0, [True])
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(None, v_weighted, sub_out, [c0], identity_map, [True])
                            yield_([])

                        dma_memcpy_nd(h_out, l1_out, dst_offsets=[row_off],
                                      dst_sizes=[head_dim], dst_strides=[1])
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)
                    DeallocOp(l1_w)
                    DeallocOp(l1_acc)
                    DeallocOp(l1_sq)


@module_builder
def _build_qknorm_1d(n_rows, head_dim, np_dtype, eps, herd_x=8, vector_size=16):
    """Decode per-head RMSNorm with 1D func args (M=1 token).

    Func signature: (in_1d: [n_rows*head_dim], weight: [head_dim], out_1d: [n_rows*head_dim]).
    The herd processes n_rows rows (= n_heads or n_kv_heads) of head_dim each.
    Mirrors _build_qknorm_2d math but with no collapse (args are already 1D).
    """
    xrt_dtype = type_mapper(np_dtype)
    total = n_rows * head_dim
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert head_dim % vector_size == 0
    assert n_rows % total_tiles == 0
    rows_per_tile = n_rows // total_tiles

    f32 = F32Type.get()
    vecTy = VectorType.get([vector_size], xrt_dtype)
    vecTyF32 = VectorType.get([vector_size], f32)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_1d_ty = MemRefType.get([total], xrt_dtype)
    l3_w_ty = MemRefType.get([head_dim], xrt_dtype)
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([head_dim], xrt_dtype, memory_space=l1_mem_space)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1_mem_space)
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    row_offset_map = AffineMap.get(
        0, 3,
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
                AffineConstantExpr.get(head_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_1d_ty, l3_w_ty, l3_1d_ty)
    def qknorm_1d(arg0_in, arg1_w, arg2_out):
        @launch(operands=[arg0_in, arg1_w, arg2_out])
        def qkn_launch(l_in, l_w, l_out):
            @segment(name="qkn1_seg", operands=[l_in, l_w, l_out])
            def qkn_seg(s_in, s_w, s_out):
                @herd(name="qkn1_herd", sizes=[herd_x, herd_y],
                      operands=[s_in, s_w, s_out])
                def qkn_body(_tx, _ty, _sx, _sy, h_in, h_w, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])
                    l1_w = AllocOp(l1RowTy, [], [])
                    l1_acc = AllocOp(l1VecTyF32, [], [])
                    l1_sq = AllocOp(l1SqTy, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    cst0_f32 = arith.ConstantOp(f32, 0.0)
                    n_f = arith.ConstantOp(f32, float(head_dim))
                    eps_f = arith.ConstantOp(f32, eps)
                    v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)

                    dma_memcpy_nd(l1_w, h_w, src_offsets=[0],
                                  src_sizes=[head_dim], src_strides=[1])

                    for local_row in range_(rows_per_tile):
                        row_off = affine_apply(row_offset_map, [local_row, _tx, _ty])
                        dma_memcpy_nd(l1_in, h_in, src_offsets=[row_off],
                                      src_sizes=[head_dim], src_strides=[1])

                        transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            v_x = transfer_read(vecTy, sub_in, [c0], identity_map, cst0, [True])
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(None, v_sq, l1_sq, [c0], identity_map, [True])
                            v_sq_rd = transfer_read(vecTy, l1_sq, [c0], identity_map, cst0, [True])
                            v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                            v_acc = transfer_read(vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True])
                            v_sum = arith.addf(v_acc, v_sq_f32)
                            transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                            yield_([])

                        v_final = transfer_read(vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True])
                        total_sum = vector_reduction(f32, "add", v_final)
                        rms = arith.divf(total_sum, n_f)
                        rms_eps = arith.addf(rms, eps_f)
                        rstd_f32 = math_dialect.rsqrt(rms_eps)
                        rstd = arith.truncf(xrt_dtype, rstd_f32)
                        v_rstd = BroadcastOp(vecTy, rstd)

                        for j in range_(0, head_dim, vector_size):
                            sub_in = subview(l1_in.result, [j], [vector_size], [1])
                            sub_w = subview(l1_w.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(vecTy, sub_in, [c0], identity_map, cst0, [True])
                            v_w = transfer_read(vecTy, sub_w, [c0], identity_map, cst0, [True])
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(None, v_weighted, sub_out, [c0], identity_map, [True])
                            yield_([])

                        dma_memcpy_nd(h_out, l1_out, dst_offsets=[row_off],
                                      dst_sizes=[head_dim], dst_strides=[1])
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)
                    DeallocOp(l1_w)
                    DeallocOp(l1_acc)
                    DeallocOp(l1_sq)


def build_rms_qkv_qknorm_rope_module(
    seq_len,
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    herd_m=8,
    herd_n=4,
    rope_herd_x=8,
    qknorm_eps=1e-6,
    qknorm_herd_x=8,
    gemm_spec_fn=None,
):
    """Build the 8-launch fused prefill attention-input ELF.

    Func args:
      %arg0  x_in     (seq_len, emb_dim)
      %arg1  norm_w   (emb_dim,)
      %arg2  normed   (seq_len, emb_dim)
      %arg3  wq       (emb_dim, q_dim)
      %arg4  q        (seq_len, q_dim)            Q GEMM out (pre-QK-norm)
      %arg5  wk       (emb_dim, kv_dim)
      %arg6  k        (seq_len, kv_dim)           K GEMM out (pre-QK-norm)
      %arg7  wv       (emb_dim, kv_dim)
      %arg8  v        (seq_len, kv_dim)           V GEMM out (final)
      %arg9  q_norm   (head_dim,)                 QK-norm Q weight
      %arg10 k_norm   (head_dim,)                 QK-norm K weight
      %arg11 q_n      (seq_len, q_dim)            QK-norm Q out (RoPE input)
      %arg12 k_n      (seq_len, kv_dim)           QK-norm K out (RoPE input)
      %arg13 lut_q    (n_heads*seq_len*head_dim,) RoPE Q LUT (1D, seq-first)
      %arg14 lut_k    (n_kv_heads*seq_len*head_dim,) RoPE K LUT (1D)
      %arg15 q_roped  (seq_len, q_dim)            final RoPE Q
      %arg16 k_roped  (seq_len, kv_dim)           final RoPE K
      [+ registry-driven f32 C-scratch tail args for fused-cast GEMMs]

    gemm_spec_fn: optional callable (m, k, n) -> spec dict shaped like
      gemm_registry_config() (keys: method, tile_m/k_l2/k_l1/n, sym_suffix,
      build_kwargs, needs_f32_scratch). When None (default), the per-GEMM spec
      is looked up from the kernel registry via gemm_registry_config — used by
      qwen3_0_6b / qwen3_1_7b whose shapes are in the registry. Models whose
      attention-input GEMM shapes are NOT in the registry (e.g. qwen3_4b with
      emb=2560) pass their own gemm_spec so the Q/K/V GEMMs use the same
      validated method+tiles the model's split rms_qkv ELF already used.

    Returns (module, scratch_for).
    """
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from shared.builders.rms_gemms_rope_multi import _build_rope_2d
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    q_total = seq_len * q_dim
    k_total = seq_len * kv_dim
    assert q_dim == n_heads * head_dim, (q_dim, n_heads, head_dim)
    assert kv_dim == n_kv_heads * head_dim, (kv_dim, n_kv_heads, head_dim)

    # Per-GEMM config: caller-injected spec fn (off-registry shapes) or the
    # registry lookup (default).
    if gemm_spec_fn is not None:
        q_spec = gemm_spec_fn(seq_len, emb_dim, q_dim)
        k_spec = gemm_spec_fn(seq_len, emb_dim, kv_dim)
        v_spec = gemm_spec_fn(seq_len, emb_dim, kv_dim)
    else:
        q_spec = gemm_registry_config(seq_len, emb_dim, q_dim, "bf16", "high")
        k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
        v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")

    def _kw_tiles(spec):
        return (
            dict(spec["build_kwargs"]),
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
        )

    # ---- Build sub-kernels ----
    print("  [1/8] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)))

    _q_kw, _q_tm, _q_k2, _q_k1, _q_tn = _kw_tiles(q_spec)
    _k_kw, _k_tm, _k_k2, _k_k1, _k_tn = _kw_tiles(k_spec)
    _v_kw, _v_tm, _v_k2, _v_k1, _v_tn = _kw_tiles(v_spec)
    print(f"  [2/8] Q GEMM ({q_spec['method']})  {seq_len}x{emb_dim}x{q_dim}...")
    q_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, q_dim, _q_tm, _q_k2, _q_k1, _q_tn, herd_m, herd_n, **_q_kw
        )
    )
    print(f"  [3/8] K GEMM ({k_spec['method']})  {seq_len}x{emb_dim}x{kv_dim}...")
    k_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, kv_dim, _k_tm, _k_k2, _k_k1, _k_tn, herd_m, herd_n, **_k_kw
        )
    )
    print(f"  [4/8] V GEMM ({v_spec['method']})  {seq_len}x{emb_dim}x{kv_dim}...")
    v_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, kv_dim, _v_tm, _v_k2, _v_k1, _v_tn, herd_m, herd_n, **_v_kw
        )
    )

    # 5-6. QK-norm: per-head RMSNorm over head_dim with eps=1e-6. Uses a
    #   dedicated 2D-in/out builder (collapse inside launch) — see
    #   _build_qknorm_2d for why expand_shape on the func arg is illegal.
    qn_rows = seq_len * n_heads
    kn_rows = seq_len * n_kv_heads
    print(f"  [5/8] QK-norm Q (rows={qn_rows} dim={head_dim} eps={qknorm_eps})...")
    qkn_q_ir = str(
        _build_qknorm_2d(seq_len, q_dim, head_dim, bfloat16, qknorm_eps, qknorm_herd_x)
    )
    print(f"  [6/8] QK-norm K (rows={kn_rows} dim={head_dim} eps={qknorm_eps})...")
    qkn_k_ir = str(
        _build_qknorm_2d(seq_len, kv_dim, head_dim, bfloat16, qknorm_eps, qknorm_herd_x)
    )

    # 7-8. RoPE Q/K (2D in/out, head_dim wide).
    print(f"  [7/8] RoPE Q (outer={seq_len}x{q_dim}, dim={head_dim})...")
    rope_q_ir = str(_build_rope_2d(seq_len, q_dim, head_dim, bfloat16, rope_herd_x))
    print(f"  [8/8] RoPE K (outer={seq_len}x{kv_dim}, dim={head_dim})...")
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    # ---- Scratch (fused-cast GEMM f32 C-tail) ----
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (q_spec, seq_len, q_dim),
            (k_spec, seq_len, kv_dim),
            (v_spec, seq_len, kv_dim),
        ],
        base_arg_count=17,
    )

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}

    def _gemm_externs(spec):
        sfx = spec["sym_suffix"]
        return {
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{q_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{q_total}xbf16>"),
        FuncArg("%arg14", f"memref<{k_total}xbf16>"),
        FuncArg("%arg15", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg16", f"memref<{seq_len}x{kv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}),
        KernelSlice(
            q_ir, "q", _gemm_arg_map(2, 3, 4, scratch_for[0]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(q_spec),
        ),
        KernelSlice(
            k_ir, "k", _gemm_arg_map(2, 5, 6, scratch_for[1]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(k_spec),
        ),
        KernelSlice(
            v_ir, "v", _gemm_arg_map(2, 7, 8, scratch_for[2]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(v_spec),
        ),
        # QK-norm Q: in=q(arg4), weight=q_norm(arg9), out=q_n(arg11).
        KernelSlice(qkn_q_ir, "qn", {0: 4, 1: 9, 2: 11}, private_from=False),
        # QK-norm K: in=k(arg6), weight=k_norm(arg10), out=k_n(arg12).
        KernelSlice(qkn_k_ir, "kn", {0: 6, 1: 10, 2: 12}, private_from=False),
        # RoPE consumes the QK-norm outputs (arg11/arg12), not the raw GEMM outs.
        KernelSlice(rope_q_ir, "rq", {0: 11, 1: 13, 2: 15}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 12, 1: 14, 2: 16}, extern_syms={"@rope"}),
    ]

    module = stitch_elf(
        "rms_qkv_qknorm_rope",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_rms_qkv_qknorm_rope.mlir",
    )
    print(f"  rms_qkv_qknorm_rope module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


# ===========================================================================
# DECODE (M=1) fused builder: RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE.
# 8-launch 1D ELF. Mirrors the prefill builder at M=1 (GEMV instead of GEMM).
# ===========================================================================


def build_rms_qkv_qknorm_rope_gemv_module(
    emb_dim, q_dim, kv_dim, n_heads, n_kv_heads, head_dim,
    tile_m=8, m_input=4, herd_m=8, qknorm_eps=1e-6, qknorm_herd_x=8,
):
    """8-launch decode ELF (all 1D — M=1 token):

      %arg0  x_in     (emb_dim,)
      %arg1  norm_w   (emb_dim,)
      %arg2  normed   (emb_dim,)
      %arg3  wq       (q_dim, emb_dim)    GEMV weight (out, in)
      %arg4  q        (q_dim,)            Q GEMV out (pre-QK-norm)
      %arg5  wk       (kv_dim, emb_dim)
      %arg6  k        (kv_dim,)
      %arg7  wv       (kv_dim, emb_dim)
      %arg8  v        (kv_dim,)           V GEMV out (final)
      %arg9  q_norm   (head_dim,)
      %arg10 k_norm   (head_dim,)
      %arg11 q_n      (q_dim,)            QK-norm Q out (RoPE input)
      %arg12 k_n      (kv_dim,)           QK-norm K out (RoPE input)
      %arg13 lut_q    (q_dim,)            RoPE Q LUT (n_heads*head_dim, position-dependent)
      %arg14 lut_k    (kv_dim,)           RoPE K LUT
      %arg15 q_roped  (q_dim,)            final RoPE Q
      %arg16 k_roped  (kv_dim,)           final RoPE K
    """
    import shared.builders.rms_gemv_rope_multi as rgr
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    from matvec import build_module as build_gemv

    assert q_dim == n_heads * head_dim
    assert kv_dim == n_kv_heads * head_dim

    # RMSNorm (decode 1D) reads rgr.EPS; Qwen3 = 1e-6.
    _saved_eps = rgr.EPS
    rgr.EPS = qknorm_eps
    try:
        print("  [1/8] RMSNorm (decode 1D, eps=%g)..." % qknorm_eps)
        rms_ir = str(rgr._build_rms_1d(emb_dim, bfloat16, 16))
    finally:
        rgr.EPS = _saved_eps

    print(f"  [2/8] Q GEMV M={q_dim} K={emb_dim}...")
    q_ir = str(build_gemv(q_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))
    print(f"  [3/8] K GEMV M={kv_dim} K={emb_dim}...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))
    print(f"  [4/8] V GEMV M={kv_dim} K={emb_dim}...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [5/8] QK-norm Q (rows={n_heads} dim={head_dim} eps={qknorm_eps})...")
    qkn_q_ir = str(_build_qknorm_1d(n_heads, head_dim, bfloat16, qknorm_eps, qknorm_herd_x))
    print(f"  [6/8] QK-norm K (rows={n_kv_heads} dim={head_dim} eps={qknorm_eps})...")
    qkn_k_ir = str(_build_qknorm_1d(n_kv_heads, head_dim, bfloat16, qknorm_eps,
                                    herd_x=min(qknorm_herd_x, n_kv_heads)))

    print(f"  [7/8] RoPE Q (rows={n_heads} dim={head_dim})...")
    rope_q_ir = str(rgr._build_rope_1d(n_heads, head_dim, bfloat16,
                                       herd_x=min(qknorm_herd_x, n_heads)))
    print(f"  [8/8] RoPE K (rows={n_kv_heads} dim={head_dim})...")
    rope_k_ir = str(rgr._build_rope_1d(n_kv_heads, head_dim, bfloat16,
                                       herd_x=min(qknorm_herd_x, n_kv_heads)))

    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{q_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{kv_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{kv_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{head_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg15", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg16", f"memref<{kv_dim}xbf16>"),
    ]
    # GEMV func args: {0: weight (MxK), 1: input (K,), 2: output (M,)}.
    slices = [
        KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, private_from=False),
        KernelSlice(q_ir, "q", {0: 3, 1: 2, 2: 4}),
        KernelSlice(k_ir, "k", {0: 5, 1: 2, 2: 6}, private_from=False),
        KernelSlice(v_ir, "v", {0: 7, 1: 2, 2: 8}, private_from=False),
        KernelSlice(qkn_q_ir, "qn", {0: 4, 1: 9, 2: 11}, private_from=False),
        KernelSlice(qkn_k_ir, "kn", {0: 6, 1: 10, 2: 12}, private_from=False),
        KernelSlice(rope_q_ir, "rq", {0: 11, 1: 13, 2: 15}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 12, 1: 14, 2: 16}, extern_syms={"@rope"}),
    ]
    module = stitch_elf(
        "rms_qkv_qknorm_rope_gemv",
        base_args,
        slices,
        extra_externs={
            "@zero_vectorized_bf16",
            "@matvec_vectorized_bf16_bf16",
            "@linalg_fill_bf16",
            "@rope",
        },
        debug_dump_path="/tmp/debug_rms_qkv_qknorm_rope_gemv.mlir",
    )
    print(f"  rms_qkv_qknorm_rope_gemv module: {len(str(module).splitlines())} lines, parsed OK")
    return module
