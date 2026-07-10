# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMMs + per-channel QKV bias-add + RoPE Q+K — fused prefill ELF.

This is the Qwen2.5 (Qwen2 family, attention_bias=True) analogue of
`rms_qkv_qknorm_rope_multi` (Qwen3). Qwen2.5 adds a per-channel BIAS to Q/K/V
AFTER the projection GEMM and BEFORE RoPE (HF Qwen2 order: proj -> +bias ->
RoPE(Q,K) -> attention; V is bias-added and used directly). The bias-add is a
broadcast `out[seq, D] = in[seq, D] + bias[D]` — NOT the registry EltwiseAdd
(same-shape a+b). We express it with a tiny per-row broadcast-add launch
(`_build_bias_add_2d`) modeled on the qwen3 QK-norm slice (which broadcasts a
(head_dim,) weight across rows) — the bias-add is even simpler (a single add,
no reduction).

Layout note (Qwen2.5-0.5B): the Q/K/V GEMMs are N-padded (e.g. to 1024) so the
drain path can use the only numerically-correct tile_n=128. The bias-add slice
reads the PADDED GEMM-output buffer row-by-row (row stride = in_cols), adds the
real-width (D,) bias, and writes to an UN-PADDED contiguous (seq, D) buffer that
RoPE then consumes via collapse_shape. So padding stays internal to the GEMM and
RoPE/attention see clean un-padded tensors — exactly the split path's behaviour,
just moved on-device.

8 launches (prefill):
  1. RMSNorm    x_in x norm_w -> normed
  2. Q GEMM     normed x wq -> q_pad   (seq, q_pad)   N-padded
  3. K GEMM     normed x wk -> k_pad   (seq, kv_pad)  N-padded
  4. V GEMM     normed x wv -> v_pad   (seq, kv_pad)  N-padded
  5. bias Q     q_pad[:, :q_dim] + bq -> q_b   (seq, q_dim)   <-- NEW
  6. bias K     k_pad[:, :kv_dim] + bk -> k_b  (seq, kv_dim)  <-- NEW
  7. bias V     v_pad[:, :kv_dim] + bv -> v_b  (seq, kv_dim)  <-- NEW (no RoPE)
  8. RoPE Q     q_b(2D->1D) x lut_q -> q_roped(1D->2D)
  9. RoPE K     k_b(2D->1D) x lut_k -> k_roped(1D->2D)

(9 launches when V bias counts; "8" matches the qwen3 builder naming loosely.)
"""

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    MemRefType,
    IntegerAttr,
    AffineMap,
    AffineExpr,
    AffineSymbolExpr,
    AffineConstantExpr,
    AffineMapAttr,
    VectorType,
)
from air.dialects.air import (
    module_builder,
    launch,
    segment,
    herd,
    dma_memcpy_nd,
    MemorySpace,
    T,
)
from air.dialects.affine import apply as affine_apply
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_ as range_, yield_
from air.backend.xrt_runner import type_mapper

# ---------------------------------------------------------------------------
# Per-channel broadcast bias-add, 2D in/out (prefill, M=seq).
#
#   out[row, c] = in[row, c] + bias[c]   for c in [0, real_cols)
#
# `in` may be N-padded to `in_cols` >= real_cols (the GEMM output width); only
# the first `real_cols` columns of each row are read (DMA with row stride
# in_cols) and written to the un-padded contiguous out buffer. bias (real_cols,)
# is broadcast across all rows (DMA'd once per tile, like the QK-norm weight).
# Models _build_qknorm_2d (broadcast weight) + _build_padded_residual_add_2d_ir
# (padded-row read).
# ---------------------------------------------------------------------------


@module_builder
def _build_bias_add_2d(seq_len, in_cols, real_cols, np_dtype, herd_x, vector_size=16):
    """Build a broadcast bias-add launch with 2D in/out args.

    Func signature:
      (in_2d:  [seq_len, in_cols]   — N-padded GEMM output,
       bias_1d:[real_cols],
       out_2d: [seq_len, real_cols] — un-padded contiguous)
    The herd splits the seq rows across herd_x tiles; each row reads the first
    real_cols of `in`, adds the broadcast bias, writes real_cols to `out`.
    """
    xrt_dtype = type_mapper(np_dtype)
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert real_cols % vector_size == 0, (real_cols, vector_size)
    assert seq_len % total_tiles == 0, (seq_len, total_tiles)
    rows_per_tile = seq_len // total_tiles

    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    in_2d_ty = MemRefType.get([seq_len, in_cols], xrt_dtype)
    bias_ty = MemRefType.get([real_cols], xrt_dtype)
    out_2d_ty = MemRefType.get([seq_len, real_cols], xrt_dtype)
    out_flat_ty = MemRefType.get([seq_len * real_cols], xrt_dtype)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_row_ty = MemRefType.get([real_cols], xrt_dtype, memory_space=l1_space)

    # row index r = tile_id * rows_per_tile + local_row
    row_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(
                    AffineExpr.get_add(
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(1), AffineConstantExpr.get(herd_y)
                        ),
                        AffineSymbolExpr.get(2),
                    ),
                    AffineConstantExpr.get(rows_per_tile),
                ),
                AffineSymbolExpr.get(0),
            )
        ],
    )

    @FuncOp.from_py_func(in_2d_ty, bias_ty, out_2d_ty)
    def bias_add_2d(arg0_in, arg1_bias, arg2_out):
        @launch(operands=[arg0_in, arg1_bias, arg2_out])
        def ba_launch(l_in, l_bias, l_out):
            from air.dialects.memref import collapse_shape as memref_collapse_shape

            out_flat = memref_collapse_shape(out_flat_ty, l_out, [[0, 1]])

            @segment(name="ba_seg", operands=[l_in, l_bias, out_flat])
            def ba_seg(s_in, s_bias, s_out):
                @herd(
                    name="ba_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_bias, s_out],
                )
                def ba_body(_tx, _ty, _sx, _sy, h_in, h_bias, h_out):
                    l1_in = AllocOp(l1_row_ty, [], [])
                    l1_bias = AllocOp(l1_row_ty, [], [])
                    l1_out = AllocOp(l1_row_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                    # bias DMA once per tile (broadcast across rows).
                    dma_memcpy_nd(
                        l1_bias,
                        h_bias,
                        src_offsets=[0],
                        src_sizes=[real_cols],
                        src_strides=[1],
                    )

                    for local_row in range_(rows_per_tile):
                        r = affine_apply(row_map, [local_row, _tx, _ty])
                        # in: 2D (seq, in_cols) — read row r, first real_cols.
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[r, c0],
                            src_sizes=[1, real_cols],
                            src_strides=[in_cols, 1],
                        )
                        for j in range_(0, real_cols, vector_size):
                            si = subview(l1_in.result, [j], [vector_size], [1])
                            sb = subview(l1_bias.result, [j], [vector_size], [1])
                            so = subview(l1_out.result, [j], [vector_size], [1])
                            v_i = transfer_read(
                                vec_ty, si, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vec_ty, sb, [c0], identity_map, cst0, [True]
                            )
                            transfer_write(
                                None,
                                arith.addf(v_i, v_b),
                                so,
                                [c0],
                                identity_map,
                                [True],
                            )
                            yield_([])
                        # out: flat 1D (seq*real_cols), row offset r*real_cols.
                        off = arith.muli(r, arith.ConstantOp.create_index(real_cols))
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[off],
                            dst_sizes=[real_cols],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_bias)
                    DeallocOp(l1_out)


# ---------------------------------------------------------------------------
# Per-channel broadcast bias-add, 1D in/out (decode, M=1 token).
#
#   out[c] = in[c] + bias[c]   for c in [0, n_cols)
#
# Decode GEMV outputs are NOT N-padded (plain matvec), so in_cols == n_cols.
# Single row processed by tile 0; trivially cheap.
# ---------------------------------------------------------------------------


@module_builder
def _build_bias_add_1d(n_cols, np_dtype, herd_x=1, vector_size=16):
    """Build a 1D broadcast bias-add launch (decode, M=1).

    Func signature: (in_1d: [n_cols], bias_1d: [n_cols], out_1d: [n_cols]).
    """
    xrt_dtype = type_mapper(np_dtype)
    assert n_cols % vector_size == 0, (n_cols, vector_size)
    assert n_cols % herd_x == 0, (n_cols, herd_x)
    cols_per_tile = n_cols // herd_x

    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_ty = MemRefType.get([n_cols], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([cols_per_tile], xrt_dtype, memory_space=l1_space)

    # offset = tile_id * cols_per_tile
    off_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0), AffineConstantExpr.get(1)
                    ),
                    AffineSymbolExpr.get(1),
                ),
                AffineConstantExpr.get(cols_per_tile),
            )
        ],
    )

    @FuncOp.from_py_func(l3_ty, l3_ty, l3_ty)
    def bias_add_1d(arg0_in, arg1_bias, arg2_out):
        @launch(operands=[arg0_in, arg1_bias, arg2_out])
        def ba_launch(l_in, l_bias, l_out):
            @segment(name="ba1_seg", operands=[l_in, l_bias, l_out])
            def ba_seg(s_in, s_bias, s_out):
                @herd(
                    name="ba1_herd", sizes=[herd_x, 1], operands=[s_in, s_bias, s_out]
                )
                def ba_body(_tx, _ty, _sx, _sy, h_in, h_bias, h_out):
                    l1_in = AllocOp(l1_ty, [], [])
                    l1_bias = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    off = affine_apply(off_map, [_tx, _ty])
                    dma_memcpy_nd(
                        l1_in,
                        h_in,
                        src_offsets=[off],
                        src_sizes=[cols_per_tile],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(
                        l1_bias,
                        h_bias,
                        src_offsets=[off],
                        src_sizes=[cols_per_tile],
                        src_strides=[1],
                    )
                    for j in range_(0, cols_per_tile, vector_size):
                        si = subview(l1_in.result, [j], [vector_size], [1])
                        sb = subview(l1_bias.result, [j], [vector_size], [1])
                        so = subview(l1_out.result, [j], [vector_size], [1])
                        v_i = transfer_read(
                            vec_ty, si, [c0], identity_map, cst0, [True]
                        )
                        v_b = transfer_read(
                            vec_ty, sb, [c0], identity_map, cst0, [True]
                        )
                        transfer_write(
                            None, arith.addf(v_i, v_b), so, [c0], identity_map, [True]
                        )
                        yield_([])
                    dma_memcpy_nd(
                        h_out,
                        l1_out,
                        dst_offsets=[off],
                        dst_sizes=[cols_per_tile],
                        dst_strides=[1],
                    )
                    DeallocOp(l1_in)
                    DeallocOp(l1_bias)
                    DeallocOp(l1_out)


# ===========================================================================
# PREFILL fused builder (M=seq): RMSNorm + Q/K/V GEMM + bias×3 + RoPE Q+K.
# ===========================================================================


def build_rms_qkv_bias_rope_module(
    seq_len,
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    q_pad=None,
    kv_pad=None,
    herd_m=8,
    herd_n=4,
    rope_herd_x=8,
    bias_herd_x=8,
    gemm_spec_fn=None,
):
    """Build the fused prefill attention-input ELF for Qwen2.5.

    Func args:
      %arg0  x_in     (seq_len, emb_dim)
      %arg1  norm_w   (emb_dim,)
      %arg2  normed   (seq_len, emb_dim)
      %arg3  wq       (emb_dim, q_pad)             N-padded Q weight
      %arg4  q_pad    (seq_len, q_pad)             Q GEMM out (N-padded)
      %arg5  wk       (emb_dim, kv_pad)
      %arg6  k_pad    (seq_len, kv_pad)            K GEMM out (N-padded)
      %arg7  wv       (emb_dim, kv_pad)
      %arg8  v_pad    (seq_len, kv_pad)            V GEMM out (N-padded)
      %arg9  bq       (q_dim,)                     Q bias
      %arg10 bk       (kv_dim,)                    K bias
      %arg11 bv       (kv_dim,)                    V bias
      %arg12 q_b      (seq_len, q_dim)             Q after bias (RoPE input)
      %arg13 k_b      (seq_len, kv_dim)            K after bias (RoPE input)
      %arg14 v_b      (seq_len, kv_dim)            V after bias (final, no RoPE)
      %arg15 lut_q    (n_heads*seq_len*head_dim,)  RoPE Q LUT (1D)
      %arg16 lut_k    (n_kv_heads*seq_len*head_dim,) RoPE K LUT (1D)
      %arg17 q_roped  (seq_len, q_dim)             final RoPE Q
      %arg18 k_roped  (seq_len, kv_dim)            final RoPE K
      [+ registry-driven f32 C-scratch tail args for fused-cast GEMMs]

    q_pad/kv_pad default to q_dim/kv_dim (no padding) when None. Qwen2.5-0.5B
    passes q_pad=kv_pad=1024 (drain tile_n=128 correctness). The bias-add reads
    the padded GEMM output and writes un-padded (seq, *_dim) buffers.

    gemm_spec_fn: optional callable (m, k, n) -> spec dict (keys: method,
      tile_m/k_l2/k_l1/n, sym_suffix, build_kwargs, needs_f32_scratch). When
      None, per-GEMM spec is the registry lookup. Models whose attention-input
      GEMM shapes are NOT in the registry pass their own.

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

    if q_pad is None:
        q_pad = q_dim
    if kv_pad is None:
        kv_pad = kv_dim

    q_total = seq_len * q_dim
    k_total = seq_len * kv_dim
    assert q_dim == n_heads * head_dim, (q_dim, n_heads, head_dim)
    assert kv_dim == n_kv_heads * head_dim, (kv_dim, n_kv_heads, head_dim)

    # Per-GEMM config (off-registry shapes: injected spec fn; else registry).
    # Build at the PADDED N (q_pad/kv_pad), forcing tile_n=128 like the split
    # rms_qkv builder (the only numerically-correct drain tile_n).
    if gemm_spec_fn is not None:
        q_spec = dict(gemm_spec_fn(seq_len, emb_dim, q_pad))
        k_spec = dict(gemm_spec_fn(seq_len, emb_dim, kv_pad))
        v_spec = dict(gemm_spec_fn(seq_len, emb_dim, kv_pad))
    else:
        q_spec = dict(gemm_registry_config(seq_len, emb_dim, q_dim, "bf16", "high"))
        k_spec = dict(gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high"))
        v_spec = dict(gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high"))
    # Force tile_n=128 for the N-padded drain GEMMs (matches split rms_qkv).
    if q_pad != q_dim:
        q_spec["tile_n"] = 128
    if kv_pad != kv_dim:
        k_spec["tile_n"] = 128
        v_spec["tile_n"] = 128

    def _kw_tiles(spec):
        return (
            dict(spec["build_kwargs"]),
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
        )

    # ---- Build sub-kernels ----
    print("  [1/9] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    _q_kw, _q_tm, _q_k2, _q_k1, _q_tn = _kw_tiles(q_spec)
    _k_kw, _k_tm, _k_k2, _k_k1, _k_tn = _kw_tiles(k_spec)
    _v_kw, _v_tm, _v_k2, _v_k1, _v_tn = _kw_tiles(v_spec)
    print(
        f"  [2/9] Q GEMM ({q_spec['method']})  {seq_len}x{emb_dim}x{q_pad} (pad {q_dim})..."
    )
    q_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, q_pad, _q_tm, _q_k2, _q_k1, _q_tn, herd_m, herd_n, **_q_kw
        )
    )
    print(
        f"  [3/9] K GEMM ({k_spec['method']})  {seq_len}x{emb_dim}x{kv_pad} (pad {kv_dim})..."
    )
    k_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_pad,
            _k_tm,
            _k_k2,
            _k_k1,
            _k_tn,
            herd_m,
            herd_n,
            **_k_kw,
        )
    )
    print(
        f"  [4/9] V GEMM ({v_spec['method']})  {seq_len}x{emb_dim}x{kv_pad} (pad {kv_dim})..."
    )
    v_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_pad,
            _v_tm,
            _v_k2,
            _v_k1,
            _v_tn,
            herd_m,
            herd_n,
            **_v_kw,
        )
    )

    # 5-7. bias-add (broadcast (D,) over rows, padded->unpadded).
    print(f"  [5/9] bias Q (in_cols={q_pad} real={q_dim})...")
    bias_q_ir = str(_build_bias_add_2d(seq_len, q_pad, q_dim, bfloat16, bias_herd_x))
    print(f"  [6/9] bias K (in_cols={kv_pad} real={kv_dim})...")
    bias_k_ir = str(_build_bias_add_2d(seq_len, kv_pad, kv_dim, bfloat16, bias_herd_x))
    print(f"  [7/9] bias V (in_cols={kv_pad} real={kv_dim})...")
    bias_v_ir = str(_build_bias_add_2d(seq_len, kv_pad, kv_dim, bfloat16, bias_herd_x))

    # 8-9. RoPE Q/K (2D in/out, head_dim wide), on the un-padded bias outputs.
    print(f"  [8/9] RoPE Q (outer={seq_len}x{q_dim}, dim={head_dim})...")
    rope_q_ir = str(_build_rope_2d(seq_len, q_dim, head_dim, bfloat16, rope_herd_x))
    print(f"  [9/9] RoPE K (outer={seq_len}x{kv_dim}, dim={head_dim})...")
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    # ---- Scratch (fused-cast GEMM f32 C-tail) ----
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (q_spec, seq_len, q_pad),
            (k_spec, seq_len, kv_pad),
            (v_spec, seq_len, kv_pad),
        ],
        base_arg_count=19,
    )

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}

    def _gemm_externs(spec):
        if spec["method"] == "direct":
            return set()
        sfx = spec["sym_suffix"]
        return {
            "@matmul_bf16",
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{q_pad}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{q_pad}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}x{kv_pad}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{kv_pad}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{kv_pad}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{kv_pad}xbf16>"),
        FuncArg("%arg9", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg15", f"memref<{q_total}xbf16>"),
        FuncArg("%arg16", f"memref<{k_total}xbf16>"),
        FuncArg("%arg17", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg18", f"memref<{seq_len}x{kv_dim}xbf16>"),
    ]

    # Collect the GEMM private func decls from the FIRST slice of each distinct
    # method+suffix. Q/K/V may use different methods (e.g. 1.5B: Q fused-cast
    # _m64, K/V drain _m32) → each suffix's decls (zero_f32_mn_*, f32_to_bf16_*)
    # must be emitted exactly once. direct-codegen GEMMs carry no external decls.
    def _gemm_key(spec):
        return "direct" if spec["method"] == "direct" else spec["sym_suffix"]

    _seen_keys = set()

    def _pf(spec):
        key = _gemm_key(spec)
        first = key not in _seen_keys
        _seen_keys.add(key)
        return first

    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(
            q_ir,
            "q",
            _gemm_arg_map(2, 3, 4, scratch_for[0]),
            extern_syms=_gemm_externs(q_spec),
            private_from=_pf(q_spec),
        ),
        KernelSlice(
            k_ir,
            "k",
            _gemm_arg_map(2, 5, 6, scratch_for[1]),
            extern_syms=_gemm_externs(k_spec),
            private_from=_pf(k_spec),
        ),
        KernelSlice(
            v_ir,
            "v",
            _gemm_arg_map(2, 7, 8, scratch_for[2]),
            extern_syms=_gemm_externs(v_spec),
            private_from=_pf(v_spec),
        ),
        # bias Q: in=q_pad(arg4), bias=bq(arg9), out=q_b(arg12).
        KernelSlice(bias_q_ir, "bq", {0: 4, 1: 9, 2: 12}, private_from=False),
        # bias K: in=k_pad(arg6), bias=bk(arg10), out=k_b(arg13).
        KernelSlice(bias_k_ir, "bk", {0: 6, 1: 10, 2: 13}, private_from=False),
        # bias V: in=v_pad(arg8), bias=bv(arg11), out=v_b(arg14).
        KernelSlice(bias_v_ir, "bv", {0: 8, 1: 11, 2: 14}, private_from=False),
        # RoPE consumes the bias outputs (arg12/arg13), not the raw GEMM outs.
        KernelSlice(rope_q_ir, "rq", {0: 12, 1: 15, 2: 17}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 13, 1: 16, 2: 18}, extern_syms={"@rope"}),
    ]

    module = stitch_elf(
        "rms_qkv_bias_rope",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_rms_qkv_bias_rope.mlir",
    )
    print(
        f"  rms_qkv_bias_rope module: {len(str(module).splitlines())} lines, parsed OK"
    )
    return module, scratch_for


# ===========================================================================
# DECODE (M=1) fused builder: RMSNorm + Q/K/V GEMV + bias×3 + RoPE Q+K.
# ===========================================================================


def build_rms_qkv_bias_rope_gemv_module(
    emb_dim,
    q_dim,
    kv_dim,
    n_heads,
    n_kv_heads,
    head_dim,
    tile_m=8,
    m_input=8,
    herd_m=8,
    eps=1e-6,
    bias_herd_x=8,
):
    """9-launch decode ELF (all 1D — M=1 token):

    %arg0  x_in     (emb_dim,)
    %arg1  norm_w   (emb_dim,)
    %arg2  normed   (emb_dim,)
    %arg3  wq       (q_dim, emb_dim)    GEMV weight (out, in)
    %arg4  q        (q_dim,)            Q GEMV out (pre-bias)
    %arg5  wk       (kv_dim, emb_dim)
    %arg6  k        (kv_dim,)
    %arg7  wv       (kv_dim, emb_dim)
    %arg8  v        (kv_dim,)           V GEMV out (pre-bias)
    %arg9  bq       (q_dim,)
    %arg10 bk       (kv_dim,)
    %arg11 bv       (kv_dim,)
    %arg12 q_b      (q_dim,)            Q after bias (RoPE input)
    %arg13 k_b      (kv_dim,)           K after bias (RoPE input)
    %arg14 v_b      (kv_dim,)           V after bias (final)
    %arg15 lut_q    (q_dim,)            RoPE Q LUT (position-dependent)
    %arg16 lut_k    (kv_dim,)           RoPE K LUT
    %arg17 q_roped  (q_dim,)            final RoPE Q
    %arg18 k_roped  (kv_dim,)           final RoPE K
    """
    import shared.builders.rms_gemv_rope_multi as rgr
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    from matvec import build_module as build_gemv

    assert q_dim == n_heads * head_dim
    assert kv_dim == n_kv_heads * head_dim

    _saved_eps = rgr.EPS
    rgr.EPS = eps
    try:
        print("  [1/9] RMSNorm (decode 1D, eps=%g)..." % eps)
        rms_ir = str(rgr._build_rms_1d(emb_dim, bfloat16, 16))
    finally:
        rgr.EPS = _saved_eps

    print(f"  [2/9] Q GEMV M={q_dim} K={emb_dim}...")
    q_ir = str(build_gemv(q_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))
    print(f"  [3/9] K GEMV M={kv_dim} K={emb_dim}...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))
    print(f"  [4/9] V GEMV M={kv_dim} K={emb_dim}...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    def _ba_hx(n):
        # herd_x must divide n_cols and n_cols/herd_x stays vector_size-aligned.
        for d in range(min(n // 16, bias_herd_x), 0, -1):
            if n % d == 0 and (n // d) % 16 == 0:
                return d
        return 1

    print(f"  [5/9] bias Q (n={q_dim})...")
    bias_q_ir = str(_build_bias_add_1d(q_dim, bfloat16, herd_x=_ba_hx(q_dim)))
    print(f"  [6/9] bias K (n={kv_dim})...")
    bias_k_ir = str(_build_bias_add_1d(kv_dim, bfloat16, herd_x=_ba_hx(kv_dim)))
    print(f"  [7/9] bias V (n={kv_dim})...")
    bias_v_ir = str(_build_bias_add_1d(kv_dim, bfloat16, herd_x=_ba_hx(kv_dim)))

    def _largest_divisor(n, cap):
        for d in range(min(n, cap), 0, -1):
            if n % d == 0:
                return d
        return 1

    rq_hx = _largest_divisor(n_heads, bias_herd_x)
    rk_hx = _largest_divisor(n_kv_heads, bias_herd_x)
    print(f"  [8/9] RoPE Q (rows={n_heads} dim={head_dim} herd_x={rq_hx})...")
    rope_q_ir = str(rgr._build_rope_1d(n_heads, head_dim, bfloat16, herd_x=rq_hx))
    print(f"  [9/9] RoPE K (rows={n_kv_heads} dim={head_dim} herd_x={rk_hx})...")
    rope_k_ir = str(rgr._build_rope_1d(n_kv_heads, head_dim, bfloat16, herd_x=rk_hx))

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
        FuncArg("%arg9", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg15", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg16", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg17", f"memref<{q_dim}xbf16>"),
        FuncArg("%arg18", f"memref<{kv_dim}xbf16>"),
    ]
    # GEMV func args: {0: weight (MxK), 1: input (K,), 2: output (M,)}.
    slices = [
        KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, private_from=False),
        KernelSlice(q_ir, "q", {0: 3, 1: 2, 2: 4}),
        KernelSlice(k_ir, "k", {0: 5, 1: 2, 2: 6}, private_from=False),
        KernelSlice(v_ir, "v", {0: 7, 1: 2, 2: 8}, private_from=False),
        KernelSlice(bias_q_ir, "bq", {0: 4, 1: 9, 2: 12}, private_from=False),
        KernelSlice(bias_k_ir, "bk", {0: 6, 1: 10, 2: 13}, private_from=False),
        KernelSlice(bias_v_ir, "bv", {0: 8, 1: 11, 2: 14}, private_from=False),
        KernelSlice(rope_q_ir, "rq", {0: 12, 1: 15, 2: 17}, extern_syms={"@rope"}),
        KernelSlice(rope_k_ir, "rk", {0: 13, 1: 16, 2: 18}, extern_syms={"@rope"}),
    ]
    module = stitch_elf(
        "rms_qkv_bias_rope_gemv",
        base_args,
        slices,
        extra_externs={
            "@zero_vectorized_bf16",
            "@matvec_vectorized_bf16_bf16",
            "@linalg_fill_bf16",
            "@rope",
        },
        debug_dump_path="/tmp/debug_rms_qkv_bias_rope_gemv.mlir",
    )
    print(
        f"  rms_qkv_bias_rope_gemv module: {len(str(module).splitlines())} lines, parsed OK"
    )
    return module
