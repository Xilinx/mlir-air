# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-0.5B Prefill on MLIR-AIR (NPU2) — single-block (Phase 2) path.

Qwen2.5 diverges from LLAMA-3.2 in two ways:

  1. QKV bias (attention_bias=True, the Qwen2 family). After the Q/K/V
     projection GEMM, a per-channel bias is added: q=x@wq+bq, k=x@wk+bk,
     v=x@wv+bv, AFTER the projection GEMM and BEFORE RoPE (the HF Qwen2 order:
     proj -> +bias -> RoPE(Q,K) -> attention; V is bias-added and used directly).
     The bias-add is fused ON-DEVICE inside the rms_qkv_bias_rope ELF as a
     broadcast bias-add slice (bq/bk/bv passed as static args), so RMSNorm +
     Q/K/V GEMM + bias + RoPE(Q,K) all run in one stitched module.
     NO QK-norm (that is Qwen3 — do NOT add it).

  2. Non-aligned dims + mixed GEMM precision.
        emb=896, q_dim=896 (14 heads x 64), kv_dim=128 (2 heads x 64),
        hidden=4864.  o_proj is SQUARE (q_dim==emb_dim==896), unlike Qwen3.
     Registry-selected methods for these shapes (seq=2048):
        Q/K/V  (896x{896,128})  -> drain      (_m32)
        O      (896x896)        -> drain      (_m32)
        Gate/Up(896x4864)       -> DIRECT (low-precision) — high-prec tier RAISES
                                   (near-zero-ref atol artifact); direct is in
                                   the bf16 tier at mean_rel_L1=1.11e-2.
        Down   (4864x896)       -> fused-cast (_m64)
     The o_ffn ELF therefore mixes THREE GEMM methods in one stitched module:
     drain (mm_m32.o), direct (no external .o), and fused-cast (mm_m64.o).

Attention uses the CPU fallback (cpu_attn=True), matching llama/qwen3 prefill.
head_dim=64 → no FA hang risk.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Add programming_examples/ and llms/ to path for shared.* + registry imports.
_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from qwen25_0_5b_weights import LlamaConfig
from qwen25_0_5b_cpu_helpers import attention_reference


# ---------------------------------------------------------------------------
# Generic per-GEMM slice builder. Supports the three bf16-out methods the
# Qwen2.5 GEMMs resolve to (drain / fused-cast / direct), so each GEMM in a
# stitched ELF independently picks its registry method. Returns the GEMM IR,
# the set of extern symbols to preserve, and whether it needs an f32 C-scratch.
# ---------------------------------------------------------------------------


def _gemm_spec(m, k, n, precision):
    """Registry config for one GEMM. precision: 'high' or 'low'."""
    from shared.builders.gemm_builder import gemm_registry_config, gemm_method_spec

    if precision == "low":
        # 'low' best is 'direct' for the Gate/Up shape; synthesize a spec since
        # gemm_method_spec only knows the high-prec external methods.
        from kernel_registry.registry_lookup import gemm_config

        cfg = gemm_config(m, k, n, "bf16", "low")
        assert cfg["method"] == "direct", (
            f"expected direct for low-prec {m}x{k}x{n}, got {cfg['method']}"
        )
        tile = cfg["tile"]
        return {
            "method": "direct",
            "tile_m": tile["tile_m"],
            "tile_k_l2": tile["tile_k_l2"],
            "tile_k_l1": tile["tile_k_l1"],
            "tile_n": tile["tile_n"],
            "n_launches": 1,
            "needs_f32_scratch": False,
            "sym_suffix": "",
            "build_kwargs": {},
        }
    spec = gemm_registry_config(m, k, n, "bf16", "high")
    return spec


def _build_gemm_ir(m, k, n, spec, herd_m=8, herd_n=4):
    """Build the lowered IR for one GEMM by its method spec."""
    method = spec["method"]
    tm, k2, k1, tn = spec["tile_m"], spec["tile_k_l2"], spec["tile_k_l1"], spec["tile_n"]
    if method == "direct":
        from matrix_multiplication.bf16_in_bf16_out.run import build_module_lowered

        return str(
            build_module_lowered(
                m, k, n, tm, k2, k1, tn, herd_m, herd_n, bfloat16, bfloat16, arch="aie2p"
            )
        )
    from shared.builders.gemm_builder import _build_gemm_module

    return str(
        _build_gemm_module(
            m, k, n, tm, k2, k1, tn, herd_m, herd_n, **dict(spec["build_kwargs"])
        )
    )


def _gemm_externs(spec):
    """Extern symbols a GEMM slice contributes (empty for direct-codegen)."""
    method = spec["method"]
    if method == "direct":
        return set()  # fully lowered, no external mm.o call
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


# ---------------------------------------------------------------------------
# RMSNorm + Q/K/V GEMM padding helper (shared by the fused builder).
# ---------------------------------------------------------------------------


def _padded_qkv_dims(q_dim, kv_dim):
    """Pad Q AND K/V output N to 1024 (tile_n=128, herd_n=4: 1024/(128*4)=2).

    tile_n=32 (the registry config for N=896/128) is numerically broken for the
    drain f32-accumulate path (mean_rel_L1 ~0.2). tile_n=128 is the only correct
    tile_n but needs a 128*herd_n-aligned N. Q (896) and K/V (128) are BOTH padded
    to the SAME width 1024 so their drain `_m32` private decls (whose 6D strides
    depend on N) are byte-identical and dedup in one stitched ELF — distinct N
    would force distinct mm.o suffix copies. Host zero-pads weight columns and
    slices the GEMM output back to the real width. (Cost: K/V do extra work, but
    Phase 4 can split rms_qkv or re-sweep a correct tile_n=128 KV config.)"""
    return 1024, 1024


# ---------------------------------------------------------------------------
# Builder 1b (FUSED): RMSNorm + Q/K/V GEMM + bias-add(Q,K,V) + RoPE(Q,K).
#   One ELF replaces rms_qkv + host bias-add + rope_q + rope_k. The QKV bias is
#   moved on-device as a broadcast bias-add slice (see
#   shared/builders/rms_qkv_bias_rope_multi.py). Q/K/V GEMM N stays padded to
#   q_pad/kv_pad (tile_n=128 correctness); the bias-add reads the padded GEMM
#   output and writes un-padded (seq, *_dim) buffers that RoPE then consumes.
# ---------------------------------------------------------------------------


def build_rms_qkv_bias_rope_module(seq_len, config):
    from shared.builders.rms_qkv_bias_rope_multi import (
        build_rms_qkv_bias_rope_module as _build,
    )

    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    q_pad, kv_pad = _padded_qkv_dims(q_dim, kv_dim)
    return _build(
        seq_len,
        config.emb_dim,
        q_dim,
        kv_dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        q_pad=q_pad,
        kv_pad=kv_pad,
    )


# ---------------------------------------------------------------------------
# Builder 3: O proj (SQUARE) + Residual + FFN, SPLIT into TWO ELFs.
#
#   Why split? The Down GEMM (K=hidden=4864, NON-1024-aligned) produces ALL-NaN
#   when it is NOT the first launch of a stitched ELF (confirmed by bisect: Down
#   as launch 0 → correct; Down as launch >=1 after ANY prior GEMM → all NaN,
#   independent of Down's method (drain or fused-cast) or the prior GEMM's method
#   (drain/direct/fused-cast)). This is a multi-launch resource collision tied to
#   the non-1024-aligned K=4864 reduction consuming/contending BD/L2 state left
#   by a preceding launch. Qwen3's o_ffn did not hit it because hidden=3072 is
#   1024-aligned. Documented remedy (debug-multi-launch-merge H1/H3): keep the
#   offending GEMM as its own XRT call so it is always launch 0.
#
#   ELF A  o_ffn_head : O(drain) + Residual + RMSNorm + Gate(direct) + Up(direct)
#                       + SwiGLU.  Outputs swiglu(seq,hidden) AND res1(seq,emb).
#   ELF B  down_add   : Down(fused-cast, launch 0) + FFN-Add.  Output (seq*emb,).
# ---------------------------------------------------------------------------


def _build_ffn_add_2d_to_1d_ir(seq_len, emb_dim):
    """Eltwise add (2D,2D)->1D, used as the FFN residual tail in down_add."""
    from air.ir import MemRefType, IntegerAttr, AffineMap, AffineExpr
    from air.ir import AffineSymbolExpr, AffineConstantExpr, AffineMapAttr, VectorType
    from air.dialects.air import module_builder, launch, segment, herd, dma_memcpy_nd
    from air.dialects.air import MemorySpace, T
    from air.dialects.affine import apply as affine_apply
    from air.dialects import arith
    from air.dialects.memref import AllocOp, DeallocOp, subview
    from air.dialects.vector import transfer_read, transfer_write
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_ as range_, yield_
    from air.backend.xrt_runner import type_mapper

    n_total = seq_len * emb_dim

    @module_builder
    def _build():
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
                        0, 3,
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
                            dma_memcpy_nd(l1_a, h_a, src_offsets=[offset], src_sizes=[tile_n], src_strides=[1])
                            dma_memcpy_nd(l1_b, h_b, src_offsets=[offset], src_sizes=[tile_n], src_strides=[1])
                            for j in range_(0, tile_n, 16):
                                sub_a = subview(l1_a.result, [j], [16], [1])
                                sub_b = subview(l1_b.result, [j], [16], [1])
                                sub_out = subview(l1_out.result, [j], [16], [1])
                                v_a = transfer_read(vec_ty, sub_a, [c0], identity_map, cst0, [True])
                                v_b = transfer_read(vec_ty, sub_b, [c0], identity_map, cst0, [True])
                                v_sum = arith.addf(v_a, v_b)
                                transfer_write(None, v_sum, sub_out, [c0], identity_map, [True])
                                yield_([])
                            dma_memcpy_nd(h_out, l1_out, dst_offsets=[offset], dst_sizes=[tile_n], dst_strides=[1])
                            yield_([])
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_out)

    return str(_build())


def _build_padded_residual_add_2d_ir(seq_len, emb_dim, n_pad):
    """Residual add: proj(seq,n_pad)[:, :emb] + x_resid(seq,emb) -> res1(seq,emb) 2D.

    `proj` is the O GEMM output widened to n_pad; only its first emb columns are
    read. res1 is kept 2D so downstream slices read it without expand_shape.
    """
    from air.ir import MemRefType, IntegerAttr, AffineMap, AffineExpr
    from air.ir import AffineSymbolExpr, AffineConstantExpr, AffineMapAttr, VectorType
    from air.dialects.air import module_builder, launch, segment, herd, dma_memcpy_nd
    from air.dialects.air import MemorySpace, T
    from air.dialects.affine import apply as affine_apply
    from air.dialects import arith
    from air.dialects.memref import AllocOp, DeallocOp, subview
    from air.dialects.vector import transfer_read, transfer_write
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_ as range_, yield_
    from air.backend.xrt_runner import type_mapper

    n_total = seq_len * emb_dim

    @module_builder
    def _build():
        from air.dialects.memref import collapse_shape as memref_collapse_shape

        xrt_dtype = type_mapper(bfloat16)
        proj_2d_ty = MemRefType.get([seq_len, n_pad], xrt_dtype)
        res_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        flat_ty = MemRefType.get([n_total], xrt_dtype)
        rows_per_pe = seq_len // 8
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([emb_dim], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(proj_2d_ty, res_2d_ty, res_2d_ty)
        def res_add(proj_2d, xres_2d, out_2d):
            @launch(operands=[proj_2d, xres_2d, out_2d])
            def add_launch(l_proj, l_xres, l_out):
                xres_flat = memref_collapse_shape(flat_ty, l_xres, [[0, 1]])
                out_flat = memref_collapse_shape(flat_ty, l_out, [[0, 1]])

                @segment(name="radd_seg", operands=[l_proj, xres_flat, out_flat])
                def add_seg(s_proj, s_xres, s_out):
                    row_map = AffineMap.get(
                        0, 2,
                        [AffineExpr.get_add(
                            AffineExpr.get_mul(AffineSymbolExpr.get(0),
                                               AffineConstantExpr.get(rows_per_pe)),
                            AffineSymbolExpr.get(1))],
                    )

                    @herd(name="radd_herd", sizes=[8, 1], operands=[s_proj, s_xres, s_out])
                    def add_body(_tx, _ty, _sx, _sy, h_proj, h_xres, h_out):
                        l1_p = AllocOp(l1_ty, [], [])
                        l1_x = AllocOp(l1_ty, [], [])
                        l1_o = AllocOp(l1_ty, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        for iv in range_(0, rows_per_pe, 1):
                            r = affine_apply(row_map, [_tx, iv])
                            dma_memcpy_nd(l1_p, h_proj, src_offsets=[r, c0],
                                          src_sizes=[1, emb_dim], src_strides=[n_pad, 1])
                            off = arith.muli(r, arith.ConstantOp.create_index(emb_dim))
                            dma_memcpy_nd(l1_x, h_xres, src_offsets=[off], src_sizes=[emb_dim], src_strides=[1])
                            for j in range_(0, emb_dim, 16):
                                sp = subview(l1_p.result, [j], [16], [1])
                                sx = subview(l1_x.result, [j], [16], [1])
                                so = subview(l1_o.result, [j], [16], [1])
                                vp = transfer_read(vec_ty, sp, [c0], identity_map, cst0, [True])
                                vx = transfer_read(vec_ty, sx, [c0], identity_map, cst0, [True])
                                transfer_write(None, arith.addf(vp, vx), so, [c0], identity_map, [True])
                                yield_([])
                            dma_memcpy_nd(h_out, l1_o, dst_offsets=[off], dst_sizes=[emb_dim], dst_strides=[1])
                            yield_([])
                        DeallocOp(l1_p)
                        DeallocOp(l1_x)
                        DeallocOp(l1_o)

    return str(_build())


def build_o_ffn_head_module(
    seq_len, emb_dim, hidden_dim,
    o_herd_m=8, o_herd_n=4, gate_herd_m=8, gate_herd_n=4,
    swiglu_tile_n=4864, swiglu_herd_x=8, swiglu_herd_y=1,
):
    """ELF A: O(drain) + Residual + RMSNorm + Gate(direct) + Up(direct) + SwiGLU.

    Func args:
      %arg0 attn_out (seq,emb)  %arg1 wo (emb,emb)  %arg2 proj (seq,emb)
      %arg3 x_resid (seq,emb)   %arg4 res1 (seq,emb)  <- OUTPUT (feeds down_add)
      %arg5 ffn_norm (emb,)     %arg6 normed2 (seq,emb)
      %arg7 w_gate (emb,hid)    %arg8 gate (seq,hid)
      %arg9 w_up (emb,hid)      %arg10 up (seq,hid)
      %arg11 swiglu (seq,hid)   <- OUTPUT (feeds down_add)
      [+ f32 C-scratch tail for any fused-cast GEMM — none here: O=drain,
       Gate/Up=direct]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch, stitch_elf, KernelSlice, FuncArg, alloc_gemm_scratch,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from silu_and_mul.silu_and_mul import build_module_2d as build_swiglu

    # O GEMM N is padded to 1024 (same tile_n=32-broken reason as Down): proj is
    # (seq, n_pad); the residual add reads only the first emb columns.
    n_pad = _padded_n_for_down(emb_dim)
    o_spec = _gemm_spec(seq_len, emb_dim, emb_dim, "high")  # method=drain
    o_spec = dict(o_spec)
    o_spec["tile_n"] = 128
    g_spec = _gemm_spec(seq_len, emb_dim, hidden_dim, "low")
    print(f"  [head] GEMM methods: O={o_spec['method']} Gate/Up={g_spec['method']}")

    print(f"  [1/6] O GEMM ({o_spec['method']})  {seq_len}x{emb_dim}x{n_pad} "
          f"(N padded from {emb_dim}, tile_n=128)...")
    o_ir = _build_gemm_ir(seq_len, emb_dim, n_pad, o_spec, o_herd_m, o_herd_n)
    print("  [2/6] Residual Add (padded proj)...")
    res_add_ir = _build_padded_residual_add_2d_ir(seq_len, emb_dim, n_pad)
    print("  [3/6] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)))
    print(f"  [4/6] Gate GEMM ({g_spec['method']})  {seq_len}x{emb_dim}x{hidden_dim}...")
    gate_ir = _build_gemm_ir(seq_len, emb_dim, hidden_dim, g_spec, gate_herd_m, gate_herd_n)
    print(f"  [5/6] Up GEMM ({g_spec['method']})...")
    up_ir = _build_gemm_ir(seq_len, emb_dim, hidden_dim, g_spec, gate_herd_m, gate_herd_n)
    print("  [6/6] SwiGLU...")
    swiglu_ir = _wrap_ir_in_launch(
        str(build_swiglu(seq_len, hidden_dim, swiglu_tile_n, bfloat16, swiglu_herd_x, swiglu_herd_y))
    )

    scratch_args, scratch_for = alloc_gemm_scratch(
        [(o_spec, seq_len, n_pad), (g_spec, seq_len, hidden_dim), (g_spec, seq_len, hidden_dim)],
        base_arg_count=12,
    )

    def _amap(i, w, o, sc):
        return {0: i, 1: w, 2: sc, 3: o} if sc is not None else {0: i, 1: w, 2: o}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),   # attn_out
        FuncArg("%arg1", f"memref<{emb_dim}x{n_pad}xbf16>"),     # wo (padded N)
        FuncArg("%arg2", f"memref<{seq_len}x{n_pad}xbf16>"),     # proj (padded N)
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),   # x_resid
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),   # res1 (out)
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),             # ffn_norm
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),   # normed2
        FuncArg("%arg7", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # w_gate
        FuncArg("%arg8", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # gate
        FuncArg("%arg9", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # w_up
        FuncArg("%arg10", f"memref<{seq_len}x{hidden_dim}xbf16>"), # up
        FuncArg("%arg11", f"memref<{seq_len}x{hidden_dim}xbf16>"), # swiglu (out)
    ]
    slices = [
        KernelSlice(o_ir, "og", _amap(0, 1, 2, scratch_for[0]), extern_syms=_gemm_externs(o_spec)),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(gate_ir, "gg", _amap(6, 7, 8, scratch_for[1]), extern_syms=_gemm_externs(g_spec), private_from=False),
        KernelSlice(up_ir, "ug", _amap(6, 9, 10, scratch_for[2]), extern_syms=_gemm_externs(g_spec), private_from=False),
        KernelSlice(swiglu_ir, "sw", {0: 8, 1: 10, 2: 11}, extern_syms={"@silu_and_mul_bf16"}),
    ]
    module = stitch_elf("o_ffn_head", base_args, slices, scratch_args=scratch_args,
                        debug_dump_path="/tmp/debug_o_ffn_head.mlir")
    print(f"  o_ffn_head module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def _padded_n_for_down(emb_dim):
    """Down GEMM N must be 1024-aligned to admit tile_n=128 (the only fused-cast
    tile_n that is numerically correct — tile_n=32 truncates the f32 accumulator,
    giving mean_rel_L1 ~0.2). N=896 is padded up to 1024."""
    if emb_dim % 1024 == 0:
        return emb_dim
    return ((emb_dim + 1023) // 1024) * 1024


def _build_down_add_2d_padded_ir(seq_len, emb_dim, n_pad):
    """Eltwise add: down(seq,n_pad)[:, :emb] + res1(seq,emb) -> output(seq*emb,).

    `down` is the Down GEMM output buffer, widened to n_pad (1024-aligned) so the
    GEMM can use tile_n=128. The padded columns [emb:n_pad] are garbage/zero and
    are dropped here (we read only the first emb columns of each down row).
    """
    from air.ir import MemRefType, IntegerAttr, AffineMap, AffineExpr
    from air.ir import AffineSymbolExpr, AffineConstantExpr, AffineMapAttr, VectorType
    from air.dialects.air import module_builder, launch, segment, herd, dma_memcpy_nd
    from air.dialects.air import MemorySpace, T
    from air.dialects.affine import apply as affine_apply
    from air.dialects import arith
    from air.dialects.memref import AllocOp, DeallocOp, subview
    from air.dialects.vector import transfer_read, transfer_write
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_ as range_, yield_
    from air.backend.xrt_runner import type_mapper

    n_total = seq_len * emb_dim

    @module_builder
    def _build():
        from air.dialects.memref import collapse_shape as memref_collapse_shape

        xrt_dtype = type_mapper(bfloat16)
        down_2d_ty = MemRefType.get([seq_len, n_pad], xrt_dtype)   # padded down
        res_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)  # res1
        out_1d_ty = MemRefType.get([n_total], xrt_dtype)
        rows_per_pe = seq_len // 8
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([emb_dim], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(down_2d_ty, res_2d_ty, out_1d_ty)
        def down_add(down_2d, res_2d, out_1d):
            @launch(operands=[down_2d, res_2d, out_1d])
            def add_launch(l_down, l_res, l_out):
                res_flat = memref_collapse_shape(out_1d_ty, l_res, [[0, 1]])

                @segment(name="dadd_seg", operands=[l_down, res_flat, l_out])
                def add_seg(s_down, s_res, s_out):
                    # row index r = pe*rows_per_pe + iv ; out/res offset = r*emb ;
                    # down offset = r*n_pad (the padded row stride).
                    row_map = AffineMap.get(
                        0, 2,
                        [AffineExpr.get_add(
                            AffineExpr.get_mul(AffineSymbolExpr.get(0),
                                               AffineConstantExpr.get(rows_per_pe)),
                            AffineSymbolExpr.get(1))],
                    )

                    @herd(name="dadd_herd", sizes=[8, 1], operands=[s_down, s_res, s_out])
                    def add_body(_tx, _ty, _sx, _sy, h_down, h_res, h_out):
                        l1_d = AllocOp(l1_ty, [], [])
                        l1_r = AllocOp(l1_ty, [], [])
                        l1_o = AllocOp(l1_ty, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        for iv in range_(0, rows_per_pe, 1):
                            r = affine_apply(row_map, [_tx, iv])
                            # down: 2D (seq, n_pad) — DMA row r, first emb cols.
                            dma_memcpy_nd(
                                l1_d, h_down,
                                src_offsets=[r, c0], src_sizes=[1, emb_dim],
                                src_strides=[n_pad, 1],
                            )
                            # res1 / out: flat 1D, row offset r*emb.
                            off = arith.muli(r, arith.ConstantOp.create_index(emb_dim))
                            dma_memcpy_nd(l1_r, h_res, src_offsets=[off], src_sizes=[emb_dim], src_strides=[1])
                            for j in range_(0, emb_dim, 16):
                                sd = subview(l1_d.result, [j], [16], [1])
                                sr = subview(l1_r.result, [j], [16], [1])
                                so = subview(l1_o.result, [j], [16], [1])
                                vd = transfer_read(vec_ty, sd, [c0], identity_map, cst0, [True])
                                vr = transfer_read(vec_ty, sr, [c0], identity_map, cst0, [True])
                                transfer_write(None, arith.addf(vd, vr), so, [c0], identity_map, [True])
                                yield_([])
                            dma_memcpy_nd(h_out, l1_o, dst_offsets=[off], dst_sizes=[emb_dim], dst_strides=[1])
                            yield_([])
                        DeallocOp(l1_d)
                        DeallocOp(l1_r)
                        DeallocOp(l1_o)

    return str(_build())


def build_down_add_module(seq_len, emb_dim, hidden_dim, down_herd_m=8, down_herd_n=4):
    """ELF B: Down(fused-cast, launch 0, N-padded to 1024) + FFN-Add.

    The Down GEMM output N is padded to 1024 (``_padded_n_for_down``) so it can
    use tile_n=128 (mean_rel_L1 0.0099); tile_n=32 at N=896 is numerically broken
    (0.2). The FFN-add reads only the first emb columns of the padded down buffer.

    Func args:
      %arg0 swiglu (seq,hid)  %arg1 w_down (hid,n_pad)  %arg2 down (seq,n_pad)
      %arg3 res1 (seq,emb)    %arg4 output (seq*emb,)
      [+ f32 C-scratch tail for the fused-cast Down]
    """
    from shared.infra.stitching import (
        stitch_elf, KernelSlice, FuncArg, alloc_gemm_scratch,
    )
    n_total = seq_len * emb_dim
    n_pad = _padded_n_for_down(emb_dim)
    # Down GEMM: M=seq, K=hidden, N=n_pad (padded). tile_n=128 herd_n=4 valid:
    # 1024/(128*4)=2. Use the fused-cast method spec at the registry tiles for
    # the 1024-N shape (proven correct); force tile_n=128.
    d_spec = _gemm_spec(seq_len, hidden_dim, emb_dim, "high")  # method=fused-cast
    d_spec = dict(d_spec)
    d_spec["tile_n"] = 128
    print(f"  [down_add] Down GEMM ({d_spec['method']}) {seq_len}x{hidden_dim}x{n_pad} "
          f"(N padded from {emb_dim}, tile_n=128)...")
    down_ir = _build_gemm_ir(seq_len, hidden_dim, n_pad, d_spec, down_herd_m, down_herd_n)
    print("  [down_add] FFN Add (padded down -> 1D)...")
    ffn_add_ir = _build_down_add_2d_padded_ir(seq_len, emb_dim, n_pad)

    scratch_args, scratch_for = alloc_gemm_scratch(
        [(d_spec, seq_len, n_pad)], base_arg_count=5,
    )

    def _amap(i, w, o, sc):
        return {0: i, 1: w, 2: sc, 3: o} if sc is not None else {0: i, 1: w, 2: o}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # swiglu
        FuncArg("%arg1", f"memref<{hidden_dim}x{n_pad}xbf16>"),    # w_down (padded N)
        FuncArg("%arg2", f"memref<{seq_len}x{n_pad}xbf16>"),       # down (padded N)
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),     # res1
        FuncArg("%arg4", f"memref<{n_total}xbf16>"),              # output
    ]
    slices = [
        KernelSlice(down_ir, "dg", _amap(0, 1, 2, scratch_for[0]), extern_syms=_gemm_externs(d_spec), private_from=True),
        KernelSlice(ffn_add_ir, "fa", {0: 2, 1: 3, 2: 4}, private_from=False),
    ]
    module = stitch_elf("down_add", base_args, slices, scratch_args=scratch_args,
                        debug_dump_path="/tmp/debug_down_add.mlir")
    print(f"  down_add module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _rms_qkv_bias_rope_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_bias_rope",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _o_ffn_head_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn_head",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _down_add_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "down_add",
        "runtime_loop_tiling_sizes": [2, 2],
    }


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# Set by compile_all_kernels so the block runner knows scratch-arg indices.
_FUSED_SCRATCH_FOR = None  # rms_qkv_bias_rope (Q/K/V) — all drain → [None,None,None]
_HEAD_SCRATCH_FOR = None   # o_ffn_head (O,Gate,Up) — all drain/direct → [None,None,None]
_DOWN_SCRATCH_FOR = None   # down_add (Down) — fused-cast → [5]


def _resolve_scratch_for():
    """Recompute scratch_for lists from the registry (for --run-only path)."""
    global _FUSED_SCRATCH_FOR, _HEAD_SCRATCH_FOR, _DOWN_SCRATCH_FOR
    cfg = LlamaConfig()
    seq = 2048
    q_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim

    def _alloc(specs, base):
        nxt = base
        out = []
        for s in specs:
            if s["needs_f32_scratch"]:
                out.append(nxt); nxt += 1
            else:
                out.append(None)
        return out

    # Fused rms_qkv_bias_rope: Q/K/V all drain (padded N) → no scratch.
    _FUSED_SCRATCH_FOR = _alloc([
        _gemm_spec(seq, cfg.emb_dim, q_dim, "high"),
        _gemm_spec(seq, cfg.emb_dim, kv_dim, "high"),
        _gemm_spec(seq, cfg.emb_dim, kv_dim, "high"),
    ], 19)
    _HEAD_SCRATCH_FOR = _alloc([
        _gemm_spec(seq, cfg.emb_dim, cfg.emb_dim, "high"),
        _gemm_spec(seq, cfg.emb_dim, cfg.hidden_dim, "low"),
        _gemm_spec(seq, cfg.emb_dim, cfg.hidden_dim, "low"),
    ], 12)
    _DOWN_SCRATCH_FOR = _alloc([
        _gemm_spec(seq, cfg.hidden_dim, cfg.emb_dim, "high"),
    ], 5)
    return _HEAD_SCRATCH_FOR, _DOWN_SCRATCH_FOR


# Backend kwargs for the seq-first NPU FlashAttention ELF (head_dim=64). Mirrors
# llama32_1b's _ATTN_BACKEND_KWARGS. omit_while_true_loop=False because
# lkp==head_dim==64 (shared L1 buffers enabled).
_ATTN_BACKEND_KWARGS = {
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=True):
    global _FUSED_SCRATCH_FOR, _HEAD_SCRATCH_FOR, _DOWN_SCRATCH_FOR
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}\nCompiling Qwen2.5 prefill kernels (seq_len={seq_len})...\n{'='*60}\n")

    from shared.infra.external_kernels import compile_gemm_mm, compile_rope

    # mm.o variants for the external GEMMs (drain _m32, fused-cast _m64).
    # Gate/Up direct-codegen needs NO external .o. rope.o for head_dim=64.
    compile_gemm_mm(tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o")
    compile_gemm_mm(tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o")
    compile_rope()

    print("\n--- rms_qkv_bias_rope (FUSED: RMSNorm+QKV+bias+RoPE, 9 launches) ---")
    fused_mod, fused_scratch = build_rms_qkv_bias_rope_module(seq_len, config)
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_bias_rope", fused_mod, _rms_qkv_bias_rope_backend(verbose)
    )

    print("\n--- o_ffn_head (O proj + Residual + RMSNorm + Gate + Up + SwiGLU) ---")
    head_mod, head_scratch = build_o_ffn_head_module(seq_len, emb_dim, hidden_dim)
    _HEAD_SCRATCH_FOR = head_scratch
    cache.compile_and_cache("o_ffn_head", head_mod, _o_ffn_head_backend(verbose))

    print("\n--- down_add (Down GEMM [launch 0] + FFN Add) ---")
    down_mod, down_scratch = build_down_add_module(seq_len, emb_dim, hidden_dim)
    _DOWN_SCRATCH_FOR = down_scratch
    cache.compile_and_cache("down_add", down_mod, _down_add_backend(verbose))

    # Flash Attention GQA (seq-first NPU FA). head_dim=64 → seq-first works
    # directly, no FA-hang risk. Skipped when cpu_attn=True (CPU fallback).
    if not cpu_attn:
        print("\n--- flash_attn (seq-first NPU FlashAttention, head_dim=64) ---")
        from shared.infra.external_kernels import compile_attn_npu2

        compile_attn_npu2(head_dim=head_dim)
        from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
            build_module as build_attn,
        )

        lkp = head_dim  # 64
        lqp = 256
        enable_shared_buffers = lkp == head_dim
        cache.compile_and_cache(
            "flash_attn",
            build_attn(
                lk=seq_len,
                lkp=lkp,
                lq=seq_len,
                lqp=lqp,
                dk=head_dim,
                dv=head_dim,
                num_q_tiles=4,
                num_cascade_stages=4,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                causal=True,
            ),
            {
                "verbose": verbose,
                "omit_while_true_loop": not enable_shared_buffers,
                "omit_pingpong": "all",
                "runtime_loop_tiling_sizes": [1, 1],
                "output_format": "elf",
                "instance_name": "attention_bf16",
            },
        )
    else:
        print("  Skipping flash_attn compilation (using CPU attention fallback)")

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")


# ---------------------------------------------------------------------------
# Pre-load prefill weights into per-layer BOs (one-time warmup)
# ---------------------------------------------------------------------------


def _fused_bias_rope_call(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """Issue one rms_qkv_bias_rope ELF call (fused prefill attention-input).

    Used by BOTH preload_prefill_weights (warmup, x_in zeroed) and the block
    runner (x_in = real hidden). Single owner of the fused arg layout + index
    sets so the warmup BO set lines up exactly. Returns the load_and_run result
    tuple; output_indices=[14, 17, 18] -> v_b, q_roped, k_roped.

    Q/K/V GEMM N stays padded (q_pad/kv_pad): wq->(emb,q_pad), wk/wv->(emb,kv_pad),
    q/k/v output buffers widened. The on-device bias-add reads the padded buffers
    and writes un-padded (seq, *_dim) bias outputs that RoPE consumes; q_roped/
    k_roped/v_b come back at the real (un-padded) widths.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q_pad, kv_pad = _padded_qkv_dims(q_dim, kv_dim)

    wq_pad = np.zeros((emb_dim, q_pad), dtype=bfloat16)
    wq_pad[:, :q_dim] = np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim)
    wk_pad = np.zeros((emb_dim, kv_pad), dtype=bfloat16)
    wk_pad[:, :kv_dim] = np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim)
    wv_pad = np.zeros((emb_dim, kv_pad), dtype=bfloat16)
    wv_pad[:, :kv_dim] = np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim)

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),         # 0 x_in (dynamic)
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),          # 1 norm_w (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                       # 2 normed (inter)
        wq_pad,                                                             # 3 wq (static, padded N)
        np.zeros((seq_len, q_pad), dtype=bfloat16),                         # 4 q_pad (inter)
        wk_pad,                                                             # 5 wk (static, padded N)
        np.zeros((seq_len, kv_pad), dtype=bfloat16),                        # 6 k_pad (inter)
        wv_pad,                                                             # 7 wv (static, padded N)
        np.zeros((seq_len, kv_pad), dtype=bfloat16),                        # 8 v_pad (inter)
        np.asarray(lw.bq, dtype=bfloat16).reshape(q_dim),                  # 9 bq (static)
        np.asarray(lw.bk, dtype=bfloat16).reshape(kv_dim),                 # 10 bk (static)
        np.asarray(lw.bv, dtype=bfloat16).reshape(kv_dim),                 # 11 bv (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),                         # 12 q_b (inter)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                        # 13 k_b (inter)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                        # 14 v_b (inter/out)
        lut_q,                                                             # 15 lut_q (static)
        lut_k,                                                             # 16 lut_k (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),                         # 17 q_roped (inter/out)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                        # 18 k_roped (inter/out)
    ]
    inter = {2, 4, 6, 8, 12, 13, 14, 17, 18}
    nxt = 19
    for sc, cols in zip(_FUSED_SCRATCH_FOR or [], (q_pad, kv_pad, kv_pad)):
        if sc is not None:
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return cache.load_and_run(
        "rms_qkv_bias_rope",
        _rms_qkv_bias_rope_backend(verbose),
        *args,
        output_indices=[14, 17, 18],
        static_input_indices={1, 3, 5, 7, 9, 10, 11, 15, 16},
        intermediate_indices=inter,
        bo_key=f"rms_qkv_bias_rope_L{layer_idx}",
    )


def preload_prefill_weights(weights, config, cache, seq_len, rope_lut_bf16):
    """Pre-load all prefill block weights into per-layer BOs once.

    A warmup XRT call per layer per ELF allocates the bo_key-keyed BO set and
    performs the host->device write of the *static* weight args. During the real
    prefill pass ``static_input_indices`` then skips those weight writes.

    The warmup call layout MUST match ``run_transformer_block_qwen25`` exactly
    (same arg count, static/intermediate indices, bo_key) or the reused BO set
    would not line up at inference time. Qwen2.5 has THREE prefill ELFs:
    rms_qkv_bias_rope (fused RMSNorm+QKV+bias+RoPE), o_ffn_head, down_add (the
    O+FFN split into two ELFs because the K=4864 Down GEMM must be launch 0; see
    build_down_add_module).
    """
    if hasattr(weights, "_prefill_weights_preloaded"):
        return

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim
    n_pad = _padded_n_for_down(emb_dim)

    head_scratch = _HEAD_SCRATCH_FOR or [None, None, None]
    down_scratch = _DOWN_SCRATCH_FOR or [None]
    if (_HEAD_SCRATCH_FOR is None or _DOWN_SCRATCH_FOR is None
            or _FUSED_SCRATCH_FOR is None):
        head_scratch, down_scratch = _resolve_scratch_for()

    print("Pre-loading prefill block weights (per-layer BOs)...")
    profiler_enabled = cache.profiler.enabled
    cache.profiler.enabled = False

    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # One fused ELF: RMSNorm + Q/K/V GEMM + bias-add + RoPE.
        _fused_bias_rope_call(
            cache, lw, config, seq_len, lut_q, lut_k, layer_idx,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
        )

        # o_ffn_head (O GEMM N padded): static {1,5,7,9}.
        wo_pad = np.zeros((emb_dim, n_pad), dtype=bfloat16)
        wo_pad[:, :emb_dim] = np.asarray(lw.wo, dtype=bfloat16).reshape(emb_dim, emb_dim)
        head_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            wo_pad,
            np.zeros((seq_len, n_pad), dtype=bfloat16),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        ]
        head_inter = set()
        for sc, cols in zip(head_scratch, (n_pad, hidden_dim, hidden_dim)):
            if sc is not None:
                head_args.append(np.zeros((seq_len, cols), dtype=np.float32))
                head_inter.add(sc)
        cache.load_and_run(
            "o_ffn_head", _o_ffn_head_backend(), *head_args,
            output_indices=[4, 11],
            static_input_indices={1, 5, 7, 9},
            intermediate_indices={2, 4, 6, 8, 10, 11} | head_inter,
            bo_key=f"o_ffn_head_L{layer_idx}",
        )

        # down_add (Down GEMM N padded): static {1}.
        w_down_pad = np.zeros((hidden_dim, n_pad), dtype=bfloat16)
        w_down_pad[:, :emb_dim] = np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim)
        down_args = [
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            w_down_pad,
            np.zeros((seq_len, n_pad), dtype=bfloat16),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros(n_total, dtype=bfloat16),
        ]
        down_inter = set()
        for sc, cols in zip(down_scratch, (n_pad,)):
            if sc is not None:
                down_args.append(np.zeros((seq_len, cols), dtype=np.float32))
                down_inter.add(sc)
        cache.load_and_run(
            "down_add", _down_add_backend(), *down_args,
            output_indices=[4],
            static_input_indices={1},
            intermediate_indices={2, 4} | down_inter,
            bo_key=f"down_add_L{layer_idx}",
        )

    cache.profiler.enabled = profiler_enabled
    weights._prefill_weights_preloaded = True
    print(f"  Pre-loaded {config.n_layers} prefill layers.")


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------


def run_transformer_block_qwen25(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=True,
    verbose=False,
):
    """Run one Qwen2.5 transformer block on NPU (kernels pre-compiled in cache)."""
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim

    inter = {}

    # RoPE LUTs (seq-first, repeated per head).
    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # ---- Stages A-C (FUSED): one ELF = RMSNorm + Q/K/V GEMM + bias(Q,K,V)
    # + RoPE(Q,K). Output: v (bias-only), q_roped, k_roped. ----
    res = _fused_bias_rope_call(
        cache, layer_weights, config, seq_len, lut_q, lut_k, layer_idx,
        x_bf16, verbose=verbose,
    )
    v = res[14].reshape(seq_len, kv_dim)
    q_roped = res[17].reshape(seq_len, q_dim)
    k_roped = res[18].reshape(seq_len, kv_dim)
    inter["v"] = v
    inter["q_roped"] = q_roped
    inter["k_roped"] = k_roped

    # ---- Stage D: GQA attention ----
    # q_roped (seq,q_dim=896), k_roped (seq,kv_dim=128), v (seq,kv_dim=128) are
    # the SLICED, un-padded, seq-first bias+RoPE tensors (V is bias-only).
    if cpu_attn:
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)
    else:
        # Seq-first NPU FlashAttention (head_dim=64 → no hang risk). Inputs are
        # already seq-first contiguous: q (seq, n_heads*head_dim), k/v
        # (seq, n_kv_heads*head_dim). Output is seq-first (seq, n_heads*head_dim).
        q_attn = np.ascontiguousarray(q_roped).reshape(seq_len, q_dim)
        k_attn = np.ascontiguousarray(k_roped).reshape(seq_len, kv_dim)
        v_attn = np.ascontiguousarray(v).reshape(seq_len, kv_dim)
        attn_output = np.zeros((seq_len, q_dim), dtype=bfloat16)
        attn_res = cache.load_and_run(
            "flash_attn",
            _ATTN_BACKEND_KWARGS,
            q_attn,
            k_attn,
            v_attn,
            attn_output,
        )
        attn_out = attn_res[-1].reshape(seq_len, q_dim)
    inter["attn_out"] = attn_out

    # ---- Stage E: O proj + Residual + FFN ----
    head_scratch = _HEAD_SCRATCH_FOR
    down_scratch = _DOWN_SCRATCH_FOR
    if head_scratch is None or down_scratch is None:
        head_scratch, down_scratch = _resolve_scratch_for()

    # ELF A: o_ffn_head → swiglu (arg11) + res1 (arg4). O GEMM N is padded to
    # 1024; wo columns zero-padded emb->n_pad, proj buffer widened to n_pad.
    n_pad = _padded_n_for_down(emb_dim)
    wo_pad = np.zeros((emb_dim, n_pad), dtype=bfloat16)
    wo_pad[:, :emb_dim] = np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim)
    head_args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim),       # 0 attn_out
        wo_pad,                                                               # 1 wo (static, padded N)
        np.zeros((seq_len, n_pad), dtype=bfloat16),                            # 2 proj (inter, padded N)
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),          # 3 x_resid
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 4 res1 (out)
        np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),   # 5 ffn_norm (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 6 normed2 (inter)
        np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 7 w_gate (static)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 8 gate (inter)
        np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),    # 9 w_up (static)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 10 up (inter)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 11 swiglu (out)
    ]
    head_inter = set()
    for sc, cols in zip(head_scratch, (n_pad, hidden_dim, hidden_dim)):
        if sc is not None:
            head_args.append(np.zeros((seq_len, cols), dtype=np.float32))
            head_inter.add(sc)
    head_res = cache.load_and_run(
        "o_ffn_head",
        _o_ffn_head_backend(verbose),
        *head_args,
        output_indices=[4, 11],
        static_input_indices={1, 5, 7, 9},
        intermediate_indices={2, 4, 6, 8, 10, 11} | head_inter,
        bo_key=f"o_ffn_head_L{layer_idx}",
    )
    res1 = head_res[4].reshape(seq_len, emb_dim)
    swiglu = head_res[11].reshape(seq_len, hidden_dim)

    # ELF B: down_add → output (arg4). Down GEMM N is padded to 1024 (the
    # only fused-cast tile_n that is numerically correct). w_down columns are
    # zero-padded emb->n_pad; the FFN-add drops the padded columns.
    n_pad = _padded_n_for_down(emb_dim)
    w_down_pad = np.zeros((hidden_dim, n_pad), dtype=bfloat16)
    w_down_pad[:, :emb_dim] = np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim)
    down_args = [
        np.asarray(swiglu, dtype=bfloat16).reshape(seq_len, hidden_dim),       # 0 swiglu
        w_down_pad,                                                            # 1 w_down (static, padded N)
        np.zeros((seq_len, n_pad), dtype=bfloat16),                            # 2 down (inter, padded N)
        np.asarray(res1, dtype=bfloat16).reshape(seq_len, emb_dim),            # 3 res1
        np.zeros(n_total, dtype=bfloat16),                                    # 4 output (out)
    ]
    down_inter = set()
    for sc, cols in zip(down_scratch, (n_pad,)):
        if sc is not None:
            down_args.append(np.zeros((seq_len, cols), dtype=np.float32))
            down_inter.add(sc)
    down_res = cache.load_and_run(
        "down_add",
        _down_add_backend(verbose),
        *down_args,
        output_indices=[4],
        static_input_indices={1},
        intermediate_indices={2, 4} | down_inter,
        bo_key=f"down_add_L{layer_idx}",
    )
    output_bf16 = down_res[4].reshape(seq_len, emb_dim)
    inter["ffn_out"] = output_bf16
    return output_bf16, inter
