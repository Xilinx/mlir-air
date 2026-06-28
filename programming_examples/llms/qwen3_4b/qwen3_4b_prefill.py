# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-4B Prefill on MLIR-AIR (NPU2).

Direct re-parameterization of qwen3_0_6b (SAME architecture: Qwen3 QK-norm
host-side, NO bias, eps=1e-6, tied embeddings, head_dim=128, DECOUPLED O
proj where q_dim != emb_dim). Only the dims grow:

    qwen3_0_6b : emb=1024, q_dim=2048, kv_dim=1024, hidden=3072, 28 layers
    qwen3_4b   : emb=2560, q_dim=4096, kv_dim=1024, hidden=9728, 36 layers

The two Qwen3 deltas vs LLAMA-3.2 (handled exactly as in qwen3_0_6b):

  1. QK-norm: per-head RMSNorm over head_dim on Q and K AFTER the projection
     GEMM and BEFORE RoPE. RoPE's linearity does not commute the nonlinear
     QK-norm past it, so we build a Qwen-specific 8-launch ELF that does
     RMSNorm + Q/K/V GEMM + per-head QK-norm(Q,K) + RoPE(Q,K) all on the NPU
     (rms_qkv_qknorm_rope).

  2. Decoupled head_dim: n_heads*head_dim = 4096 != hidden_size = 2560.
        q_proj : 2560 -> 4096   (32 heads x 128)
        k/v    : 2560 -> 1024   (8 heads x 128)
        o_proj : 4096 -> 2560   (NOT square)
     We build a Qwen-specific o_ffn ELF whose O GEMM is K=q_dim=4096,
     N=emb_dim=2560; the residual/RMSNorm/FFN tail stays emb_dim=2560.

ALIGNMENT + the Gate/Up un-merge (Phase-1 NPU finding):
Every GEMM N and K is divisible by 512 (emb=2560=512x5, q_dim=4096=512x8,
kv_dim=1024, hidden=9728=512x19). Phase-1 standalone GEMM sweep on NPU2:
    Q/K/V/O/Down -> fused-cast high (mm_m64.o, tile_m=64, tk_l2=256, tn=128) PASS
    Gate/Up (2048x2560x9728) -> high-precision fused-cast FAILS AT COMPILE TIME
        ("aie.dma_bd op Stride 2 exceeds the [1:1048576] range" on the
        f32-out B-tile DMA at N=9728). Only the DIRECT low-precision path
        (tile_m=64, tk_l2=128, tk_l1=32, tn=64) compiles + PASSES (max_abs
        2.9e-3, within the bf16 low-prec tolerance).

Because Gate/Up cannot be fused-cast, the merged qwen3_0_6b 8-launch o_ffn
ELF cannot compile here. We adopt the qwen25_3b un-merge: the O+FFN tail is
FIVE pieces (matching the N=11008 case), with the O GEMM kept DECOUPLED:
    o_res_norm : O(fused-cast, K=q_dim=4096 N=emb=2560) + Residual + FFN-RMSNorm
    gate       : Gate(direct low-prec) alone
    up         : Up(direct low-prec) alone
    swiglu     : NPU SwiGLU silu(gate)*up   (standalone ELF, tile_n tuned to
                  fit both L1 and the BD pool at hidden=9728)
    down_add   : Down(fused-cast, launch 0) + FFN-Add

Attention uses the CPU fallback (cpu_attn=True), matching llama / qwen3_0_6b
(head_dim=128 -> NPU FlashAttention hang risk).
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

from qwen3_4b_weights import LlamaConfig
from qwen3_4b_cpu_helpers import attention_reference


# ---------------------------------------------------------------------------
# GEMM method/tile spec. The kernel registry has no emb=2560 entries yet, so
# this picks the method + tiles directly (validated by the Phase-1 standalone
# GEMM sweep on NPU2). All qwen3_4b GEMMs are 512-aligned -> high-precision
# fused-cast (mm_m64.o, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=128).
# Signature mirrors gemm_registry_config() so the qwen3_0_6b builders transfer
# verbatim (same keys: method, tile_m/k_l2/k_l1/n, needs_f32_scratch,
# sym_suffix, build_kwargs).
# ---------------------------------------------------------------------------

_FUSED = {
    "method": "fused-cast",
    "tile_m": 64,
    "tile_k_l2": 256,
    "tile_k_l1": 32,
    "tile_n": 128,
    "n_launches": 2,
    "needs_f32_scratch": True,
    "sym_suffix": "_m64",
    "build_kwargs": {
        "external_fused_cast": True,
        "sym_suffix": "_m64",
        "link_with_name": "mm_m64.o",
    },
}

# Gate/Up (N=9728): fused-cast fails the AIE DMA-stride limit at compile time,
# so they use the direct-codegen low-precision GEMM (no external mm.o, no f32
# scratch — fully lowered in the example builder). Phase-1 verified PASS.
_DIRECT = {
    "method": "direct",
    "tile_m": 64,
    "tile_k_l2": 128,
    "tile_k_l1": 32,
    "tile_n": 64,
    "n_launches": 1,
    "needs_f32_scratch": False,
    "sym_suffix": "",
    "build_kwargs": {},
}


def gemm_spec(m, k, n, precision="high"):
    """Per-GEMM build recipe for one qwen3_4b shape.

    'high' -> fused-cast (Q/K/V/O/Down, all PASS the high-prec path).
    'low'  -> direct low-precision (Gate/Up at N=9728, the only shape whose
              fused-cast path overflows the AIE DMA stride at compile time)."""
    if precision == "low":
        return dict(_DIRECT)
    return dict(_FUSED)


def _build_gemm_ir(m, k, n, spec, herd_m=8, herd_n=4):
    """Build lowered IR for one GEMM by its method spec.

    direct -> fully-lowered example builder (no external mm.o); fused-cast ->
    the shared _build_gemm_module (links mm_m64.o)."""
    tm, k2, k1, tn = spec["tile_m"], spec["tile_k_l2"], spec["tile_k_l1"], spec["tile_n"]
    if spec["method"] == "direct":
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
    if spec["method"] == "direct":
        return set()
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


# ---------------------------------------------------------------------------
# Builder 1 (FUSED): RMSNorm + Q/K/V GEMM + per-head QK-norm(Q,K) + RoPE(Q,K).
#   8-launch ELF that does the entire attention-input stage on the NPU. See
#   shared/builders/rms_qkv_qknorm_rope_multi.py. qwen3_4b's attention-input
#   GEMM shapes (emb=2560) are NOT in the kernel registry, so we inject this
#   model's gemm_spec (all Q/K/V are 512-aligned fused-cast _m64, validated by
#   the Phase-1 standalone sweep) via gemm_spec_fn.
# ---------------------------------------------------------------------------


def build_rms_qkv_qknorm_rope_module(seq_len, config):
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_module as _build,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    return _build(
        seq_len, emb_dim, q_dim, kv_dim, n_heads, n_kv_heads, head_dim,
        qknorm_eps=1e-6,
        gemm_spec_fn=gemm_spec,
    )


# ---------------------------------------------------------------------------
# Builder 2: O proj (DECOUPLED) + Residual + FFN, SPLIT into FIVE pieces.
#   Gate/Up must be direct low-prec (fused-cast overflows the AIE DMA stride at
#   N=9728), so the merged 8-launch o_ffn won't compile. We adopt the qwen25_3b
#   un-merge, keeping the O GEMM DECOUPLED (K=q_dim=4096, N=emb=2560):
#     o_res_norm : O(fused-cast decoupled) + Residual + FFN-RMSNorm
#     gate       : Gate(direct) alone
#     up         : Up(direct)   alone
#     HOST SwiGLU: silu(gate)*up
#     down_add   : Down(fused-cast, launch 0) + FFN-Add
# ---------------------------------------------------------------------------


def _build_residual_add_2d_ir(seq_len, emb_dim):
    """Residual add: proj(seq,emb) + x_resid(seq,emb) -> res1(seq,emb) 2D."""
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
        twod_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        flat_ty = MemRefType.get([n_total], xrt_dtype)
        rows_per_pe = seq_len // 8
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([emb_dim], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(twod_ty, twod_ty, twod_ty)
        def res_add(proj_2d, xres_2d, out_2d):
            @launch(operands=[proj_2d, xres_2d, out_2d])
            def add_launch(l_proj, l_xres, l_out):
                proj_flat = memref_collapse_shape(flat_ty, l_proj, [[0, 1]])
                xres_flat = memref_collapse_shape(flat_ty, l_xres, [[0, 1]])
                out_flat = memref_collapse_shape(flat_ty, l_out, [[0, 1]])

                @segment(name="radd_seg", operands=[proj_flat, xres_flat, out_flat])
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
                            off = arith.muli(r, arith.ConstantOp.create_index(emb_dim))
                            dma_memcpy_nd(l1_p, h_proj, src_offsets=[off], src_sizes=[emb_dim], src_strides=[1])
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


def build_o_res_norm_module(seq_len, emb_dim, q_dim, o_herd_m=8, o_herd_n=4):
    """ELF A1: O(fused-cast, DECOUPLED K=q_dim N=emb) + Residual + FFN-RMSNorm.

    Func args:
      %arg0 attn_out (seq,q_dim)  <- DECOUPLED
      %arg1 wo (q_dim,emb)        <- DECOUPLED
      %arg2 proj (seq,emb)
      %arg3 x_resid (seq,emb)     %arg4 res1 (seq,emb)   <- OUTPUT (feeds down_add)
      %arg5 ffn_norm (emb,)       %arg6 normed2 (seq,emb) <- OUTPUT (feeds gate/up)
      [+ f32 C-scratch tail for the fused-cast O GEMM]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch, stitch_elf, KernelSlice, FuncArg, alloc_gemm_scratch,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    o_spec = gemm_spec(seq_len, q_dim, emb_dim, "high")   # fused-cast _m64
    print(f"  [1/3] O GEMM ({o_spec['method']})  {seq_len}x{q_dim}x{emb_dim} (DECOUPLED)...")
    o_ir = _build_gemm_ir(seq_len, q_dim, emb_dim, o_spec, o_herd_m, o_herd_n)
    print("  [2/3] Residual Add...")
    res_add_ir = _build_residual_add_2d_ir(seq_len, emb_dim)
    print("  [3/3] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)))

    scratch_args, scratch_for = alloc_gemm_scratch(
        [(o_spec, seq_len, emb_dim)], base_arg_count=7,
    )

    def _amap(i, w, o, sc):
        return {0: i, 1: w, 2: sc, 3: o} if sc is not None else {0: i, 1: w, 2: o}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{q_dim}xbf16>"),     # attn_out (DECOUPLED)
        FuncArg("%arg1", f"memref<{q_dim}x{emb_dim}xbf16>"),     # wo (DECOUPLED)
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),   # proj
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),   # x_resid
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),   # res1 (out)
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),             # ffn_norm
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),   # normed2 (out)
    ]
    slices = [
        KernelSlice(o_ir, "og", _amap(0, 1, 2, scratch_for[0]), extern_syms=_gemm_externs(o_spec)),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
    ]
    module = stitch_elf("o_res_norm", base_args, slices, scratch_args=scratch_args,
                        debug_dump_path="/tmp/debug_o_res_norm_4b.mlir")
    print(f"  o_res_norm module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def build_gate_module(seq_len, emb_dim, hidden_dim, gate_herd_m=8, gate_herd_n=4):
    """ELF A2: Gate(direct low-prec) alone -> gate (seq,hidden)."""
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    g_spec = gemm_spec(seq_len, emb_dim, hidden_dim, "low")  # direct
    print(f"  [gate] Gate GEMM ({g_spec['method']}) {seq_len}x{emb_dim}x{hidden_dim} (tn={g_spec['tile_n']})...")
    gate_ir = _build_gemm_ir(seq_len, emb_dim, hidden_dim, g_spec, gate_herd_m, gate_herd_n)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),     # normed2
        FuncArg("%arg1", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # w_gate
        FuncArg("%arg2", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # gate (out)
    ]
    slices = [KernelSlice(gate_ir, "gg", {0: 0, 1: 1, 2: 2}, extern_syms=_gemm_externs(g_spec))]
    module = stitch_elf("gate", base_args, slices)
    print(f"  gate module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def build_up_module(seq_len, emb_dim, hidden_dim, up_herd_m=8, up_herd_n=4):
    """ELF A3: Up(direct low-prec) alone -> up (seq,hidden)."""
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    g_spec = gemm_spec(seq_len, emb_dim, hidden_dim, "low")  # direct
    print(f"  [up] Up GEMM ({g_spec['method']}) {seq_len}x{emb_dim}x{hidden_dim} (tn={g_spec['tile_n']})...")
    up_ir = _build_gemm_ir(seq_len, emb_dim, hidden_dim, g_spec, up_herd_m, up_herd_n)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),     # normed2
        FuncArg("%arg1", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # w_up
        FuncArg("%arg2", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # up (out)
    ]
    slices = [KernelSlice(up_ir, "ug", {0: 0, 1: 1, 2: 2}, extern_syms=_gemm_externs(g_spec))]
    module = stitch_elf("up", base_args, slices)
    print(f"  up module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# SwiGLU runs on the NPU as a standalone ELF (silu_and_mul.build_module_2d).
# It compiles at hidden=9728 once tile_n is in the BD/L1 sweet spot (band
# ~[3584, 5120]; too large overflows L1, too small exhausts the BD pool).
# Verified standalone on NPU2: mean_rel_L1=1.02e-2 (registry SwiGLU tier).

# Largest compiling tile_n per hidden (fewest loop iters), from a compile sweep.
_SWIGLU_TILE_N = {8960: 5120, 9728: 4864, 11008: 4096}


def _swiglu_tile_n(seq_len, hidden_dim, herd_x=8):
    if hidden_dim in _SWIGLU_TILE_N:
        return _SWIGLU_TILE_N[hidden_dim]
    half = (seq_len * hidden_dim) // herd_x
    for t in range(5120, 1024, -64):
        if half % t == 0:
            return t
    raise RuntimeError(f"No SwiGLU tile_n for seq={seq_len} hidden={hidden_dim}")


def build_swiglu_module(seq_len, hidden_dim, herd_x=8, herd_y=1):
    """Standalone NPU SwiGLU ELF: silu(gate)*up -> swiglu (seq x hidden)."""
    from silu_and_mul.silu_and_mul import build_module_2d as build_swiglu
    tile_n = _swiglu_tile_n(seq_len, hidden_dim, herd_x)
    print(f"  [swiglu] SwiGLU {seq_len}x{hidden_dim} (tile_n={tile_n}, "
          f"iters={seq_len * hidden_dim // (tile_n * herd_x)})...")
    module = build_swiglu(seq_len, hidden_dim, tile_n, bfloat16, herd_x, herd_y)
    print(f"  swiglu module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def _build_down_add_2d_to_1d_ir(seq_len, emb_dim):
    """Eltwise add: down(seq,emb) + res1(seq,emb) -> output(seq*emb,)."""
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
        twod_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        out_1d_ty = MemRefType.get([n_total], xrt_dtype)
        total_tiles = 8
        chunk_size = n_total // total_tiles
        tile_n = emb_dim
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(twod_ty, twod_ty, out_1d_ty)
        def down_add(down_2d, res_2d, out_1d):
            @launch(operands=[down_2d, res_2d, out_1d])
            def add_launch(l_down, l_res, l_out):
                down_flat = memref_collapse_shape(out_1d_ty, l_down, [[0, 1]])
                res_flat = memref_collapse_shape(out_1d_ty, l_res, [[0, 1]])

                @segment(name="dadd_seg", operands=[down_flat, res_flat, l_out])
                def add_seg(s_down, s_res, s_out):
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

                    @herd(name="dadd_herd", sizes=[8, 1], operands=[s_down, s_res, s_out])
                    def add_body(_tx, _ty, _sx, _sy, h_down, h_res, h_out):
                        l1_d = AllocOp(l1_ty, [], [])
                        l1_r = AllocOp(l1_ty, [], [])
                        l1_o = AllocOp(l1_ty, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        for loop_iv in range_(0, chunk_size, tile_n):
                            offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                            dma_memcpy_nd(l1_d, h_down, src_offsets=[offset], src_sizes=[tile_n], src_strides=[1])
                            dma_memcpy_nd(l1_r, h_res, src_offsets=[offset], src_sizes=[tile_n], src_strides=[1])
                            for j in range_(0, tile_n, 16):
                                sd = subview(l1_d.result, [j], [16], [1])
                                sr = subview(l1_r.result, [j], [16], [1])
                                so = subview(l1_o.result, [j], [16], [1])
                                vd = transfer_read(vec_ty, sd, [c0], identity_map, cst0, [True])
                                vr = transfer_read(vec_ty, sr, [c0], identity_map, cst0, [True])
                                transfer_write(None, arith.addf(vd, vr), so, [c0], identity_map, [True])
                                yield_([])
                            dma_memcpy_nd(h_out, l1_o, dst_offsets=[offset], dst_sizes=[tile_n], dst_strides=[1])
                            yield_([])
                        DeallocOp(l1_d)
                        DeallocOp(l1_r)
                        DeallocOp(l1_o)

    return str(_build())


def build_down_add_module(seq_len, emb_dim, hidden_dim, down_herd_m=8, down_herd_n=4):
    """ELF B: Down(fused-cast, launch 0) + FFN-Add.

    Func args:
      %arg0 swiglu (seq,hid)  %arg1 w_down (hid,emb)  %arg2 down (seq,emb)
      %arg3 res1 (seq,emb)    %arg4 output (seq*emb,)
      [+ f32 C-scratch tail for the fused-cast Down]
    """
    from shared.infra.stitching import (
        stitch_elf, KernelSlice, FuncArg, alloc_gemm_scratch,
    )
    n_total = seq_len * emb_dim
    d_spec = gemm_spec(seq_len, hidden_dim, emb_dim, "high")  # fused-cast _m64
    print(f"  [down_add] Down GEMM ({d_spec['method']}) {seq_len}x{hidden_dim}x{emb_dim} "
          f"(tn={d_spec['tile_n']})...")
    down_ir = _build_gemm_ir(seq_len, hidden_dim, emb_dim, d_spec, down_herd_m, down_herd_n)
    print("  [down_add] FFN Add (2D -> 1D)...")
    ffn_add_ir = _build_down_add_2d_to_1d_ir(seq_len, emb_dim)

    scratch_args, scratch_for = alloc_gemm_scratch(
        [(d_spec, seq_len, emb_dim)], base_arg_count=5,
    )

    def _amap(i, w, o, sc):
        return {0: i, 1: w, 2: sc, 3: o} if sc is not None else {0: i, 1: w, 2: o}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # swiglu
        FuncArg("%arg1", f"memref<{hidden_dim}x{emb_dim}xbf16>"),  # w_down
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),     # down
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),     # res1
        FuncArg("%arg4", f"memref<{n_total}xbf16>"),              # output
    ]
    slices = [
        KernelSlice(down_ir, "dg", _amap(0, 1, 2, scratch_for[0]), extern_syms=_gemm_externs(d_spec), private_from=True),
        KernelSlice(ffn_add_ir, "fa", {0: 2, 1: 3, 2: 4}, private_from=False),
    ]
    module = stitch_elf("down_add", base_args, slices, scratch_args=scratch_args,
                        debug_dump_path="/tmp/debug_down_add_4b.mlir")
    print(f"  down_add module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _rms_qkv_qknorm_rope_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _o_res_norm_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_res_norm",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _gate_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gate",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _up_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "up",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _swiglu_backend(verbose=False):
    # ELF symbol name must match the top func name in build_module_2d.
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "silu_and_mul_2d",
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
# Gate/Up are direct-codegen (no f32 scratch). o_res_norm (O fused-cast) -> [7];
# down_add (Down fused-cast) -> [5]; rms_qkv_qknorm_rope (Q/K/V fused-cast).
_ORES_SCRATCH_FOR = None
_DOWN_SCRATCH_FOR = None
# Scratch-arg indices for the FUSED rms_qkv_qknorm_rope ELF (Q/K/V fused-cast).
_FUSED_SCRATCH_FOR = None


def _resolve_scratch_for(config, seq_len=2048):
    """Recompute all scratch_for lists from the spec (for --run-only path)."""
    global _ORES_SCRATCH_FOR, _DOWN_SCRATCH_FOR, _FUSED_SCRATCH_FOR
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim

    def _alloc(specs, base):
        nxt = base
        out = []
        for s in specs:
            if s["needs_f32_scratch"]:
                out.append(nxt); nxt += 1
            else:
                out.append(None)
        return out

    # Fused rms_qkv_qknorm_rope ELF: Q/K/V specs, scratch tail starts at 17.
    _FUSED_SCRATCH_FOR = _alloc([
        gemm_spec(seq_len, config.emb_dim, q_dim),
        gemm_spec(seq_len, config.emb_dim, kv_dim),
        gemm_spec(seq_len, config.emb_dim, kv_dim),
    ], 17)
    _ORES_SCRATCH_FOR = _alloc([
        gemm_spec(seq_len, q_dim, config.emb_dim),
    ], 7)
    _DOWN_SCRATCH_FOR = _alloc([
        gemm_spec(seq_len, config.hidden_dim, config.emb_dim),
    ], 5)
    return _ORES_SCRATCH_FOR, _DOWN_SCRATCH_FOR


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=False):
    global _ORES_SCRATCH_FOR, _DOWN_SCRATCH_FOR, _FUSED_SCRATCH_FOR
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}\nCompiling Qwen3-4B prefill kernels (seq_len={seq_len})...\n{'='*60}\n")

    from shared.infra.external_kernels import compile_gemm_mm, compile_rope

    # mm.o variants for the external (fused-cast) GEMMs; Gate/Up direct-codegen
    # needs NO external .o. rope.o for head_dim=128.
    compile_gemm_mm(tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o")
    compile_gemm_mm(tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o")
    compile_rope()
    from shared.infra.external_kernels import compile_silu_and_mul
    compile_silu_and_mul()  # silu_and_mul.o for the standalone NPU SwiGLU ELF

    print("\n--- rms_qkv_qknorm_rope (FUSED: RMSNorm+QKV+QK-norm+RoPE, 8 launches) ---")
    fused_mod, fused_scratch = build_rms_qkv_qknorm_rope_module(seq_len, config)
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope", fused_mod, _rms_qkv_qknorm_rope_backend(verbose)
    )

    print("\n--- o_res_norm (O proj decoupled + Residual + FFN RMSNorm) ---")
    ores_mod, ores_scratch = build_o_res_norm_module(seq_len, emb_dim, q_dim)
    _ORES_SCRATCH_FOR = ores_scratch
    cache.compile_and_cache("o_res_norm", ores_mod, _o_res_norm_backend(verbose))

    print("\n--- gate (Gate GEMM, direct) ---")
    cache.compile_and_cache("gate", build_gate_module(seq_len, emb_dim, hidden_dim),
                            _gate_backend(verbose))

    print("\n--- up (Up GEMM, direct) ---")
    cache.compile_and_cache("up", build_up_module(seq_len, emb_dim, hidden_dim),
                            _up_backend(verbose))

    print("\n--- swiglu (NPU SwiGLU: silu(gate)*up) ---")
    cache.compile_and_cache("swiglu", build_swiglu_module(seq_len, hidden_dim),
                            _swiglu_backend(verbose))

    print("\n--- down_add (Down GEMM [launch 0] + FFN Add) ---")
    down_mod, down_scratch = build_down_add_module(seq_len, emb_dim, hidden_dim)
    _DOWN_SCRATCH_FOR = down_scratch
    cache.compile_and_cache("down_add", down_mod, _down_add_backend(verbose))

    # Flash Attention (head-first, head_dim=128). Skip if using CPU fallback.
    if not cpu_attn:
        print("\n--- flash_attn (head-first FA, head_dim=128) ---")
        from shared.infra.fa_headfirst import compile_headfirst_fa

        compile_headfirst_fa(cache, seq_len, n_heads, n_kv_heads, head_dim, verbose)
    else:
        print("\n--- Skipping flash_attn (CPU attention fallback) ---")

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")


# ---------------------------------------------------------------------------
# Prefill weight pre-load (BO reuse)
# ---------------------------------------------------------------------------


def _fused_qknorm_rope_call(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """Issue one rms_qkv_qknorm_rope ELF call (fused prefill attention-input).

    Single owner of the fused arg layout + index sets, shared by the preload
    warmup (x_in zeroed) and the block runner (x_in = real hidden). Returns the
    cache.load_and_run result tuple (output_indices=[8, 15, 16] -> v, q_roped,
    k_roped).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    fused_scratch = _FUSED_SCRATCH_FOR
    if fused_scratch is None:
        _resolve_scratch_for(config, seq_len)
        fused_scratch = _FUSED_SCRATCH_FOR

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),          # 0 x_in (dynamic)
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),           # 1 norm_w (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                        # 2 normed (inter)
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),           # 3 wq (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),                          # 4 q (inter)
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),          # 5 wk (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                         # 6 k (inter)
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),          # 7 wv (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                         # 8 v (inter/out)
        np.asarray(lw.q_norm, dtype=bfloat16).reshape(head_dim),            # 9 q_norm (static)
        np.asarray(lw.k_norm, dtype=bfloat16).reshape(head_dim),            # 10 k_norm (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),                          # 11 q_n (inter)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                         # 12 k_n (inter)
        lut_q,                                                              # 13 lut_q (static)
        lut_k,                                                              # 14 lut_k (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),                          # 15 q_roped (inter/out)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                         # 16 k_roped (inter/out)
    ]
    inter = {2, 4, 6, 8, 11, 12, 15, 16}
    for sc, cols in zip(fused_scratch or [], (q_dim, kv_dim, kv_dim)):
        if sc is not None:
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(sc)
    return cache.load_and_run(
        "rms_qkv_qknorm_rope",
        _rms_qkv_qknorm_rope_backend(verbose),
        *args,
        output_indices=[8, 15, 16],
        static_input_indices={1, 3, 5, 7, 9, 10, 13, 14},
        intermediate_indices=inter,
        bo_key=f"rms_qkv_qknorm_rope_L{layer_idx}",
    )


def preload_prefill_weights(weights, config, cache, seq_len, rope_lut_bf16):
    """Pre-load all prefill block weights into per-layer BOs once.

    The warmup call layout MUST match run_transformer_block_qwen3 exactly.
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

    print("Pre-loading prefill block weights (per-layer BOs)...")
    profiler_enabled = cache.profiler.enabled
    cache.profiler.enabled = False

    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    ores_scratch = _ORES_SCRATCH_FOR
    down_scratch = _DOWN_SCRATCH_FOR
    if ores_scratch is None or down_scratch is None:
        ores_scratch, down_scratch = _resolve_scratch_for(config, seq_len)

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # One fused ELF: RMSNorm + Q/K/V GEMM + per-head QK-norm + RoPE.
        _fused_qknorm_rope_call(
            cache, lw, config, seq_len, lut_q, lut_k, layer_idx,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
        )

        # o_res_norm: static {1,5}. Outputs res1 (arg4), normed2 (arg6).
        ores_args = [
            np.zeros((seq_len, q_dim), dtype=bfloat16),                           # 0 attn_out (DECOUPLED)
            np.asarray(lw.wo, dtype=bfloat16).reshape(q_dim, emb_dim),            # 1 wo (static, DECOUPLED)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                         # 2 proj (inter)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                         # 3 x_resid
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                         # 4 res1 (out)
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),            # 5 ffn_norm (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                         # 6 normed2 (out)
        ]
        ores_inter = set()
        for sc, cols in zip(ores_scratch, (emb_dim,)):
            if sc is not None:
                ores_args.append(np.zeros((seq_len, cols), dtype=np.float32))
                ores_inter.add(sc)
        cache.load_and_run(
            "o_res_norm", _o_res_norm_backend(), *ores_args,
            output_indices=[4, 6],
            static_input_indices={1, 5},
            intermediate_indices={2, 4, 6} | ores_inter,
            bo_key=f"o_res_norm_L{layer_idx}",
        )

        # gate: static {1}. Output gate (arg2).
        cache.load_and_run(
            "gate", _gate_backend(),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 0 normed2
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),    # 1 w_gate (static)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 2 gate (out)
            output_indices=[2], static_input_indices={1}, intermediate_indices={2},
            bo_key=f"gate_L{layer_idx}",
        )

        # up: static {1}. Output up (arg2).
        cache.load_and_run(
            "up", _up_backend(),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 0 normed2
            np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),      # 1 w_up (static)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 2 up (out)
            output_indices=[2], static_input_indices={1}, intermediate_indices={2},
            bo_key=f"up_L{layer_idx}",
        )

        # SwiGLU: NPU ELF (no static weights — only warms the per-layer BO set).
        cache.load_and_run(
            "swiglu", _swiglu_backend(),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            output_indices=[2],
            intermediate_indices={2},
            bo_key=f"swiglu_L{layer_idx}",
        )

        # down_add: static {1}.
        down_args = [
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 0 swiglu
            np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),    # 1 w_down (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 2 down (inter)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 3 res1
            np.zeros(n_total, dtype=bfloat16),                                    # 4 output (out)
        ]
        down_inter = set()
        for sc, cols in zip(down_scratch, (emb_dim,)):
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


def run_transformer_block_qwen3(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=True,
    verbose=False,
):
    """Run one Qwen3-4B transformer block on NPU (kernels pre-compiled)."""
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

    # ---- Stages A-C: one ELF = RMSNorm + Q/K/V GEMM + QK-norm(Q,K) + RoPE(Q,K).
    res = _fused_qknorm_rope_call(
        cache, layer_weights, config, seq_len, lut_q, lut_k, layer_idx,
        x_bf16, verbose=verbose,
    )
    v = res[8].reshape(seq_len, kv_dim)
    q_roped = res[15].reshape(seq_len, q_dim)
    k_roped = res[16].reshape(seq_len, kv_dim)
    inter["v"] = v
    inter["q_roped"] = q_roped
    inter["k_roped"] = k_roped

    # ---- Stage D: GQA attention ----
    if cpu_attn:
        # HOST cpu fallback (FP32 causal GQA reference).
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)
    else:
        # NPU head-first FlashAttention (head_dim=128). Decoupled O: q_roped is
        # [seq, q_dim=4096] (32*128), v is [seq, kv_dim=1024] (8*128); the
        # wrapper returns [seq, n_heads*head_dim=4096], which feeds the
        # decoupled O proj (4096 -> emb_dim=2560). q_roped/k_roped are
        # post-QK-norm post-RoPE seq-first; v is the raw projection seq-first.
        from shared.infra.fa_headfirst import npu_fa_headfirst

        attn_out = npu_fa_headfirst(
            cache,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(k_roped),
            np.ascontiguousarray(v),
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            verbose=verbose,
        )
    inter["attn_out"] = attn_out

    # ---- Stage E: O proj + Residual + FFN ----
    ores_scratch = _ORES_SCRATCH_FOR
    down_scratch = _DOWN_SCRATCH_FOR
    if ores_scratch is None or down_scratch is None:
        ores_scratch, down_scratch = _resolve_scratch_for(config, seq_len)

    # ELF A1: o_res_norm (O decoupled) -> res1 (arg4) + normed2 (arg6).
    ores_args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, q_dim),          # 0 attn_out (DECOUPLED)
        np.asarray(layer_weights.wo, dtype=bfloat16).reshape(q_dim, emb_dim),  # 1 wo (static, DECOUPLED)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 2 proj (inter)
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),          # 3 x_resid
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 4 res1 (out)
        np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),   # 5 ffn_norm (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 6 normed2 (out)
    ]
    ores_inter = set()
    for sc, cols in zip(ores_scratch, (emb_dim,)):
        if sc is not None:
            ores_args.append(np.zeros((seq_len, cols), dtype=np.float32))
            ores_inter.add(sc)
    ores_res = cache.load_and_run(
        "o_res_norm",
        _o_res_norm_backend(verbose),
        *ores_args,
        output_indices=[4, 6],
        static_input_indices={1, 5},
        intermediate_indices={2, 4, 6} | ores_inter,
        bo_key=f"o_res_norm_L{layer_idx}",
    )
    res1 = ores_res[4].reshape(seq_len, emb_dim)
    normed2 = ores_res[6].reshape(seq_len, emb_dim)

    # ELF A2: gate -> gate (arg2).
    gate_res = cache.load_and_run(
        "gate",
        _gate_backend(verbose),
        np.asarray(normed2, dtype=bfloat16).reshape(seq_len, emb_dim),
        np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        output_indices=[2],
        static_input_indices={1},
        intermediate_indices={2},
        bo_key=f"gate_L{layer_idx}",
    )
    gate = gate_res[2].reshape(seq_len, hidden_dim)

    # ELF A3: up -> up (arg2).
    up_res = cache.load_and_run(
        "up",
        _up_backend(verbose),
        np.asarray(normed2, dtype=bfloat16).reshape(seq_len, emb_dim),
        np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        output_indices=[2],
        static_input_indices={1},
        intermediate_indices={2},
        bo_key=f"up_L{layer_idx}",
    )
    up = up_res[2].reshape(seq_len, hidden_dim)

    # SwiGLU: NPU ELF (silu(gate)*up).
    sw_res = cache.load_and_run(
        "swiglu",
        _swiglu_backend(verbose),
        np.asarray(gate, dtype=bfloat16).reshape(seq_len, hidden_dim),   # 0 gate
        np.asarray(up, dtype=bfloat16).reshape(seq_len, hidden_dim),     # 1 up
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),                 # 2 swiglu (out)
        output_indices=[2],
        intermediate_indices={2},
        bo_key=f"swiglu_L{layer_idx}",
    )
    swiglu = sw_res[2].reshape(seq_len, hidden_dim)

    # ELF B: down_add -> output (arg4). Down GEMM is launch 0.
    down_args = [
        np.asarray(swiglu, dtype=bfloat16).reshape(seq_len, hidden_dim),       # 0 swiglu
        np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),  # 1 w_down (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 2 down (inter)
        np.asarray(res1, dtype=bfloat16).reshape(seq_len, emb_dim),            # 3 res1
        np.zeros(n_total, dtype=bfloat16),                                    # 4 output (out)
    ]
    down_inter = set()
    for sc, cols in zip(down_scratch, (emb_dim,)):
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
