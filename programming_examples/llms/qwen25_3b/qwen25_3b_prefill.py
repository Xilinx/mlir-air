# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-3B Prefill on MLIR-AIR (NPU2) — single-block (Phase 2) path.

Direct re-parameterization of qwen25_1_5b (SAME architecture family: Qwen2.5
QKV bias, NO QK-norm, eps=1e-6, tied embeddings; head_dim=128 cpu_attn prefill).
The o_ffn tail is SPLIT into FIVE NPU ELFs + 1 host op (o_res_norm, gate, up,
HOST SwiGLU, down_add) because at this hidden size the merged tail overflows L2,
and the Down GEMM (K=hidden, NON-512-aligned) must be launch 0 of its ELF — see
build_down_add_module.

Qwen2.5 deltas vs LLAMA-3.2:

  1. QKV bias (attention_bias=True, the Qwen2 family). The per-channel bias is
     fused into the rms_qkv_bias_rope ELF on-device (bq/bk/bv passed as static
     ELF args), applied AFTER the Q/K/V projection and BEFORE RoPE (HF Qwen2
     order: proj -> +bias -> RoPE(Q,K) -> attention; V is bias-added and used
     directly). NO QK-norm (that is Qwen3).

  2. Dims (3B): emb=2048, q_dim=2048 (16 heads x 128), kv_dim=256
     (2 heads x 128), hidden=11008, head_dim=128. o_proj is SQUARE
     (q_dim==emb_dim==2048).

emb=q_dim=2048 is 1024-aligned (Q/O/Down N=2048 -> stock tile_n=128 herd_n=4).
The non-aligned dim is hidden=11008=256*43 (Gate/Up N=11008 -> tile_n=64;
Down K=11008 -> tile_k_l2=256). Every GEMM is driven straight from
gemm_registry_config, with the Down-launch-0 split layered on top.

Registry-selected methods (seq=2048):
    Q/O    (2048x2048)   -> fused-cast (mm_m64.o, tile_n=128) — needs f32 scratch
    K/V    (2048x256)    -> drain      (mm_m32.o, tile_n=64)
    Gate/Up(2048x11008)  -> direct (low-precision, tile_n=64, tile_k_l2=128)
    Down   (11008x2048)  -> fused-cast (mm_m64.o, tile_n=128, launch 0)

Attention uses the CPU fallback (cpu_attn=True). head_dim=128 -> FA hang risk,
so prefill never uses NPU FlashAttention (mirrors qwen3_0_6b).
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

from qwen25_3b_weights import LlamaConfig
from qwen25_3b_cpu_helpers import attention_reference


# ---------------------------------------------------------------------------
# Generic per-GEMM spec + IR builder. Supports the three bf16-out methods the
# Qwen2.5 GEMMs resolve to (fused-cast / drain / direct), each picking its own
# registry tiles. NO padding — registry tile_n aligns with the real dims.
# ---------------------------------------------------------------------------


def _gemm_spec(m, k, n, precision):
    """Registry config for one GEMM. precision: 'high' or 'low'."""
    from shared.builders.gemm_builder import gemm_registry_config

    if precision == "low":
        # 'low' best is 'direct' for the Gate/Up shape; synthesize a spec since
        # gemm_registry_config's high tier does not know the direct method.
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
    return gemm_registry_config(m, k, n, "bf16", "high")


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
    if spec["method"] == "direct":
        return set()  # fully lowered, no external mm.o call
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


# ---------------------------------------------------------------------------
# Builder 1 (FUSED): RMSNorm + Q/K/V GEMM + bias-add(Q,K,V) + RoPE(Q,K).
#   One ELF replaces rms_qkv + host bias-add + rope_q + rope_k. NO N-padding
#   (registry tile_n aligns). Q is fused-cast (needs an f32 C-scratch); K/V drain.
# ---------------------------------------------------------------------------


def build_rms_qkv_bias_rope_module(seq_len, config):
    from shared.builders.rms_qkv_bias_rope_multi import (
        build_rms_qkv_bias_rope_module as _build,
    )

    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim

    def _spec_fn(m, k, n):
        return _gemm_spec(m, k, n, "high")

    return _build(
        seq_len,
        config.emb_dim,
        q_dim,
        kv_dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        q_pad=q_dim,
        kv_pad=kv_dim,
        gemm_spec_fn=_spec_fn,
    )


# ---------------------------------------------------------------------------
# Builder 3: O proj (SQUARE) + Residual + FFN, SPLIT into multiple ELFs.
#
#   Why split? The Down GEMM (K=hidden, NON-512-aligned) produces ALL-NaN when
#   it is NOT the first launch of a stitched ELF (the 0.5B K=4864 bug). Remedy:
#   keep the offending GEMM as its own XRT call so it is always launch 0.
#
#   The O+FFN tail is FIVE ELFs (L2/L1/BD limits; see per-builder docstrings):
#     o_res_norm : O(fused-cast) + Residual + FFN-RMSNorm -> res1, normed2
#     gate       : Gate(direct) alone -> gate
#     up         : Up(direct) alone   -> up
#     swiglu     : NPU silu(gate)*up  -> swiglu
#     down_add   : Down(fused-cast, launch 0) + FFN-Add -> output (seq*emb,)
# ---------------------------------------------------------------------------


def _build_residual_add_2d_ir(seq_len, emb_dim):
    """Residual add: proj(seq,emb) + x_resid(seq,emb) -> res1(seq,emb) 2D.

    res1 is kept 2D so downstream slices read it without expand_shape.
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


def build_o_res_norm_module(seq_len, emb_dim, o_herd_m=8, o_herd_n=4):
    """ELF A1: O(fused-cast) + Residual + FFN-RMSNorm.

    The 0.5B single o_ffn_head merge (O fused-cast + Gate/Up direct + SwiGLU)
    overflows L2 MemTiles at hidden=8960: the fused-cast O f32 C-scratch and the
    two direct Gate/Up f32 L2 staging buffers (1x4x64x128xf32 = 128 KB each, 8
    columns) cannot co-reside. So the O+FFN tail is split into o_res_norm, gate,
    up, HOST SwiGLU, down_add. This ELF holds only the fused-cast O scratch.

    Func args:
      %arg0 attn_out (seq,emb)  %arg1 wo (emb,emb)  %arg2 proj (seq,emb)
      %arg3 x_resid (seq,emb)   %arg4 res1 (seq,emb)   <- OUTPUT (feeds down_add)
      %arg5 ffn_norm (emb,)     %arg6 normed2 (seq,emb) <- OUTPUT (feeds gate_up)
      [+ f32 C-scratch tail for the fused-cast O GEMM]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch, stitch_elf, KernelSlice, FuncArg, alloc_gemm_scratch,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    o_spec = _gemm_spec(seq_len, emb_dim, emb_dim, "high")   # fused-cast _m64
    print(f"  [o_res_norm] O GEMM method={o_spec['method']} (tn={o_spec['tile_n']})")

    print(f"  [1/3] O GEMM ({o_spec['method']})  {seq_len}x{emb_dim}x{emb_dim}...")
    o_ir = _build_gemm_ir(seq_len, emb_dim, emb_dim, o_spec, o_herd_m, o_herd_n)
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
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),   # attn_out
        FuncArg("%arg1", f"memref<{emb_dim}x{emb_dim}xbf16>"),   # wo
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
                        debug_dump_path="/tmp/debug_o_res_norm_3b.mlir")
    print(f"  o_res_norm module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def build_gate_module(seq_len, emb_dim, hidden_dim, gate_herd_m=8, gate_herd_n=4):
    """ELF A2: Gate(direct) alone.

    Even isolated from O's fused-cast scratch, the TWO direct Gate/Up GEMMs
    cannot co-reside in one ELF: each direct codegen GEMM stages its full f32
    output (1x4x64x128xf32 = 128 KB per column × 8) in L2, and two of them
    overflow the MemTiles. So Gate and Up are each their own ELF; SwiGLU is fused
    onto Up. (Standalone, one direct GEMM fits — Phase-1 verified.)

    Func args:
      %arg0 normed2 (seq,emb)  %arg1 w_gate (emb,hid)  %arg2 gate (seq,hid) <- OUT
    """
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    g_spec = _gemm_spec(seq_len, emb_dim, hidden_dim, "low")  # direct
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
    """ELF A3: Up(direct) alone.

    A single direct GEMM at N=8960 already saturates the L2 MemTiles (its f32
    staging is 128 KB/column), so even Up+SwiGLU overflow — SwiGLU is its own ELF.

    Func args:
      %arg0 normed2 (seq,emb)  %arg1 w_up (emb,hid)  %arg2 up (seq,hid) <- OUT
    """
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
    g_spec = _gemm_spec(seq_len, emb_dim, hidden_dim, "low")  # direct
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


# NPU SwiGLU (silu_and_mul.build_module_2d) compiles at hidden=11008 once tile_n
# is in the BD/L1 sweet spot. Failure horns:
#   - tile_n too large (>= ~7168) -> 3 L1 buffers overflow per-core L1
#   - tile_n too small (<= ~2048)  -> the chunk loop unrolls into >1k shim DMA
#     tasks and exhausts the buffer-descriptor pool
# Compiling band tile_n in ~[3584, 5120]. Verified standalone on NPU2:
# mean_rel_L1=1.02e-2 (registry SwiGLU tier).

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
    d_spec = _gemm_spec(seq_len, hidden_dim, emb_dim, "high")  # fused-cast _m64
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
                        debug_dump_path="/tmp/debug_down_add_3b.mlir")
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
# Gate/Up are direct-codegen (bf16-out, no f32 scratch), so each is its own ELF
# with no scratch args.
_FUSED_SCRATCH_FOR = None     # rms_qkv_bias_rope (Q/K/V): Q fused-cast → [19, None, None]
_ORES_SCRATCH_FOR = None     # o_res_norm (O): fused-cast → [7]
_DOWN_SCRATCH_FOR = None     # down_add (Down): fused-cast → [5]


def _resolve_scratch_for():
    """Recompute scratch_for lists from the registry (for --run-only path)."""
    global _FUSED_SCRATCH_FOR, _ORES_SCRATCH_FOR, _DOWN_SCRATCH_FOR
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

    _FUSED_SCRATCH_FOR = _alloc([
        _gemm_spec(seq, cfg.emb_dim, q_dim, "high"),
        _gemm_spec(seq, cfg.emb_dim, kv_dim, "high"),
        _gemm_spec(seq, cfg.emb_dim, kv_dim, "high"),
    ], 19)
    _ORES_SCRATCH_FOR = _alloc([
        _gemm_spec(seq, cfg.emb_dim, cfg.emb_dim, "high"),
    ], 7)
    _DOWN_SCRATCH_FOR = _alloc([
        _gemm_spec(seq, cfg.hidden_dim, cfg.emb_dim, "high"),
    ], 5)
    return _ORES_SCRATCH_FOR, _DOWN_SCRATCH_FOR


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=False):
    global _FUSED_SCRATCH_FOR, _ORES_SCRATCH_FOR, _DOWN_SCRATCH_FOR
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}\nCompiling Qwen2.5-3B prefill kernels (seq_len={seq_len})...\n{'='*60}\n")

    from shared.infra.external_kernels import compile_gemm_mm, compile_rope

    # mm.o variants for the external GEMMs:
    #   _m32 drain    tile_n=64  (K/V projections)
    #   _m64 fused    tile_n=128 (Q, O, Down projections)
    # Gate/Up direct-codegen needs NO external .o. rope.o for head_dim=128.
    compile_gemm_mm(tile_m=32, tile_n=64, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o")
    compile_gemm_mm(tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o")
    compile_rope()
    from shared.infra.external_kernels import compile_silu_and_mul
    compile_silu_and_mul()  # silu_and_mul.o for the standalone NPU SwiGLU ELF

    print("\n--- rms_qkv_bias_rope (FUSED: RMSNorm+QKV+bias+RoPE) ---")
    fused_mod, fused_scratch = build_rms_qkv_bias_rope_module(seq_len, config)
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_bias_rope", fused_mod, _rms_qkv_bias_rope_backend(verbose)
    )

    print("\n--- o_res_norm (O proj + Residual + FFN RMSNorm) ---")
    ores_mod, ores_scratch = build_o_res_norm_module(seq_len, emb_dim)
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
# Pre-load prefill weights into per-layer BOs (one-time warmup)
# ---------------------------------------------------------------------------


def _fused_bias_rope_call(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """Issue one rms_qkv_bias_rope ELF call (fused prefill attention-input).

    Used by BOTH preload_prefill_weights (warmup, x_in zeroed) and the block
    runner. Single owner of the fused arg layout. NO N-padding. Q is fused-cast
    (f32 C-scratch tail, tracked by _FUSED_SCRATCH_FOR). output_indices=
    [14,17,18] -> v_b, q_roped, k_roped.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),         # 0 x_in
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),          # 1 norm_w (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),                       # 2 normed (inter)
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),          # 3 wq (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),                         # 4 q (inter)
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),         # 5 wk (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                        # 6 k (inter)
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),         # 7 wv (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),                        # 8 v (inter)
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
    for sc, cols in zip(_FUSED_SCRATCH_FOR or [], (q_dim, kv_dim, kv_dim)):
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

    The warmup call layout MUST match ``run_transformer_block_qwen25`` exactly
    (same arg count, static/intermediate indices, bo_key). Qwen2.5-3B has SIX
    prefill ELFs: rms_qkv_bias_rope (fused), o_res_norm, gate, up, swiglu,
    down_add (the O+FFN tail split because the K=hidden Down GEMM must be launch
    0; see build_down_add_module).
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

    ores_scratch = _ORES_SCRATCH_FOR
    down_scratch = _DOWN_SCRATCH_FOR
    if (ores_scratch is None or down_scratch is None
            or _FUSED_SCRATCH_FOR is None):
        ores_scratch, down_scratch = _resolve_scratch_for()

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

        # o_res_norm: static {1,5}. Outputs res1 (arg4), normed2 (arg6).
        ores_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 0 attn_out
            np.asarray(lw.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),           # 1 wo (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 2 proj (inter)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 3 x_resid
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 4 res1 (out)
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),              # 5 ffn_norm (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 6 normed2 (out)
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
        gate_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 0 normed2
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),    # 1 w_gate (static)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 2 gate (out)
        ]
        cache.load_and_run(
            "gate", _gate_backend(), *gate_args,
            output_indices=[2], static_input_indices={1}, intermediate_indices={2},
            bo_key=f"gate_L{layer_idx}",
        )

        # up: static {1}. Output up (arg2).
        up_args = [
            np.zeros((seq_len, emb_dim), dtype=bfloat16),                          # 0 normed2
            np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),      # 1 w_up (static)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 2 up (out)
        ]
        cache.load_and_run(
            "up", _up_backend(), *up_args,
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
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros(n_total, dtype=bfloat16),
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
    """Run one Qwen2.5-3B transformer block on NPU (kernels pre-compiled)."""
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
        # NPU head-first FlashAttention (head_dim=128). q_roped/k_roped are
        # post-bias post-RoPE seq-first; v is the post-bias raw V projection
        # seq-first. Real (un-padded) dims — Qwen2.5-3B does not N-pad.
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
        ores_scratch, down_scratch = _resolve_scratch_for()

    # ELF A1: o_res_norm → res1 (arg4) + normed2 (arg6).
    ores_args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim),        # 0 attn_out
        np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),  # 1 wo (static)
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

    # ELF A2: gate → gate (arg2).
    gate_res = cache.load_and_run(
        "gate",
        _gate_backend(verbose),
        np.asarray(normed2, dtype=bfloat16).reshape(seq_len, emb_dim),         # 0 normed2
        np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 1 w_gate (static)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 2 gate (out)
        output_indices=[2],
        static_input_indices={1},
        intermediate_indices={2},
        bo_key=f"gate_L{layer_idx}",
    )
    gate = gate_res[2].reshape(seq_len, hidden_dim)

    # ELF A3: up → up (arg2).
    up_res = cache.load_and_run(
        "up",
        _up_backend(verbose),
        np.asarray(normed2, dtype=bfloat16).reshape(seq_len, emb_dim),         # 0 normed2
        np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),    # 1 w_up (static)
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),                       # 2 up (out)
        output_indices=[2],
        static_input_indices={1},
        intermediate_indices={2},
        bo_key=f"up_L{layer_idx}",
    )
    up = up_res[2].reshape(seq_len, hidden_dim)

    # ELF A4: swiglu (NPU silu(gate)*up).
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

    # ELF B: down_add → output (arg4). Down GEMM is launch 0 (K=hidden NaN fix).
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
