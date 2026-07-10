# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-1.7B Prefill on MLIR-AIR (NPU2) — single-block (Phase 2) path.

Qwen3 diverges from LLAMA-3.2 in two ways that break the llama
`build_rms_gemms_rope_module` fusion:

  1. QK-norm: a per-head RMSNorm over head_dim is applied to Q and K AFTER
     the projection GEMM and BEFORE RoPE. RoPE linearity does NOT let us
     commute the (nonlinear) QK-norm past RoPE, so the llama RMSNorm+QKV+RoPE
     fused ELF (RoPE immediately after the GEMM) is wrong. We instead build a
     Qwen-specific 8-launch ELF that does RMSNorm + Q/K/V GEMM + per-head
     QK-norm(Q,K) + RoPE(Q,K) all on the NPU (rms_qkv_qknorm_rope).

  2. GQA dims (Qwen3-1.7B): emb_dim=2048, q_dim=n_heads*head_dim=16*128=2048,
     kv_dim=n_kv_heads*head_dim=8*128=1024, hidden=6144.
        q_proj : 2048 -> 2048   (16 heads x 128)
        k/v    : 2048 -> 1024   (8 heads x 128)
        o_proj : 2048 -> 2048   (SQUARE — q_dim == emb_dim)
     Because q_dim == emb_dim the O GEMM is square (2048x2048x2048); the
     generic `build_o_ffn_qwen_module` (written with q_dim separate from
     emb_dim) handles this directly — it simply emits a square O GEMM. All
     dims (2048/1024/6144) are 1024-aligned, so every GEMM uses the stock
     fused-cast TILE_N=128 HERD_N=4 config (sym `_m64`); no low-precision
     tier or padding is needed.

Attention runs NPU FlashAttention by default (cpu_attn=False builds the
flash_attn ELF); CPU attention is an optional fallback (cpu_attn=True).
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

from qwen3_1_7b_weights import LlamaConfig
from qwen3_1_7b_cpu_helpers import attention_reference
from shared.infra.cache import KernelCache, Profiler

# ---------------------------------------------------------------------------
# Builder 1 (FUSED): RMSNorm + Q/K/V GEMM + per-head QK-norm(Q,K) + RoPE(Q,K).
#   8-launch ELF that does the entire attention-input stage on the NPU. See
#   shared/builders/rms_qkv_qknorm_rope_multi.py.
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
        seq_len,
        emb_dim,
        q_dim,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        qknorm_eps=1e-6,
    )


# ---------------------------------------------------------------------------
# Builder 2: O proj (wo: q_dim->emb_dim) + Residual + FFN.
#   Generic O GEMM is K=q_dim, N=emb_dim, attn_out (seq, q_dim), wo (q_dim, emb_dim).
#   For Qwen3-1.7B q_dim==emb_dim==2048 so this O GEMM is SQUARE (2048x2048x2048)
#   — the same generic builder, no special-casing. Tail (residual/RMSNorm/FFN)
#   is emb_dim throughout.
# ---------------------------------------------------------------------------


def build_o_ffn_qwen_module(
    seq_len,
    emb_dim,
    q_dim,
    hidden_dim,
    o_herd_m=8,
    o_herd_n=4,
    gate_herd_m=8,
    gate_herd_n=4,
    down_herd_m=8,
    down_herd_n=4,
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
):
    """O-proj(q_dim->emb_dim) + Residual + FFN, 8 launches.

    Func args:
      %arg0  attn_out  (seq, q_dim)         <- DECOUPLED (q_dim, not emb_dim)
      %arg1  wo        (q_dim, emb_dim)      <- DECOUPLED
      %arg2  proj      (seq, emb_dim)
      %arg3  x_resid   (seq, emb_dim)
      %arg4  res1      (seq, emb_dim)
      %arg5  ffn_norm  (emb_dim,)
      %arg6  normed2   (seq, emb_dim)
      %arg7  w_gate    (emb_dim, hidden)
      %arg8  gate      (seq, hidden)
      %arg9  w_up      (emb_dim, hidden)
      %arg10 up        (seq, hidden)
      %arg11 swiglu    (seq, hidden)
      %arg12 w_down    (hidden, emb_dim)
      %arg13 down      (seq, emb_dim)
      %arg14 output    (seq*emb_dim,)
      %arg15..18  f32 C-scratch (proj[seq,emb], gate[seq,hid], up[seq,hid], down[seq,emb])
    """
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from shared.builders.o_ffn_multi import _build_add_2d_to_2d
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from silu_and_mul.silu_and_mul import build_module_2d as build_swiglu
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

    # O GEMM is decoupled: M=seq, K=q_dim, N=emb_dim.
    o_spec = gemm_registry_config(seq_len, q_dim, emb_dim, "bf16", "high")
    g_spec = gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")

    def _tiles(spec):
        return (
            dict(spec["build_kwargs"]),
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
        )

    _o_kw, _o_m, _o_k2, _o_k1, _o_n = _tiles(o_spec)
    _g_kw, _g_m, _g_k2, _g_k1, _g_n = _tiles(g_spec)
    _d_kw, _d_m, _d_k2, _d_k1, _d_n = _tiles(d_spec)

    print(f"  [1/8] O GEMM ({o_spec['method']})  {seq_len}x{q_dim}x{emb_dim}...")
    o_ir = str(
        _build_gemm_module(
            seq_len,
            q_dim,
            emb_dim,
            _o_m,
            _o_k2,
            _o_k1,
            _o_n,
            o_herd_m,
            o_herd_n,
            **_o_kw,
        )
    )
    print("  [2/8] Residual Add...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))
    print("  [3/8] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )
    print(
        f"  [4/8] Gate GEMM ({g_spec['method']})  {seq_len}x{emb_dim}x{hidden_dim}..."
    )
    gate_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            _g_m,
            _g_k2,
            _g_k1,
            _g_n,
            gate_herd_m,
            gate_herd_n,
            **_g_kw,
        )
    )
    print(f"  [5/8] Up GEMM ({g_spec['method']})...")
    up_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            _g_m,
            _g_k2,
            _g_k1,
            _g_n,
            gate_herd_m,
            gate_herd_n,
            **_g_kw,
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
        f"  [7/8] Down GEMM ({d_spec['method']})  {seq_len}x{hidden_dim}x{emb_dim}..."
    )
    down_ir = str(
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            _d_m,
            _d_k2,
            _d_k1,
            _d_n,
            down_herd_m,
            down_herd_n,
            **_d_kw,
        )
    )

    print("  [8/8] FFN Add (2D -> 1D)...")

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

    # All GEMMs here resolve to fused-cast (large shapes). One mm.o suffix.
    _gemm_sym = o_spec["sym_suffix"]
    _gemm_externs = {
        "@op_has_no_registered_library_name" + _gemm_sym,
        "@zero_f32_mn" + _gemm_sym,
        "@f32_to_bf16_mn" + _gemm_sym,
    }
    assert g_spec["sym_suffix"] == _gemm_sym and d_spec["sym_suffix"] == _gemm_sym, (
        "Qwen o_ffn assumes all 4 GEMMs share the fused-cast mm_m64.o suffix; "
        f"got O={o_spec['method']} G={g_spec['method']} D={d_spec['method']}"
    )

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{q_dim}xbf16>"),  # attn_out (DECOUPLED)
        FuncArg("%arg1", f"memref<{q_dim}x{emb_dim}xbf16>"),  # wo       (DECOUPLED)
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),  # proj
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),  # x_resid
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),  # res1
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),  # ffn_norm
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),  # normed2
        FuncArg("%arg7", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # w_gate
        FuncArg("%arg8", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # gate
        FuncArg("%arg9", f"memref<{emb_dim}x{hidden_dim}xbf16>"),  # w_up
        FuncArg("%arg10", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # up
        FuncArg("%arg11", f"memref<{seq_len}x{hidden_dim}xbf16>"),  # swiglu
        FuncArg("%arg12", f"memref<{hidden_dim}x{emb_dim}xbf16>"),  # w_down
        FuncArg("%arg13", f"memref<{seq_len}x{emb_dim}xbf16>"),  # down
        FuncArg("%arg14", f"memref<{n_total}xbf16>"),  # output
    ]
    scratch_args = [
        FuncArg("%arg15", f"memref<{seq_len}x{emb_dim}xf32>"),
        FuncArg("%arg16", f"memref<{seq_len}x{hidden_dim}xf32>"),
        FuncArg("%arg17", f"memref<{seq_len}x{hidden_dim}xf32>"),
        FuncArg("%arg18", f"memref<{seq_len}x{emb_dim}xf32>"),
    ]

    slices = [
        KernelSlice(o_ir, "og", {0: 0, 1: 1, 2: 15, 3: 2}, extern_syms=_gemm_externs),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(
            gate_ir,
            "gg",
            {0: 6, 1: 7, 2: 16, 3: 8},
            extern_syms=_gemm_externs,
            private_from=False,
        ),
        KernelSlice(
            up_ir,
            "ug",
            {0: 6, 1: 9, 2: 17, 3: 10},
            extern_syms=_gemm_externs,
            private_from=False,
        ),
        KernelSlice(
            swiglu_ir, "sw", {0: 8, 1: 10, 2: 11}, extern_syms={"@silu_and_mul_bf16"}
        ),
        KernelSlice(
            down_ir,
            "dg",
            {0: 11, 1: 12, 2: 18, 3: 13},
            extern_syms=_gemm_externs,
            private_from=False,
        ),
        KernelSlice(ffn_add_ir, "fa", {0: 13, 1: 4, 2: 14}, private_from=False),
    ]

    module = stitch_elf(
        "o_ffn_qwen",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_o_ffn_qwen.mlir",
    )
    print(f"  o_ffn_qwen module: {len(str(module).splitlines())} lines, parsed OK")
    return module


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


def _o_ffn_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn_qwen",
        "runtime_loop_tiling_sizes": [2, 2],
    }


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# Scratch-arg indices for the rms_qkv_qknorm_rope ELF (registry-driven;
# GQA -> 1 fused-cast scratch on Q). Set by compile_all_kernels so the block
# runner + preload know the fused ELF's scratch-arg layout.
_FUSED_SCRATCH_FOR = None


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=False):
    global _FUSED_SCRATCH_FOR
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(
        f"\n{'='*60}\nCompiling Qwen3 prefill kernels (seq_len={seq_len})...\n{'='*60}\n"
    )

    from shared.infra.external_kernels import compile_gemm_mm, compile_rope

    # mm.o variants for GEMM co-linking; rope.o (head_dim=128) for the rope ELFs.
    compile_gemm_mm(
        tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )
    compile_rope()

    print("\n--- rms_qkv_qknorm_rope (FUSED: RMSNorm+QKV+QK-norm+RoPE, 8 launches) ---")
    fused_mod, fused_scratch = build_rms_qkv_qknorm_rope_module(seq_len, config)
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope", fused_mod, _rms_qkv_qknorm_rope_backend(verbose)
    )

    print("\n--- o_ffn_qwen (O proj decoupled + Residual + FFN) ---")
    cache.compile_and_cache(
        "o_ffn_qwen",
        build_o_ffn_qwen_module(seq_len, emb_dim, q_dim, hidden_dim),
        _o_ffn_backend(verbose),
    )

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
# Prefill weight pre-load (BO reuse — opt-buffer-object-reuse B1)
# ---------------------------------------------------------------------------


def _fused_qknorm_rope_call(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """Issue one rms_qkv_qknorm_rope ELF call (fused prefill attention-input).

    Used by BOTH preload_prefill_weights (warmup, x_in zeroed) and the block
    runner (x_in = real hidden). Returns the cache.load_and_run result tuple
    (output_indices=[8, 15, 16] -> v, q_roped, k_roped). The single owner of
    the fused arg layout + index sets so the warmup BO set lines up exactly.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0 x_in (dynamic)
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),  # 1 norm_w (static)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2 normed (inter)
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),  # 3 wq (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 4 q (inter)
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 5 wk (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 6 k (inter)
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 7 wv (static)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 8 v (inter/out)
        np.asarray(lw.q_norm, dtype=bfloat16).reshape(head_dim),  # 9 q_norm (static)
        np.asarray(lw.k_norm, dtype=bfloat16).reshape(head_dim),  # 10 k_norm (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 11 q_n (inter)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 12 k_n (inter)
        lut_q,  # 13 lut_q (static)
        lut_k,  # 14 lut_k (static)
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 15 q_roped (inter/out)
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 16 k_roped (inter/out)
    ]
    inter = {2, 4, 6, 8, 11, 12, 15, 16}
    nxt = 17
    for sc, cols in zip(_FUSED_SCRATCH_FOR or [], (q_dim, kv_dim, kv_dim)):
        if sc is not None:
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
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

    Mirrors llama's preload_prefill_weights / IRON's prepare_runtime: a warmup
    XRT call per layer per ELF allocates the bo_key-keyed BO set and performs
    the host->device write of the *static* weight args. During the real prefill
    pass, ``static_input_indices`` then skips those weight writes (they are
    unchanged), so only the small dynamic activation inputs are re-written.

    Without this, the first (and only) prefill pass writes every weight inside
    the timed region: o_ffn_qwen alone moves 154 MB/layer of BO data on call 1.

    The warmup call layout MUST match ``run_transformer_block_qwen3`` exactly
    (same arg count, same static_input_indices / intermediate_indices, same
    bo_key) or the reused BO set would not line up at inference time.
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

    # RoPE LUTs (seq-first, repeated per head) — same for all layers.
    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]

        # One fused ELF: RMSNorm + Q/K/V GEMM + per-head QK-norm + RoPE.
        _fused_qknorm_rope_call(
            cache,
            lw,
            config,
            seq_len,
            lut_q,
            lut_k,
            layer_idx,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
        )

        # o_ffn_qwen: allocate + write wo/ffn_norm/w_gate/w_up/w_down ({1,5,7,9,12}).
        offn_args = [
            np.zeros((seq_len, q_dim), dtype=bfloat16),  # 0 attn_out (dynamic)
            np.asarray(lw.wo, dtype=bfloat16).reshape(q_dim, emb_dim),  # 1 wo (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2 proj (inter)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 3 x_resid (dynamic)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 4 res1 (inter)
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(
                emb_dim
            ),  # 5 ffn_norm (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 6 normed2 (inter)
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(
                emb_dim, hidden_dim
            ),  # 7 w_gate (static)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 8 gate (inter)
            np.asarray(lw.w_up, dtype=bfloat16).reshape(
                emb_dim, hidden_dim
            ),  # 9 w_up (static)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 10 up (inter)
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 11 swiglu (inter)
            np.asarray(lw.w_down, dtype=bfloat16).reshape(
                hidden_dim, emb_dim
            ),  # 12 w_down (static)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 13 down (inter)
            np.zeros(n_total, dtype=bfloat16),  # 14 output (inter)
            np.zeros((seq_len, emb_dim), dtype=np.float32),  # 15 scratch (inter)
            np.zeros((seq_len, hidden_dim), dtype=np.float32),  # 16 scratch (inter)
            np.zeros((seq_len, hidden_dim), dtype=np.float32),  # 17 scratch (inter)
            np.zeros((seq_len, emb_dim), dtype=np.float32),  # 18 scratch (inter)
        ]
        cache.load_and_run(
            "o_ffn_qwen",
            _o_ffn_backend(),
            *offn_args,
            output_indices=[14],
            static_input_indices={1, 5, 7, 9, 12},
            intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18},
            bo_key=f"o_ffn_qwen_L{layer_idx}",
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
    """Run one Qwen3 transformer block on NPU (kernels pre-compiled in cache)."""
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
        cache,
        layer_weights,
        config,
        seq_len,
        lut_q,
        lut_k,
        layer_idx,
        x_bf16,
        verbose=verbose,
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
        # NPU head-first FlashAttention (head_dim=128). q_roped/k_roped are
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
    offn_args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, q_dim),
        np.asarray(layer_weights.wo, dtype=bfloat16).reshape(q_dim, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.zeros(n_total, dtype=bfloat16),
        np.zeros((seq_len, emb_dim), dtype=np.float32),
        np.zeros((seq_len, hidden_dim), dtype=np.float32),
        np.zeros((seq_len, hidden_dim), dtype=np.float32),
        np.zeros((seq_len, emb_dim), dtype=np.float32),
    ]
    results = cache.load_and_run(
        "o_ffn_qwen",
        _o_ffn_backend(verbose),
        *offn_args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18},
        bo_key=f"o_ffn_qwen_L{layer_idx}",
    )
    output_bf16 = results[14].reshape(seq_len, emb_dim)
    inter["ffn_out"] = output_bf16
    return output_bf16, inter
