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
import tempfile

import filelock

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper
from air.backend.xrt import XRTBackend

from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
)

range_ = for_


# ---------------------------------------------------------------------------
# 2D→2D eltwise add (all 3 args 2D, collapse inside launch)
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_2d(rows, cols, np_dtype, vector_size=16, herd_x=8, herd_y=1):
    """Eltwise add (bf16): all 3 args are 2D memrefs, collapsed to 1D inside launch.

    Unlike the standard _build_add_2d (2D inputs, 1D output), this version keeps the
    OUTPUT as 2D too, so subsequent launches can read it as 2D without expand_shape.
    The collapse_shape inside the launch gives a 1D view for the DMA writes — same
    bytes in DDR, just different type. Used for the residual add (proj + x_residual),
    both bf16 (the fused-cast GEMM already produced bf16).
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
    print_kernels=False,
    herd_m=8,
):
    """Build the fused O-proj + Residual + FFN ELF (12 launches, 19 args).

    The 4 GEMMs (O/Gate/Up/Down) use external mm.o kernels whose method + tiles come
    from the kernel_registry JSON per shape (gemm_registry_config). All 4 are large
    (M*K*N>=4e9) so the registry resolves them to fused-cast (tile_m=64, f32 C scratch
    + on-chip cast launch); 4 extra f32-scratch func args (15..18) carry the scratch.
    Same 9.3e-3 GPU-standard precision as drain, but faster.

    herd_m: M-dimension herd size for the O/Gate/Up/Down GEMMs (default 8). Must
    satisfy seq_len % (tile_m * herd_m) == 0; small contexts (seq_len < 512) need
    herd_m=4 since fused-cast tile_m=64 makes 64*8=512 > seq_len.
    """
    return _build_o_ffn(
        seq_len=seq_len,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        o_herd_m=herd_m,
        gate_herd_m=herd_m,
        down_herd_m=herd_m,
        print_kernels=print_kernels,
    )


def _build_o_ffn(
    seq_len=2048,
    emb_dim=2048,
    hidden_dim=8192,
    o_herd_m=8,
    o_herd_n=4,
    gate_herd_m=8,
    gate_herd_n=4,
    down_herd_m=8,
    down_herd_n=4,
    # SwiGLU config
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    print_kernels=False,
):
    """O-proj + Residual + FFN, all-bf16 outward buffers, 12 launches / 19 args.

    The 4 GEMMs (O/Gate/Up/Down) get their method + tiles from the kernel_registry
    JSON per shape (gemm_registry_config, "bf16"/"high"). All 4 are large so they
    resolve to fused-cast (external mm.o GEMM with an f32 C scratch + a separate
    on-chip cast launch each = @gemm_cast_bf16, 2 launches/GEMM). The 4 f32 scratch
    buffers are func args 15..18. GPU-standard 9.3e-3 precision. Needs mm_m64.o +
    runtime_loop_tiling_sizes=[2,2] (BD-ID recycling).
    """
    from shared.builders.gemm_builder import (
        _build_gemm_module,
        gemm_registry_config,
        disambiguate_by_tile_n,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    # Per-GEMM config from the kernel_registry JSON (single source of truth): method
    # (fused-cast vs drain) AND all tiles are looked up per shape — never hardcoded.
    # The lookup is per-shape so this adapts automatically to other models. Distinct
    # _m64/_m32 symbols + mm_*.o let any method mix co-link in one ELF.
    o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    g_spec = gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")

    # Guard against a genuine ELF-merge hazard: two GEMMs sharing a method
    # (drain or fused-cast) but resolving to DIFFERENT tile_n cannot safely
    # share the same sym_suffix/mm.o (DIM_N is baked at compile time into the
    # external object). This is a no-op (identical specs returned) whenever
    # all GEMMs of a given method already share one tile_n -- true for every
    # existing caller of this builder (e.g. llama's uniform tile_n=128) -- and
    # only rewrites sym_suffix/obj when a real collision exists (e.g. SmolVLA's
    # O/Down at tile_n=80 vs Gate/Up at tile_n=128, all "drain").
    o_spec, g_spec, d_spec = disambiguate_by_tile_n([o_spec, g_spec, d_spec])

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

    n_total = seq_len * emb_dim

    # ---- Build sub-kernels ----

    # L1: O GEMM
    print(f"  [1/8] O GEMM ({o_spec['method']})...")
    o_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
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

    # L2: Residual Add (2D → 2D, all args 2D, collapse inside)
    print("  [2/8] Residual Add (2D -> 2D)...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    # L3: FFN RMSNorm (bare herd → wrap)
    print("  [3/8] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # L4: Gate GEMM
    print(f"  [4/8] Gate GEMM ({g_spec['method']})...")
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

    # L5: Up GEMM (same shape as Gate → same registry config)
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

    # L6: SwiGLU (bare herd → wrap)
    print("  [6/8] SwiGLU...")
    from silu_and_mul.silu_and_mul import build_module_2d as build_swiglu

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
    print(f"  [7/8] Down GEMM ({d_spec['method']})...")
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

    # L8: FFN Add (2D inputs, 1D output — same as ffn_full's add)
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

    # ---- Stitch (declarative via stitch_elf) ----
    # Each of the 4 GEMMs (O/Gate/Up/Down) independently resolves to drain or
    # fused-cast per its OWN shape (gemm_registry_config) -- at llama's seq_len
    # (2048) all 4 happen to be large enough to land on fused-cast, but at
    # smaller seq_len (e.g. SmolVLA's 256) they resolve to drain instead, which
    # has a different launch arg count (3: A,B,out) than fused-cast (4: A,B,
    # C-f32-scratch,D-bf16-out). Mirrors rms_gemms_rope_multi.py's per-GEMM
    # gemm_registry_config + alloc_gemm_scratch pattern so this builder adapts
    # to any shape instead of hardcoding "always fused-cast".
    from shared.infra.stitching import alloc_gemm_scratch

    def _gemm_extern_syms(spec):
        sfx = spec["sym_suffix"]
        return {
            "@matmul_bf16",
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:  # fused-cast: {0:in, 1:w, 2:Cf32-scratch, 3:bf16-out}
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}  # drain: {0:in, 1:w, 2:bf16-out}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{hidden_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{emb_dim}x{hidden_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{n_total}xbf16>"),
    ]
    # Registry-driven f32 C-scratch args, in builder order (O, Gate, Up, Down).
    # One per fused-cast GEMM (indices continue from 15); drain GEMMs get none.
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (o_spec, seq_len, emb_dim),
            (g_spec, seq_len, hidden_dim),  # gate
            (g_spec, seq_len, hidden_dim),  # up (same shape/spec as gate)
            (d_spec, seq_len, emb_dim),  # down
        ],
        base_arg_count=15,
    )

    # Privates come from every GEMM slice (og/gg/ug/dg), not just the first:
    # when disambiguate_by_tile_n() (above) gives two GEMMs distinct sym_suffix
    # values (e.g. SmolVLA's O/Down at "_m32_n80" vs Gate/Up at "_m32_n128"),
    # each distinct suffix's private decls must come from a slice that actually
    # uses it -- collecting only from `og` (as when every GEMM shared one
    # suffix) would silently drop the second suffix's declarations. `all_privates`
    # in stitch_elf is a set, so slices sharing an identical suffix (og/dg both
    # "_m32_n80" here) safely dedupe instead of double-declaring. SwiGLU still
    # contributes its own (@silu_and_mul_bf16).
    slices = [
        KernelSlice(
            o_ir,
            "og",
            _gemm_arg_map(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_extern_syms(o_spec),
        ),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(
            gate_ir,
            "gg",
            _gemm_arg_map(6, 7, 8, scratch_for[1]),
            extern_syms=_gemm_extern_syms(g_spec),
        ),
        KernelSlice(
            up_ir,
            "ug",
            _gemm_arg_map(6, 9, 10, scratch_for[2]),
            extern_syms=_gemm_extern_syms(g_spec),
            private_from=False,
        ),
        KernelSlice(
            swiglu_ir, "sw", {0: 8, 1: 10, 2: 11}, extern_syms={"@silu_and_mul_bf16"}
        ),
        KernelSlice(
            down_ir,
            "dg",
            _gemm_arg_map(11, 12, 13, scratch_for[3]),
            extern_syms=_gemm_extern_syms(d_spec),
        ),
        KernelSlice(ffn_add_ir, "fa", {0: 13, 1: 4, 2: 14}, private_from=False),
    ]

    module = stitch_elf(
        "o_ffn",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_o_ffn.mlir",
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
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

    from shared.infra.external_kernels import (
        compile_silu_and_mul,
        compile_gemm_mm,
    )

    # The 4 GEMMs are fused-cast (tile_m=64) per the registry → mm_m64.o.
    print("Compiling external kernels (mm_m64.o, silu_and_mul.o)...")
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )
    compile_silu_and_mul()

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

    # fused-cast GEMM herds need BD-ID recycling.
    extra_backend = {"runtime_loop_tiling_sizes": [2, 2]}

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="o_ffn",
            **extra_backend,
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

    # All GEMMs emit bf16 (fused-cast does the cast on-chip); proj/gate/up/down
    # buffers are bf16. The 4 f32 C-scratch buffers are args 15..18; output arg14.
    proj_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    res1_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    normed2_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    gate_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    up_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    swiglu_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    down_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    output_buf = np.zeros(N_TOTAL, dtype=bfloat16)

    # The stitched @o_ffn has its output at arg14 (NOT last) with 4 f32 C-scratch
    # buffers at args 15..18. XRTRunner.run_test() appends output placeholders
    # AFTER inputs, so it can only handle output-is-last functions — it would
    # misalign every arg from 14 onward here. Drive the module directly with the
    # full 19-arg list in position (matching the production cache.load_and_run
    # path), then read back arg14.
    args_in_order = [
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
        output_buf,  # arg14: output
        # arg15..18: per-GEMM f32 C-scratch (proj, gate, up, down)
        np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32),
        np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.float32),
        np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.float32),
        np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32),
    ]

    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="o_ffn",
        **extra_backend,
    )
    compiled = backend.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        module_function = backend.load(compiled)
        results = module_function(*args_in_order)
    backend.unload()

    actual = np.asarray(results[14]).reshape(-1).astype(np.float32)
    expected = output_ref.flatten().astype(np.float32)
    corr = np.corrcoef(actual, expected)[0, 1]
    print(f"correlation = {corr:.6f}")
    if corr >= 0.99:
        print("PASS!")
        exit(0)
    print("FAIL!")
    exit(1)
