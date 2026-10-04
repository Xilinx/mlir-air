# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Backbone-specific o_ffn variant: Gate+Up fused into ONE GEMM (measured
-10.4% device time vs 2 separate launches).

7 launches instead of shared/builders/o_ffn_multi.py's 8 (O, Residual-Add,
FFN-RMSNorm, GateUp-fused-GEMM, SwiGLU-from-wide, Down, FFN-Add). O, Residual,
RMSNorm, Down, FFN-Add are UNCHANGED, reused as-is from o_ffn_multi.py -- this
file only replaces Gate-GEMM + Up-GEMM + SwiGLU (3 launches -> 2).

New isolated file: does NOT touch shared/infra/stitching.py or
shared/builders/o_ffn_multi.py, so llama32_1b/smollm2_1_7b/qwen builds are
untouched. The custom SwiGLU-from-wide builder reuses o_ffn_multi.py's own
build_padded_add pattern (row-iterate, read a column-slice of one wide buffer
via `A[r, lo:hi]`) -- already proven correct there for a padded residual add,
same trick applied to the SiLU*mul activation instead.
"""
from __future__ import annotations
import dataclasses, os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent.parent
for p in (str(_PROG), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ml_dtypes import bfloat16
from air import api as air
from air.api import ops
from air.api.types import i32
from shared.builders.rms_gemms_rope_multi import _api_dtype
from shared.builders.o_ffn_multi import _build_add_2d_to_2d, _build_add_2d_to_1d
from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
    alloc_gemm_scratch,
)


def _build_swiglu_from_wide(
    rows, hidden_dim, np_dtype, herd_x=8, herd_y=1, target="npu2"
):
    """SiLU(gate) * up, reading gate/up as column-slices of ONE wide
    (rows, 2*hidden_dim) buffer instead of two separate tensors -- same
    row-iterate + column-slice pattern as o_ffn_multi.py's build_padded_add,
    applied to the silu_and_mul extern instead of an add.
    """
    total_tiles = herd_x * herd_y
    assert rows % total_tiles == 0, (rows, total_tiles)
    rows_per_tile = rows // total_tiles

    dtype = _api_dtype(np_dtype)
    GATE_UP = air.tensor([rows, 2 * hidden_dim], dtype)
    OUT = air.tensor([rows, hidden_dim], dtype)

    activation = air.extern(
        "silu_and_mul_bf16", link_with="silu_and_mul.o", scalars=[i32]
    )

    with air.launch(name="swiglu_from_wide") as launch:

        @launch.body
        def _():
            with air.segment(name="swiglu_wide_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="swiglu_wide_herd",
                        shape=(herd_x, herd_y),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_gate = air.alloc([hidden_dim], dtype, scope=h.private())
                            l1_up = air.alloc([hidden_dim], dtype, scope=h.private())
                            l1_out = air.alloc([hidden_dim], dtype, scope=h.private())

                            for iv in air.sequential(0, rows_per_tile):
                                r = (tx * herd_y + ty) * rows_per_tile + iv
                                ops.load(l1_gate, GATE_UP[r, 0:hidden_dim])
                                ops.load(l1_up, GATE_UP[r, hidden_dim : 2 * hidden_dim])
                                activation(l1_gate, l1_up, l1_out, hidden_dim)
                                ops.store(l1_out, OUT[r, 0:hidden_dim])

    return launch.build(target=target)


def interleave_gate_up(w_gate, w_up, half):
    """(K, H) gate/up -> (K, 2H) where every 2*half-column block is [gate block | up block],
    the B layout build_module(epilogue_swiglu=True) expects at tile_n = 2*half."""
    k, h = w_gate.shape
    assert w_up.shape == (k, h) and h % half == 0, (w_gate.shape, w_up.shape, half)
    nb = h // half
    return np.ascontiguousarray(
        np.stack(
            [w_gate.reshape(k, nb, half), w_up.reshape(k, nb, half)], axis=2
        ).reshape(k, 2 * h)
    )


def _compile_mm_swiglu(tile_m, tile_n, tile_k_l1, sym_suffix, out_name):
    """compile_gemm_mm's flags, on kernels_swiglu/mm_swiglu.cc (mm_aie2p.cc + the
    f32_to_bf16_swiglu_mn drain)."""
    from shared.infra.external_kernels import _compile_kernel, _PROJ_ROOT

    extra = [
        f"-I{_PROJ_ROOT / 'matrix_multiplication' / 'bf16_in_fp32_out'}",
        "-DBIT_WIDTH=8",
        f"-DDIM_M={tile_m}",
        f"-DDIM_N={tile_n}",
        f"-DDIM_K={tile_k_l1}",
        f"-DDIM_N_DIV_4={tile_n // 4}",
        f"-DDIM_M_DIV_4={tile_m // 4}",
        f"-DDIM_N_DIV_8={tile_n // 8}",
        f"-DDIM_M_DIV_8={tile_m // 8}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DSYM_SUFFIX={sym_suffix}",
    ]
    _compile_kernel(
        _HERE / "kernels_swiglu" / "mm_swiglu.cc",
        out_name,
        extra_flags=extra,
        force=True,
    )


def build_o_ffn_module_fused_gu(
    seq_len,
    emb_dim,
    hidden_dim,
    herd_m=4,
    herd_n=4,
    print_kernels=False,
    gu_tile_n=80,
    gu_b_stationary=False,
    od_b_stationary=False,
    od_tile_n=80,
    o_b_stationary=None,
    dn_b_stationary=None,
    dn_herd_m=None,
    dn_tile_m=32,
    dn_tile_n=None,
    dup=(),
    gu_swiglu=False,
    bfp16=None,
):
    """O-proj + Residual + FFN with Gate+Up fused into one GEMM.

    7 launches, args (base, before scratch tail):
      %arg0  attn_out    (seq_len, emb_dim)              O-GEMM input
      %arg1  wo          (emb_dim, emb_dim)
      %arg2  proj        (seq_len, emb_dim)               O-GEMM output
      %arg3  x_residual  (seq_len, emb_dim)
      %arg4  res1        (seq_len, emb_dim)               residual output (shared w/ FFN-Add)
      %arg5  ffn_norm_w  (emb_dim,)
      %arg6  normed2     (seq_len, emb_dim)                FFN RMSNorm output
      %arg7  w_gateup    (emb_dim, 2*hidden_dim)           [Wgate|Wup] concatenated
      %arg8  gate_up     (seq_len, 2*hidden_dim)           fused GEMM output
      %arg9  swiglu      (seq_len, hidden_dim)             SwiGLU-from-wide output
      %arg10 w_down      (hidden_dim, emb_dim)
      %arg11 down        (seq_len, emb_dim)                Down-GEMM output
      %arg12 output      (seq_len*emb_dim,)                FFN Add output

    bfp16: {"o" | "gu" | "dn": (tile_n, tile_k_l2, tile_k_l1)} -- those GEMMs
    take bfp16ebs8 weights (gemm_bfp16.py) and their weight arg is the packed
    matrix (pack_b_bfp16ebs8; w_gateup interleaved first when gu_swiglu).
    """
    bfp16 = bfp16 or {}
    from shared.builders.gemm_builder import (
        _build_gemm_module,
        gemm_registry_config,
        disambiguate_by_tile_n,
    )
    from shared.infra.external_kernels import compile_gemm_mm
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    n_total = seq_len * emb_dim

    def _custom_extern_syms(sfx):
        return {
            "@matmul_bf16",
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    def _registry_extern_syms(spec):
        return _custom_extern_syms(spec["sym_suffix"])

    o_b_stationary = od_b_stationary if o_b_stationary is None else o_b_stationary
    dn_b_stationary = od_b_stationary if dn_b_stationary is None else dn_b_stationary
    dn_herd_m = dn_herd_m or herd_m
    dn_tile_n = dn_tile_n or od_tile_n

    w_types = {
        "o": f"memref<{emb_dim}x{emb_dim}xbf16>",
        "gu": f"memref<{emb_dim}x{2 * hidden_dim}xbf16>",
        "dn": f"memref<{hidden_dim}x{emb_dim}xbf16>",
    }
    if bfp16:
        from gemm_bfp16 import (
            bfp16_extern_syms,
            bfp16_weight_type,
            build_gemm_bfp16,
            compile_mm_bfp16,
        )

        def _bfp16_gemm(key, k, n, swiglu=False):
            # Optional 4th value: spread N over that many array columns (cols_n).
            tn, tk2, tk1, *cols = bfp16[key]
            sfx, obj = f"_{key}b", f"mm_{key}b.o"
            print(
                f"  {key} GEMM, bfp16 weights (tile_n {tn}, K {tk2}x{tk1}"
                f"{f', N over {cols[0]} columns' if cols else ''}{', SwiGLU drain' if swiglu else ''})..."
            )
            compile_mm_bfp16(32, tn, tk1, sfx, obj)
            w_types[key] = bfp16_weight_type(k, n, tn, tk1)
            ir = str(
                build_gemm_bfp16(
                    seq_len,
                    k,
                    n,
                    32,
                    tk2,
                    tk1,
                    tn,
                    herd_m,
                    cols[0] if cols else herd_n,
                    sfx,
                    obj,
                    swiglu=swiglu,
                    cols_n=bool(cols),
                )
            )
            return ir, bfp16_extern_syms(sfx, swiglu)

    if "o" in bfp16:
        o_ir, o_extern_syms = _bfp16_gemm("o", emb_dim, emb_dim)
    elif o_b_stationary:
        # Bypass the registry (tile_k_l2=320, not full-K) with explicit
        # full-K + b_stationary=True, same lever as the QKV/GateUp fusions.
        print("  [1/7] O GEMM, B-stationary (drain)...")
        compile_gemm_mm(
            tile_m=32,
            tile_n=od_tile_n,
            tile_k_l1=32,
            sym_suffix="_o",
            out_name="mm_o.o",
        )
        o_ir = str(
            _build_gemm_module(
                seq_len,
                emb_dim,
                emb_dim,
                32,
                emb_dim,
                32,
                od_tile_n,
                herd_m,
                herd_n,
                external_bf16_out=True,
                sym_suffix="_o",
                link_with_name="mm_o.o",
                b_stationary=True,
            )
        )
        o_extern_syms = _custom_extern_syms("_o")
    else:
        o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
        d_spec_probe = gemm_registry_config(
            seq_len, hidden_dim, emb_dim, "bf16", "high"
        )
        o_spec, _ = disambiguate_by_tile_n([o_spec, d_spec_probe])
        _o_kw, _o_m, _o_k2, _o_k1, _o_n = (
            dict(o_spec["build_kwargs"]),
            o_spec["tile_m"],
            o_spec["tile_k_l2"],
            o_spec["tile_k_l1"],
            o_spec["tile_n"],
        )
        print("  [1/7] O GEMM (drain)...")
        o_ir = str(
            _build_gemm_module(
                seq_len,
                emb_dim,
                emb_dim,
                _o_m,
                _o_k2,
                _o_k1,
                _o_n,
                herd_m,
                herd_n,
                **_o_kw,
            )
        )
        o_extern_syms = _registry_extern_syms(o_spec)

    print("  [2/7] Residual Add (2D -> 2D)...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    print("  [3/7] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # Fused GateUp GEMM: same tiles validated in the standalone A/B
    # (measured: -10.4% vs 2 separate launches).
    gu_n = 2 * hidden_dim
    gu_tile_k1 = 32
    if "gu" in bfp16:
        gu_ir, gu_extern_syms = _bfp16_gemm("gu", emb_dim, gu_n, swiglu=gu_swiglu)
    else:
        if gu_swiglu:
            # w_gateup must be block-interleaved on the host (interleave_gate_up with
            # half=gu_tile_n//2); the GEMM then writes SiLU(gate)*up straight to arg9.
            print("  [4/7] GateUp GEMM, fused, SwiGLU drain epilogue...")
            _compile_mm_swiglu(32, gu_tile_n, gu_tile_k1, "_gu", "mm_gu_sw.o")
            gu_link, gu_drain_sym = "mm_gu_sw.o", "@f32_to_bf16_swiglu_mn_gu"
        else:
            print("  [4/7] GateUp GEMM, fused (drain)...")
            compile_gemm_mm(
                tile_m=32,
                tile_n=gu_tile_n,
                tile_k_l1=gu_tile_k1,
                sym_suffix="_gu",
                out_name="mm_gu.o",
            )
            gu_link, gu_drain_sym = "mm_gu.o", "@f32_to_bf16_mn_gu"
        gu_ir = str(
            _build_gemm_module(
                seq_len,
                emb_dim,
                gu_n,
                32,
                emb_dim,
                gu_tile_k1,
                gu_tile_n,
                herd_m,
                herd_n,
                external_bf16_out=True,
                sym_suffix="_gu",
                link_with_name=gu_link,
                b_stationary=gu_b_stationary,
                epilogue_swiglu=gu_swiglu,
            )
        )
        gu_extern_syms = {
            "@matmul_bf16",
            "@op_has_no_registered_library_name_gu",
            "@zero_f32_mn_gu",
            gu_drain_sym,
        }

    if not gu_swiglu:
        print("  [5/7] SwiGLU (from fused GateUp buffer)...")
        swiglu_ir = _wrap_ir_in_launch(
            str(_build_swiglu_from_wide(seq_len, hidden_dim, bfloat16, herd_x=8))
        )

    if "dn" in bfp16:
        down_ir, down_extern_syms = _bfp16_gemm("dn", hidden_dim, emb_dim)
    elif dn_b_stationary:
        # Memtile budget per herd column (512 KB): B slab K*tile_n + 2x ping-pong
        # A tile_m*K + C. At K=2560, tile_m=32 overflows at any tile_n >= 40.
        print(
            f"  [6/7] Down GEMM, B-stationary (drain, tile_m={dn_tile_m}, tile_n={dn_tile_n}, herd_m={dn_herd_m})..."
        )
        compile_gemm_mm(
            tile_m=dn_tile_m,
            tile_n=dn_tile_n,
            tile_k_l1=32,
            sym_suffix="_dn",
            out_name="mm_dn.o",
        )
        down_ir = str(
            _build_gemm_module(
                seq_len,
                hidden_dim,
                emb_dim,
                dn_tile_m,
                hidden_dim,
                32,
                dn_tile_n,
                dn_herd_m,
                herd_n,
                external_bf16_out=True,
                sym_suffix="_dn",
                link_with_name="mm_dn.o",
                b_stationary=True,
            )
        )
        down_extern_syms = _custom_extern_syms("_dn")
    else:
        d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")
        o_spec_probe = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
        _, d_spec = disambiguate_by_tile_n([o_spec_probe, d_spec])
        _d_kw, _d_m, _d_k2, _d_k1, _d_n = (
            dict(d_spec["build_kwargs"]),
            d_spec["tile_m"],
            d_spec["tile_k_l2"],
            d_spec["tile_k_l1"],
            d_spec["tile_n"],
        )
        print("  [6/7] Down GEMM (drain)...")
        down_ir = str(
            _build_gemm_module(
                seq_len,
                hidden_dim,
                emb_dim,
                _d_m,
                _d_k2,
                _d_k1,
                _d_n,
                herd_m,
                herd_n,
                **_d_kw,
            )
        )
        down_extern_syms = _registry_extern_syms(d_spec)

    print("  [7/7] FFN Add (2D -> 1D)...")
    ffn_add_ir = str(_build_add_2d_to_1d(seq_len, emb_dim, bfloat16))

    if print_kernels:
        for name, ir in [
            ("O GEMM", o_ir),
            ("Res Add", res_add_ir),
            ("FFN RMSNorm", rms_ir),
            ("GateUp GEMM", gu_ir),
            ("SwiGLU-wide", None if gu_swiglu else swiglu_ir),
            ("Down GEMM", down_ir),
            ("FFN Add", ffn_add_ir),
        ]:
            if ir is None:
                continue
            print(
                f"\n{'='*60}\n  Sub-kernel: {name} ({len(ir.splitlines())} lines)\n{'='*60}"
            )
            print(ir)

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", w_types["o"]),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", w_types["gu"]),
        FuncArg("%arg8", f"memref<{seq_len}x{gu_n}xbf16>"),
        FuncArg("%arg9", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg10", w_types["dn"]),
        FuncArg("%arg11", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{n_total}xbf16>"),
    ]
    # Both O and Down always resolve to drain here (registry-driven or the
    # explicit b_stationary bypass), so neither needs an f32 fused-cast scratch.
    scratch_args, scratch_for = [], (None, None)

    slices = [
        KernelSlice(
            o_ir,
            "og",
            _gemm_arg_map(0, 1, 2, scratch_for[0]),
            extern_syms=o_extern_syms,
        ),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(
            gu_ir,
            "gu",
            {0: 6, 1: 7, 2: 9 if gu_swiglu else 8},
            extern_syms=gu_extern_syms,
        ),
        (
            None
            if gu_swiglu
            else KernelSlice(
                swiglu_ir, "sw", {0: 8, 1: 9}, extern_syms={"@silu_and_mul_bf16"}
            )
        ),
        KernelSlice(
            down_ir,
            "dg",
            _gemm_arg_map(9, 10, 11, scratch_for[1]),
            extern_syms=down_extern_syms,
        ),
        KernelSlice(ffn_add_ir, "fa", {0: 11, 1: 4, 2: 12}, private_from=False),
    ]
    slices = [sl for sl in slices if sl is not None]
    # Timing probe: run the named launches twice back to back. Every launch here
    # recomputes its output from unchanged inputs, so the result is the same and
    # the time delta is that launch's in-situ cost.
    assert set(dup) <= {sl.prefix for sl in slices}, dup
    slices = [
        s
        for sl in slices
        for s in (
            [sl, dataclasses.replace(sl, prefix=sl.prefix + "x", private_from=False)]
            if sl.prefix in dup
            else [sl]
        )
    ]

    # arg8 (the wide gate|up buffer) is dead with the SwiGLU epilogue; kept so the
    # 13-arg ABI matches the non-epilogue variant.
    module = stitch_elf(
        "o_ffn_fused_gu",
        base_args,
        slices,
        scratch_args=scratch_args,
        allow_unreferenced_args={8} if gu_swiglu else (),
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module
