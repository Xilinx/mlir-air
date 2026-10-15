# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SigLIP ViT fused multi-launch ELF builders (A3-6b perf hill-climb).

Two fused ELFs, mirroring the SmolVLA backbone's rms_gemms_rope / o_ffn builders
but re-parameterized for the SigLIP vision encoder (affine LayerNorm instead of
RMSNorm, GELU-tanh instead of SwiGLU, a per-Linear BIAS on every projection, and
NO RoPE). Building these two ELFs collapses the unfused 10-dispatch/layer path
(LN1, q, k, v, +host bias×3, +host residual, LN2, fc1, +host bias, gelu, fc2,
+host bias, +host residual) into 3 on-device dispatches/layer:

  vit_ln_qkv  (2 launches): LN1 + one fused [Q|K|V] GEMM (bias on the
                            weight stream)              -> qkv (seq, 3*emb)
  flash_attn  (1 launch,  unchanged registry ELF)
  vit_o_ffn   (6 launches): O GEMM + residual + LN2 + fc1 GEMM (bias+GELU
                            in the epilogue) + fc2 GEMM + residual

All the per-Linear bias-adds and the two residual adds — host f32 glue in the
unfused driver — move on-device here (Lever 3, folded into the fusion). They are
math-equivalent within the accepted BFP16 ceiling: the fused path adds bias/residual
in bf16 vector lanes instead of host f32, a per-element rounding difference far
below the 0.945 systematic-bias floor (verified: post_ln cosine unchanged).

Reuses shared/infra/stitching.py (stitch_elf) + the on-device bias-add primitive
from shared/builders/rms_qkv_bias_rope_multi.py (_build_bias_add_2d, broadcast
(D,) over rows) + o_ffn_multi's 2D->2D residual add.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# smolvla/ -> llms (for shared.infra / shared.builders)


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
from air.dialects import math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_ as range_, yield_
from air.backend.xrt_runner import type_mapper

from air import api as air
from air.api import ops

# programming_examples/ is published as the air_examples package rather than put
# on sys.path: every directory under it would otherwise become a top-level
# module name and shadow any installed package that shares it. Registered
# before the first air_examples import below, which needs it.
import types
from pathlib import Path

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]

from air_examples.llms.shared.builders.rms_gemms_rope_multi import _api_dtype
from air.dialects.arith import ConstantOp

from air_examples.llms.shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
)
from air_examples.llms.shared.builders.gemm_builder import (
    _build_gemm_module,
    BIAS_PAD_ROWS,
    packed_k,
    gemm_registry_config,
    disambiguate_by_tile_n,
)
from air_examples.llms.shared.builders.rms_qkv_bias_rope_multi import _build_bias_add_2d
from air_examples.llms.shared.builders.o_ffn_multi import _build_add_2d_to_2d
from air_examples.layer_norm.layer_norm import build_module as build_layer_norm
from smolvla_fuse import LN_EXT, LN_ROWS

import types
from pathlib import Path


def _layer_norm_ir(seq_len, emb_dim):
    """The affine LayerNorm launch as text: the C++ row kernel when LN_EXT, else the
    air.api loop. Either way it is already a launch, which `_wrap_ir_in_launch`
    leaves as is."""
    return _wrap_ir_in_launch(
        str(
            build_layer_norm(
                seq_len, emb_dim, bfloat16, 16, herd_x=8, ext=LN_EXT, rows=LN_ROWS
            )
        )
    )


def _ln_externs():
    """Symbols an LN slice links from outside (never prefix-renamed)."""
    return {"@zero_vectorized_bf16"} | ({"@ln_rows_bf16"} if LN_EXT else set())


# GELU-tanh constants (match the gelu/gelu.py registry kernel).
_GELU_BETA = 0.044715
_SQRT_2_OVER_PI = 0.7978845608028654


# ---------------------------------------------------------------------------
# 2D GELU-tanh (all 2D args, collapse to 1D inside launch) — mirrors
# silu_and_mul.build_module_2d / o_ffn_multi._build_add_2d_to_2d layout but for
# the single-input GELU activation (no gate*up). Inline math (no external .o),
# same tanh-approximation the standalone gelu ELF uses.
# ---------------------------------------------------------------------------


def _build_gelu_2d(
    rows, cols, tile_n, np_dtype, herd_x=8, herd_y=2, vector_size=16, target="npu2"
):
    """Tanh-approximation GELU over a 2-D tensor, walked as a flat array.

        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Written out rather than as ``ops.gelu``, which composes the same identity
    from the same constants: the DSL associates it as ``(x*x)*x`` and ``0.5*x``
    where the predecessor has ``x*(x*x)`` and ``x*0.5``. Both are the same
    number, but this conversion is gated on emitting a byte-identical
    air.insts.bin, so the association is kept as it was. A future cleanup can
    switch to ops.gelu and re-gate.
    """
    n = rows * cols
    total_tiles = herd_x * herd_y
    assert n % (tile_n * total_tiles) == 0, (n, tile_n, total_tiles)
    assert tile_n % vector_size == 0

    dtype = _api_dtype(np_dtype)
    IN = air.tensor([rows, cols], dtype)
    OUT = air.tensor([rows, cols], dtype)

    with air.launch(name="gelu2d") as launch:

        @launch.body
        def _():
            with air.segment(name="gelu2d_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="gelu2d_herd",
                        shape=(herd_x, herd_y),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_in = air.alloc(
                                [tile_n], dtype, scope=h.private(), vector=vector_size
                            )
                            l1_out = air.alloc(
                                [tile_n], dtype, scope=h.private(), vector=vector_size
                            )

                            in_flat = IN.reshape(n)
                            out_flat = OUT.reshape(n)

                            for iv in air.sequential(0, n, tile_n * total_tiles):
                                off = iv + (tx * herd_y + ty) * tile_n
                                ops.load(l1_in, in_flat[off : off + tile_n])

                                x = l1_in[:]
                                x2 = x * x
                                x3 = x * x2
                                inner = (x + x3 * _GELU_BETA) * _SQRT_2_OVER_PI
                                l1_out[:] = (x * 0.5) * (ops.tanh(inner) + 1.0)

                                ops.store(l1_out, out_flat[off : off + tile_n])

    return launch.build(target=target)


def _bcast(vec_ty, xrt_dtype, val):
    from air.dialects.vector import BroadcastOp

    c = arith.ConstantOp(xrt_dtype, float(val))
    return BroadcastOp(vec_ty, c).result


# ---------------------------------------------------------------------------
# GEMM slice helpers (drain-only for vision; all 3 vision GEMM shapes resolve
# to drain per the registry). Mirrors o_ffn_multi's extern/arg-map helpers.
# ---------------------------------------------------------------------------


def _force_tile_n_suffix(spec):
    """Rewrite a drain spec's sym_suffix/obj/build_kwargs to key off tile_n
    (`_m32_n{tile_n}` / `mm_m32_n{tile_n}.o`), matching disambiguate_by_tile_n's
    naming. Ensures an ELF whose GEMM is built in isolation still links the
    SAME compiled object as another ELF that resolved the same (method, tile_n)
    via disambiguation — avoids colliding on the generic `mm_m32.o` whose baked
    DIM_N could belong to a different shape."""
    s = dict(spec)
    tag = f"m{s['tile_m']}"
    suffix = f"_{tag}_n{s['tile_n']}"
    obj = f"mm_{tag}_n{s['tile_n']}.o"
    s["sym_suffix"] = suffix
    s["obj"] = obj
    s["build_kwargs"] = dict(s["build_kwargs"])
    s["build_kwargs"]["sym_suffix"] = suffix
    s["build_kwargs"]["link_with_name"] = obj
    return s


def _herd(spec, herd_m, herd_n):
    """(herd_m, herd_n) to build `spec` with: a swapped spec puts 4 rows on M and 8 columns on N."""
    return (herd_n, herd_m) if spec.get("herd_swap") else (herd_m, herd_n)


def vit_gemm_spec(m, k, n):
    """Registry spec of one vision GEMM at the batched M, keyed by (tile_m, tile_n)
    so GEMMs that resolve to the same tile share one compiled object."""
    spec = _force_tile_n_suffix(gemm_registry_config(m, k, n, "bf16", "high"))
    assert spec["method"] == "drain", spec["method"]
    return spec


def vit_gemm_specs(m, emb_dim, hidden_dim):
    """{qkv, o, fc1, fc2} registry specs; asserts that specs sharing an object agree."""
    specs = {
        "qkv": vit_gemm_spec(m, emb_dim, 3 * emb_dim),
        "o": vit_gemm_spec(m, emb_dim, emb_dim),
        "fc1": vit_gemm_spec(m, emb_dim, hidden_dim),
        "fc2": vit_gemm_spec(m, hidden_dim, emb_dim),
    }
    by_obj = {}
    for s in specs.values():
        key = (s["tile_m"], s["tile_n"], s["tile_k_l1"])
        assert by_obj.setdefault(s["obj"], key) == key, (
            s["obj"],
            by_obj[s["obj"]],
            key,
        )
    return specs


def _b_build(spec, k):
    """(tile_k_l2, B-side build kwargs) for `spec`'s b_stationary / b_panel."""
    kw = {}
    if spec["b_stationary"]:
        assert spec["tile_k_l2"] == k, (spec["tile_k_l2"], k)
        kw["b_stationary"] = True
    if spec["b_panel"]:
        kw["b_panel_layout"] = True
    return spec["tile_k_l2"], kw


def b_arg_type(spec, k, n, herd_n):
    """memref type of the bias-packed B func arg; `herd_n` as passed to the builder."""
    tk1 = spec["tile_k_l1"]
    if spec["b_panel"]:
        shape = _panel_arg_shape(k, n, tk1, spec["tile_k_l2"], spec["tile_n"], herd_n)
        return f"memref<{shape}xbf16>"
    return f"memref<{packed_k(k, tk1)}x{n}xbf16>"


def pack_b(packed, spec, k, herd_m=8, herd_n=4):
    """Host side of b_arg_type: the bias-packed (packed_k, N) weight, panel-reshuffled
    when the spec says b_panel (same herd the module builders resolve)."""
    if not spec["b_panel"]:
        return packed
    from air_examples.llms.shared.builders.gemm_builder import (
        repack_gemm_b_panel_layout,
    )

    tk1, tk2 = spec["tile_k_l1"], spec["tile_k_l2"]
    tk2_pad = (tk2 // tk1) * (tk1 + BIAS_PAD_ROWS)
    hn = _herd(spec, herd_m, herd_n)[1]
    return repack_gemm_b_panel_layout(packed, tk2_pad, spec["tile_n"], herd_n=hn)


def _gemm_externs(spec, bias_fused=False, gelu=False):
    sfx = spec["sym_suffix"]
    syms = {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
    }
    # The bias-fused GEMM folds the per-channel bias into the drain herd's
    # epilogue cast, so it links a different cast symbol plus the extractor
    # that lifts the bias out of B's trailing k-block.
    if bias_fused:
        syms.add("@extract_bias_from_b" + sfx)
        syms.add(
            ("@f32_to_bf16_bias_gelu_mn" if gelu else "@f32_to_bf16_bias_mn") + sfx
        )
    else:
        syms.add("@f32_to_bf16_mn" + sfx)
    return syms


def _gemm_tiles(spec):
    return (
        dict(spec["build_kwargs"]),
        spec["tile_m"],
        spec["tile_k_l2"],
        spec["tile_k_l1"],
        spec["tile_n"],
    )


def _panel_arg_shape(k, n, tk1, tk2, tn, herd_n, pad_rows=BIAS_PAD_ROWS):
    """memref shape string for a bias-packed B weight laid out panel-contiguous:
    mirrors build_module's b_size for b_panel_layout=True (bf16_in_bf16_out/run.py)."""
    tk2_pad = (tk2 // tk1) * (tk1 + pad_rows)
    n_kchunks, n_nblocks = k // tk2, n // (tn * herd_n)
    return f"{n_kchunks * n_nblocks}x{herd_n}x{tk2_pad}x{tn}"


def _gemm_grid_extents(m, n, spec, herd_m, herd_n):
    """(grid_m, grid_n): how many launch-grid iterations a drain GEMM at this
    shape/tiling needs. A `runtime_loop_tiling_sizes` request >= this per-axis
    fully unrolls the GEMM's control stream, matching AIRDependencyScheduleOpt's
    findLargestFactor(trip_count, requested) -- which saturates at trip_count
    once requested >= trip_count, leaving no runtime loop and so no BD IDs to
    recycle."""
    return m // (spec["tile_m"] * herd_m), n // (spec["tile_n"] * herd_n)


def vit_ln_qkv_runtime_tiling(
    seq_len, emb_dim, herd_m=8, herd_n=4, registry_seq_len=None
):
    """`runtime_loop_tiling_sizes` that fully unrolls vit_ln_qkv's one GEMM
    launch -- its own (M, N) grid extents. `seq_len` is the BATCHED length
    (n_images * per-image seq); `registry_seq_len`, if given, is the per-image
    length the registry was measured at (see build_vit_ln_qkv_module)."""
    reg_len = registry_seq_len or seq_len
    spec = gemm_registry_config(reg_len, emb_dim, emb_dim, "bf16", "high")
    return list(_gemm_grid_extents(seq_len, 3 * emb_dim, spec, herd_m, herd_n))


def vit_o_ffn_runtime_tiling(
    seq_len, emb_dim, hidden_dim, herd_m=8, herd_n=4, registry_seq_len=None
):
    """`runtime_loop_tiling_sizes` that fully unrolls vit_o_ffn's three GEMM
    launches (O/fc1/fc2) -- the elementwise max of their (M, N) grid extents,
    since one tiling vector is shared by the whole stitched ELF and each
    launch's own loop clips to its own trip count (see _gemm_grid_extents)."""
    reg_len = registry_seq_len or seq_len
    # The per-image registry tiles, not the batched ones: those would raise the request to
    # 12,8, which hangs the device (ERT_CMD_STATE_TIMEOUT); with 12,6 each launch's loop clips
    # to its own trip count.
    o_spec, g_spec, d_spec = (
        gemm_registry_config(reg_len, kk, nn, "bf16", "high")
        for kk, nn in ((emb_dim, emb_dim), (emb_dim, hidden_dim), (hidden_dim, emb_dim))
    )
    grids = [
        _gemm_grid_extents(seq_len, emb_dim, o_spec, herd_m, herd_n),
        _gemm_grid_extents(seq_len, hidden_dim, g_spec, herd_m, herd_n),
        _gemm_grid_extents(seq_len, emb_dim, d_spec, herd_m, herd_n),
    ]
    return [max(g[0] for g in grids), max(g[1] for g in grids)]


# ===========================================================================
# Group A: vit_ln_qkv — LN1 + one fused [Q|K|V] GEMM (2 launches)
# ===========================================================================


def build_vit_ln_qkv_module(
    seq_len, emb_dim, n_heads, head_dim, herd_m=8, herd_n=4, registry_seq_len=None
):
    """Fused LN1 + Q/K/V GEMM + per-channel Q/K/V bias-add for one SigLIP block.

    Func args (MHA, no GQA: q_dim == kv_dim == emb_dim):
      %arg0  x_in     (seq, emb)            block input (bf16)
      %arg1  ln_param (2*emb,)              LN1 gamma||beta packed
      %arg2  normed   (seq, emb)            LN1 out (intermediate)
      %arg3  wqkv     (packed_k, 3*emb)     [Wq|Wk|Wv], bias-packed
      %arg4  qkv      (seq, 3*emb)          [Q|K|V], bias folded in — OUTPUT

    Two mechanisms collapse this from 7 launches to 2:
      - the Q/K/V weights are the BIAS-PACKED form (repack_gemm_b_with_bias), so
        the drain herd folds the bias into the epilogue cast and the three
        bias-add launches are gone;
      - [X.Wq | X.Wk | X.Wv] == X . [Wq|Wk|Wv], so the three GEMMs become ONE
        wide GEMM. They share A, so the group axis just rides the existing N
        launch axis — no builder change. This matters because a launch costs
        ~337 us of control-stream replay (the whole device configuration is
        cloned inline at every launch) while a launch ITERATION costs ~13 us.
    FlashAttention reads Q/K/V straight out of the concatenated buffer
    (`fused_qkv=True`), so nothing is copied apart to feed it.
    """
    qkv_dim = 3 * emb_dim
    spec = vit_gemm_spec(seq_len, emb_dim, qkv_dim)

    print("  [ln_qkv 1/2] LN1 (affine)...")
    ln_ir = _layer_norm_ir(seq_len, emb_dim)

    kw, tm, _, tk1, tn = _gemm_tiles(spec)
    hm, hn = _herd(spec, herd_m, herd_n)
    tk2, bkw = _b_build(spec, emb_dim)
    print(
        f"  [ln_qkv 2/2] fused QKV GEMM (drain {tm}x{tn}x{tk1}, N={qkv_dim}, bias fused)..."
    )
    qkv_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            qkv_dim,
            tm,
            tk2,
            tk1,
            tn,
            hm,
            hn,
            b_pad_rows=BIAS_PAD_ROWS,
            **bkw,
            **kw,
        )
    )

    arg3_type = b_arg_type(spec, emb_dim, qkv_dim, hn)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{2*emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", arg3_type),
        FuncArg("%arg4", f"memref<{seq_len}x{qkv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(ln_ir, "ln", {0: 0, 1: 1, 2: 2}, extern_syms=_ln_externs()),
        KernelSlice(
            qkv_ir, "qkv", {0: 2, 1: 3, 2: 4}, extern_syms=_gemm_externs(spec, True)
        ),
    ]

    module = stitch_elf(
        "vit_ln_qkv",
        base_args,
        slices,
        debug_dump_path="/tmp/debug_vit_ln_qkv.mlir",
    )
    print(f"  vit_ln_qkv module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# ===========================================================================
# Group B: vit_o_ffn — O GEMM + bias + residual + LN2 + fc1 + bias + GELU
#          + fc2 + bias + residual (6 launches)
# ===========================================================================


def build_vit_o_ffn_module(
    seq_len, emb_dim, hidden_dim, herd_m=8, herd_n=4, registry_seq_len=None
):
    """Fused O-proj + residual + LN2 + FFN(fc1/GELU/fc2) + residual for one block.

    Func args:
      %arg0  attn     (seq, emb)          FA output (block-local input)
      %arg1  wo       (packed_k, emb)     O weight, bias-packed
      %arg2  o_b      (seq, emb)          O incl. bias (intermediate)
      %arg3  x_res    (seq, emb)          residual (block input x)
      %arg4  res1     (seq, emb)          o_b + x_res (intermediate; +FFN residual)
      %arg5  ln2_param(2*emb,)            LN2 gamma||beta packed
      %arg6  normed2  (seq, emb)          LN2 out (intermediate)
      %arg7  w_fc1    (packed_k, hidden)  fc1 weight, bias-packed
      %arg8  gelu_out (seq, hidden)       fc1 incl. bias AND GELU (intermediate)
      %arg9  w_fc2    (packed_k, emb)     fc2 weight, bias-packed
      %arg10 fc2_b    (seq, emb)          fc2 incl. bias (intermediate)
      %arg11 output   (seq, emb)          fc2_b + res1 — OUTPUT

    The three bias-add launches AND the GELU launch are gone: each GEMM carries
    its bias on the weight stream, and fc1 additionally applies GELU, all inside
    the drain herd's epilogue cast.
    """
    specs = vit_gemm_specs(seq_len, emb_dim, hidden_dim)
    o_spec, g_spec, d_spec = specs["o"], specs["fc1"], specs["fc2"]

    def gemm(spec, k, n, **extra):
        kw, tm, _, tk1, tn = _gemm_tiles(spec)
        hm, hn = _herd(spec, herd_m, herd_n)
        tk2, bkw = _b_build(spec, k)
        ir = _build_gemm_module(
            seq_len,
            k,
            n,
            tm,
            tk2,
            tk1,
            tn,
            hm,
            hn,
            b_pad_rows=BIAS_PAD_ROWS,
            **bkw,
            **extra,
            **kw,
        )
        return str(ir), b_arg_type(spec, k, n, hn)

    def tag(spec):
        return f"{spec['tile_m']}x{spec['tile_n']}x{spec['tile_k_l1']}"

    print(f"  [o_ffn 1/6] O GEMM (drain {tag(o_spec)}, bias fused)...")
    o_ir, o_b = gemm(o_spec, emb_dim, emb_dim)
    print("  [o_ffn 2/6] residual add (o_b + x_res)...")
    res1_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))
    print("  [o_ffn 3/6] LN2 (affine)...")
    ln2_ir = _layer_norm_ir(seq_len, emb_dim)
    print(f"  [o_ffn 4/6] fc1 GEMM (drain {tag(g_spec)}, bias+GELU fused)...")
    fc1_ir, g_b = gemm(g_spec, emb_dim, hidden_dim, epilogue_gelu=True)
    print(f"  [o_ffn 5/6] fc2 GEMM (drain {tag(d_spec)}, bias fused)...")
    fc2_ir, d_b = gemm(d_spec, hidden_dim, emb_dim)
    print("  [o_ffn 6/6] residual add (fc2_b + res1)...")
    res2_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", o_b),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{2*emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", g_b),
        FuncArg("%arg8", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg9", d_b),
        FuncArg("%arg10", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{emb_dim}xbf16>"),
    ]

    # GEMM privates: emit each distinct sym_suffix's decls exactly once; fc1's GELU cast
    # is its own decl, so it is keyed apart even when fc1 shares O's or fc2's object.
    _seen = set()

    def _pf(spec, kind="plain"):
        key = (spec["sym_suffix"], kind)
        first = key not in _seen
        _seen.add(key)
        return first

    slices = [
        KernelSlice(
            o_ir,
            "o",
            {0: 0, 1: 1, 2: 2},
            extern_syms=_gemm_externs(o_spec, True),
            private_from=_pf(o_spec),
        ),
        KernelSlice(res1_ir, "r1", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(
            ln2_ir,
            "ln2",
            {0: 4, 1: 5, 2: 6},
            extern_syms=_ln_externs(),
            # The C++ kernel's declaration is not among the GEMM slices', so this
            # slice has to bring it (the air.api version's externs are already in).
            private_from=LN_EXT,
        ),
        KernelSlice(
            fc1_ir,
            "g",
            {0: 6, 1: 7, 2: 8},
            extern_syms=_gemm_externs(g_spec, True, gelu=True),
            private_from=_pf(g_spec, "gelu"),
        ),
        KernelSlice(
            fc2_ir,
            "d",
            {0: 8, 1: 9, 2: 10},
            extern_syms=_gemm_externs(d_spec, True),
            private_from=_pf(d_spec),
        ),
        KernelSlice(res2_ir, "r2", {0: 10, 1: 4, 2: 11}, private_from=False),
    ]

    module = stitch_elf(
        "vit_o_ffn",
        base_args,
        slices,
        debug_dump_path="/tmp/debug_vit_o_ffn.mlir",
    )
    print(f"  vit_o_ffn module: {len(str(module).splitlines())} lines, parsed OK")
    return module
