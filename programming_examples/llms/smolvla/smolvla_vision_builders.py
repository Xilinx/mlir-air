# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SigLIP ViT fused multi-launch ELF builders (A3-6b perf hill-climb).

Two fused ELFs, mirroring the SmolVLA backbone's rms_gemms_rope / o_ffn builders
but re-parameterized for the SigLIP vision encoder (affine LayerNorm instead of
RMSNorm, GELU-tanh instead of SwiGLU, a per-Linear BIAS on every projection, and
NO RoPE). Building these two ELFs collapses the unfused 10-dispatch/layer path
(LN1, q, k, v, +host bias×3, +host residual, LN2, fc1, +host bias, gelu, fc2,
+host bias, +host residual) into 3 on-device dispatches/layer:

  vit_ln_qkv  (7 launches): LN1 + Q/K/V GEMM + Q/K/V bias-add  -> q_b, k_b, v_b
  flash_attn  (1 launch,  unchanged registry ELF)
  vit_o_ffn  (10 launches): O GEMM + O bias + residual + LN2 + fc1 GEMM +
                            fc1 bias + GELU + fc2 GEMM + fc2 bias + residual

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

# smolvla/ -> programming_examples (for kernel example imports: layer_norm, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# smolvla/ -> llms (for shared.infra / shared.builders)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from shared.builders.rms_gemms_rope_multi import _api_dtype
from air.dialects.arith import ConstantOp

from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
)
from shared.builders.gemm_builder import (
    _build_gemm_module,
    gemm_registry_config,
    disambiguate_by_tile_n,
)
from shared.builders.rms_qkv_bias_rope_multi import _build_bias_add_2d
from shared.builders.o_ffn_multi import _build_add_2d_to_2d
from layer_norm.layer_norm import build_module as build_layer_norm

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
    tag = "m32" if s["method"] == "drain" else "m64"
    suffix = f"_{tag}_n{s['tile_n']}"
    obj = f"mm_{tag}_n{s['tile_n']}.o"
    s["sym_suffix"] = suffix
    s["obj"] = obj
    s["build_kwargs"] = dict(s["build_kwargs"])
    s["build_kwargs"]["sym_suffix"] = suffix
    s["build_kwargs"]["link_with_name"] = obj
    return s


def _gemm_externs(spec):
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


def _gemm_tiles(spec):
    return (
        dict(spec["build_kwargs"]),
        spec["tile_m"],
        spec["tile_k_l2"],
        spec["tile_k_l1"],
        spec["tile_n"],
    )


# ===========================================================================
# Group A: vit_ln_qkv — LN1 + Q/K/V GEMM + Q/K/V bias-add (7 launches)
# ===========================================================================


def build_vit_ln_qkv_module(seq_len, emb_dim, n_heads, head_dim, herd_m=8, herd_n=4):
    """Fused LN1 + Q/K/V GEMM + per-channel Q/K/V bias-add for one SigLIP block.

    Func args (MHA, no GQA: q_dim == kv_dim == emb_dim):
      %arg0  x_in     (seq, emb)          block input (bf16)
      %arg1  ln_param (2*emb,)            LN1 gamma||beta packed
      %arg2  normed   (seq, emb)          LN1 out (intermediate)
      %arg3  wq       (emb, emb)          Q weight
      %arg4  q_raw    (seq, emb)          Q GEMM out (intermediate)
      %arg5  wk       (emb, emb)
      %arg6  k_raw    (seq, emb)          (intermediate)
      %arg7  wv       (emb, emb)
      %arg8  v_raw    (seq, emb)          (intermediate)
      %arg9  bq       (emb,)              Q bias
      %arg10 bk       (emb,)
      %arg11 bv       (emb,)
      %arg12 q_b      (seq, emb)          Q after bias (FA input) — OUTPUT
      %arg13 k_b      (seq, emb)          K after bias (FA input) — OUTPUT
      %arg14 v_b      (seq, emb)          V after bias (FA input) — OUTPUT
    """
    spec = dict(gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high"))
    assert spec["method"] == "drain", spec["method"]  # vision qkvo is drain
    # Force a tile_n-keyed suffix (`_m32_n96`) so this ELF links the SAME
    # correctly-compiled object as vit_o_ffn's O GEMM (both tile_n=96 drain),
    # NOT the generic `mm_m32.o` (whose baked DIM_N may be stale from another
    # model's build). Mirrors disambiguate_by_tile_n's naming.
    spec = _force_tile_n_suffix(spec)

    print("  [ln_qkv 1/7] LN1 (affine)...")
    ln_ir = _wrap_ir_in_launch(
        str(build_layer_norm(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    kw, tm, tk2, tk1, tn = _gemm_tiles(spec)
    print(f"  [ln_qkv 2-4/7] Q/K/V GEMM ({spec['method']} tile_n={tn})...")
    q_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, emb_dim, tm, tk2, tk1, tn, herd_m, herd_n, **kw
        )
    )
    k_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, emb_dim, tm, tk2, tk1, tn, herd_m, herd_n, **kw
        )
    )
    v_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, emb_dim, tm, tk2, tk1, tn, herd_m, herd_n, **kw
        )
    )

    print("  [ln_qkv 5-7/7] Q/K/V bias-add (broadcast)...")
    # in_cols == real_cols == emb_dim (drain tile_n=96 divides 768, no N padding).
    bias_q_ir = str(_build_bias_add_2d(seq_len, emb_dim, emb_dim, bfloat16, 8))
    bias_k_ir = str(_build_bias_add_2d(seq_len, emb_dim, emb_dim, bfloat16, 8))
    bias_v_ir = str(_build_bias_add_2d(seq_len, emb_dim, emb_dim, bfloat16, 8))

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{2*emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{seq_len}x{emb_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            ln_ir, "ln", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(q_ir, "q", {0: 2, 1: 3, 2: 4}, extern_syms=_gemm_externs(spec)),
        KernelSlice(
            k_ir,
            "k",
            {0: 2, 1: 5, 2: 6},
            extern_syms=_gemm_externs(spec),
            private_from=False,
        ),
        KernelSlice(
            v_ir,
            "v",
            {0: 2, 1: 7, 2: 8},
            extern_syms=_gemm_externs(spec),
            private_from=False,
        ),
        KernelSlice(bias_q_ir, "bq", {0: 4, 1: 9, 2: 12}, private_from=False),
        KernelSlice(bias_k_ir, "bk", {0: 6, 1: 10, 2: 13}, private_from=False),
        KernelSlice(bias_v_ir, "bv", {0: 8, 1: 11, 2: 14}, private_from=False),
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
#          + fc2 + bias + residual (10 launches)
# ===========================================================================


def build_vit_o_ffn_module(seq_len, emb_dim, hidden_dim, herd_m=8, herd_n=4):
    """Fused O-proj + residual + LN2 + FFN(fc1/GELU/fc2) + residual for one block.

    Func args:
      %arg0  attn     (seq, emb)          FA output (block-local input)
      %arg1  wo       (emb, emb)          O weight
      %arg2  o_raw    (seq, emb)          O GEMM out (intermediate)
      %arg3  bo       (emb,)              O bias
      %arg4  o_b      (seq, emb)          O after bias (intermediate)
      %arg5  x_res    (seq, emb)          residual (block input x)
      %arg6  res1     (seq, emb)          o_b + x_res (intermediate; +FFN residual)
      %arg7  ln2_param(2*emb,)            LN2 gamma||beta packed
      %arg8  normed2  (seq, emb)          LN2 out (intermediate)
      %arg9  w_fc1    (emb, hidden)       fc1 weight
      %arg10 fc1_raw  (seq, hidden)       fc1 GEMM out (intermediate)
      %arg11 b_fc1    (hidden,)           fc1 bias
      %arg12 fc1_b    (seq, hidden)       fc1 after bias (intermediate)
      %arg13 gelu_out (seq, hidden)       GELU out (intermediate)
      %arg14 w_fc2    (hidden, emb)       fc2 weight
      %arg15 fc2_raw  (seq, emb)          fc2 GEMM out (intermediate)
      %arg16 b_fc2    (emb,)              fc2 bias
      %arg17 fc2_b    (seq, emb)          fc2 after bias (intermediate)
      %arg18 output   (seq, emb)          fc2_b + res1 — OUTPUT
    """
    o_spec = dict(gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high"))
    g_spec = dict(gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high"))
    d_spec = dict(gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high"))
    # O (tn96) + fc1 (tn128) + fc2 (tn96), all drain → disambiguate by tile_n.
    o_spec, g_spec, d_spec = disambiguate_by_tile_n([o_spec, g_spec, d_spec])
    for s in (o_spec, g_spec, d_spec):
        assert s["method"] == "drain", s["method"]

    ok, otm, ok2, ok1, otn = _gemm_tiles(o_spec)
    gk, gtm, gk2, gk1, gtn = _gemm_tiles(g_spec)
    dk, dtm, dk2, dk1, dtn = _gemm_tiles(d_spec)

    print(f"  [o_ffn 1/10] O GEMM (drain tn={otn})...")
    o_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, emb_dim, otm, ok2, ok1, otn, herd_m, herd_n, **ok
        )
    )
    print("  [o_ffn 2/10] O bias-add...")
    bias_o_ir = str(_build_bias_add_2d(seq_len, emb_dim, emb_dim, bfloat16, 8))
    print("  [o_ffn 3/10] residual add (o_b + x_res)...")
    res1_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))
    print("  [o_ffn 4/10] LN2 (affine)...")
    ln2_ir = _wrap_ir_in_launch(
        str(build_layer_norm(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )
    print(f"  [o_ffn 5/10] fc1 GEMM (drain tn={gtn})...")
    fc1_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, hidden_dim, gtm, gk2, gk1, gtn, herd_m, herd_n, **gk
        )
    )
    print("  [o_ffn 6/10] fc1 bias-add...")
    bias_fc1_ir = str(_build_bias_add_2d(seq_len, hidden_dim, hidden_dim, bfloat16, 8))
    print("  [o_ffn 7/10] GELU-tanh (2D)...")
    gelu_ir = str(
        _build_gelu_2d(seq_len, hidden_dim, 4096, bfloat16, herd_x=8, herd_y=2)
    )
    print(f"  [o_ffn 8/10] fc2 GEMM (drain tn={dtn})...")
    fc2_ir = str(
        _build_gemm_module(
            seq_len, hidden_dim, emb_dim, dtm, dk2, dk1, dtn, herd_m, herd_n, **dk
        )
    )
    print("  [o_ffn 9/10] fc2 bias-add...")
    bias_fc2_ir = str(_build_bias_add_2d(seq_len, emb_dim, emb_dim, bfloat16, 8))
    print("  [o_ffn 10/10] residual add (fc2_b + res1)...")
    res2_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{2*emb_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{emb_dim}x{hidden_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{hidden_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg15", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg16", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg17", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg18", f"memref<{seq_len}x{emb_dim}xbf16>"),
    ]

    # GEMM privates: emit each distinct sym_suffix's decls exactly once.
    _seen = set()

    def _pf(spec):
        first = spec["sym_suffix"] not in _seen
        _seen.add(spec["sym_suffix"])
        return first

    slices = [
        KernelSlice(
            o_ir,
            "o",
            {0: 0, 1: 1, 2: 2},
            extern_syms=_gemm_externs(o_spec),
            private_from=_pf(o_spec),
        ),
        KernelSlice(bias_o_ir, "bo", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(res1_ir, "r1", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(
            ln2_ir,
            "ln2",
            {0: 6, 1: 7, 2: 8},
            extern_syms={"@zero_vectorized_bf16"},
            private_from=False,
        ),
        KernelSlice(
            fc1_ir,
            "g",
            {0: 8, 1: 9, 2: 10},
            extern_syms=_gemm_externs(g_spec),
            private_from=_pf(g_spec),
        ),
        KernelSlice(bias_fc1_ir, "bg", {0: 10, 1: 11, 2: 12}, private_from=False),
        KernelSlice(gelu_ir, "ge", {0: 12, 1: 13}, private_from=False),
        KernelSlice(
            fc2_ir,
            "d",
            {0: 13, 1: 14, 2: 15},
            extern_syms=_gemm_externs(d_spec),
            private_from=_pf(d_spec),
        ),
        KernelSlice(bias_fc2_ir, "bd", {0: 15, 1: 16, 2: 17}, private_from=False),
        KernelSlice(res2_ir, "r2", {0: 17, 1: 6, 2: 18}, private_from=False),
    ]

    module = stitch_elf(
        "vit_o_ffn",
        base_args,
        slices,
        debug_dump_path="/tmp/debug_vit_o_ffn.mlir",
    )
    print(f"  vit_o_ffn module: {len(str(module).splitlines())} lines, parsed OK")
    return module
