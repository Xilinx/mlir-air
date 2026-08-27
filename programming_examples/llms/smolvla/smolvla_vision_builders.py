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
from air.dialects.arith import ConstantOp

from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
)
from shared.builders.gemm_builder import (
    _build_gemm_module,
    BIAS_PAD_ROWS,
    packed_k,
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


@module_builder
def _build_gelu_2d(rows, cols, tile_n, np_dtype, herd_x=8, herd_y=2, vector_size=16):
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    n = rows * cols
    total_tiles = herd_x * herd_y
    assert n % (tile_n * total_tiles) == 0, (n, tile_n, total_tiles)
    assert tile_n % vector_size == 0

    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
    index_type = T.index()

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty)
    def gelu2d(arg0_2d, arg1_2d):
        @launch(operands=[arg0_2d, arg1_2d])
        def g_launch(l_in, l_out):
            in_flat = memref_collapse_shape(l3_1d_ty, l_in, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out, [[0, 1]])

            @segment(name="gelu2d_seg", operands=[in_flat, out_flat])
            def g_seg(s_in, s_out):
                @herd(
                    name="gelu2d_herd", sizes=[herd_x, herd_y], operands=[s_in, s_out]
                )
                def g_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    l1_in = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    for _iv in range_(0, n, tile_n * total_tiles):
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
                                        AffineConstantExpr.get(tile_n),
                                    ),
                                )
                            ],
                        )
                        offset = affine_apply(offset_map, [_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        c0 = ConstantOp(index_type, 0)
                        cVec = ConstantOp(index_type, vector_size)
                        cTile = ConstantOp(index_type, tile_n)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        v_half = _bcast(vec_ty, xrt_dtype, 0.5)
                        v_one = _bcast(vec_ty, xrt_dtype, 1.0)
                        v_beta = _bcast(vec_ty, xrt_dtype, _GELU_BETA)
                        v_s2opi = _bcast(vec_ty, xrt_dtype, _SQRT_2_OVER_PI)
                        for j in range_(c0, cTile, cVec):
                            si = subview(l1_in.result, [j], [vector_size], [1])
                            so = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(
                                vec_ty, si, [c0], identity_map, cst0, [True]
                            )
                            v_x2 = arith.mulf(v_x, v_x)
                            v_x3 = arith.mulf(v_x, v_x2)
                            v_bx3 = arith.mulf(v_x3, v_beta)
                            v_inner = arith.addf(v_x, v_bx3)
                            v_scaled = arith.mulf(v_inner, v_s2opi)
                            v_tanh = math_dialect.tanh(v_scaled)
                            v_opt = arith.addf(v_tanh, v_one)
                            v_hx = arith.mulf(v_x, v_half)
                            v_gelu = arith.mulf(v_hx, v_opt)
                            transfer_write(None, v_gelu, so, [c0], identity_map, [True])
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(l1_in)
                    DeallocOp(l1_out)


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


# ===========================================================================
# Group A: vit_ln_qkv — LN1 + one fused [Q|K|V] GEMM (2 launches)
# ===========================================================================


def build_vit_ln_qkv_module(seq_len, emb_dim, n_heads, head_dim, herd_m=8, herd_n=4):
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
    spec = dict(gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high"))
    assert spec["method"] == "drain", spec["method"]  # vision qkvo is drain
    # Force a tile_n-keyed suffix (`_m32_n96`) so this ELF links the SAME
    # correctly-compiled object as vit_o_ffn's O GEMM (both tile_n=96 drain),
    # NOT the generic `mm_m32.o` (whose baked DIM_N may be stale from another
    # model's build). Mirrors disambiguate_by_tile_n's naming.
    spec = _force_tile_n_suffix(spec)

    print("  [ln_qkv 1/2] LN1 (affine)...")
    ln_ir = _wrap_ir_in_launch(
        str(build_layer_norm(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    kw, tm, tk2, tk1, tn = _gemm_tiles(spec)
    qkv_dim = 3 * emb_dim
    assert qkv_dim % (tn * herd_n) == 0, (qkv_dim, tn, herd_n)
    print(
        f"  [ln_qkv 2/2] fused QKV GEMM ({spec['method']} tile_n={tn}, "
        f"N={qkv_dim}, bias fused)..."
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
            herd_m,
            herd_n,
            b_pad_rows=BIAS_PAD_ROWS,
            **kw,
        )
    )

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{2*emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{packed_k(emb_dim, tk1)}x{qkv_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{qkv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            ln_ir, "ln", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
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


def build_vit_o_ffn_module(seq_len, emb_dim, hidden_dim, herd_m=8, herd_n=4):
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

    print(f"  [o_ffn 1/6] O GEMM (drain tn={otn}, bias fused)...")
    o_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            otm,
            ok2,
            ok1,
            otn,
            herd_m,
            herd_n,
            b_pad_rows=BIAS_PAD_ROWS,
            **ok,
        )
    )
    print("  [o_ffn 2/6] residual add (o_b + x_res)...")
    res1_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))
    print("  [o_ffn 3/6] LN2 (affine)...")
    ln2_ir = _wrap_ir_in_launch(
        str(build_layer_norm(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )
    print(f"  [o_ffn 4/6] fc1 GEMM (drain tn={gtn}, bias+GELU fused)...")
    fc1_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gtm,
            gk2,
            gk1,
            gtn,
            herd_m,
            herd_n,
            b_pad_rows=BIAS_PAD_ROWS,
            epilogue_gelu=True,
            **gk,
        )
    )
    print(f"  [o_ffn 5/6] fc2 GEMM (drain tn={dtn}, bias fused)...")
    fc2_ir = str(
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            dtm,
            dk2,
            dk1,
            dtn,
            herd_m,
            herd_n,
            b_pad_rows=BIAS_PAD_ROWS,
            **dk,
        )
    )
    print("  [o_ffn 6/6] residual add (fc2_b + res1)...")
    res2_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{packed_k(emb_dim, ok1)}x{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{2*emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{packed_k(emb_dim, gk1)}x{hidden_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{packed_k(hidden_dim, dk1)}x{emb_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{emb_dim}xbf16>"),
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
            extern_syms=_gemm_externs(o_spec, True),
            private_from=_pf(o_spec),
        ),
        KernelSlice(res1_ir, "r1", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(
            ln2_ir,
            "ln2",
            {0: 4, 1: 5, 2: 6},
            extern_syms={"@zero_vectorized_bf16"},
            private_from=False,
        ),
        KernelSlice(
            fc1_ir,
            "g",
            {0: 6, 1: 7, 2: 8},
            extern_syms=_gemm_externs(g_spec, True, gelu=True),
            private_from=_pf(g_spec),
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
