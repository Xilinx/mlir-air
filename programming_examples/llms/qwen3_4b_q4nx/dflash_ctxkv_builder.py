# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The DFlash drafter's CONTEXT K/V, as one multi-launch AIR module.
#
#     target_hidden [CTX, 2560]  --k/v_proj_L-->  [CTX, 2048] per layer L
#
# WHY THIS EXISTS AT ALL. A drafter layer projects K and V from
# cat[target_hidden, hidden_states] where a target layer projects them from its
# own hidden state alone (_dflash_upstream/model.py:384-387). The block half of
# that concatenation is what the decode engine already appends every dispatch;
# the CONTEXT half is not, and it cannot be produced by simply running the
# context rows through a decode pass, because the engine RMS-norms its input
# before projecting and `target_hidden` reaches k/v_proj RAW (model.py:446-453
# hands it to self_attn without input_layernorm).
#
# Pre-compensating the input to cancel that norm does not work: RMSNorm is not
# invertible that way, and the leftover scale would land in v_ctx, where
# k_norm's scale-invariance cannot absorb it. So the context projection is its
# own pass.
#
# WHAT MAKES IT CHEAP: dflash_draft_decomp.py measures that `target_hidden` is
# LAYER-INVARIANT -- it never flows through the stack -- so all five layers'
# context K/V are a function of the taps alone and can be computed once, before
# the layer loop, and written into the KV cache. After that the drafter is an
# ordinary bidirectional decode (section 3.2).
#
# K AND V ARE SEPARATE GEMMs, and that is forced by what comes after. Fusing
# them into one [2048, 2560] projection halves the launch count and was the
# first thing built (it passes on device at 9.6e-03). But `k_norm` is an
# RMSNorm over head_dim, i.e. over k viewed as [ctx*8, 128] -- and in a fused
# [ctx, 2048] output the k half is a STRIDED region, not a contiguous
# [ctx*8, 128]. Splitting the projection is cheaper than teaching the norm to
# stride; the weight traffic is identical either way.
#
# Tiles/herd follow the same constraints the fc launch ran into: tile_m is
# forced to 32 by gemm_method_spec('drain'), mm_aie2p.cc static-asserts
# DIM_M % (2*r) == 0, and M % (tile_m*herd_m) == 0 then forces a 1x4 herd with
# M a multiple of 32.

CTX_PAD = 32
D = 2560
KV_DIM = 1024  # 8 kv heads x 128
KV2 = 2 * KV_DIM  # k and v, when fused (build_ctxkv_module)
HEAD_DIM = 128
N_KV_HEADS = KV_DIM // HEAD_DIM  # 8
N_LAYERS = 5

TILE_M = 32
HERD_M = 1
HERD_N = 4
TILE_K_L2 = 128  # 2560 = 128 * 20
TILE_K_L1 = 32  # 128 = 32 * 4
TILE_N = 64  # divides both 2048 and 1024; tile_k_l2*tile_n*2 = 16 KB of L1


def _paths():
    import sys
    from pathlib import Path

    pe = Path(__file__).resolve().parent.parent.parent
    for p in (pe / "llms", pe, pe / "weighted_rms_norm"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return pe


def gemm_externs():
    _paths()
    from shared.builders.gemm_builder import gemm_method_spec

    sfx = gemm_method_spec("drain")["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


def build_ctxkv_module(ctx_pad=CTX_PAD, n_layers=N_LAYERS):
    """`n_layers` air.launch ops in one func: the per-layer context K/V.

    Args (bf16):
        %arg0                 target_hidden [ctx_pad, 2560]  (fc's output)
        %arg(1+2L)            kv_w_L        [2560, 2048]     concat(k_proj, v_proj), transposed
        %arg(2+2L)            kv_out_L      [ctx_pad, 2048]  [k_ctx | v_ctx]
    """
    _paths()
    from shared.builders.gemm_builder import _build_gemm_module, gemm_method_spec
    from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

    kw = dict(gemm_method_spec("drain")["build_kwargs"])
    gemm_ir = str(
        _build_gemm_module(
            ctx_pad,
            D,
            KV2,
            TILE_M,
            TILE_K_L2,
            TILE_K_L1,
            TILE_N,
            herd_m=HERD_M,
            herd_n=HERD_N,
            **kw,
        )
    )
    ext = gemm_externs()

    base_args = [FuncArg("%arg0", f"memref<{ctx_pad}x{D}xbf16>")]
    slices = []
    for L in range(n_layers):
        base_args.append(FuncArg(f"%arg{1+2*L}", f"memref<{D}x{KV2}xbf16>"))
        base_args.append(FuncArg(f"%arg{2+2*L}", f"memref<{ctx_pad}x{KV2}xbf16>"))
        slices.append(
            KernelSlice(
                gemm_ir, f"kv{L}", {0: 0, 1: 1 + 2 * L, 2: 2 + 2 * L}, extern_syms=ext
            )
        )
    return stitch_elf("dflash_ctxkv", base_args, slices)


def build_ctxkv_split_module(ctx_pad=CTX_PAD, n_layers=N_LAYERS, with_knorm=True):
    """Per layer: a K GEMM, a V GEMM, and (optionally) k_norm over K.

    K lands in its own contiguous [ctx_pad, 1024] buffer precisely so it can be
    viewed as [ctx_pad*8, 128] and normed per head without striding.

    Args (bf16):
        %arg0                target_hidden [ctx_pad, 2560]
        per layer L, base b = 1 + 4*L:
        %arg(b+0)            k_w_L   [2560, 1024]
        %arg(b+1)            k_raw_L [ctx_pad, 1024]   pre-norm K
        %arg(b+2)            v_w_L   [2560, 1024]
        %arg(b+3)            v_ctx_L [ctx_pad, 1024]   V is final here
        then, if with_knorm, after all layers, base nb = 1 + 4*N:
        %arg(nb+2L)          k_norm_w_L [128]            PER LAYER, not shared
        %arg(nb+2L+1)        k_ctx_L    [ctx_pad*8, 128] normed K
    """
    _paths()
    from ml_dtypes import bfloat16

    from shared.builders.gemm_builder import _build_gemm_module, gemm_method_spec
    from shared.infra.stitching import (
        FuncArg,
        KernelSlice,
        stitch_elf,
        _wrap_ir_in_launch,
    )
    from weighted_rms_norm import build_module as build_rms

    kw = dict(gemm_method_spec("drain")["build_kwargs"])
    gemm_ir = str(
        _build_gemm_module(
            ctx_pad,
            D,
            KV_DIM,
            TILE_M,
            TILE_K_L2,
            TILE_K_L1,
            TILE_N,
            herd_m=HERD_M,
            herd_n=HERD_N,
            **kw,
        )
    )
    ext = gemm_externs()
    rows = ctx_pad * N_KV_HEADS

    base_args = [FuncArg("%arg0", f"memref<{ctx_pad}x{D}xbf16>")]
    slices = []
    for L in range(n_layers):
        b = 1 + 4 * L
        base_args += [
            FuncArg(f"%arg{b+0}", f"memref<{D}x{KV_DIM}xbf16>"),
            FuncArg(f"%arg{b+1}", f"memref<{ctx_pad}x{KV_DIM}xbf16>"),
            FuncArg(f"%arg{b+2}", f"memref<{D}x{KV_DIM}xbf16>"),
            FuncArg(f"%arg{b+3}", f"memref<{ctx_pad}x{KV_DIM}xbf16>"),
        ]
        slices += [
            KernelSlice(gemm_ir, f"k{L}", {0: 0, 1: b + 0, 2: b + 1}, extern_syms=ext),
            KernelSlice(gemm_ir, f"v{L}", {0: 0, 1: b + 2, 2: b + 3}, extern_syms=ext),
        ]

    prelude = ""
    if with_knorm:
        # k_norm is per-head -- Qwen3RMSNorm is constructed on head_dim, so one
        # 128-vector serves all 8 heads of a layer. It is NOT shared BETWEEN
        # layers: an earlier version of this builder passed layer 0's weight to
        # all five, and every layer still gated clean because the reference made
        # the same assumption. dflash_ctxkv_gate.py now compares the five
        # checkpoint tensors directly, which is what caught it.
        nb = 1 + 4 * n_layers
        rms_ir = _wrap_ir_in_launch(
            str(build_rms(rows, HEAD_DIM, bfloat16, 16, herd_x=1))
        )
        # The K GEMM writes [ctx_pad, 1024]; k_norm wants the SAME bytes as
        # [ctx_pad*8, 128], one row per (position, head). Rather than spend a
        # second buffer and a copy, the prelude reshapes the GEMM's output in
        # place and the norm launch is aliased onto it -- which is what
        # arg_aliases is for.
        # reinterpret_cast, NOT collapse_shape + expand_shape: `aie.dma_bd`
        # accepts a buffer rooted at a subview/view/cast/reinterpret_cast chain
        # and rejects anything else, so the reshape pair lowers fine right up to
        # DMA lowering and then fails with "Buffer argument must be a constant
        # aie.buffer, a runtime sequence input argument, or a supported chain".
        prelude = "\n".join(
            f"    %k2d{L} = memref.reinterpret_cast %arg{1 + 4 * L + 1} to "
            f"offset: [0], sizes: [{rows}, {HEAD_DIM}], strides: [{HEAD_DIM}, 1] "
            f": memref<{ctx_pad}x{KV_DIM}xbf16> to memref<{rows}x{HEAD_DIM}xbf16>"
            for L in range(n_layers)
        )
        for L in range(n_layers):
            base_args += [
                FuncArg(f"%arg{nb+2*L}", f"memref<{HEAD_DIM}xbf16>"),
                FuncArg(f"%arg{nb+2*L+1}", f"memref<{rows}x{HEAD_DIM}xbf16>"),
            ]
            slices.append(
                KernelSlice(
                    rms_ir,
                    f"kn{L}",
                    {1: nb + 2 * L, 2: nb + 2 * L + 1},
                    arg_aliases={0: f"%k2d{L}"},
                    extern_syms={"@zero_vectorized_bf16"},
                )
            )
    return stitch_elf("dflash_ctxkv_split", base_args, slices, prelude=prelude)


def layer_kv_weight(dw, layer):
    """[2560, 2048] -- concat(k_proj, v_proj) transposed into the GEMM's [K, N].

    The checkpoint stores both as [out, in]; the GEMM wants B as [K, N], so the
    concatenation happens along the OUTPUT axis and the whole thing is
    transposed once. k occupies columns 0..1023 and v 1024..2047, which is the
    order `reference` and the consumer both assume.
    """
    import numpy as np

    k = np.asarray(dw.bf16(f"layers.{layer}.self_attn.k_proj.weight"))
    v = np.asarray(dw.bf16(f"layers.{layer}.self_attn.v_proj.weight"))
    return np.ascontiguousarray(np.concatenate([k, v], axis=0).T)


def reference(target_hidden, kv_w_T):
    """numpy [ctx, 2048] = target_hidden @ kv_w_T, in f32."""
    import numpy as np

    return np.asarray(target_hidden, np.float32) @ np.asarray(kv_w_T, np.float32)


if __name__ == "__main__":
    import sys

    m = build_ctxkv_module()
    txt = str(m)
    n = txt.count("air.launch")
    print(f"[ctxkv] {len(txt.splitlines())} lines, {n} air.launch ops, parsed OK")
    print(
        f"[ctxkv] M={CTX_PAD} K={D} N={KV2} x {N_LAYERS} layers, herd {HERD_M}x{HERD_N}"
    )
    sys.exit(0 if n == N_LAYERS else 1)
