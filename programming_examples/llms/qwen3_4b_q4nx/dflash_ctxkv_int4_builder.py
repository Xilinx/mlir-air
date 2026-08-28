# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The DFlash drafter's CONTEXT K/V in int4-AWQ -- the quantized form of
# dflash_ctxkv_builder.build_ctxkv_split_module.
#
#     target_hidden [CTX, 2560] --k_proj_L--> k_raw_L [CTX, 1024] --k_norm_L--> k_ctx_L
#                               --v_proj_L--> v_ctx_L [CTX, 1024]
#
# Read dflash_ctxkv_builder.py first: it carries the reasons this pass exists at
# all (the drafter projects K/V from cat[target_hidden, hidden_states] where the
# decode engine RMS-norms its input first, and `target_hidden` reaches k/v_proj
# raw), why `target_hidden` being layer-invariant makes it a single pre-pass,
# why K and V are separate GEMMs (k_norm needs a contiguous [ctx*8, 128] view),
# and why k_norm's weight is per layer.
#
# WHAT CHANGES IN int4. Only the GEMMs, and only their B operand: 52.4 MB of
# bf16 k/v weight becomes 13.8 MB packed. K is already 2560 here, so unlike fc
# there is nothing to split -- tile_k_l2 == K fits L2 at 168 KB. k_norm is
# untouched (bf16, one 128-vector per layer).
#
# THE LAUNCH SHAPE IS ALREADY AT THE LIMIT. A herd tile takes at most two
# incoming L3 streams (measured; see dflash_int4_fc_builder.py). Every launch
# here is two-in/one-out already -- a GEMM is (A, B, C) and k_norm is
# (in, weight, out) -- so the 15-launch split form ports across unchanged.

CTX_PAD = 32
D = 2560
KV_DIM = 1024  # 8 kv heads x 128
HEAD_DIM = 128
N_KV_HEADS = KV_DIM // HEAD_DIM  # 8
N_LAYERS = 5
# rope_rows = ctx_pad * n_kv_heads = 256 must divide by the tile count.
ROPE_HERD = 4


def ctxkv_parts(
    ctx_pad=CTX_PAD,
    n_layers=N_LAYERS,
    with_knorm=True,
    with_rope=True,
    base=0,
    th_arg=None,
):
    """(base_args, slices, prelude) -- the context K/V launches, ready to stitch.

    Split out from `build_ctxkv_int4_module` so the drafter pre-pass
    (dflash_draft_prepass.py) can put these launches in the SAME func as fc's
    and hand `target_hidden` straight from one to the other. `base` shifts every
    arg number; `th_arg` names an existing arg to read `target_hidden` from (fc's
    output) instead of declaring one.

    Args (offset by `base`):
        %arg0                 target_hidden [ctx_pad, 2560] bf16  (unless th_arg)
        per layer L, base b = 1 + 4*L:
        %arg(b+0)             k_w_L   packed int4                i8
        %arg(b+1)             k_raw_L [ctx_pad, 1024]            bf16  pre-norm K
        %arg(b+2)             v_w_L   packed int4                i8
        %arg(b+3)             v_ctx_L [ctx_pad, 1024]            bf16  V is final
        then, if with_knorm, nb = 1 + 4*n_layers:
        %arg(nb+2L)           k_norm_w_L [128]                   bf16  PER LAYER
        %arg(nb+2L+1)         k_nrm_L    [ctx_pad*8, 128]        bf16  normed K
        then, if with_rope, rb = nb + 2*n_layers:
        %arg(rb)              rope_lut   [ctx_pad*8*128]         bf16  SHARED
        %arg(rb+1+L)          k_ctx_L    [ctx_pad*8, 128]        bf16  final K
    """
    import sys

    import dflash_int4 as I

    pe = I.paths()
    # I.paths() covers the int4 GEMM builder and packer; k_norm and RoPE are
    # elsewhere.
    if str(pe / "weighted_rms_norm") not in sys.path:
        sys.path.insert(0, str(pe / "weighted_rms_norm"))
    from ml_dtypes import bfloat16

    from shared.builders.rms_gemms_rope_multi import _build_rope_2d as build_rope_2d
    from shared.infra.stitching import (
        FuncArg,
        KernelSlice,
        stitch_elf,
        _wrap_ir_in_launch,
    )
    from weighted_rms_norm import build_module as build_rms

    gemm_ir = I.build_int4_gemm_ir(ctx_pad, D, KV_DIM)
    bshape = I.packed_shape(D, KV_DIM)
    b_ty = f"memref<{bshape[0]}x{bshape[1]}x{bshape[2]}xi8>"
    rows = ctx_pad * N_KV_HEADS

    # When th_arg is given the caller already owns that buffer, so the slot is
    # not declared here and every later arg number shifts down by one.
    th = base if th_arg is None else th_arg
    off = base + (1 if th_arg is None else 0)
    base_args = (
        [FuncArg(f"%arg{base}", f"memref<{ctx_pad}x{D}xbf16>")]
        if th_arg is None
        else []
    )
    slices = []
    for L in range(n_layers):
        b = off + 4 * L
        base_args += [
            FuncArg(f"%arg{b+0}", b_ty),
            FuncArg(f"%arg{b+1}", f"memref<{ctx_pad}x{KV_DIM}xbf16>"),
            FuncArg(f"%arg{b+2}", b_ty),
            FuncArg(f"%arg{b+3}", f"memref<{ctx_pad}x{KV_DIM}xbf16>"),
        ]
        slices += [
            KernelSlice(
                gemm_ir,
                f"k{L}",
                {0: th, 1: b + 0, 2: b + 1},
                extern_syms=I.EXTERN_SYMS,
            ),
            KernelSlice(
                gemm_ir,
                f"v{L}",
                {0: th, 1: b + 2, 2: b + 3},
                extern_syms=I.EXTERN_SYMS,
            ),
        ]

    prelude = ""
    if with_knorm:
        nb = off + 4 * n_layers
        rms_ir = _wrap_ir_in_launch(
            str(build_rms(rows, HEAD_DIM, bfloat16, 16, herd_x=1))
        )
        # reinterpret_cast, NOT collapse_shape + expand_shape: `aie.dma_bd`
        # accepts a buffer rooted at a subview/view/cast/reinterpret_cast chain
        # and rejects anything else.
        prelude = "\n".join(
            f"    %k2d{L} = memref.reinterpret_cast %arg{off + 4 * L + 1} to "
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

        if with_rope:
            # AFTER k_norm, not before: upstream normalises the concatenated
            # [k_ctx | k_noise] and only then rotates (model.py:388-393).
            #
            # ONE LUT FOR ALL FIVE LAYERS. Qwen3 builds cos/sin from position
            # alone, once per forward (model.py:585), so the layers share it --
            # it is a read-only third stream, and each rope launch is still
            # two-in/one-out.
            rb = nb + 2 * n_layers
            rope_ir = str(build_rope_2d(rows, HEAD_DIM, HEAD_DIM, bfloat16, ROPE_HERD))
            base_args.append(FuncArg(f"%arg{rb}", f"memref<{rows * HEAD_DIM}xbf16>"))
            for L in range(n_layers):
                base_args.append(
                    FuncArg(f"%arg{rb+1+L}", f"memref<{rows}x{HEAD_DIM}xbf16>")
                )
                slices.append(
                    KernelSlice(
                        rope_ir,
                        f"rp{L}",
                        {0: nb + 2 * L + 1, 1: rb, 2: rb + 1 + L},
                        extern_syms={"@rope"},
                    )
                )

    return base_args, slices, prelude


def build_ctxkv_int4_module(
    ctx_pad=CTX_PAD, n_layers=N_LAYERS, with_knorm=True, with_rope=True
):
    """The context K/V launches on their own, as a standalone ELF module."""
    import dflash_int4 as I

    I.paths()
    from shared.infra.stitching import stitch_elf

    base_args, slices, prelude = ctxkv_parts(ctx_pad, n_layers, with_knorm, with_rope)
    return stitch_elf("dflash_ctxkv_int4", base_args, slices, prelude=prelude)


def rope_lut(positions, head_dim=HEAD_DIM, n_kv_heads=N_KV_HEADS, theta=1000000.0):
    """[len(positions) * n_kv_heads * head_dim] bf16, laid out to match k_ctx.

    k_ctx is [ctx_pad, 1024] viewed as [ctx_pad*8, 128], so row r is
    (position r // 8, head r % 8) and the LUT repeats each position's row once
    per KV head. Each row is [cos(64) | sin(64)] -- the concatenated half-split
    layout rope.cc expects, NOT the interleaved one.

    `positions` are ABSOLUTE: the drafter is called with
    position_ids[start - ctx_len : start + block] (model.py:246), so a context
    row's position is its own place in the sequence, not its index in the block.
    """
    import numpy as np
    from ml_dtypes import bfloat16

    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.outer(np.asarray(positions, np.float64), inv)
    lut = np.empty((len(positions), head_dim), np.float64)
    lut[:, :half] = np.cos(ang)
    lut[:, half:] = np.sin(ang)
    return np.repeat(lut, n_kv_heads, axis=0).astype(bfloat16).reshape(-1)


def layer_kv_weights(dw, layer):
    """(k_proj, v_proj) as [out, in] = [1024, 2560], the layout awq_quantize wants.

    NOT transposed: the bf16 path hands `_build_gemm_module` a [K, N] B operand,
    but the int4 packer takes the checkpoint's own output-major [N, K] and
    produces the tiled B itself.
    """
    import numpy as np

    return (
        np.asarray(dw.bf16(f"layers.{layer}.self_attn.k_proj.weight")),
        np.asarray(dw.bf16(f"layers.{layer}.self_attn.v_proj.weight")),
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    m = build_ctxkv_int4_module()
    txt = str(m)
    n = txt.count("air.launch")
    print(f"[ctxkv int4] {len(txt.splitlines())} lines, {n} air.launch ops, parsed OK")
    print(
        f"[ctxkv int4] M={CTX_PAD} K={D} N={KV_DIM}, {N_LAYERS} layers, "
        f"2 GEMM + k_norm + RoPE each"
    )
    sys.exit(0 if n == 4 * N_LAYERS else 1)
