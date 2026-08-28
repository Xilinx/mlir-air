# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The DFlash drafter's tap fusion in int4-AWQ, as four air.launch ops:
#
#     A_0, A_1 [CTX, 6400]  --W_i-->  P_0, P_1 [CTX, 2560]   two int4 GEMMs
#     S = P_0 + P_1                                          one add
#     out = hidden_norm(S)                                   one norm
#
# WHY THE K AXIS IS SPLIT AT ALL. `matmul_int4_packed.build_module` puts the
# whole herd INSIDE the K-outer loop (`for i in for_(0, k // tile_k_l2)`), and
# the herd body allocates its L1 f32 accumulator, zeroes it, converts and drains
# to L2 on every iteration. The accumulator does not survive a K-outer step: the
# L2 C tile is OVERWRITTEN by each chunk and the result is the LAST chunk's
# partial product alone. Measured on device (M=64, N=128, herd 2x4): tile_k_l2
# == K gives 7.2-7.4e-03 at K=128, 256 and 1280; K=256 with tile_k_l2=128 gives
# 1.12 and K=1280 with tile_k_l2=128 gives 1.03 -- uncorrelated. Every lit test
# in matrix_multiplication/int4_awq uses K=128 with tile_k_l2=128, so the
# K-outer path is never exercised there. M and N are innocent: N=2560 and M=32
# both pass at K=128.
#
# So tile_k_l2 == K is mandatory, and at fc's K=12800 that stages 400 KB of A
# plus 429 KB of B against a 512 KB memtile. Splitting K is the way out, and it
# is exact: fc over a concatenation is a sum of per-chunk GEMMs, which
# dflash_draft_decomp.py verifies against the real checkpoint at 3.4e-05 (there
# on the natural five-tap boundary; any K partition works the same way). The
# only cost is the partial buffers and the adds.
#
# WHY TWO CHUNKS AND NOT FIVE. Each partial needs an add, and adds are launches
# -- see the operand rule below. K=6400 stages 200 KB of A + 209 KB of B = 406
# KB, which fits, and two chunks means one add. Five K=2560 chunks would need
# four.
#
# A HERD TILE TAKES AT MOST TWO INCOMING L3 STREAMS. This is the rule that
# shapes the tail, and it fails in two different ways depending on how far past
# it you go. Bisected on the sum/norm herd (dflash_sumnorm.py), counting L3
# operands as (inputs + weight):
#     2 in, 1 out  -> correct        (rel 0.0 for a copy, 2.6e-03 for a sum)
#     3 in, 1 out  -> SILENTLY WRONG (rel 1.12-1.17, and the padded rows fill
#                                     with other rows' data; the value varies
#                                     run to run, so it is a race, not a
#                                     miscompile)
#     5 in, 1 out  -> aircc dies with 0xC0000005 and no diagnostic
# A single AIE tile has two S2MM channels, which is exactly where the first
# boundary sits. So every launch here is at most two-in/one-out.

CTX_PAD = 32
D = 2560  # fc's output width, and each tap's width
N_TAPS = 5
FC_IN = N_TAPS * D  # 12800
N_CHUNKS = 2
K_CHUNK = FC_IN // N_CHUNKS  # 6400

TILE_K_L2 = K_CHUNK  # == K. See above; this is a correctness constraint.


def fc_parts(ctx_pad=CTX_PAD, n_chunks=N_CHUNKS, with_norm=True, base=0):
    """(base_args, slices, out_arg) -- the fc launches, ready to stitch.

    Split out from `build_int4_fc_module` so the drafter pre-pass
    (dflash_draft_prepass.py) can concatenate these launches with the context
    K/V's in ONE func instead of dispatching two ELFs and bouncing
    `target_hidden` through the host. `base` shifts every arg number, and
    `out_arg` is where `target_hidden` lands.

    Args (in order), with p = n_chunks and k = FC_IN // p, all offset by `base`:
        %arg{i}                 i in [0, p)    A_i   [ctx_pad, k]     bf16
        %arg{p + i}                            B_i   packed int4      i8
        %arg{2p + i}                           P_i   [ctx_pad, 2560]  bf16
        then, if with_norm:
        %arg{3p + j}            j in [0, p-1)  S_j   [ctx_pad, 2560]  bf16
        %arg{4p - 1}                           hn_w  [2560]           bf16
        %arg{4p}                               out   [ctx_pad, 2560]  bf16
    """
    import dflash_int4 as I

    I.paths()
    from ml_dtypes import bfloat16

    from shared.infra.stitching import (
        FuncArg,
        KernelSlice,
        stitch_elf,
        _wrap_ir_in_launch,
    )
    import dflash_sumnorm

    assert FC_IN % n_chunks == 0, (FC_IN, n_chunks)
    k_chunk = FC_IN // n_chunks

    gemm_ir = I.build_int4_gemm_ir(ctx_pad, k_chunk, D, tile_k_l2=k_chunk)
    bshape = I.packed_shape(k_chunk, D)

    base_args, slices = [], []
    for i in range(n_chunks):
        base_args.append(
            FuncArg(f"%arg{base + i}", f"memref<{ctx_pad}x{k_chunk}xbf16>")
        )
    for i in range(n_chunks):
        base_args.append(
            FuncArg(
                f"%arg{base + n_chunks + i}",
                f"memref<{bshape[0]}x{bshape[1]}x{bshape[2]}xi8>",
            )
        )
    for i in range(n_chunks):
        base_args.append(
            FuncArg(f"%arg{base + 2 * n_chunks + i}", f"memref<{ctx_pad}x{D}xbf16>")
        )
    for i in range(n_chunks):
        slices.append(
            KernelSlice(
                gemm_ir,
                f"fc{i}",
                {0: base + i, 1: base + n_chunks + i, 2: base + 2 * n_chunks + i},
                extern_syms=I.EXTERN_SYMS,
            )
        )

    if not with_norm:
        return base_args, slices, base + 2 * n_chunks

    add_ir = _wrap_ir_in_launch(
        str(dflash_sumnorm.build_module(2, ctx_pad, D, bfloat16, 16, with_norm=False))
    )
    norm_ir = _wrap_ir_in_launch(
        str(dflash_sumnorm.build_module(1, ctx_pad, D, bfloat16, 16, with_norm=True))
    )

    s0 = base + 3 * n_chunks
    for j in range(n_chunks - 1):
        base_args.append(FuncArg(f"%arg{s0 + j}", f"memref<{ctx_pad}x{D}xbf16>"))
    wa = s0 + n_chunks - 1
    base_args += [
        FuncArg(f"%arg{wa}", f"memref<{D}xbf16>"),
        FuncArg(f"%arg{wa + 1}", f"memref<{ctx_pad}x{D}xbf16>"),
    ]

    # Left fold: S_0 = P_0 + P_1, S_j = S_{j-1} + P_{j+1}. Two in, one out each.
    for j in range(n_chunks - 1):
        lhs = base + 2 * n_chunks if j == 0 else s0 + j - 1
        slices.append(
            KernelSlice(
                add_ir,
                f"add{j}",
                {0: lhs, 1: base + 2 * n_chunks + j + 1, 2: s0 + j},
                extern_syms={"@zero_vectorized_bf16"},
            )
        )
    src = s0 + n_chunks - 2 if n_chunks > 1 else base + 2 * n_chunks
    slices.append(
        KernelSlice(
            norm_ir,
            "hn",
            {0: src, 1: wa, 2: wa + 1},
            extern_syms={"@zero_vectorized_bf16"},
        )
    )

    return base_args, slices, wa + 1


def build_int4_fc_module(ctx_pad=CTX_PAD, n_chunks=N_CHUNKS, with_norm=True):
    """The fc launches on their own, as a standalone ELF module."""
    import dflash_int4 as I

    I.paths()
    from shared.infra.stitching import stitch_elf

    base_args, slices, _ = fc_parts(ctx_pad, n_chunks, with_norm)
    return stitch_elf("dflash_int4_fc", base_args, slices)


def split_fc_weight(fc_w, n_chunks=N_CHUNKS):
    """fc.weight [2560, 12800] -> `n_chunks` [2560, K_CHUNK] column blocks.

    Column blocks, not output-axis blocks: the K axis is what the decomposition
    splits. Slicing the output axis instead would be silent at n_chunks=5, where
    the pieces are square.
    """
    import numpy as np

    n_in = fc_w.shape[1]
    w = n_in // n_chunks
    return [np.ascontiguousarray(fc_w[:, i * w : (i + 1) * w]) for i in range(n_chunks)]


def split_taps(taps, n_chunks=N_CHUNKS):
    """[M, 12800] -> `n_chunks` contiguous [M, K_CHUNK].

    `taps` is the tap CONCATENATION in the drafter's own order (tap 0 first);
    the chunk boundary need not fall on a tap boundary, and at n_chunks=2 it
    does not -- chunk 0 is taps 0, 1 and half of 2.
    """
    import numpy as np

    w = taps.shape[1] // n_chunks
    return [np.ascontiguousarray(taps[:, i * w : (i + 1) * w]) for i in range(n_chunks)]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    m = build_int4_fc_module()
    txt = str(m)
    n = txt.count("air.launch")
    print(f"[int4 fc] {len(txt.splitlines())} lines, {n} air.launch ops, parsed OK")
    print(
        f"[int4 fc] {N_CHUNKS} x (M={CTX_PAD} K={K_CHUNK} N={D}, "
        f"tile_k_l2={TILE_K_L2}) + {N_CHUNKS - 1} add + norm"
    )
    sys.exit(0 if n == 2 * N_CHUNKS else 1)
