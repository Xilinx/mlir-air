# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""bf16 A x bfp16ebs8 B -> bf16 C mixed-precision GEMM on NPU2, on air.api.

B is ``uint8`` at the AIR boundary -- MLIR has no bfp16ebs8 element type -- and
``mm_bf16_x_bfp16.cc`` reinterprets it through ``aie::block_vector<bfp16ebs8>``.
So every bit of arithmetic here is in a hand-written kernel reached by
``air.extern``, and what moves onto the DSL is the schedule and the data
movement::

    air.launch (m_outer, n_outer)
      air.segment
        L2: A [herd_m, tile_m, tile_k_l2]      (bf16)
            B [herd_n, k_per_l2, tile_bytes]   (packed bfp16ebs8 bytes)
            C [herd_m, herd_n, tile_m, tile_n] (bf16)
        L1, segment lifetime, one slab per core:
            acc   [herd_m, herd_n, tile_n/t, tile_m/r, r, t]  (f32)
            drain [same shape]                                (bf16)
        herd            zero the accumulator
        for k2 in air.sequential(k / tile_k_l2)
            L3 -> L2 for A and B
            herd
                for j in air.sequential(k_per_l2)
                    L2 -> L1, then one mmul into the accumulator
        herd            narrow f32 -> bf16 and drain to L2
        L2 -> L3

**The accumulator is shared, and a core is handed its own slab.** It is zeroed
once, added into across every trip of a K loop that sits at *segment* scope --
so the herd is entered once per trip -- and narrowed at the end. That is what
``<segment>.shared()`` is for: L1, but with the segment's lifetime, carrying one
leading dimension per herd axis. The predecessor wrote a ``memref.subview`` at
each of the four call sites to pick out the calling core's slab; there is only
ever one slab a given core may touch, so the DSL emits that subview itself and
``zero_acc(acc)`` means what the six-line subview meant. It is the same helper
``ops.fill`` and ``ops.dot`` already used -- a hand-written kernel is simply the
third way to write an accumulator.

**The micro-tiled L1 layouts are a walk, not a memref layout.** The mmul
consumes ``[.., tile_m/r, tile_k_l1/s, r, s]`` blocks while the L2 staging
buffer is plain row-major; the reordering is a ``reshape`` and a ``transpose``
of the region being read, which costs nothing because both are views and what
they produce is the DMA's access pattern. Note the A permutation is
``(0, 1, 2, 4, 3, 5)``, not the ``(0, 1, 4, 2, 3, 5)`` its bf16 sibling uses:
this kernel's L1 A is M-tile-outer where that one's is K-tile-outer, so the
transpose follows the layout rather than a habit.

**Only axes that mean something are declared.** The predecessor's L2 A and B
each carried a unit axis whose stride was written by hand, and a size-1 axis is
never stepped, so that stride says nothing -- but asking a reshape to reproduce
one makes it *invent* a value, which downstream passes can then act on. The
staging buffers here declare the per-core axis and the tile and nothing else. C
keeps all four of its axes, because the drain out of it is a genuine transpose.

The ping-pong opt-out is gone. Both bfp16 matmuls used to carry
``air.disable_ping_pong`` on the K-l1 fill loop purely to supply a fact the pass
could not see -- that unrolling by two would put 74 KiB of L1 buffers on a 64
KiB tile. ``air-label-scf-for-to-ping-pong`` now does that arithmetic itself
(#1928), so there is nothing left for the kernel to declare.

``build_module`` returns the module rather than the launch, because the bfp16
prefill stitchers under ``llms/llama32_1b_int4/`` call it for one and match its
signature positionally.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i8
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

KERNEL_OBJ_NAME = "mm_bf16_x_bfp16.o"
BFP16_BLOCK = 8
BFP16_BYTES_PER_BLOCK = 9


def bfp_tile_bytes(tile_n, tile_k_l1):
    """Bytes in one (tile_n, tile_k_l1) B tile, packed bfp16ebs8."""
    return (tile_n * tile_k_l1 // BFP16_BLOCK) * BFP16_BYTES_PER_BLOCK


def _bf16_block_to_bfp16ebs8(block_f32):
    """Reference per-block scalar packer; use pack_b_bfp16ebs8 in production."""
    bits = block_f32.astype(np.float32).view(np.uint32)
    sign = (bits & 0x80000000) != 0
    exp = ((bits >> 23) & 0xFF).astype(np.int32)
    mant_explicit = (bits & 0x007FFFFF) | np.where(exp != 0, 0x00800000, 0)
    max_exp = exp.max()
    signed_mant = np.where(
        sign, (~mant_explicit + 1) & 0xFFFFFFFF, mant_explicit
    ).astype(np.int64)
    # Truncate to 8 bits FIRST, then arithmetic-shift the int8 by (max_exp - exp).
    # This is the order floatToBfp16() uses in mlir-aie
    # programming_examples/ml/block_datatypes/helper.h. Shifting the 32-bit value
    # first and truncating afterwards diverges for NEGATIVE values whose in-block
    # exponent spread reaches 8, producing a packing the hardware would not.
    b8 = (signed_mant >> (23 - 7 + 1)).astype(np.uint8).view(np.int8).astype(np.int32)
    shift = (max_exp - exp).astype(np.int32)
    aligned = np.where(
        shift >= 32, np.where(sign, -1, 0), b8 >> np.minimum(shift, 31)
    ).astype(np.int8)
    out = np.empty(BFP16_BYTES_PER_BLOCK, dtype=np.uint8)
    out[0] = np.uint8(max_exp)
    out[1:] = aligned.view(np.uint8)
    return out


def pack_b_bfp16ebs8(B_bf16, tile_n, tile_k_l1):
    """[K, N] bf16 -> [N/tile_n, K/tile_k_l1, tile_bytes] uint8 (3D BO).

    Vectorized over all blocks at once. Each 9-byte record packs 8
    K-contiguous elements for one N row (B is consumed transposed).
    """
    K, N = B_bf16.shape
    r = s = t = 8
    assert tile_n % t == 0 and tile_k_l1 % s == 0
    assert K % tile_k_l1 == 0 and N % tile_n == 0
    Nb = N // tile_n
    Kb = K // tile_k_l1
    NB_in = tile_n // t  # n MMUL sub-tiles per tile_n
    KB_in = tile_k_l1 // s  # k MMUL sub-tiles per tile_k_l1
    n_blocks_total = Nb * Kb * NB_in * KB_in  # one MMUL sub-block per row of output

    # Permute B into per-block 8-element views: [Nb, Kb, NB_in, KB_in, t, s].
    # sub[..., n_i, k_i] = B[kb * tile_k_l1 + kbi * s + k_i, nb * tile_n + nbi * t + n_i]
    Bf = B_bf16.astype(np.float32)
    Bv = Bf.reshape(Kb, KB_in, s, Nb, NB_in, t)  # [Kb, KB_in, s, Nb, NB_in, t]
    Bv = np.transpose(Bv, (3, 0, 4, 1, 5, 2))  # [Nb, Kb, NB_in, KB_in, t, s]
    # Each (block_idx, n_i) line of 8 elements gets one bfp16ebs8 9-byte record.
    blocks = np.ascontiguousarray(Bv).reshape(n_blocks_total * t, s)  # [n_records, 8]

    # Vectorized bit math (same per-element logic as the scalar reference).
    bits = blocks.view(np.uint32)  # reinterpret f32 bits
    sign = (bits & 0x80000000) != 0  # [n_records, 8] bool
    exp = ((bits >> 23) & 0xFF).astype(np.int32)  # [n_records, 8]
    mant_explicit = (bits & 0x007FFFFF) | np.where(
        exp != 0, np.uint32(0x00800000), np.uint32(0)
    )
    max_exp = exp.max(axis=1, keepdims=True)  # [n_records, 1] shared exp per block
    signed_mant = np.where(
        sign, (~mant_explicit + np.uint32(1)) & np.uint32(0xFFFFFFFF), mant_explicit
    ).astype(np.int64)
    # 24-bit -> 8-bit-with-sign. Truncate to 8 bits FIRST, then arithmetic-shift
    # the int8 by (max_exp - exp) -- the order floatToBfp16() uses in helper.h.
    # See _bf16_block_to_bfp16ebs8 for why the order matters.
    b8 = (signed_mant >> (23 - 7 + 1)).astype(np.uint8).view(np.int8).astype(np.int32)
    shift = (max_exp - exp).astype(np.int32)  # [n_records, 8] >= 0
    aligned = np.where(
        shift >= 32, np.where(sign, -1, 0), b8 >> np.minimum(shift, 31)
    ).astype(np.int8)

    # Interleave [shared_exp, m0..m7] per record into 9-byte stride.
    records = np.empty((n_blocks_total * t, BFP16_BYTES_PER_BLOCK), dtype=np.uint8)
    records[:, 0] = max_exp.flatten().astype(np.uint8)
    records[:, 1:] = aligned.view(np.uint8)

    # Reassemble: per tile we have NB_in * KB_in * t records, each 9 bytes.
    tb = bfp_tile_bytes(tile_n, tile_k_l1)
    out = records.reshape(Nb, Kb, NB_in * KB_in * t * BFP16_BYTES_PER_BLOCK)
    assert out.shape[2] == tb
    return out


def cpu_reference_from_bfp_packed(B_packed, A_bf16, m, k, n, tile_n, tile_k_l1):
    r = s = t = 8
    NB_in = tile_n // t
    KB_in = tile_k_l1 // s
    Bf = np.zeros((k, n), dtype=np.float32)
    for nb in range(n // tile_n):
        for kb in range(k // tile_k_l1):
            cursor = 0
            for nbi in range(NB_in):
                for kbi in range(KB_in):
                    n0 = nb * tile_n + nbi * t
                    k0 = kb * tile_k_l1 + kbi * s
                    sub_T = np.zeros((t, s), dtype=np.float32)
                    for n_i in range(t):
                        block = B_packed[
                            nb, kb, cursor : cursor + BFP16_BYTES_PER_BLOCK
                        ]
                        cursor += BFP16_BYTES_PER_BLOCK
                        shared_exp = int(block[0])
                        mults = (
                            (1.0 * (1 << (shared_exp - 127)))
                            if shared_exp >= 127
                            else (1.0 / (1 << (127 - shared_exp)))
                        ) / 64.0
                        mants = block[1:].view(np.int8).astype(np.int32)
                        sub_T[n_i, :] = mants.astype(np.float32) * mults
                    Bf[k0 : k0 + s, n0 : n0 + t] = sub_T.T
    C = A_bf16.astype(np.float32) @ Bf
    return C.astype(bfloat16)


def build_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
):
    """bf16 A x bfp16ebs8 B mixed-precision GEMM with f32 L1 accumulator."""
    r, s, t = 8, 8, 8
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert tile_m % (2 * r) == 0
    assert tile_n % (2 * t) == 0
    assert tile_k_l1 % s == 0

    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    k_per_l2 = tile_k_l2 // tile_k_l1
    N_div = n // tile_n
    K_div = k // tile_k_l1

    # The output block one segment covers: herd_m x herd_n L1 tiles.
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n

    A = air.tensor([m, k], bf16)
    # bfp16ebs8 has no MLIR element type, so B is bytes on this side of the
    # call: one contiguous run per (n_outer, k_outer) tile, unpacked inside
    # mm_bf16_x_bfp16.cc as aie::block_vector<bfp16ebs8>.
    B = air.tensor([N_div, K_div, tile_bytes], i8)
    C = air.tensor([m, n], bf16)

    zero_acc = air.extern("zero_vectorized_f32_mn", link_with=KERNEL_OBJ_NAME)
    matmul = air.extern("matmul_bf16_x_bfp16_packed_f32", link_with=KERNEL_OBJ_NAME)
    to_bf16 = air.extern("f32_to_bf16_mn", link_with=KERNEL_OBJ_NAME)

    with air.launch(
        [range(m // tile_m // herd_m), range(N_div // herd_n)],
        name="matmul_bf16_x_bfp16",
    ) as launch:

        @launch.body
        def _(li, lj):
            with air.segment(name="matmul_seg") as seg:

                @seg.body
                def _():
                    l2_a = air.alloc(
                        [herd_m, tile_m, tile_k_l2], bf16, scope=seg.private()
                    )
                    l2_b = air.alloc(
                        [herd_n, k_per_l2, tile_bytes], i8, scope=seg.private()
                    )
                    l2_c = air.alloc(
                        [herd_m, herd_n, tile_m, tile_n], bf16, scope=seg.private()
                    )
                    # One slab per core, with the segment's lifetime: the K loop
                    # below is at segment scope, so the herd is entered once per
                    # trip and a herd-body buffer would not survive it.
                    acc = air.alloc(
                        [herd_m, herd_n, tile_n // t, tile_m // r, r, t],
                        f32,
                        scope=seg.shared(),
                    )
                    drain = air.alloc(
                        [herd_m, herd_n, tile_n // t, tile_m // r, r, t],
                        bf16,
                        scope=seg.shared(),
                    )

                    row, col = li * l2_m, lj * l2_n
                    n_outer = lj * herd_n

                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as zero_herd:

                        @zero_herd.body
                        def _(tx, ty):
                            zero_acc(acc)

                    for k2 in air.sequential(k // tile_k_l2):
                        k_l2_off = k2 * tile_k_l2
                        k_chunk_off = k2 * k_per_l2

                        # Split the row axis of a plain [m, k] region into
                        # (core, within-tile). The split is the access pattern.
                        ops.load(
                            l2_a,
                            A[
                                row : row + l2_m, k_l2_off : k_l2_off + tile_k_l2
                            ].reshape(herd_m, tile_m, tile_k_l2),
                        )
                        ops.load(
                            l2_b,
                            B[
                                n_outer : n_outer + herd_n,
                                k_chunk_off : k_chunk_off + k_per_l2,
                                :,
                            ],
                        )

                        with air.herd(
                            [range(herd_m), range(herd_n)],
                            name="herd_0",
                            shape=(herd_m, herd_n),
                        ) as h:

                            @h.body
                            def _(tx, ty):
                                l1_a = air.alloc(
                                    [1, 1, tile_m // r, tile_k_l1 // s, r, s],
                                    bf16,
                                    scope=h.private(),
                                )
                                l1_b = air.alloc([tile_bytes], i8, scope=h.private())
                                for j in air.sequential(k_per_l2):
                                    k1 = j * tile_k_l1
                                    # [M, K] read block-first: split each axis
                                    # into (tiles, within-tile), then bring the
                                    # M tile axis outside the K one.
                                    ops.load(
                                        l1_a,
                                        l2_a[tx, :, k1 : k1 + tile_k_l1]
                                        .reshape(
                                            1,
                                            1,
                                            tile_m // r,
                                            r,
                                            tile_k_l1 // s,
                                            s,
                                        )
                                        .transpose(0, 1, 2, 4, 3, 5),
                                    )
                                    ops.load(l1_b, l2_b[ty, j, :])
                                    matmul(l1_a, l1_b, acc)

                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as drain_herd:

                        @drain_herd.body
                        def _(tx, ty):
                            to_bf16(acc, drain)
                            # The inverse walk: put the M axes back outside the
                            # N ones. The 6-D form is deliberate -- it lets the
                            # BD optimizer collapse (m_b, m_i) into a single
                            # dimension, which is what fits an AIE2P 3-dim BD.
                            ops.store(
                                drain[tx, ty, :, :, :, :].transpose(0, 1, 3, 4, 2, 5),
                                l2_c[tx, ty, :, :],
                            )

                    ops.store(
                        l2_c.transpose(0, 2, 1, 3),
                        C[row : row + l2_m, col : col + l2_n],
                    )

    # bfp16ebs8 is an AIE2P datatype and both lits are REQUIRES: ryzen_ai_npu2,
    # so the module is npu2-specific before the herd is even sized.
    return launch.build(target="npu2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=32, dest="tile_m")
    parser.add_argument("--tile-k-l2", type=int, default=128, dest="tile_k_l2")
    parser.add_argument("--tile-k-l1", type=int, default=128, dest="tile_k_l1")
    parser.add_argument("--tile-n", type=int, default=32, dest="tile_n")
    parser.add_argument("--herd-m", type=int, default=2, dest="herd_m")
    parser.add_argument("--herd-n", type=int, default=4, dest="herd_n")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-and-run", "compile-only", "compile-and-xclbin"],
        default="compile-and-run",
        dest="compile_mode",
    )
    args = parser.parse_args()

    module = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    A = (np.random.randn(args.m, args.k) * (1.0 / np.sqrt(args.k))).astype(bfloat16)
    B = (np.random.randn(args.k, args.n) * (1.0 / np.sqrt(args.k))).astype(bfloat16)

    B_packed = pack_b_bfp16ebs8(B, args.tile_n, args.tile_k_l1)
    C_ref = cpu_reference_from_bfp_packed(
        B_packed, A, args.m, args.k, args.n, args.tile_n, args.tile_k_l1
    )

    # runtime_loop_tiling_sizes=[2,2] keeps the runtime DMA loop within the
    # ~4-task shim BD pool at large launch axes.
    common_kwargs = dict(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="xclbin",
        stack_size=2048,
        runtime_loop_tiling_sizes=[2, 2],
        instance_name="matmul_bf16_x_bfp16",
    )

    if args.compile_mode == "compile-only":
        backend = XRTBackend(**common_kwargs)
        backend.compile(module)
        backend.unload()
        sys.exit(0)

    if args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(**common_kwargs)
        backend.compile(module)
        backend.unload()
        sys.exit(0)

    runner = XRTRunner(**common_kwargs)
    sys.exit(
        runner.run_test(
            module,
            inputs=[A, B_packed],
            expected_outputs=[C_ref],
            rtol=0.1,
            atol=0.05,
            max_mismatch_percentage=0.05,
            min_correlation=0.999,
        )
    )
