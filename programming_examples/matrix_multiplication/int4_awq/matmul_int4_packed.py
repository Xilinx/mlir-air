# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""int4-AWQ GEMM (prefill), on air.api.

A 2-D herd over (M, N), with K accumulated per-PE inside the herd and a per-PE
drain into a 4-D L2 C ``[herd_m, herd_n, tile_m, tile_n]``. The arithmetic is
entirely in ``mv_int4_bf16.cc`` -- dequantisation, the mmul and the f32->bf16
convert are three hand-written AIE kernels -- so what the DSL expresses here is
the *schedule* and the *data movement*, and nothing else::

    air.launch (m_outer, n_outer)
      air.segment
        L2: A [herd_m, tile_m, tile_k_l2] (bf16)
            B [herd_n, k_per_l2, tile_bytes] (packed Q+S+Z bytes)
            C [herd_m, herd_n, tile_m, tile_n] (bf16)
        for m_o in air.sequential(m_per_segment)
            for k2 in air.sequential(k / tile_k_l2)
                L3 -> L2 for A and B
                herd
                    zero the f32 accumulator
                    for j in air.sequential(k_per_l2)
                        L2 -> L1, then dequant + mmul into the accumulator
                    convert to bf16 and drain to L2
            L2 -> L3

Three things are worth pointing at.

**The packed weight tensor is already blocked, so its transfers are plain
subscripts.** ``pack_inputs`` lays the Q+S+Z bytes out as ``[N_div, K_div,
tile_bytes]``, one contiguous run per ``(n_outer, k_outer)`` tile, which is why
the L1 weight buffer is a flat ``[tile_bytes]`` and its fill is
``l2_b[ty, j, :]``. The int4 nibbles, the scales and the zero points never
appear as types in the IR: to the DMA they are bytes, and to the kernel they are
whatever ``mv_int4_bf16.cc`` says they are.

**Every L2 staging axis here is one somebody divides along.** A's fill is
``.reshape(herd_m, tile_m, tile_k_l2)`` over a plain ``[m, k]`` region: the
split of the row axis into a per-core axis and a within-tile axis *is* the DMA's
access pattern, and nothing is copied. It is tempting to go further, since
row-major ``[a, b, c]`` and ``[a*b, c]`` hold the same bytes in the same order
and ``l2_a[tx*tile_m : ...]`` would walk the same addresses -- but at Llama
prefill scale that buffer is 512 KB, over what one memtile holds, and
``air-split-l2-memref-for-buffer-constraint`` has to divide it. It can do that
along a declared ``herd_m`` axis and not along a fused one: flattened, it aborts
in ``tileChannelOpByFactor`` on an invalid affine map.

The predecessor's two *unit* axes go the other way, and for the same reason
inverted -- nothing is ever split along them. Their strides were hand-written
and arbitrary (a size-1 axis is never stepped), which is exactly why they cannot
be derived: asking for ``[herd_m, 1, tile_m, tile_k_l2]`` makes the reshape
invent a stride, and at ``M=32`` the invented one happens to equal its
neighbour's, which is enough for the L2 split to fuse two per-core puts into one
4-D BD instead. Declaring only the axes that mean something removes the choice.
C keeps all four of its axes because the drain out of it is a genuine transpose.

**The accumulator is f32 and lives in L1 across the whole K loop.** Partial sums
must not round to bf16 between kernel calls, so ``zero_vectorized_f32_mn`` seeds
it once per herd entry and ``f32_to_bf16_mn`` converts once at the end. This is
also why ``TILE_K_L2`` defaults to ``K``: with one segment-level K iteration the
accumulator survives every ``K_CHUNK`` step.

``m_per_segment`` is unchanged and still load-bearing. Shim DMA BD chains on
AIE2P fold zero-stride launch-axis loops into a single multi-dim BD but unroll
non-zero-stride ones into a BD per iteration. M iterations have a non-zero
stride, so at full Llama prefill (M=2048, tile_m=16, herd_m=8) the launch M axis
would need 16 BDs per chain -- past the per-shim-channel BD ID pool. Moving
those iterations into the segment's ``air.sequential`` makes the launch M axis 1
and the loop a single BD repeated at runtime.

The module is built for **npu2**: the kernel object is compiled
``--target=aie2p``, both lits are ``REQUIRES: ryzen_ai_npu2``, and the widest
configuration wants an 8-column part. ``build_module`` returns the module rather
than the launch, because the int4 prefill stitchers under
``llms/llama32_1b_int4/`` call it for one and match its signature positionally.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16, f32, i8
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

KERNEL_OBJ_NAME = "mv_int4_bf16.o"


def packed_tile_bytes(n_tile, k_chunk, gs):
    n_gpc = k_chunk // gs
    q_bytes = n_tile * (k_chunk // 2)
    s_bytes = n_gpc * n_tile * 2
    z_bytes = n_gpc * n_tile
    return q_bytes, s_bytes, z_bytes, q_bytes + s_bytes + z_bytes


def pack_inputs(W_q, W_s, W_z, M, K, N, GS, N_TILE, K_CHUNK):
    """Pack per-(n_outer, k_outer) Q+S+Z tiles into [N_div, K_div, tile_bytes].

    W_q [N, K/2] u8 (output-major), W_s [K/GS, N] bf16, W_z [K/GS, N] u8.
    """
    n_gpc = K_CHUNK // GS
    q_bytes, s_bytes, _, tile_bytes = packed_tile_bytes(N_TILE, K_CHUNK, GS)
    N_div = N // N_TILE
    K_div = K // K_CHUNK
    packed = np.zeros((N_div, K_div, tile_bytes), dtype=np.uint8)
    for n_outer in range(N_div):
        col_off = n_outer * N_TILE
        for k_outer in range(K_div):
            q_col_byte = k_outer * (K_CHUNK // 2)
            g_off = k_outer * n_gpc
            q_tile = W_q[
                col_off : col_off + N_TILE,
                q_col_byte : q_col_byte + (K_CHUNK // 2),
            ]
            s_tile = W_s[g_off : g_off + n_gpc, col_off : col_off + N_TILE]
            z_tile = W_z[g_off : g_off + n_gpc, col_off : col_off + N_TILE]
            p = packed[n_outer, k_outer]
            p[0:q_bytes] = np.ascontiguousarray(q_tile).view(np.uint8).reshape(-1)
            p[q_bytes : q_bytes + s_bytes] = (
                np.ascontiguousarray(s_tile).view(np.uint8).reshape(-1)
            )
            p[q_bytes + s_bytes :] = (
                np.ascontiguousarray(z_tile).view(np.uint8).reshape(-1)
            )
    return packed


def cpu_reference(W_q, W_s, W_z, A):
    N_ = W_q.shape[0]
    K_ = A.shape[1]
    n_groups = W_s.shape[0]
    gs = K_ // n_groups
    Af = A.astype(np.float32)
    W_s_f = W_s.astype(np.float32)
    W_z_i = W_z.astype(np.int32)
    W_dq = np.zeros((K_, N_), dtype=np.float32)
    for n in range(N_):
        for kk in range(K_):
            byte = int(W_q[n, kk // 2])
            nib = (byte & 0x0F) if (kk % 2 == 0) else ((byte >> 4) & 0x0F)
            g = kk // gs
            W_dq[kk, n] = (nib - W_z_i[g, n]) * W_s_f[g, n]
    C = Af @ W_dq
    return C.astype(bfloat16)


def build_module(
    m, k, n, gs, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n, m_per_segment=1
):
    """Build the int4-AWQ packed GEMM and return the MLIR module.

    `m_per_segment` lets the segment body iterate M outer tiles inside one
    launch step, instead of putting every M outer tile on its own launch axis
    position. The total per-PE M tile count is unchanged
    (`m // (tile_m * herd_m)`); only WHERE the iteration lives moves -- see the
    module docstring for why that matters to the shim BD budget.
    """
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    m_outer_total = m // (tile_m * herd_m)
    assert m_outer_total % m_per_segment == 0, (
        f"m_outer_total ({m_outer_total}) must be divisible by m_per_segment "
        f"({m_per_segment})"
    )
    launch_m_outer = m_outer_total // m_per_segment
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert tile_k_l1 % gs == 0
    # Kernel-side static_assert constraints from mv_int4_bf16.cc:
    #   mm_int4_bf16_mmul_impl: tile_m/n/k_chunk % 8 (mmul dims), gs % R=32
    #   zero_vectorized_bf16_mn: (tile_m * tile_n) % VW=32
    assert (
        tile_m % 8 == 0 and tile_n % 8 == 0 and tile_k_l1 % 8 == 0
    ), "tile_m, tile_n, tile_k_l1 must each be multiples of 8 (mmul tile size)"
    assert gs % 32 == 0, "gs must be a multiple of dequant inner-vector width 32"
    assert (tile_m * tile_n) % 32 == 0, (
        f"tile_m*tile_n ({tile_m}*{tile_n}={tile_m * tile_n}) must be a multiple "
        f"of vector width 32 for zero_vectorized_bf16_mn"
    )

    _, _, _, tile_bytes = packed_tile_bytes(tile_n, tile_k_l1, gs)
    k_per_l2 = tile_k_l2 // tile_k_l1
    N_div = n // tile_n
    K_div = k // tile_k_l1

    # The output block one segment covers: herd_m x herd_n L1 tiles.
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n

    A = air.tensor([m, k], bf16)
    # Packed weights are bytes to everything on this side of the call: the
    # nibbles, the bf16 scales and the u8 zero points are unpacked inside
    # mv_int4_bf16.cc, from one contiguous run per (n_outer, k_outer) tile.
    B = air.tensor([N_div, K_div, tile_bytes], i8)
    C = air.tensor([m, n], bf16)

    zero_acc = air.extern("zero_vectorized_f32_mn", link_with=KERNEL_OBJ_NAME)
    matmul = air.extern("matmul_int4_bf16_packed_f32", link_with=KERNEL_OBJ_NAME)
    to_bf16 = air.extern("f32_to_bf16_mn", link_with=KERNEL_OBJ_NAME)

    with air.launch(
        [range(launch_m_outer), range(N_div // herd_n)], name="matmul_int4_packed"
    ) as launch:

        @launch.body
        def _(li, lj):
            with air.segment(name="seg") as seg:

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

                    n_outer = lj * herd_n
                    col = lj * l2_n

                    for m_o in air.sequential(m_per_segment):
                        row = li * (l2_m * m_per_segment) + m_o * l2_m

                        for k2 in air.sequential(k // tile_k_l2):
                            k_l2_off = k2 * tile_k_l2
                            k_chunk_off = k2 * k_per_l2

                            # Split the region's row axis into (core,
                            # within-tile). The split is the DMA's access
                            # pattern; the buffer stays contiguous.
                            air.ops.load(
                                l2_a,
                                A[
                                    row : row + l2_m,
                                    k_l2_off : k_l2_off + tile_k_l2,
                                ].reshape(herd_m, tile_m, tile_k_l2),
                            )
                            air.ops.load(
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
                                        [tile_m, tile_k_l1], bf16, scope=h.private()
                                    )
                                    l1_b = air.alloc(
                                        [tile_bytes], i8, scope=h.private()
                                    )
                                    # f32, and kept across the whole K loop so
                                    # partial sums never round to bf16 between
                                    # kernel calls.
                                    acc = air.alloc(
                                        [tile_m, tile_n], f32, scope=h.private()
                                    )
                                    drain = air.alloc(
                                        [tile_m, tile_n], bf16, scope=h.private()
                                    )

                                    zero_acc(acc)
                                    for j in air.sequential(k_per_l2):
                                        k1_off = j * tile_k_l1
                                        air.ops.load(
                                            l1_a,
                                            l2_a[tx, :, k1_off : k1_off + tile_k_l1],
                                        )
                                        air.ops.load(l1_b, l2_b[ty, j, :])
                                        matmul(l1_b, l1_a, acc)
                                    to_bf16(acc, drain)

                                    air.ops.store(drain, l2_c[tx, ty, :, :])

                        # The drain is the inverse walk: bring the M axes back
                        # outside the N ones, so [herd_m, herd_n, tile_m,
                        # tile_n] lands as a plain [l2_m, l2_n] block of C.
                        air.ops.store(
                            l2_c.transpose(0, 2, 1, 3),
                            C[row : row + l2_m, col : col + l2_n],
                        )

    # --target=aie2p built the kernel object and both lits are npu2-only, so the
    # module is npu2-specific before the herd is even sized.
    return launch.build(target="npu2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=16, dest="tile_m")
    parser.add_argument("--tile-k-l2", type=int, default=128, dest="tile_k_l2")
    parser.add_argument("--tile-k-l1", type=int, default=128, dest="tile_k_l1")
    parser.add_argument("--tile-n", type=int, default=16, dest="tile_n")
    parser.add_argument("--herd-m", type=int, default=2, dest="herd_m")
    parser.add_argument("--herd-n", type=int, default=4, dest="herd_n")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-and-run", "compile-only"],
        default="compile-and-run",
        dest="compile_mode",
    )
    args = parser.parse_args()

    module = build_module(
        args.m,
        args.k,
        args.n,
        args.gs,
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
    W_q_unp = np.random.randint(0, 16, size=(args.n, args.k), dtype=np.uint8)
    W_q = (W_q_unp[:, 0::2] | (W_q_unp[:, 1::2] << 4)).astype(np.uint8)
    n_groups = args.k // args.gs
    W_s = np.random.uniform(0.005, 0.02, size=(n_groups, args.n)).astype(bfloat16)
    W_z = np.random.randint(7, 9, size=(n_groups, args.n), dtype=np.uint8)
    A = np.random.randn(args.m, args.k).astype(bfloat16)

    PACKED = pack_inputs(
        W_q, W_s, W_z, args.m, args.k, args.n, args.gs, args.tile_n, args.tile_k_l1
    )
    C_ref = cpu_reference(W_q, W_s, W_z, A)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            runtime_loop_tiling_sizes=[2, 2],
            stack_size=16384,
        )
        backend.compile(module)
        backend.unload()
        sys.exit(0)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="xclbin",
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=16384,
    )
    sys.exit(
        runner.run_test(
            module,
            inputs=[A, PACKED],
            expected_outputs=[C_ref],
            rtol=0.1,
            atol=0.05,
            # bf16 floor: at large K and tight atol a small fraction of
            # elements land just outside atol while correlation stays > 0.9999.
            max_mismatch_percentage=0.05,
            min_correlation=0.999,
        )
    )
