# ./python/test/api/pack.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Micro-tiled (packed) layouts: shapes, DMA patterns, and the contraction.

Every CHECK here is transcribed from the IR of the example this models:

    programming_examples/matrix_multiplication/bf16/run.py \
        --herd-m 1 --herd-n 1 --m 64 --n 64 --k 64 --tile-m 32 \
        --tile-k-l2 32 --tile-k-l1 16 --tile-n 32 --arch aie2 --print-module-only

so a failure here means the DSL has drifted from a form known to run on
hardware, not merely from a form that looked reasonable when this was written.
"""

import air.api as air
import air.api.ops as ops
from air.api import bf16

M = N = K = 64
TILE_M = TILE_N = TILE_K_L2 = 32
TILE_K_L1 = 16
HERD_M = HERD_N = 1


def build():
    mm = air.micro_tile(m=4, k=8, n=4)

    A = air.tensor([M, K], bf16)
    B = air.tensor([K, N], bf16)
    C = air.tensor([M, N], bf16)

    with air.launch(
        [range(0, M, TILE_M * HERD_M), range(0, N, TILE_N * HERD_N)],
        name="matmul",
        target="npu1",
    ) as launch:

        @launch.body
        def _(ss, st):
            with air.segment(name="matmul_seg") as seg:

                @seg.body
                def _():
                    row, col = ss * TILE_M * HERD_M, st * TILE_N * HERD_N

                    l2_a = air.alloc(
                        [1, 1, TILE_M, TILE_K_L2], bf16, scope=seg.private()
                    )
                    l2_b = air.alloc(
                        [1, HERD_N, TILE_K_L2, TILE_N], bf16, scope=seg.private()
                    )
                    l2_c = air.alloc([1, 1, TILE_M, TILE_N], bf16, scope=seg.private())
                    acc = air.alloc(
                        mm.c(TILE_M, TILE_N, lead=(HERD_M, HERD_N)),
                        bf16,
                        scope=seg.shared(),
                    )

                    with air.herd([range(HERD_M), range(HERD_N)], name="fill") as h0:

                        @h0.body
                        def _(tx, ty):
                            acc[:] = 0.0

                    for k2 in air.sequential(0, K, TILE_K_L2):
                        ops.load(l2_a, A[row : row + TILE_M, k2 : k2 + TILE_K_L2])
                        ops.load(l2_b, B[k2 : k2 + TILE_K_L2, col : col + TILE_N])

                        with air.herd([range(HERD_M), range(HERD_N)], name="mm") as h:

                            @h.body
                            def _(tx, ty):
                                l1_a = air.alloc(
                                    mm.a(TILE_M, TILE_K_L1), bf16, scope=h.private()
                                )
                                l1_b = air.alloc(
                                    mm.b(TILE_K_L1, TILE_N), bf16, scope=h.private()
                                )
                                for k1 in air.sequential(0, TILE_K_L2, TILE_K_L1):
                                    ops.load(l1_a, l2_a[tx, 0, :, k1 : k1 + TILE_K_L1])
                                    ops.load(l1_b, l2_b[0, ty, k1 : k1 + TILE_K_L1, :])
                                    ops.dot(l1_a, l1_b, acc=acc)

                    with air.herd([range(HERD_M), range(HERD_N)], name="drain") as h2:

                        @h2.body
                        def _(tx, ty):
                            ops.store(acc[tx, ty, :, :], l2_c[tx, ty, :, :])

                    ops.store(l2_c, C[row : row + TILE_M, col : col + TILE_N])

    return launch


# The three indexing maps of the contraction, byte-identical to the reference's.
# CHECK-DAG: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
# CHECK-DAG: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>
# CHECK-DAG: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d0, d4, d3, d6, d7)>

# The launch carries the M/N tiling: 64/32 = 2 in each direction. The L2
# staging buffers are refilled per point, so this cannot be the herd's job.
# CHECK-LABEL: func.func @matmul
# CHECK: air.launch (%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) in (%{{[a-z0-9_]+}}=%c2{{[a-z0-9_]*}}, %{{[a-z0-9_]+}}=%c2{{[a-z0-9_]*}})

# The launch induction variables are passed into the segment as operands --
# air.segment is IsolatedFromAbove, so they cannot be referenced through it.
# CHECK: air.segment @matmul_seg args({{.*}}) : index, index, memref<64x64xbf16>

# Packed shapes. A is [1,1,K/k,M/m,m,k], B is [1,1,N/n,K/k,k,n], C is
# [herd_m,herd_n,N/n,M/m,m,n]. All contiguous: packing is not a layout.
# CHECK: memref.alloc() : memref<1x1x32x32xbf16, 1 : i32>
# CHECK: memref.alloc() : memref<1x1x8x8x4x4xbf16, 2 : i32>

# Zeroing a packed accumulator is one linalg.fill on the core's own slab, not a
# six-deep scalar loop nest.
# CHECK: %[[SV0:[a-z0-9_]+]] = memref.subview %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1]
# CHECK-SAME: to memref<1x1x8x8x4x4xbf16, strided<[1024, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
# CHECK: linalg.fill ins(%{{[a-z0-9_]+}} : bf16) outs(%[[SV0]]

# L1 A tile: the pack. Six offsets/sizes/strides against a rank-4 L2 memref --
# the DMA walks it in micro-tile order and lands it contiguously in L1.
# CHECK: air.dma_memcpy_nd (%{{[a-z0-9_]+}}[] [] [], %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, 0, 0, 0, 0, %{{[a-z0-9_]+}}]
# CHECK-SAME: [1, 1, 2, 8, 4, 8] [1024, 1024, 8, 128, 32, 1])
# CHECK-SAME: (memref<1x1x2x8x4x8xbf16, 2 : i32>, memref<1x1x32x32xbf16, 1 : i32>)

# L1 B tile: same, with the K offset on the k_in dimension.
# CHECK: air.dma_memcpy_nd (%{{[a-z0-9_]+}}[] [] [], %{{[a-z0-9_]+}}[0, %{{[a-z0-9_]+}}, 0, 0, %{{[a-z0-9_]+}}, 0]
# CHECK-SAME: [1, 1, 8, 2, 8, 4] [1024, 1024, 4, 256, 32, 1])
# CHECK-SAME: (memref<1x1x8x2x8x4xbf16, 2 : i32>, memref<1x1x32x32xbf16, 1 : i32>)

# The contraction: a 9-dimensional linalg.generic over the micro-tile grid,
# accumulating into the core's subview of the shared accumulator.
# CHECK: %[[SV1:[a-z0-9_]+]] = memref.subview %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4]
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
# CHECK-SAME: ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<1x1x2x8x4x8xbf16, 2 : i32>, memref<1x1x8x2x8x4xbf16, 2 : i32>)
# CHECK-SAME: outs(%[[SV1]]

# The drain: the pattern moves to the packed side and walks it back in logical
# row-major order, which is the unpack.
# CHECK: air.dma_memcpy_nd (%{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1],
# CHECK-SAME: %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1])


print(build().build(target="npu1"))


# ---------------------------------------------------------------------------
# ops.dot(kernel=...): naming the external function.
#
# Without it a micro-tiled contraction lowers to MLIR's
# "op_has_no_registered_library_name" placeholder -- the OpDSL emitter hardcodes
# library_call=None -- so every such contraction in the tree resolves to one
# symbol. A kernel compiled for the wrong tile dimensions then links anyway and
# computes silently wrong results.
# ---------------------------------------------------------------------------


def build_named_kernel():
    mm = air.micro_tile(m=4, k=8, n=4)
    A = air.tensor([32, 32], bf16)
    C = air.tensor([32, 32], bf16)

    with air.launch([range(0, 32, 32)], name="named", target="npu1") as launch:

        @launch.body
        def _(si):
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    acc = air.alloc(mm.c(32, 32), bf16, scope=seg.shared())
                    l2 = air.alloc([32, 32], bf16, scope=seg.private())
                    air.ops.load(l2, A[0:32, 0:32])

                    with air.herd([range(1), range(1)], name="mm") as h:

                        @h.body
                        def _(tx, ty):
                            a = air.alloc(mm.a(32, 16), bf16, scope=h.private())
                            b = air.alloc(mm.b(16, 32), bf16, scope=h.private())
                            air.ops.dot(a, b, acc=acc, kernel="matmul_bf16_m32k16n32")

                    with air.herd([range(1), range(1)], name="drain") as h2:

                        @h2.body
                        def _(tx, ty):
                            air.ops.store(acc[tx, ty, :, :], l2[0:32, 0:32])

                    air.ops.store(l2, C[0:32, 0:32])

    return launch


# The name rides on the contraction as linalg's own attribute, so
# air-linalg-to-func picks it up without any air-specific plumbing.
# CHECK-LABEL: func.func @named
# CHECK: linalg.generic
# CHECK-SAME: library_call = "matmul_bf16_m32k16n32"

print(build_named_kernel().build(target="npu1"))
