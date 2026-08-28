# ./python/test/api/pack.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Blocked layouts: shapes, DMA patterns, and the contraction.

Every CHECK here is transcribed from the IR of the example this models:

    programming_examples/matrix_multiplication/bf16/run.py \
        --herd-m 1 --herd-n 1 --m 64 --n 64 --k 64 --tile-m 32 \
        --tile-k-l2 32 --tile-k-l1 16 --tile-n 32 --arch aie2 --print-module-only

so a failure here means the DSL has drifted from a form known to run on
hardware, not merely from a form that looked reasonable when this was written.
"""

import air.api as air
import air.api.ops as ops
from air.api import bf16, i32

M = N = K = 64
TILE_M = TILE_N = TILE_K_L2 = 32
TILE_K_L1 = 16
HERD_M = HERD_N = 1


def build():
    MM_M, MM_K, MM_N = 4, 8, 4

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
                        [
                            HERD_M,
                            HERD_N,
                            TILE_N // MM_N,
                            TILE_M // MM_M,
                            MM_M,
                            MM_N,
                        ],
                        bf16,
                        scope=seg.shared(),
                    )

                    with air.herd([range(HERD_M), range(HERD_N)], name="fill") as h0:

                        @h0.body
                        def _(tx, ty):
                            ops.fill(acc, 0.0)

                    for k2 in air.sequential(0, K, TILE_K_L2):
                        ops.load(l2_a, A[row : row + TILE_M, k2 : k2 + TILE_K_L2])
                        ops.load(l2_b, B[k2 : k2 + TILE_K_L2, col : col + TILE_N])

                        with air.herd([range(HERD_M), range(HERD_N)], name="mm") as h:

                            @h.body
                            def _(tx, ty):
                                l1_a = air.alloc(
                                    [
                                        1,
                                        1,
                                        TILE_K_L1 // MM_K,
                                        TILE_M // MM_M,
                                        MM_M,
                                        MM_K,
                                    ],
                                    bf16,
                                    scope=h.private(),
                                )
                                l1_b = air.alloc(
                                    [
                                        1,
                                        1,
                                        TILE_N // MM_N,
                                        TILE_K_L1 // MM_K,
                                        MM_K,
                                        MM_N,
                                    ],
                                    bf16,
                                    scope=h.private(),
                                )
                                for k1 in air.sequential(0, TILE_K_L2, TILE_K_L1):
                                    ops.load(
                                        l1_a,
                                        l2_a[tx, 0, :, k1 : k1 + TILE_K_L1]
                                        .reshape(
                                            1,
                                            1,
                                            TILE_M // MM_M,
                                            MM_M,
                                            TILE_K_L1 // MM_K,
                                            MM_K,
                                        )
                                        .transpose(0, 1, 4, 2, 3, 5),
                                    )
                                    ops.load(
                                        l1_b,
                                        l2_b[0, ty, k1 : k1 + TILE_K_L1, :]
                                        .reshape(
                                            1,
                                            1,
                                            TILE_K_L1 // MM_K,
                                            MM_K,
                                            TILE_N // MM_N,
                                            MM_N,
                                        )
                                        .transpose(0, 1, 4, 2, 3, 5),
                                    )
                                    ops.dot(l1_a, l1_b, acc=acc)

                    with air.herd([range(HERD_M), range(HERD_N)], name="drain") as h2:

                        @h2.body
                        def _(tx, ty):
                            ops.store(
                                acc[tx, ty, :, :, :, :].transpose(0, 1, 3, 4, 2, 5),
                                l2_c[tx, ty, :, :],
                            )

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
# Without it a blocked contraction lowers to MLIR's
# "op_has_no_registered_library_name" placeholder -- the OpDSL emitter hardcodes
# library_call=None -- so every such contraction in the tree resolves to one
# symbol. A kernel compiled for the wrong tile dimensions then links anyway and
# computes silently wrong results.
# ---------------------------------------------------------------------------


def build_named_kernel():
    A = air.tensor([32, 32], bf16)
    C = air.tensor([32, 32], bf16)

    with air.launch([range(0, 32, 32)], name="named", target="npu1") as launch:

        @launch.body
        def _(si):
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    acc = air.alloc([1, 1, 8, 8, 4, 4], bf16, scope=seg.shared())
                    l2 = air.alloc([32, 32], bf16, scope=seg.private())
                    air.ops.load(l2, A[0:32, 0:32])

                    with air.herd([range(1), range(1)], name="mm") as h:

                        @h.body
                        def _(tx, ty):
                            a = air.alloc([1, 1, 2, 8, 4, 8], bf16, scope=h.private())
                            b = air.alloc([1, 1, 8, 2, 8, 4], bf16, scope=h.private())
                            air.ops.dot(a, b, acc=acc, kernel="matmul_bf16_m32k16n32")

                    with air.herd([range(1), range(1)], name="drain") as h2:

                        @h2.body
                        def _(tx, ty):
                            air.ops.store(
                                acc[tx, ty, :, :, :, :].transpose(0, 1, 3, 4, 2, 5),
                                l2[0:32, 0:32],
                            )

                    air.ops.store(l2, C[0:32, 0:32])

    return launch


# The name rides on the contraction as linalg's own attribute, so
# air-linalg-to-func picks it up without any air-specific plumbing.
# CHECK-LABEL: func.func @named
# CHECK: linalg.generic
# CHECK-SAME: library_call = "matmul_bf16_m32k16n32"

print(build_named_kernel().build(target="npu1"))


# ---------------------------------------------------------------------------
# A hand-written kernel is the third way to write a shared accumulator.
#
# ops.fill zeroes one and ops.dot accumulates into one, and both narrow the
# buffer to the calling core's slab first -- a shared buffer spans every core
# and there is exactly one slab a given core may touch, so it is not a choice
# the caller gets to make. air.extern reaches the same accumulator through
# func.call, and does the same thing.
#
# Both directions are pinned here, because the failure modes are opposite: a
# shared buffer passed whole would let every core write every slab, and a
# private buffer narrowed anyway would index a herd coordinate into a memref
# that has no axis for it. The two bfp16 matmuls are the callers this models.
# ---------------------------------------------------------------------------


def build_extern_accumulator():
    A = air.tensor([32, 32], bf16)
    C = air.tensor([32, 32], bf16)

    zero = air.extern("zero_kernel", link_with="mm.o")
    step = air.extern("step_kernel", link_with="mm.o")

    with air.launch([range(0, 32, 32)], name="extern_acc", target="npu1") as launch:

        @launch.body
        def _(si):
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    acc = air.alloc([2, 2, 8, 8, 4, 4], bf16, scope=seg.shared())
                    l2 = air.alloc([32, 32], bf16, scope=seg.private())
                    air.ops.load(l2, A[0:32, 0:32])

                    with air.herd([range(2), range(2)], name="mm") as h:

                        @h.body
                        def _(tx, ty):
                            local = air.alloc([4, 4], bf16, scope=h.private())
                            zero(acc)
                            step(local, acc)

                    air.ops.store(l2, C[0:32, 0:32])

    return launch


# step_kernel is declared above zero_kernel because a declaration goes to the
# top of the module as it is first called, so the order is the reverse of the
# call order. Its first operand is the *private* buffer, passed whole -- no
# subview and no coordinates -- and its second is the shared accumulator,
# narrowed. Both in one signature, which is the pair this is here to pin.
# CHECK-LABEL: func.func private @step_kernel
# CHECK-SAME: memref<4x4xbf16, 2 : i32>
# CHECK-SAME: memref<1x1x8x8x4x4xbf16, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
#
# The strided type is derived, not declared: [2, 2, 8, 8, 4, 4] row-major is
# [2048, 1024, 128, 16, 4, 1], and the offset is dynamic because the core's
# coordinates are.
# CHECK-LABEL: func.func private @zero_kernel
# CHECK-SAME: memref<1x1x8x8x4x4xbf16, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
#
# At the call sites: one subview per call, at the core's own coordinates, and
# the private buffer reaching step_kernel as its bare alloc.
# CHECK-LABEL: func.func @extern_acc
# CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<4x4xbf16, 2 : i32>
# CHECK: %[[SV:.*]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1]
# CHECK: func.call @zero_kernel(%[[SV]])
# CHECK: %[[SV2:.*]] = memref.subview
# CHECK: func.call @step_kernel(%[[LOCAL]], %[[SV2]])

print(build_extern_accumulator().build(target="npu1"))


# ---------------------------------------------------------------------------
# The subview the caller DOES choose.
#
# The two above are narrowings the DSL performs on the caller's behalf, because
# there is exactly one slab a core may touch. A cascade payload is the opposite
# case: a fixed-width slice of a wider L1 buffer, and which row it is comes from
# the loop. Everywhere else in the DSL that subscript is a DMA access pattern;
# passed to a kernel it has to become a real memref, so it emits memref.subview.
#
# The offset is dynamic and the declaration carries the subview's strided type,
# so the C symbol is declared exactly as it is called -- which is the property
# the whole of air.extern is built on.
# ---------------------------------------------------------------------------


def build_extern_region():
    A = air.tensor([16, 16], bf16)
    B = air.tensor([16, 16], bf16)
    row_kernel = air.extern("row_kernel", link_with="mm.o")

    with air.launch(name="extern_region") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([16, 16], bf16, scope=h.private())
                    ops.load(buf, A)
                    for r in air.sequential(16):
                        row_kernel(buf[r, 0:16])
                    ops.store(buf, B)

    return launch


# CHECK: func.func private @row_kernel(memref<1x16xbf16, strided<[16, 1], offset: ?>, 2 : i32>)
# CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [1, 16] [1, 1]
# CHECK: func.call @row_kernel(%subview)
print(build_extern_region().build(target="npu1"))


# A reshape of *part* of a buffer still names no memref. The whole-buffer case
# does -- see the collapse tests below -- but `buf[0:16, 0:16].reshape(256)`
# describes the order a transfer would walk a sub-region, and memref has no view
# for that. This test asserted a blanket refusal of every reshape when the
# subview support was written; the rule narrowed when collapse arrived, and this
# is the half of it that still holds.
# CHECK: refused: {{.*}}does not cover the whole buffer
def build_reshaped_region_is_refused():
    A = air.tensor([16, 16], bf16)
    B = air.tensor([16, 16], bf16)
    k = air.extern("reshaped_kernel", link_with="mm.o")

    with air.launch(name="extern_reshaped") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([16, 16], bf16, scope=h.private())
                    ops.load(buf, A)
                    try:
                        k(buf[0:8, 0:16].reshape(128))
                        print("NOT REFUSED")
                    except TypeError as e:
                        print("refused:", e)
                    ops.store(buf, B)

    return launch


build_reshaped_region_is_refused().build(target="npu1")


# ---------------------------------------------------------------------------
# The other view a kernel can take: a whole buffer at a lower rank.
#
# The flash-attention kernels take their accumulator flat -- a [chunks, n] L1
# tile handed to zero_fill_g_bf16 as [chunks * n] -- because the kernel walks it
# as one run and the rank only matters to the matmul that fills it.
# `buf.reshape(n)` is how the DSL already says "the same elements at a different
# rank", and for the whole of a contiguous buffer that is memref.collapse_shape.
#
# Only a *grouping* of the existing axes is one. The refusals below are the
# three ways it can fail to be, and the transpose is the one worth having a test
# for: it has the same extents as an identity collapse, so a check on shape
# alone accepts it and emits a collapse that hands the kernel the untransposed
# buffer. The strides are what tell them apart.
# ---------------------------------------------------------------------------


def build_extern_collapsed():
    A = air.tensor([8, 64], bf16)
    B = air.tensor([8, 64], bf16)
    flat = air.extern("zero_fill_g_bf16", link_with="attn.o")

    with air.launch(name="extern_collapsed") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    g = air.alloc([8, 64], bf16, scope=h.private())
                    flat(g.reshape(512))
                    ops.load(g, A)
                    ops.store(g, B)

    return launch


# CHECK: func.func private @zero_fill_g_bf16(memref<512xbf16, 2 : i32>)
# CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]] : memref<8x64xbf16, 2 : i32> into memref<512xbf16, 2 : i32>
# CHECK: func.call @zero_fill_g_bf16

print(build_extern_collapsed().build(target="npu1"))


def refused(what, fn):
    try:
        fn()
    except Exception as e:
        print(f"{what}: {type(e).__name__}: {e}")
        return
    raise AssertionError(f"{what}: expected a diagnostic, got none")


def build_extern_collapse_refusals():
    A = air.tensor([8, 64], bf16)
    B = air.tensor([8, 64], bf16)
    k = air.extern("takes_a_memref", link_with="attn.o")

    with air.launch(name="extern_collapse_refusals") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    g = air.alloc([8, 64], bf16, scope=h.private())
                    refused("transpose", lambda: k(g.transpose(1, 0)))
                    refused("expand", lambda: k(g.reshape(8, 8, 8)))
                    refused("part of the buffer", lambda: k(g[0:4, :].reshape(256)))
                    ops.load(g, A)
                    ops.store(g, B)

    return launch


# CHECK: transpose: TypeError: {{.*}}different order (a transpose)
# CHECK: expand: TypeError: {{.*}}memref.expand_shape
# CHECK: part of the buffer: TypeError: {{.*}}does not cover the whole buffer

build_extern_collapse_refusals().build(target="npu1")


# ---------------------------------------------------------------------------
# A kernel scalar read out of a buffer.
#
# Rank decides whether a subscript is a memref operand or a value, which is the
# rule the rest of the DSL already indexes by: an integer subscript drops an
# axis and a slice keeps it, so ctr[0] is a scalar and ctr[0:1] is a
# one-element region. Until this, both arrived as a memref and the scalar count
# came out wrong.
#
# flash_attention/kernel_fusion_based is why: its cores keep a counter tile in
# L1 across launch iterations, and the causal mask takes the q-block index out
# of it -- an element of that tile plus the tile coordinate.
# ---------------------------------------------------------------------------


def build_extern_scalar_from_memory():
    A = air.tensor([64, 64], bf16)
    B = air.tensor([64, 64], bf16)
    mask = air.extern("apply_causal_mask", link_with="attn.o", scalars=[i32, i32])

    with air.launch(name="extern_scalar") as launch:

        @launch.body
        def _():
            with air.herd([range(2), range(2)], name="h", shape=(2, 2)) as h:

                @h.body
                def _(tx, ty):
                    g = air.alloc([64, 64], bf16, scope=h.private())
                    ctr = air.alloc([4], i32, scope=h.private())
                    ops.load(g, A)
                    mask(g, ctr[0] + tx, ty)
                    ops.store(g, B)

    return launch


# CHECK: func.func private @apply_causal_mask(memref<64x64xbf16, 2 : i32>, i32, i32)
# CHECK: %[[BASE:.*]] = memref.load %{{.*}}[%c0]
# CHECK: %[[TX:.*]] = arith.index_cast
# CHECK: %[[Q:.*]] = arith.addi %[[BASE]], %[[TX]]
# CHECK: func.call @apply_causal_mask(%{{.*}}, %[[Q]], %{{.*}})

print(build_extern_scalar_from_memory().build(target="npu1"))


def build_extern_scalar_refusals():
    A = air.tensor([64, 64], bf16)
    B = air.tensor([64, 64], bf16)
    one = air.extern("takes_one_i32", link_with="attn.o", scalars=[i32])

    with air.launch(name="extern_scalar_refusals") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    g = air.alloc([64, 64], bf16, scope=h.private())
                    f = air.alloc([4], bf16, scope=h.private())
                    ops.load(g, A)
                    # Stored as bf16, declared i32: passed as it is stored, so
                    # the mismatch is named rather than silently converted.
                    refused("wrong element type", lambda: one(g, f[0]))
                    # Rank 1 is a region, and a region is a memref -- so this is
                    # counted as a buffer operand and the scalar count is short.
                    refused("rank-1 region", lambda: one(g, f[0:1]))
                    ops.store(g, B)

    return launch


# CHECK: wrong element type: TypeError: {{.*}}reads air.api.bf16 out of a buffer but was declared air.api.i32
# CHECK: rank-1 region: TypeError: {{.*}}1 scalar argument type(s) but called with 0

build_extern_scalar_refusals().build(target="npu1")
