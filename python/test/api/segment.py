# ./python/test/api/segment.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api stages data through L2: L3 -> L2 -> L1 -> L2 -> L3.

A herd whose cores each stream operands straight from L3 needs a shim DMA
channel per core per tensor, which is what bounds the herd width. Staging
through a memtile lifts that: the segment moves one large tile L3 -> L2, and
each core takes its own window L2 -> L1. This is the shape every hand-written
matmul in the tree uses, e.g.
``programming_examples/matrix_vector_multiplication/bf16/matvec.py``.

Two pieces of surface appear here that the herd-only examples do not reach:

``air.segment`` is an independent level, not a wrapper the DSL forces on you.
``launch``, ``segment`` and ``herd`` each emit only when written, so a kernel
that needs no staging keeps the plain ``func`` + ``air.herd`` shape -- see
``eltwise_add.py``, whose IR this change leaves untouched.

A *partial* subscript on a buffer (``staged[tx, 0:n]``) names a DMA region
rather than a value, so it is accepted by ``ops.load``/``ops.store`` and refused
by an elementwise expression. ``buf[:]`` still means an elementwise read, and is
refused on L2, which is a memtile and has no compute core.
"""

from air import api as air
from air.api import ops  # noqa: F401
from air.api.types import f32, i32

CORES, TILE = 4, 64


def build(dtype=i32, cores=CORES, tile=TILE):
    A = air.tensor([cores, tile], dtype)
    C = air.tensor([cores, tile], dtype)

    with air.launch(name="stage") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    a_l2 = air.alloc([cores, tile], dtype, scope=seg.private())
                    c_l2 = air.alloc([cores, tile], dtype, scope=seg.private())
                    ops.load(a_l2, A[0:cores, 0:tile])

                    with air.herd(range(0, cores, 1), shape=(cores,)) as h:

                        @h.body
                        def _(tx):
                            l1 = air.alloc([tile], dtype, scope=h.private(), vector=0)
                            # The slice is [1, tile] into a rank-1 buffer: a
                            # staged tile is indexed per core, so the natural
                            # pattern carries a leading 1, which is squeezed.
                            ops.load(l1, a_l2[tx, 0:tile])
                            l1[:] = l1[:] + 1
                            ops.store(l1, c_l2[tx, 0:tile])

                    ops.store(c_l2, C[0:cores, 0:tile])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: staged_passthrough
# A segment is emitted inside an air.launch, and that is load-bearing rather
# than decorative: air-insert-launch-around-herd wraps a *bare* herd in a launch
# and a segment, and skips a herd that already sits in a segment. A segment with
# no launch above it compiles and then computes zeros for anything beyond a
# plain copy -- measured on npu1, with a raw-bindings reproduction.
# CHECK: func.func @stage(%{{.*}}: memref<4x64xi32>, %{{.*}}: memref<4x64xi32>)
# CHECK: air.launch (%{{.*}}, %{{.*}}) in ({{.*}}) args({{.*}}) : memref<4x64xi32>, memref<4x64xi32>
#
# The segment takes the L3 tensors as operands -- it is IsolatedFromAbove, so
# nothing is captured implicitly -- and the L2 buffers land in memory space 1.
# CHECK: air.segment @seg args(%[[SA:.*]]=%{{.*}}, %{{.*}}=%{{.*}}) : memref<4x64xi32>, memref<4x64xi32>
# CHECK: memref.alloc() : memref<4x64xi32, 1 : i32>
# CHECK: memref.alloc() : memref<4x64xi32, 1 : i32>
#
# L3 -> L2, in the segment body and outside the herd: staged once for every
# core, not once per core.
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[SA]][0, 0] [4, 64] [64, 1]) : (memref<4x64xi32, 1 : i32>, memref<4x64xi32>)
#
# The herd is nested inside, and carries the L2 buffers as operands -- it is
# IsolatedFromAbove too, so a memtile buffer cannot simply be referenced from
# within it. It carries *only* those: this body never touches the two L3
# tensors, and an operand it does not touch is not free. The tracer has to
# name every candidate when it creates the op, because that is before the body
# has run, but it drops the ones the body turned out not to use once it has --
# air-dependency reads a dead operand as a data dependency and would serialise
# this herd against every other one staged from the same segment.
# CHECK: air.herd @herd_0 tile (%[[TX:.*]], %{{.*}}) in ({{.*}}) args({{.*}}) : memref<4x64xi32, 1 : i32>, memref<4x64xi32, 1 : i32>
# CHECK: memref.alloc() : memref<64xi32, 2 : i32>
#
# L2 -> L1, one window per core: the tile coordinate becomes the offset, and
# the access pattern is [1, 64] into a rank-1 L1 buffer.
# CHECK: %[[OFF:.*]] = affine.apply {{.*}}[%[[TX]]]
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[OFF]], 0] [1, 64] [64, 1]) : (memref<64xi32, 2 : i32>, memref<4x64xi32, 1 : i32>)
# CHECK: memref.load
# CHECK: arith.addi
#
# L1 -> L2 writeback into the same window, then L2 -> L3 once, after the herd.
# CHECK: air.dma_memcpy_nd (%{{.*}}[%{{.*}}, 0] [1, 64] [64, 1], %{{.*}}[] [] []) : (memref<4x64xi32, 1 : i32>, memref<64xi32, 2 : i32>)
# CHECK: memref.dealloc {{.*}} : memref<64xi32, 2 : i32>
#
# Each buffer is freed after its own last use, not all of them together at the
# end of the body. The input staging buffer is dead once the herd has run, so
# it is freed there; the output one stays live across the L2 -> L3 write that
# reads it. Holding both to the end would tell the compiler two values are
# still needed when only one is.
# CHECK: memref.dealloc %[[IN:.*]] : memref<4x64xi32, 1 : i32>
# CHECK: air.dma_memcpy_nd (%{{.*}}[0, 0] [4, 64] [64, 1], %[[OUT:.*]][] [] []) : (memref<4x64xi32>, memref<4x64xi32, 1 : i32>)
# CHECK: memref.dealloc %[[OUT]] : memref<4x64xi32, 1 : i32>
@run
def staged_passthrough():
    print(build().mlir())


# CHECK-LABEL: TEST: staged_f32
# Nothing about staging is dtype-specific; f32 differs only in the element type
# and in the compute op the expression lowers to.
# CHECK: memref.alloc() : memref<4x64xf32, 1 : i32>
# CHECK: arith.addf
@run
def staged_f32():
    print(build(dtype=f32).mlir())


# CHECK-LABEL: TEST: per_core_reaches_each_herd_whole
# <segment>.shared() and <segment>.per_core() differ in what the cores see. A
# shared buffer is one allocation the cores divide between them: it carries a
# leading dimension per herd axis, and a kernel reached through air.extern is
# handed a memref.subview of this core's slab. A per_core buffer is not
# divided -- every core gets its own copy of the whole shape, and the kernel
# receives it entire.
#
# The shape below is flash_attention/dataflow_based's: a [lq, 1] running
# maximum carried across two separate herds of a 2-D herd grid. shared() cannot
# express it at all -- both its dimensions would be cores, leaving nothing for
# the tile -- which is the case checked in api/errors.py.
# CHECK: %[[UP:.*]] = memref.alloc() : memref<64x1xbf16, 2 : i32>
# CHECK: air.herd @h {{.*}}%[[A0:.*]]=%[[UP]]{{.*}} memref<64x1xbf16, 2 : i32>
# CHECK: func.call @zero_fill_sp_bf16(%[[A0]]) : (memref<64x1xbf16, 2 : i32>)
# CHECK: air.herd @h {{.*}}%[[A1:.*]]=%[[UP]]{{.*}} memref<64x1xbf16, 2 : i32>
# CHECK: func.call @zero_fill_sp_bf16(%[[A1]]) : (memref<64x1xbf16, 2 : i32>)
# CHECK-NOT: memref.subview
@run
def per_core_reaches_each_herd_whole():
    from air.api.types import bf16

    B = air.tensor([64, 1], bf16)
    fill = air.extern("zero_fill_sp_bf16", link_with="attn.o")

    with air.launch(name="carried") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    up = air.alloc([64, 1], bf16, scope=seg.per_core())

                    with air.herd([range(1), range(4)], name="h", shape=(1, 4)) as h0:

                        @h0.body
                        def _(tx, ty):
                            fill(up)

                    with air.herd([range(1), range(4)], name="h", shape=(1, 4)) as h1:

                        @h1.body
                        def _(tx, ty):
                            fill(up)
                            air.ops.store(up, B)

    print(launch.mlir())
