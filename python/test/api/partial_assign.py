# ./python/test/api/partial_assign.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Assigning into part of a buffer, down to one element.

``out[:] = ...`` writes a whole tile, and for a long time it was the only thing
that could be written. That covered every kernel whose arithmetic is
tile-shaped, and none whose arithmetic is not: a convolution accumulates into
``out[oh, ow, co]`` from ``in[oh + kh, ow + kw, ci]``, and neither of those is a
tile. Refusing them pushed such a kernel out of the DSL entirely.

What decides the shape being written is numpy's rule, applied to the subscript:
an integer selects one element and drops its axis, a slice keeps it. So
``out[0:8, :]`` is a rank-2 block, ``out[1, :]`` is a row, and ``out[i, j, k]``
is a scalar -- rank 0, with no axis left to loop over and therefore no loop.

The axis an integer took is still there in the access pattern a transfer builds
from the same subscript. Dropping it there would change every ``ops.load`` in
the tree, and the two readings do not conflict: ``staged[tx, 0:m, :]`` names
`m*k` contiguous elements either way.
"""

from air import api as air
from air.api.types import bf16, i32


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(body, out_shape, dtype=bf16, vector=None):
    OUT = air.tensor(list(out_shape), dtype)
    with air.launch(name="partial") as launch:

        @launch.body
        def _():
            with air.herd(range(1), name="herd_0", shape=(1,)) as h:

                @h.body
                def _(tx):
                    kw = {} if vector is None else {"vector": vector}
                    src = air.alloc([8, 8, 4], dtype, scope=h.private(), **kw)
                    dst = air.alloc(list(out_shape), dtype, scope=h.private(), **kw)
                    body(h, src, dst)
                    air.ops.store(dst, OUT)

    print(launch.build(target="npu1"))


# CHECK-LABEL: TEST: one_element_emits_no_loop
# The rank-0 case, and the reason the whole feature is worth having. Every axis
# was taken by an integer, so there is nothing to iterate: the assignment is
# three loads, the arithmetic, and one store, sitting directly in whatever loop
# the kernel itself wrote. No induction variable of the DSL's own appears.
# CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
# CHECK-NOT: scf.for
# CHECK: memref.load %[[SRC:.*]][%[[I]], %[[I]], %[[I]]]
# CHECK-NOT: scf.for
# CHECK: memref.store %{{.*}}, %{{.*}}[%[[I]], %[[I]], %[[I]]]
@run
def one_element_emits_no_loop():
    def body(h, src, dst):
        for i in air.sequential(4):
            dst[i, i, i] = src[i, i, i] * 2

    build(body, (8, 8, 4), i32, vector=0)


# CHECK-LABEL: TEST: a_shifted_window_reads_at_a_loop_variable
# The convolution shape: the destination element is fixed by the outer loops
# and the operand is read one kernel offset along. An offset that is a loop
# variable used to be refused outright; it is one more term in the index, and
# reaches the load as the single affine.apply every index in this DSL becomes.
# CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
# CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
# CHECK: %[[SH:.*]] = affine.apply #{{.*}}()[%[[I]]]
# CHECK: memref.load %{{.*}}[%[[SH]], %[[J]], %{{.*}}]
# CHECK: memref.store %{{.*}}, %{{.*}}[%[[I]], %[[J]], %{{.*}}]
@run
def a_shifted_window_reads_at_a_loop_variable():
    def body(h, src, dst):
        for i in air.sequential(4):
            for j in air.sequential(4):
                dst[i, j, 0] = src[i + 3, j, 1] + 1

    build(body, (8, 8, 4), i32, vector=0)


# CHECK-LABEL: TEST: a_block_keeps_its_axes_and_its_loops
# A slice subscript keeps the axis, so this is rank 2 and nests twice -- over
# the region's extents, not the buffer's -- and both ends are shifted by the
# region's own offset. The innermost axis still vectorises: nothing about
# writing a sub-block makes the write scalar.
# CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
# CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %[[W:.*]] {
# CHECK: vector.transfer_read
# CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %[[J]]]
@run
def a_block_keeps_its_axes_and_its_loops():
    def body(h, src, dst):
        dst[2:4, :] = dst[0:2, :] * 2.0

    build(body, (4, 64), bf16)


# CHECK-LABEL: TEST: one_row_named_twice_is_one_read
# The read cache keys on what identifies a read -- which buffer, from where,
# how much -- and "from where" is now an affine form rather than an integer, so
# it has to compare structurally. A row named twice under a loop variable is
# still one read.
# CHECK: memref.load
# CHECK-NOT: memref.load
# CHECK: arith.muli
# CHECK: arith.addi
@run
def one_row_named_twice_is_one_read():
    def body(h, src, dst):
        for i in air.sequential(4):
            dst[i, 0, 0] = src[i, 0, 0] * 2 + src[i, 0, 0]

    build(body, (8, 8, 4), i32, vector=0)
