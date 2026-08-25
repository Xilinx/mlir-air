# ./python/test/api/reduce.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers ops.reduce_add / reduce_max to vector.reduction.

A reduction is the DSL's only shape-changing node -- every other node is
elementwise, so the emitter checks leaf shapes against the destination. What
these pin is mostly that consequence: which axis is collapsed, that both
keepdims spellings work, and that the row is read as a *single* vector with no
accumulator, since a loop-carried vector accumulator is the construct AIE2
cannot legalize.
"""

from air import api as air
from air.api.types import bf16, i32


def build(body, dtype=bf16, M=65536, N=16, tile=256, out_shape=None):
    A = air.tensor([M, N], dtype)
    OUT = air.tensor([M], dtype)

    with air.launch(name="red") as launch:

        @launch.body
        def _():
            with air.herd(range(0, M, tile), shape=(2,)) as h:

                @h.body
                def _(tx):
                    (tm,) = h.tile_sizes
                    row = tx * tm
                    a = air.alloc([tm, N], dtype, scope=h.private())
                    o = air.alloc(out_shape or [tm], dtype, scope=h.private())
                    air.ops.load(a, A[row : row + tm, :])
                    o[:] = body(a)
                    air.ops.store(o, OUT[row : row + tm])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: reduce_add_is_one_vector_read_and_one_reduction
# The row is read whole -- vector<16xbf16> for an N of 16 -- and reduced in one
# op. No scf.for over the row, and so no loop-carried vector accumulator: that
# is the construct ops.dot documents as unlegalizable on AIE2, and reading the
# full extent is how the hand-written kernel avoids it too.
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: vector.reduction <add>, {{.*}} : vector<16xbf16> into bf16
# CHECK: memref.store
@run
def reduce_add_is_one_vector_read_and_one_reduction():
    print(build(lambda a: air.ops.reduce_add(a[:])).mlir())


# CHECK-LABEL: TEST: reduce_max_uses_maximumf_on_floats
# maximumf, not maxnumf -- the same choice the elementwise ops.maximum makes,
# and the form known to legalize on AIE2.
# CHECK: vector.reduction <maximumf>, {{.*}} : vector<16xbf16> into bf16
@run
def reduce_max_uses_maximumf_on_floats():
    print(build(lambda a: air.ops.reduce_max(a[:])).mlir())


# CHECK-LABEL: TEST: reduce_max_uses_maxsi_on_integers
# The signed integer form: air.api's integer dtypes are signed, and an unsigned
# buffer is refused before it reaches here.
# CHECK: vector.reduction <maxsi>, {{.*}} : vector<16xi32> into i32
@run
def reduce_max_uses_maxsi_on_integers():
    print(build(lambda a: air.ops.reduce_max(a[:]), dtype=i32).mlir())


# CHECK-LABEL: TEST: keepdims_stores_at_the_collapsed_index
# Destination [tm, 1] rather than [tm]: the reduced axis is kept, numpy's
# keepdims=True, and the scalar is stored at index 0 of it.
#
# Note the L3 output here is [M, 1], not [M]. ops.store squeezes *leading*
# unit dimensions but not trailing ones, so a [tm, 1] L1 tile cannot currently
# be stored into a rank-1 [m] slice -- which is exactly what the hand-written
# reduce kernel does in its DMA. That is why the converted examples drop the
# axis instead: it keeps the L1 tile the same rank as the L3 output. Both
# spellings of the reduction work; only this pairing of the two is unsupported.
# CHECK: memref.alloc() : memref<256x1xbf16, 2 : i32>
# CHECK: vector.reduction <add>
# CHECK: memref.store {{.*}}memref<256x1xbf16, 2 : i32>
@run
def keepdims_stores_at_the_collapsed_index():
    M, N, tile = 65536, 16, 256
    A = air.tensor([M, N], bf16)
    OUT = air.tensor([M, 1], bf16)

    with air.launch(name="red") as launch:

        @launch.body
        def _():
            with air.herd(range(0, M, tile), shape=(2,)) as h:

                @h.body
                def _(tx):
                    (tm,) = h.tile_sizes
                    row = tx * tm
                    a = air.alloc([tm, N], bf16, scope=h.private())
                    o = air.alloc([tm, 1], bf16, scope=h.private())
                    air.ops.load(a, A[row : row + tm, :])
                    o[:] = air.ops.reduce_add(a[:])
                    air.ops.store(o, OUT[row : row + tm, :])

    print(launch.mlir())


# CHECK-LABEL: TEST: reduce_over_an_expression_is_a_row_dot_product
# The operand of a reduction may be any elementwise expression, so a row-wise
# dot product needs no extra surface: the multiply happens on the vector,
# before the reduction collapses it.
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: vector.reduction <add>, {{.*}} : vector<16xbf16> into bf16
@run
def reduce_over_an_expression_is_a_row_dot_product():
    print(build(lambda a: air.ops.reduce_add(a[:] * a[:])).mlir())


# CHECK-LABEL: TEST: a_wider_axis_widens_the_vector_not_the_loop
# N=32 is what vector_reduce_max defaults to. The reduced extent sets the
# vector length directly, so a wider axis stays one read and one reduction
# rather than becoming a loop.
# CHECK: vector.transfer_read {{.*}} vector<32xbf16>
# CHECK: vector.reduction <add>, {{.*}} : vector<32xbf16> into bf16
# CHECK-NOT: vector.reduction
@run
def a_wider_axis_widens_the_vector_not_the_loop():
    print(build(lambda a: air.ops.reduce_add(a[:]), N=32).mlir())
