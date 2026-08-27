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


def build(
    body,
    dtype=bf16,
    M=65536,
    N=16,
    tile=256,
    out_shape=None,
    vector=None,
    scalar=False,
):
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
                    kw = {} if vector is None else {"vector": vector}
                    a = air.alloc([tm, N], dtype, scope=h.private(), **kw)
                    o = air.alloc(out_shape or [tm], dtype, scope=h.private())
                    air.ops.load(a, A[row : row + tm, :])
                    if scalar:
                        # A per-row scalar to broadcast across the reduced
                        # axis, as a variance's mean is.
                        m = air.alloc([tm, 1], dtype, scope=h.private())
                        o[:] = air.ops.reduce_add((a[:] - m[:]) * (a[:] - m[:]))
                    else:
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
# The L3 output here is [M, 1], matching the L1 tile rank for rank. Storing the
# same [tm, 1] tile into a rank-1 [m] slice also works now -- see
# keepdims_stores_into_a_rank_1_output below -- since ops.store squeezes a
# trailing unit dimension as well as a leading one.
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


# CHECK-LABEL: TEST: keepdims_stores_into_a_rank_1_output
# The pairing average_pool needs: a [tm, 1] L1 tile -- the shape a reduction
# produces -- stored into a rank-1 [m] L3 slice, which is the shape the array
# of row averages actually has. ops.store squeezes the trailing unit dimension,
# for the same reason it squeezes a leading one on a staged L2 tile: both
# spellings describe the same tm contiguous elements.
# CHECK: memref.alloc() : memref<256x1xbf16, 2 : i32>
# CHECK: vector.reduction <add>
# CHECK: air.dma_memcpy_nd ({{.*}}[256] [1], {{.*}}memref<256x1xbf16, 2 : i32>)
@run
def keepdims_stores_into_a_rank_1_output():
    M, N, tile = 65536, 16, 256
    A = air.tensor([M, N], bf16)
    OUT = air.tensor([M], bf16)

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
                    air.ops.store(o, OUT[row : row + tm])

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


# CHECK-LABEL: TEST: a_wider_axis_is_read_in_vector_width_steps
# N=32 is what vector_reduce_max defaults to. A reduction reads the axis the
# same way an elementwise assignment would: in steps of the buffer's vector
# width, accumulating the partials through a small L1 buffer.
#
# This used to widen the vector instead -- the whole axis in one read, however
# long it was -- and that does not scale: a [.., 768] row dies in the backend
# with `unable to legalize G_EXTRACT_VECTOR_ELT <768 x s16>`, which is exactly
# the shape layer_norm needs. The accumulator is an L1 round-trip and not an
# scf.for iter_arg because a loop-carried vector is what LLVM splits into
# sub-512-bit pieces AIE2 will not legalize; every hand-written kernel this
# models does the same round-trip.
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: scf.for
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: vector.reduction <add>, {{.*}} : vector<16xbf16> into bf16
@run
def a_wider_axis_is_read_in_vector_width_steps():
    print(build(lambda a: air.ops.reduce_add(a[:]), N=32).mlir())


# CHECK-LABEL: TEST: a_width_as_wide_as_the_axis_is_one_read
# The width is the caller's knob, and it means here what it means everywhere
# else: allocate the operand at the axis's own width and the reduction is a
# single read and a single vector.reduction, with no loop and no accumulator.
# CHECK: vector.transfer_read {{.*}} vector<32xbf16>
# CHECK: vector.reduction <add>, {{.*}} : vector<32xbf16> into bf16
# CHECK-NOT: vector.reduction
@run
def a_width_as_wide_as_the_axis_is_one_read():
    print(build(lambda a: air.ops.reduce_add(a[:]), N=32, vector=32).mlir())


# CHECK-LABEL: TEST: a_reduction_operand_may_broadcast
# The variance idiom: sum((x - mean)^2) with mean a per-row scalar. numpy
# stretches mean across the reduced axis and so does this -- the widest operand
# fixes what the axis means, and a [.., 1] operand beside it is a scalar being
# splatted into it, which is the memref.load + vector.broadcast below.
# CHECK: memref.load
# CHECK: vector.broadcast
# CHECK: vector.reduction <add>
@run
def a_reduction_operand_may_broadcast():
    print(
        build(
            lambda a: air.ops.reduce_add(a[:] * a[:]),
            N=16,
            scalar=True,
        ).mlir()
    )


# CHECK-LABEL: TEST: operands_broadcast_against_each_other
# [tm, 1] and [1, N] reduce over [tm, N], and neither operand has that shape.
# Taking the widest leaf instead of broadcasting them together would pick one of
# the two and then reject the other, so this pins the two-sided rule.
# CHECK: scf.for
# CHECK: vector.reduction <add>
@run
def operands_broadcast_against_each_other():
    A = air.tensor([256, 16], bf16)
    OUT = air.tensor([256], bf16)

    with air.launch(name="bc") as launch:

        @launch.body
        def _():
            with air.herd(range(0, 256, 256), shape=(1,)) as h:

                @h.body
                def _(tx):
                    col = air.alloc([256, 1], bf16, scope=h.private())
                    row = air.alloc([1, 16], bf16, scope=h.private())
                    o = air.alloc([256], bf16, scope=h.private())
                    air.ops.load(col, A[0:256, 0:1])
                    o[:] = air.ops.reduce_add(col[:] * row[:])
                    air.ops.store(o, OUT[0:256])

    print(launch.mlir())


# CHECK-LABEL: TEST: the_scratch_outlives_a_strip_mined_run
# Four tiles onto a two-wide herd, so the body is strip-mined and runs twice.
# The scratch is allocated once above the strip loop, and its dealloc has to sit
# once *below* it: freeing it inside the loop would leave the second trip
# reading a dead buffer. The alloc, the whole strip loop, and only then the
# dealloc.
# CHECK: %[[S:.*]] = memref.alloc() : memref<16xbf16, 2 : i32>
# CHECK: scf.for
# CHECK: vector.reduction <add>
# CHECK: memref.dealloc %[[S]] : memref<16xbf16, 2 : i32>
# CHECK-NOT: memref.dealloc %[[S]]
@run
def the_scratch_outlives_a_strip_mined_run():
    print(build(lambda a: air.ops.reduce_add(a[:]), M=1024, N=32, tile=256).mlir())


def build_argmax(cols, reduced):
    from air.api.types import f32, i32

    A = air.tensor([32, cols], f32)
    OUT = air.tensor([32], i32)

    with air.launch(name="am") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([32, cols], f32, scope=h.private())
                    o = air.alloc([32], i32, scope=h.private())
                    air.ops.load(a, A[0:32, :])
                    o[:] = air.ops.argmax(a[:, 0:reduced])
                    air.ops.store(o, OUT[0:32])

    return launch.mlir()


# CHECK-LABEL: TEST: argmax_carries_the_index_in_iter_args
# The one reduction that is a scalar loop: the running maximum and the index
# that produced it travel together, and no vector reduction carries an index.
# Scalars in iter_args are fine -- it is a loop-carried *vector* AIE2 refuses.
# The comparison is a strict > so ties keep the lowest index, as numpy does.
# CHECK: scf.for {{.*}} iter_args
# CHECK: arith.cmpf ogt
# CHECK: arith.index_cast
# CHECK: arith.select
# CHECK: arith.select
@run
def argmax_carries_the_index_in_iter_args():
    print(build_argmax(cols=16, reduced=16))


# CHECK-LABEL: TEST: argmax_over_part_of_an_axis
# A padded classifier output: 10 real columns in a 16-wide tile. Reducing the
# padded tail would let a zero outrank a negative logit, so the operand is a
# region and the loop stops at 10.
# CHECK: %[[N:.*]] = arith.constant 10 : index
# CHECK: scf.for %{{.*}} to %[[N]] step
@run
def argmax_over_part_of_an_axis():
    print(build_argmax(cols=16, reduced=10))
