# ./python/test/api/regions.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""A plain buffer region reads elementwise, spelled the way numpy spells it.

A partial subscript is primarily a DMA access pattern -- what ``ops.load`` takes
-- and for a long time that was all it could be. But kernels pack several
logical operands into one buffer *because* DMA channels are scarce: an AIE2P
tile has two S2MM channels, so ``swiglu`` carries its gate and up weights in a
single ``[2, N]`` buffer. Refusing to read a row of that buffer forced an unpack
copy, which is the cost the packing existed to avoid.

What a region reads as is decided by its strides. A region that walks the buffer
the way the buffer is laid out is an ordinary operand at an offset; a reshape or
a transpose walks it differently, and an index into one of those is not an index
into the buffer.

What *rank* it reads as is numpy's rule: an integer subscript selects one
element and drops its axis, a slice keeps it. ``gu[1, :]`` is a row -- rank 1 --
so the axis the ``1`` took is never walked and never needs an induction
variable. The axis is still there in the DMA access pattern, where dropping it
would change every transfer in the tree; only arithmetic sees the shorter shape.
"""

from air import api as air
from air.api.types import bf16


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(body, out_shape=(1, 16)):
    """A herd holding one packed [2, 16] buffer, plus a destination."""
    OUT = air.tensor(list(out_shape), bf16)
    whole = tuple(slice(None) for _ in out_shape)

    with air.launch(name="regions") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    packed = air.alloc([2, 16], bf16, scope=h.private())
                    dst = air.alloc(list(out_shape), bf16, scope=h.private())
                    dst[:] = body(packed)
                    air.ops.store(dst, OUT[whole])

    print(launch.build(target="npu1"))


# CHECK-LABEL: TEST: two_rows_of_one_buffer
# The swiglu case. Two reads of one memref, not one read and a copy to separate
# the halves. Each row is addressed by the constant that named it: the integer
# subscript dropped that axis, so the nest walks only the row, and neither read
# needs index arithmetic to reach its own half.
# CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
# CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
# CHECK: %[[ZERO:.*]] = arith.constant 0 : index
# CHECK: %[[R0:.*]] = vector.transfer_read %[[BUF:.*]][%[[ZERO]], %[[J]]]
# CHECK: %[[ONE:.*]] = arith.constant 1 : index
# CHECK: %[[R1:.*]] = vector.transfer_read %[[BUF]][%[[ONE]], %[[J]]]
# CHECK: arith.mulf %[[R0]], %[[R1]]
# CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[I]], %[[J]]]
@run
def two_rows_of_one_buffer():
    build(lambda packed: packed[0, :] * packed[1, :])


# CHECK-LABEL: TEST: a_region_of_a_region
# Subscripting a region gives a region, and it reads like any other. This is
# worth pinning because such a region is flagged internally as a "view" -- the
# same flag reshape and transpose set -- and gating on that flag rather than on
# the strides rejected this outright.
#
# The two subscripts also compose the way numpy composes them: the outer one
# kept the axis and the inner one took element 1 of it, so the result is row 1
# of the buffer, read at the constant. Anchored inside the vector loop, because
# the loop bounds above it are also `arith.constant 1 : index`.
# CHECK: scf.for
# CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
# CHECK: %[[ONE:.*]] = arith.constant 1 : index
# CHECK: vector.transfer_read %{{.*}}[%[[ONE]], %[[J]]]
@run
def a_region_of_a_region():
    build(lambda packed: packed[0:2, :][1, :] * 2.0)


# CHECK-LABEL: TEST: one_region_named_twice_is_one_read
# A buffer mentioned twice in an expression is read once; a region has to behave
# the same way. Each subscript builds a fresh object, so the read cache is keyed
# on what identifies the read -- which buffer, from where, how much -- and not
# on the object. Exactly one transfer_read reaches the multiply and the add.
# CHECK: %[[R:.*]] = vector.transfer_read
# CHECK-NOT: vector.transfer_read
# CHECK: arith.mulf
# CHECK: arith.addf
@run
def one_region_named_twice_is_one_read():
    build(lambda packed: packed[1, :] * 2.0 + packed[1, :])
