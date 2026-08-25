# ./python/test/api/broadcast.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api stretches an operand axis of extent 1 to the destination's.

numpy's rule, right-aligned and one-sided: the destination already exists, so
each operand broadcasts *to* it rather than the two being broadcast against
each other. That is numpy's own rule for an explicit ``out=``.

Which code comes out depends on *which* axis is stretched, and that is the
substance of this file:

* an outer axis stretched leaves the innermost one intact, so the read is the
  ordinary ``vector.transfer_read`` -- just at the operand's own rank, pinned at
  0 on the stretched axes;
* the *innermost* axis stretched has no contiguous run to read at all, so the
  emitter loads the one element and splats it with ``vector.broadcast``.

The second is what ``vector_broadcast_scalar`` needs and the first is what
``mnist_fc/broadcast_bias_add`` needs; both were hand-written before, out of
subviews and collapse_shapes, and both are two lines here.
"""

from air import api as air
from air.api.types import bf16


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(shapes, body, dtype=bf16, vector=None):
    """A herd whose body allocates one buffer per entry in ``shapes``.

    ``shapes[0]`` is the destination; the rest are operands, and ``body`` is
    handed them in order. Nothing is DMAed *in* -- the point is the compute
    loop, and an unread buffer still gets an alloc for it to index -- but the
    destination is stored out, because a kernel that writes no tensor is
    refused before it reaches the emitter.
    """
    out_shape = shapes[0]
    OUT = air.tensor(out_shape, dtype)
    whole = tuple(slice(None) for _ in out_shape)

    with air.launch(name="bc") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    bufs = [
                        air.alloc(
                            s,
                            dtype,
                            scope=h.private(),
                            **({} if vector is None else {"vector": vector}),
                        )
                        for s in shapes
                    ]
                    dst, operands = bufs[0], bufs[1:]
                    dst[:] = body(*operands)
                    air.ops.store(dst, OUT[whole])

    print(launch.build(target="npu1"))


# CHECK-LABEL: TEST: innermost_axis_is_a_scalar_load_and_a_splat
# [8, 1] into [8, 16]: the stretched axis is the contiguous one, so there is no
# run of elements to read. One memref.load at column 0 -- note the pinned %c0 --
# and one vector.broadcast, which is exactly what the hand-written
# vector_broadcast_scalar kernel spells.
# CHECK: %[[Z:.*]] = arith.constant 0 : index
# CHECK: scf.for %[[I:.*]] =
# CHECK: scf.for
# CHECK: %[[S:.*]] = memref.load %{{.*}}[%[[I]], %[[Z]]] : memref<8x1xbf16, 2 : i32>
# CHECK: %[[V:.*]] = vector.broadcast %[[S]] : bf16 to vector<16xbf16>
# CHECK: vector.transfer_write %[[V]]
# CHECK-NOT: memref.subview
# CHECK-NOT: memref.collapse_shape
@run
def innermost_axis_is_a_scalar_load_and_a_splat():
    build([[8, 16], [8, 1]], lambda a: a[:], vector=16)


# CHECK-LABEL: TEST: a_missing_leading_axis_reads_at_its_own_rank
# [16] added to [8, 16]: the innermost extent already matches, so the operand is
# read with an ordinary transfer_read -- but from a rank-1 memref, indexed by
# the *inner* induction variable alone. That is the bias vector in
# broadcast_bias_add, which the predecessor reached via a subview.
# CHECK: scf.for %[[I:.*]] = %c0{{.*}} to %c8
# CHECK: scf.for %[[J:.*]] = %c0{{.*}} to %c16
# CHECK-DAG: vector.transfer_read %{{[a-z_0-9]+}}[%[[I]], %[[J]]]{{.*}}memref<8x16xbf16, 2 : i32>, vector<16xbf16>
# CHECK-DAG: vector.transfer_read %{{[a-z_0-9]+}}[%[[J]]]{{.*}}memref<16xbf16, 2 : i32>, vector<16xbf16>
# CHECK: arith.addf
# CHECK-NOT: memref.subview
@run
def a_missing_leading_axis_reads_at_its_own_rank():
    build([[8, 16], [8, 16], [16]], lambda a, bias: a[:] + bias[:], vector=16)


# CHECK-LABEL: TEST: an_explicit_leading_one_is_the_same_as_omitting_it
# [1, 16] and [16] broadcast into [8, 16] identically -- the first axis is
# pinned at 0 rather than dropped, so the read is rank-2, but it is still one
# transfer_read of the whole row and no subview.
# CHECK: %[[Z:.*]] = arith.constant 0 : index
# CHECK: scf.for %{{.*}} = %c0{{.*}} to %c8
# CHECK: scf.for %[[J:.*]] = %c0{{.*}} to %c16
# CHECK: vector.transfer_read %{{[a-z_0-9]+}}[%[[Z]], %[[J]]]{{.*}}memref<1x16xbf16, 2 : i32>, vector<16xbf16>
@run
def an_explicit_leading_one_is_the_same_as_omitting_it():
    build([[8, 16], [1, 16]], lambda a: a[:], vector=16)


# CHECK-LABEL: TEST: a_middle_axis_can_be_stretched_too
# [4, 1, 16] into [4, 8, 16]: nothing special about the ends. The middle index
# is the pinned 0 and the other two are the destination's own variables.
# CHECK: %[[Z:.*]] = arith.constant 0 : index
# CHECK: scf.for %[[I:.*]] = %c0{{.*}} to %c4
# CHECK: scf.for %{{.*}} = %c0{{.*}} to %c8
# CHECK: scf.for %[[K:.*]] = %c0{{.*}} to %c16
# CHECK: vector.transfer_read %{{[a-z_0-9]+}}[%[[I]], %[[Z]], %[[K]]]{{.*}}memref<4x1x16xbf16, 2 : i32>
@run
def a_middle_axis_can_be_stretched_too():
    build([[4, 8, 16], [4, 1, 16]], lambda a: a[:], vector=16)


# CHECK-LABEL: TEST: the_scalar_path_broadcasts_the_same_way
# vector=0 sends the emitter down the memref.load/store path, where a broadcast
# is nothing but which index goes in which position. Both operands are loaded
# scalar; the stretched one at the pinned 0.
# CHECK: %[[Z:.*]] = arith.constant 0 : index
# CHECK: scf.for %[[I:.*]] = %c0{{.*}} to %c8
# CHECK: scf.for %[[J:.*]] = %c0{{.*}} to %c4
# CHECK-DAG: memref.load %{{.*}}[%[[I]], %[[J]]] : memref<8x4xbf16, 2 : i32>
# CHECK-DAG: memref.load %{{.*}}[%[[I]], %[[Z]]] : memref<8x1xbf16, 2 : i32>
# CHECK: arith.mulf
# CHECK: memref.store
@run
def the_scalar_path_broadcasts_the_same_way():
    build([[8, 4], [8, 4], [8, 1]], lambda a, s: a[:] * s[:], vector=0)


# CHECK-LABEL: TEST: no_broadcast_emits_no_pinned_index
# The regression pin for the whole feature: when every operand already has the
# destination's shape, nothing is stretched and the emitter must produce exactly
# the IR it produced before broadcasting existed -- in particular no leftover
# index constant hoisted above the nest. The only constants here are the loop
# bounds and the transfer_read padding value.
# CHECK: memref.alloc
# CHECK: memref.alloc
# CHECK: %[[PAD:.*]] = arith.constant 0.0{{.*}} : bf16
# CHECK-NEXT: %[[LB:.*]] = arith.constant 0 : index
# CHECK-NEXT: arith.constant 8 : index
# CHECK-NEXT: arith.constant 1 : index
# CHECK-NEXT: scf.for
@run
def no_broadcast_emits_no_pinned_index():
    build([[8, 16], [8, 16]], lambda a: a[:], vector=16)
