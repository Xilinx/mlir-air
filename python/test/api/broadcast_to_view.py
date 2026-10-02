# ./python/test/api/broadcast_to_view.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""``broadcast_to`` reads a region at a larger shape, numpy's rule: shapes
align on the right, new leading axes and axes of extent 1 repeat with stride 0,
every other axis must match. Like numpy's, the view is read-only.

The case it exists for is a source a consumer reads once per round: a GEMM's
activation, re-sent for every block of output columns. Without it the rounds
have to be a launch repeat, one shim transfer per round; with it the dispatch
is one transfer whose pattern repeats the region in place.
"""

from air import api as air
from air.api.types import i32


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(src, dst=None):
    A = air.tensor([32, 64], i32)
    B = air.tensor([16, 8], i32)
    tiles = air.channel("Tiles")
    back = air.channel("Back")

    with air.launch(name="rounds") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    tiles.put(src(A))
                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([16, 8], i32, scope=h.private())
                            for _r in air.sequential(0, 3):
                                tiles.get(buf)
                                back.put(buf)

                    back.get(dst(B) if dst else B)

    print(launch.mlir())


# CHECK-LABEL: TEST: new_leading_axis
# The region's own pattern, behind a size-3 axis of stride 0.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 0, 8] [3, 16, 8] [0, 64, 1]) : (memref<32x64xi32>)
@run
def new_leading_axis():
    build(lambda A: A[0:16, 8:16].broadcast_to(3, 16, 8))


# CHECK-LABEL: TEST: extent_one_axis
# An axis of extent 1 repeats in place: row 3 read 16 times. The repeated axis
# has stride 0, so its offset (3 rows of 64) moves to the column axis.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 200] [16, 8] [0, 1]) : (memref<32x64xi32>)
@run
def extent_one_axis():
    build(lambda A: A[3:4, 8:16].broadcast_to(16, 8))


# CHECK-LABEL: TEST: extent_one_inner_axis
# The same for an inner axis: the second row of each row pair, repeated 3
# times. Its offset (1 row of 64) lands on the column axis: 64 + 8.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 0, 0, 72] [2, 4, 3, 8] [0, 128, 0, 1]) : (memref<32x64xi32>)
@run
def extent_one_inner_axis():
    build(
        lambda A: A[0:8, 8:16].reshape(4, 2, 8)[0:4, 1:2, 0:8].broadcast_to(2, 4, 3, 8)
    )


# CHECK-LABEL: TEST: runtime_offset
# A row picked by a loop index: the offset is r * 64 + 8 on the column axis.
# CHECK: affine_map<()[s0] -> (s0 * 64 + 8)>
# CHECK: scf.for %[[R:.*]] =
# CHECK: %[[O:.*]] = affine.apply #{{.*}}()[%[[R]]]
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, %[[O]]] [16, 8] [0, 1]) : (memref<32x64xi32>)
@run
def runtime_offset():
    A = air.tensor([32, 64], i32)
    tiles = air.channel("Tiles")
    with air.launch(name="rows") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    for r in air.sequential(0, 4):
                        tiles.put(A[r : r + 1, 8:16].broadcast_to(16, 8))
                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([16, 8], i32, scope=h.private())
                            for _r in air.sequential(0, 4):
                                tiles.get(buf)

    print(launch.mlir())


# CHECK-LABEL: TEST: after_a_view
# It composes with the other views: here a flattened row block, as a tuple.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 128] [3, 128] [0, 1]) : (memref<32x64xi32>)
@run
def after_a_view():
    build(lambda A: A[2:4, 0:64].reshape(128).broadcast_to((3, 128)))


# CHECK-LABEL: TEST: whole_tensor
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 0, 0] [2, 32, 64] [0, 64, 1]) : (memref<32x64xi32>)
@run
def whole_tensor():
    build(lambda A: A.broadcast_to(2, 32, 64))


def refused(f):
    try:
        f()
    except ValueError as e:
        print(e)


# CHECK-LABEL: TEST: mismatched_axis
# CHECK: cannot broadcast region (16, 8) to (3, 16, 4): axis 1 has extent 8, not 1 or 4
@run
def mismatched_axis():
    refused(lambda: air.tensor([32, 64], i32)[0:16, 8:16].broadcast_to(3, 16, 4))


# CHECK-LABEL: TEST: offset_with_no_carrier
# One element of a 1-D tensor repeated: no axis keeps a stride, so its offset
# has nowhere to go.
# CHECK: broadcast_to(4,): the repeated axis is at an offset, and no axis with a stride is left to carry it
@run
def offset_with_no_carrier():
    refused(lambda: air.tensor([2048], i32)[5:6].broadcast_to(4))


# CHECK-LABEL: TEST: not_a_destination
# CHECK: channel.get: a broadcast_to view is read-only
@run
def not_a_destination():
    refused(
        lambda: build(
            lambda A: A[0:16, 8:16], lambda B: B[0:1, 0:8].broadcast_to(16, 8)
        )
    )


# CHECK-LABEL: TEST: not_a_load_destination
# The same rule for a buffer region filled by ops.load.
# CHECK: load: a broadcast_to view is read-only
@run
def not_a_load_destination():
    IN = air.tensor([16, 8], i32)

    def go():
        with air.launch(name="l") as launch:

            @launch.body
            def _():
                with air.herd([range(1)], name="h", shape=(1,)) as h:

                    @h.body
                    def _(tx):
                        b = air.alloc([1, 8], i32, scope=h.private())
                        air.ops.load(b.broadcast_to(16, 8), IN[0:16, 0:8])

        launch.mlir()

    refused(go)
