# ./python/test/api/broadcast_view.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""``broadcast(n)`` walks a region n times: a new outermost axis of stride 0.

The case it exists for is a source that a consumer reads once per round: a
GEMM's activation, re-sent for every block of output columns. Without it the
rounds have to be a launch repeat, one shim transfer per round; with it the
whole dispatch is one transfer whose pattern repeats the region in place.
"""

from air import api as air
from air.api.types import i32


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(src):
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

                    back.get(B)

    print(launch.mlir())


# CHECK-LABEL: TEST: region_broadcast
# The region's own pattern, behind a size-3 axis of stride 0.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 0, 8] [3, 16, 8] [0, 64, 1]) : (memref<32x64xi32>)
@run
def region_broadcast():
    build(lambda A: A[0:16, 8:16].broadcast(3))


# CHECK-LABEL: TEST: view_broadcast
# It composes with the other views: here a flattened row block.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 128] [3, 128] [0, 1]) : (memref<32x64xi32>)
@run
def view_broadcast():
    build(lambda A: A[2:4, 0:64].reshape(128).broadcast(3))


# CHECK-LABEL: TEST: tensor_broadcast
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 0, 0] [2, 32, 64] [0, 64, 1]) : (memref<32x64xi32>)
@run
def tensor_broadcast():
    build(lambda A: A.broadcast(2))


# CHECK-LABEL: TEST: count_must_be_positive
# CHECK: broadcast(n) takes a positive integer count, got 0
@run
def count_must_be_positive():
    A = air.tensor([32, 64], i32)
    try:
        A[0:16, 8:16].broadcast(0)
    except ValueError as e:
        print(e)
