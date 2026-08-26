# ./python/test/api/hierarchy.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Each hierarchy op owns its own iteration space, and the DSL writes it there.

``air.launch``, ``air.segment`` and ``air.herd`` all implement
``air_HierarchyInterface`` and each carries a ``Variadic<Index>:$sizes``. They
are symmetric in the dialect and they mean different things:

  * a **launch** point replays everything inside it -- a segment's L2 staging is
    refilled per point, which is where outer tiling belongs;
  * a **segment** point is a spatial copy of the segment body, which
    ``air-to-aie`` lays out across columns or devices (the dialect prints it as
    ``unroll(...)``; see ``programming_examples/segment_unroll``);
  * a **herd** point is a core.

This file exists because the DSL once broke that symmetry: a grid written on
``air.segment`` set the *launch's* sizes, and the segment's own sizes were
hard-wired empty. Nothing caught it. The emitted IR was correct for every case
in the tree -- the launch grid landed where it should -- so no behavioural test
could fail, and the one thing that would have exposed it, air.api emitting a
segment iteration space, was unreachable by construction.

The guard therefore has to assert *where the grid lands*, not what the kernel
computes.
"""

from air import api as air
from air.api.types import i32

M, N = 64, 64
TILE = 32


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: launch_grid_lands_on_launch
#
# A 2x2 grid written on air.launch appears as air.launch's own sizes...
# CHECK: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
#
# ...and the segment nested inside carries none of its own: no `unroll(...)`,
# no `in (...)`, just its operands. This is the line that would have failed on
# the conflated version, where the 2x2 was the segment's to give away.
# CHECK: air.segment @seg args
# CHECK-NOT: air.segment {{.*}}unroll
@run
def launch_grid_lands_on_launch():
    A = air.tensor([M, N], i32)
    B = air.tensor([M, N], i32)

    with air.launch(
        [range(0, M, TILE), range(0, N, TILE)], name="tiled", target="npu1"
    ) as launch:

        @launch.body
        def _(si, sj):
            row, col = si * TILE, sj * TILE
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    # The launch coordinates are usable here: air.segment is
                    # IsolatedFromAbove, so the DSL threads them in as operands
                    # rather than leaving a dangling reference.
                    l2 = air.alloc([TILE, TILE], i32, scope=seg.private())
                    air.ops.load(l2, A[row : row + TILE, col : col + TILE])
                    air.ops.store(l2, B[row : row + TILE, col : col + TILE])

    print(launch.mlir())


# CHECK-LABEL: TEST: herd_grid_lands_on_herd
# A grid written on air.herd is air.herd's sizes, and reaches neither of the
# other two: with no launch grid and no segment there is no air.launch at all,
# which is the shape this kernel's hand-written predecessor had.
# CHECK-NOT: air.launch
# CHECK: air.herd @h tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1{{.*}})
@run
def herd_grid_lands_on_herd():
    A = air.tensor([M, N], i32)
    B = air.tensor([M, N], i32)

    with air.launch(name="bare", target="npu1") as launch:

        @launch.body
        def _():
            with air.herd([range(4)], name="h", shape=(4,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([TILE, N], i32, scope=h.private())
                    air.ops.load(buf, A[0:TILE, 0:N])
                    air.ops.store(buf, B[0:TILE, 0:N])

    print(launch.mlir())


# CHECK-LABEL: TEST: gridless_launch_still_hosts_a_segment
# A launch with no grid says nothing, so it emits nothing -- until a segment
# needs somewhere to sit. air-insert-launch-around-herd only wraps a *bare*
# herd and skips one already inside a segment, so a segment with no launch
# above it compiles and silently computes zeros.
# CHECK: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c1{{.*}}, %{{.*}}=%c1{{.*}})
# CHECK: air.segment @seg
@run
def gridless_launch_still_hosts_a_segment():
    A = air.tensor([M, N], i32)
    B = air.tensor([M, N], i32)

    with air.launch(name="hosted", target="npu1") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2 = air.alloc([TILE, N], i32, scope=seg.private())
                    air.ops.load(l2, A[0:TILE, 0:N])
                    air.ops.store(l2, B[0:TILE, 0:N])

    print(launch.mlir())


# CHECK-LABEL: TEST: a_launch_coordinate_reaches_into_the_herd
# The same symmetry, one level further down. air.launch, air.segment and
# air.herd are each IsolatedFromAbove, so a coordinate does not simply fall
# through: it has to be passed as an operand at every boundary it crosses.
#
# The launch's coordinate was passed to the segment and stopped there, so a herd
# body that offset a transfer by it emitted an affine.apply on a value defined
# outside the region and the module failed verification. It went unnoticed
# because outer tiling normally stays at segment scope, where the L2 staging is,
# and only a kernel with no staging pushes it all the way in.
#
# The launch coordinate arrives as the herd's first argument, ahead of the
# tensors, and the offset is computed from it *inside* the herd.
# CHECK: air.launch (%[[LI:.*]], %{{.*}}) in (%{{.*}}=%c2{{.*}}
# CHECK: air.segment @seg args(%[[SI:[^=]*]]=%[[LI]]
# CHECK: air.herd @herd_0 tile (%[[TX:[^,]*]], %{{.*}}) in ({{.*}}) args(%[[HI:[^=]*]]=%[[SI]]
# CHECK: affine.apply {{.*}}[%[[HI]], %[[TX]]]
@run
def a_launch_coordinate_reaches_into_the_herd():
    A = air.tensor([M, N], i32)
    B = air.tensor([M, N], i32)

    with air.launch([range(0, M, TILE)], name="crossing", target="npu1") as launch:

        @launch.body
        def _(li):
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    base = li * TILE

                    with air.herd([range(2)], name="herd_0", shape=(2,)) as h:

                        @h.body
                        def _(tx):
                            # Both coordinates in one expression: this is the
                            # line that could not be written before.
                            row = base + tx * (TILE // 2)
                            buf = air.alloc(
                                [TILE // 2, N], i32, scope=h.private(), vector=0
                            )
                            air.ops.load(buf, A[row : row + TILE // 2, 0:N])
                            air.ops.store(buf, B[row : row + TILE // 2, 0:N])

    print(launch.mlir())
