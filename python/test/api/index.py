# ./python/test/api/index.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Index arithmetic on a tile coordinate reaches the IR as one affine.apply.

``IndexExpr`` keeps an expression as an affine *linear* form -- coefficients on
leaves, plus a constant -- because that is what makes a slice like
``A[row : row + tm]`` recover its static size by cancellation. ``//`` and ``%``
are affine but not linear, so they cannot be a coefficient on a leaf; they
become a ``DerivedLeaf``, which the linear form carries like a coordinate and
expands only at materialisation.

Two consequences are pinned below. The whole expression still lowers to a
single ``affine.apply`` -- the AIR dependency analysis and the DMA
specialisation passes read affine offsets directly and lose track of them when
the arithmetic is scattered across ``arith`` ops. And a ``DerivedLeaf`` is
compared *structurally*, not by identity, so two separately built copies of
``(tw + 1) % W`` are one term and cancel.

``channel_examples/worker_to_worker`` is the example this exists for: it builds
exactly the two maps below by hand, with ``AffineExpr.get_mod`` and
``get_floor_div``.
"""

from air import api as air
from air.api.types import i32

H, W = 2, 3
TH, TW = 2, 4


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: ring_neighbour
#
# The worker's successor on a row-major ring:
#     tw_next = (tw + 1) % W
#     th_next = (th + (tw + 1) // W) % H
# Each is one map over the herd coordinates -- the floordiv nests *inside* the
# mod rather than being materialised as its own affine.apply, because the
# DerivedLeaf expands in place.
# CHECK-DAG: #[[NEXTW:.*]] = affine_map<()[s0] -> ((s0 + 1) mod 3)>
# CHECK-DAG: #[[NEXTH:.*]] = affine_map<()[s0, s1] -> ((s0 + (s1 + 1) floordiv 3) mod 2)>
#
# Both coordinates feed one map, sharing a symbol each: a DerivedLeaf takes no
# symbol of its own.
# CHECK: air.herd @ring
# CHECK-DAG: %[[TWN:.*]] = affine.apply #[[NEXTW]]()[%[[TW:.*]]]
# CHECK-DAG: %[[THN:.*]] = affine.apply #[[NEXTH]]()[%{{.*}}, %[[TW]]]
# CHECK: air.channel.put @Ring[%{{.*}}, %{{.*}}]
@run
def ring_neighbour():
    A = air.tensor([H * TH, W * TW], i32)
    ring = air.channel("Ring", size=[H, W])
    back = air.channel("Back", size=[H, W])

    with air.launch(name="ring_launch") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(H), range(W)], name="ring", shape=(H, W)) as h:

                        @h.body
                        def _(th, tw):
                            buf = air.alloc([TH, TW], i32, scope=h.private())
                            ring.put(
                                buf, indices=[(th + (tw + 1) // W) % H, (tw + 1) % W]
                            )
                            ring.get(buf, indices=[th, tw])
                            back.put(buf, indices=[th, tw])

                    for i in range(H):
                        for j in range(W):
                            back.get(
                                A[i * TH : (i + 1) * TH, j * TW : (j + 1) * TW],
                                indices=[i, j],
                            )

    print(launch.mlir())


# CHECK-LABEL: TEST: structural_cancellation
# A DerivedLeaf is keyed on the expression it wraps, not on object identity, so
# the two independently built copies of (tw + 1) % W below are one term and
# subtract to zero. The offset is then a plain Python int and stays in the
# static half of the access pattern -- which is the whole reason the linear
# form exists, and it has to keep working with a non-linear term present.
# CHECK: air.herd @cancel
# CHECK-NOT: mod
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, %{{.*}}] [2, 4] [12, 1])
@run
def structural_cancellation():
    A = air.tensor([H * TH, W * TW], i32)
    B = air.tensor([H * TH, W * TW], i32)

    with air.launch(name="cancel_launch") as launch:

        @launch.body
        def _():
            with air.herd([range(W)], name="cancel", shape=(W,)) as h:

                @h.body
                def _(tw):
                    buf = air.alloc([TH, TW], i32, scope=h.private())
                    # Built twice, deliberately, rather than computed once and
                    # subtracted from itself: two calls produce two distinct
                    # DerivedLeaf objects, so they only collapse into one term
                    # if the key is the expression. Reusing a single value would
                    # cancel under object identity too and prove nothing.
                    row = ((tw + 1) % W) - ((tw + 1) % W)
                    col = tw * TW
                    air.ops.load(buf, A[row : row + TH, col : col + TW])
                    air.ops.store(buf, B[row : row + TH, col : col + TW])

    print(launch.mlir())


# CHECK-LABEL: TEST: trivial_divisor
# Dividing by 1 is folded rather than emitted: `x // 1` is x, and `x % 1` is
# the constant 0. So the row offset below is static and the column is the bare
# coordinate, with neither a floordiv nor a mod reaching the IR.
# CHECK: air.herd @trivial
# CHECK-NOT: floordiv
# CHECK-NOT: mod
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, %{{.*}}] [2, 4] [12, 1])
@run
def trivial_divisor():
    A = air.tensor([H * TH, W * TW], i32)
    B = air.tensor([H * TH, W * TW], i32)

    with air.launch(name="trivial_launch") as launch:

        @launch.body
        def _():
            with air.herd([range(W)], name="trivial", shape=(W,)) as h:

                @h.body
                def _(tw):
                    buf = air.alloc([TH, TW], i32, scope=h.private())
                    row = tw % 1
                    col = (tw // 1) * TW
                    air.ops.load(buf, A[row : row + TH, col : col + TW])
                    air.ops.store(buf, B[row : row + TH, col : col + TW])

    print(launch.mlir())
