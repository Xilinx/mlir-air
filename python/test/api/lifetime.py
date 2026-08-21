# ./python/test/api/lifetime.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""A buffer's life ends at its last use, and ``air.dealloc`` can say so.

``air.alloc`` has a counterpart. Most programs never need to write it: the
tracer sees every op that touches a buffer, so it knows where the buffer stops
being needed and places the ``memref.dealloc`` there. Programs that have
something to say about the point call ``air.dealloc``.

What the placement is *for* depends on the target, which is why it belongs in
the program and not in a backend heuristic. Lowered to AIE it is a scheduling
fact -- it tells the compiler the value is no longer needed, so the schedule
does not have to keep it available. Lowered through the ROCDL path it is a
free. Holding every buffer to the end of the body asserts the opposite of both,
and in ``channel_examples/worker_to_worker`` that assertion is enough to
serialise a ring of workers into a cycle and hang the herd.
"""

from air import api as air
from air.api.types import i32

N = 8


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def _kernel(release=None):
    """A -> a -> b -> B, where `release` may free `a` mid-body."""
    A = air.tensor([N, N], i32)
    B = air.tensor([N, N], i32)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([N, N], i32, scope=h.private())
                    b = air.alloc([N, N], i32, scope=h.private(), vector=0)
                    air.ops.load(a, A)
                    b[:] = a[:] + 1
                    if release:
                        air.dealloc(a)
                    air.ops.store(b, B)

    return launch


# CHECK-LABEL: TEST: inferred_at_last_use
# `a` is dead once the elementwise loop has read it, so it is freed there --
# before the store that reads `b`, not alongside it at the end of the body.
# `b` stays live across that store, because the store is its last use.
# CHECK: %[[A:.*]] = memref.alloc() : memref<8x8xi32, 2 : i32>
# CHECK: %[[B:.*]] = memref.alloc() : memref<8x8xi32, 2 : i32>
# CHECK: air.dma_memcpy_nd (%[[A]][] [] []
# CHECK: arith.addi
# CHECK: memref.dealloc %[[A]] : memref<8x8xi32, 2 : i32>
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[B]][] [] [])
# CHECK: memref.dealloc %[[B]] : memref<8x8xi32, 2 : i32>
@run
def inferred_at_last_use():
    print(_kernel().mlir())


# CHECK-LABEL: TEST: explicit_matches_inferred
# Writing the release out by hand at the point the tracer would have chosen
# produces the same module. That is the property that makes air.dealloc safe to
# add to an existing kernel: it pins the placement rather than changing it.
# CHECK: identical to inferred: True
# CHECK: memref.dealloc
# CHECK: air.dma_memcpy_nd
# CHECK: memref.dealloc
@run
def explicit_matches_inferred():
    inferred = _kernel().mlir()
    explicit = _kernel(release=True).mlir()
    print("identical to inferred:", str(inferred) == str(explicit))
    print(explicit)


# CHECK-LABEL: TEST: use_inside_a_loop_holds_it
# A use nested in a loop keeps the buffer live until the loop as a whole is
# done: the release lands after the scf.for, never inside it, where it would
# free the buffer on the first trip and leave the rest reading dead memory.
# CHECK: scf.for
# CHECK: memref.dealloc
# CHECK-NOT: scf.for
@run
def use_inside_a_loop_holds_it():
    A = air.tensor([N, N], i32)
    B = air.tensor([N, N], i32)

    with air.launch(name="looped") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([N], i32, scope=h.private())
                    b = air.alloc([N], i32, scope=h.private(), vector=0)
                    for i in air.sequential(0, N):
                        air.ops.load(a, A[i : i + 1, 0:N])
                        b[:] = a[:] + 1
                        air.ops.store(b, B[i : i + 1, 0:N])

    print(launch.mlir())
