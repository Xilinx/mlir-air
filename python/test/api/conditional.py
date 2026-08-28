# ./python/test/api/conditional.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""``ops.branch`` opens an ``scf.if``: one side runs, and it holds statements.

The DSL's other conditional, ``ops.select``, is the branchless one -- per
element, both sides evaluated, arms are values. This one decides per core, runs
one side, and its arms can be effects: a channel put, a DMA. ``errors.py`` pins
what each says when handed the other's condition.

A herd body is traced once, for the whole herd, so a Python ``if`` on a
coordinate cannot mean "this core only": the coordinate is an SSA value with no
value at trace time. ``tx == 0`` therefore builds a ``Condition`` rather than a
bool, and ``ops.branch`` turns it into a region. ``errors.py`` pins what happens
when someone writes the Python ``if`` anyway.

Three properties this file exists to hold down:

* the comparison is emitted where the region opens, not where it was written;
* ``otherwise()`` is a second region on the *same* ``scf.if``, so nesting one
  inside the other's else is how a three-way split is spelled -- which is what
  ``programming_examples/cascade_reduction`` needs for the head, middle and
  tail of a cascade; and
* a branch with no ``otherwise`` still emits the else region, holding only its
  terminator. That is not a difference in the compiled design: MLIR's
  ``RemoveEmptyElseBranch`` deletes it, and ``canonicalize`` runs ahead of
  everything in the AIR pipeline that reads the region structure. The last test
  below shows both the emitted form and the canonicalized one.
"""

from air import api as air
from air.api.types import i32

N = 64


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: branch_otherwise
#
# The condition materialises inside the herd body, immediately ahead of the
# region, even though `tx == 0` was written a few statements earlier.
# CHECK: air.herd @h
# CHECK: %[[TX:.*]] = affine.apply
# CHECK: %[[C0:.*]] = arith.constant 0 : index
# CHECK: %[[P:.*]] = arith.cmpi eq, %[[TX]], %[[C0]] : index
# CHECK: scf.if %[[P]] {
# CHECK:   linalg.fill ins(%c1_i32
# CHECK: } else {
# CHECK:   linalg.fill ins(%c2_i32
# CHECK: }
@run
def branch_otherwise():
    A = air.tensor([N], i32)
    out = air.tensor([N], i32)

    with air.launch(name="two_way") as launch:

        @launch.body
        def _():
            with air.herd([range(4)], name="h", shape=(4,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([N], i32, scope=h.private())
                    first = tx == 0  # built here, emitted below
                    air.ops.load(buf, A[0:N])

                    with air.ops.branch(first) as branch:
                        air.ops.fill(buf, 1)
                    with branch.otherwise():
                        air.ops.fill(buf, 2)

                    air.ops.store(buf, out[0:N])

    print(launch.mlir())


# CHECK-LABEL: TEST: three_way_cascade
#
# The head/middle/tail split of a cascade: the second branch sits inside the
# first's else, so the two scf.ifs nest rather than sitting side by side. This
# is the structure cascade_reduction relies on.
# CHECK: scf.if
# CHECK:   linalg.fill ins(%c1_i32
# CHECK: } else {
# CHECK:   arith.cmpi eq, %{{.*}}, %c3
# CHECK:   scf.if
# CHECK:     linalg.fill ins(%c3_i32
# CHECK:   } else {
# CHECK:     linalg.fill ins(%c2_i32
@run
def three_way_cascade():
    A = air.tensor([N], i32)
    out = air.tensor([N], i32)

    with air.launch(name="three_way") as launch:

        @launch.body
        def _():
            with air.herd([range(4)], name="h", shape=(4,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([N], i32, scope=h.private())
                    air.ops.load(buf, A[0:N])

                    with air.ops.branch(tx == 0) as head:
                        air.ops.fill(buf, 1)
                    with head.otherwise():
                        with air.ops.branch(tx == 3) as tail:
                            air.ops.fill(buf, 3)
                        with tail.otherwise():
                            air.ops.fill(buf, 2)

                    air.ops.store(buf, out[0:N])

    print(launch.mlir())


# CHECK-LABEL: TEST: every_comparison
#
# All six operators, against a constant and against another coordinate. Index
# comparison is signed: herd coordinates are non-negative, and slt/sle/sgt/sge
# are what an offset computed as a difference needs.
# CHECK: arith.cmpi eq, %{{.*}}, %c0
# CHECK: arith.cmpi ne, %{{.*}}, %c1
# CHECK: arith.cmpi slt, %{{.*}}, %c2
# CHECK: arith.cmpi sle, %{{.*}}, %c3
# CHECK: arith.cmpi sgt, %{{.*}}, %c0
# CHECK: arith.cmpi sge, %{{.*}}, %c1
#
# A reflected comparison against a Python int on the left is the same op with
# the operands the way they were written: `1 < tx` is `tx > 1`.
# CHECK: arith.cmpi sgt, %{{.*}}, %c1
#
# And a comparison between two coordinates needs no constant at all.
# CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : index
@run
def every_comparison():
    A = air.tensor([N], i32)
    out = air.tensor([N], i32)

    with air.launch(name="cmps") as launch:

        @launch.body
        def _():
            with air.herd([range(4), range(4)], name="h", shape=(4, 4)) as h:

                @h.body
                def _(tx, ty):
                    buf = air.alloc([N], i32, scope=h.private())
                    air.ops.load(buf, A[0:N])
                    for condition in (
                        tx == 0,
                        tx != 1,
                        tx < 2,
                        tx <= 3,
                        tx > 0,
                        tx >= 1,
                        1 < tx,
                        tx == ty,
                    ):
                        with air.ops.branch(condition):
                            air.ops.fill(buf, 1)
                    air.ops.store(buf, out[0:N])

    print(launch.mlir())


# CHECK-LABEL: TEST: no_otherwise_canonicalizes_away
#
# As emitted: the else region is there, empty but for its terminator.
# CHECK: scf.if %{{.*}} {
# CHECK:   linalg.fill
# CHECK: } else {
# CHECK: }
#
# After canonicalize, which the AIR pipeline runs first, it is gone -- so the
# region a branch with no `otherwise` produces is the one it looks like.
# CHECK-LABEL: canonicalized:
# CHECK: scf.if %{{.*}} {
# CHECK-NEXT: linalg.fill
# CHECK-NEXT: }
# CHECK-NEXT: air.dma_memcpy_nd
@run
def no_otherwise_canonicalizes_away():
    A = air.tensor([N], i32)
    out = air.tensor([N], i32)

    with air.launch(name="one_way") as launch:

        @launch.body
        def _():
            with air.herd([range(4)], name="h", shape=(4,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([N], i32, scope=h.private())
                    air.ops.load(buf, A[0:N])
                    with air.ops.branch(tx == 0):
                        air.ops.fill(buf, 1)
                    air.ops.store(buf, out[0:N])

    module = launch.build(target="npu1")
    print(module)

    from air.passmanager import PassManager

    with module.context:
        PassManager.parse("builtin.module(canonicalize)").run(module.operation)
    print("canonicalized:")
    print(module)


# CHECK-LABEL: TEST: a_branch_can_test_a_value_read_from_a_buffer
# Rank says which conditional was meant. ctr[0] is a single value, so comparing
# it yields one bool and can open a region; ctr[:] is one bool per element and
# is ops.select's, which the refusal below still says.
#
# This is legal for the same reason the index form is -- it decides once per
# core, not once per element -- and it is not ops.select, because what it
# guards is a func.call and a select evaluates both of its arms.
# flash_attention/kernel_fusion_based's --causal-skip is the case: a core keeps
# its q-block index in an L1 counter tile, and blocks the causal mask kills
# outright have their matmul and softmax skipped rather than computed and
# thrown away.
# CHECK: memref.load
# CHECK: arith.cmpi sge
# CHECK: scf.if
# CHECK: func.call @block_kernel
@run
def a_branch_can_test_a_value_read_from_a_buffer():
    from air.api.types import bf16, i32

    kernel = air.extern("block_kernel", link_with="attn.o")
    A = air.tensor([64, 64], bf16)
    OUT = air.tensor([64, 64], bf16)

    with air.launch(name="vcond") as launch:

        @launch.body
        def _():
            with air.herd([range(2), range(2)], shape=(2, 2), link_with="attn.o") as h:

                @h.body
                def _(tx, ty):
                    g = air.alloc([64, 64], bf16, scope=h.private())
                    ctr = air.alloc([4], i32, scope=h.private())
                    air.ops.load(g, A)
                    with air.ops.branch(ctr[0] + tx >= ty):
                        kernel(g)
                    air.ops.store(g, OUT)

    print(launch.mlir())


# CHECK-LABEL: TEST: a_wider_data_comparison_is_still_ops_select
# One bool per element cannot open one region, and the diagnostic names the op
# that can.
# CHECK: TypeError: {{.*}}ops.select
@run
def a_wider_data_comparison_is_still_ops_select():
    from air.api.types import bf16, i32

    A = air.tensor([64], bf16)
    OUT = air.tensor([64], bf16)

    with air.launch(name="vcond_wide") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], bf16, scope=h.private())
                    ctr = air.alloc([4], i32, scope=h.private())
                    air.ops.load(a, A)
                    try:
                        air.ops.branch(air.ops.equal(ctr[0:2], 0))
                        print("NOT REFUSED")
                    except TypeError as e:
                        print("TypeError:", e)
                    air.ops.store(a, OUT)

    launch.mlir()
