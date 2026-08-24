# ./python/test/api/cmp_select.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers comparisons and select.

Named cmp_select.py, not select.py: lit puts the test's own directory on
sys.path, and `select` is a stdlib module. On a Python built with select as a
shared extension rather than statically -- which is what GitHub's hosted
runners ship -- a file named select.py here wins the import, so `subprocess`
(reached via numpy -> platform) picks up this file instead. It then imports
air.api.types while air.api.types is still importing numpy, and every test in
this directory dies on a circular import. Do not name a test after a stdlib
module.

A comparison is the first expression node whose result type is *not* the element
type -- it yields i1, or vector<Wxi1> when vectorised. That is why it gets its
own node kind and why nothing but `ops.select` can consume one: every other node
in the tree feeds an arith builder that infers its result type from operand 0.

Float comparisons use the ordered predicates (OGE, not UGE), which is what the
hand-written vector_select kernel named and what C's `>=` means: false when
either operand is NaN.
"""

from air import api as air
from air.api.types import bf16, f32, i32


def build(dtype, body, N=65536, tile=1024, vector=16, herd_shape=(2,)):
    a = air.tensor([N], dtype)
    b = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="sel") as launch:

        @launch.body
        def _():
            with air.herd([range(0, N, tile)], shape=herd_shape) as h:

                @h.body
                def _(tx):
                    col = tx * tile
                    a_buf = air.alloc([tile], dtype, scope=h.private(), vector=vector)
                    b_buf = air.alloc([tile], dtype, scope=h.private(), vector=vector)
                    o_buf = air.alloc([tile], dtype, scope=h.private(), vector=vector)
                    air.ops.load(a_buf, a[col : col + tile])
                    air.ops.load(b_buf, b[col : col + tile])
                    o_buf[:] = body(a_buf, b_buf)
                    air.ops.store(o_buf, out[col : col + tile])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: select_ge_float
# The vector_select kernel's expression: cmpf OGE feeding a select, both on
# vector<16xf32>, with the predicate result typed vector<16xi1>.
# CHECK: vector.transfer_read {{.*}} vector<16xf32>
# CHECK: vector.transfer_read {{.*}} vector<16xf32>
# CHECK: arith.cmpf oge, {{.*}} : vector<16xf32>
# CHECK: arith.select {{.*}} vector<16xi1>, vector<16xf32>
# CHECK: vector.transfer_write {{.*}} vector<16xf32>
@run
def select_ge_float():
    print(build(f32, lambda a, b: air.ops.select(a[:] >= b[:], a[:], b[:])).mlir())


# CHECK-LABEL: TEST: ordering_operators_are_ordered_predicates
# All four ordering operators, and each maps to the *ordered* float predicate.
# CHECK-DAG: arith.cmpf olt
# CHECK-DAG: arith.cmpf ole
# CHECK-DAG: arith.cmpf ogt
# CHECK-DAG: arith.cmpf oge
@run
def ordering_operators_are_ordered_predicates():
    def body(a, b):
        return (
            air.ops.select(a[:] < b[:], a[:], b[:])
            + air.ops.select(a[:] <= b[:], a[:], b[:])
            + air.ops.select(a[:] > b[:], a[:], b[:])
            + air.ops.select(a[:] >= b[:], a[:], b[:])
        )

    print(build(f32, body).mlir())


# CHECK-LABEL: TEST: equal_and_not_equal_are_functions
# `==` on a buffer expression is Python identity, so equality is spelled as a
# function. Both map to the ordered float predicates.
# CHECK-DAG: arith.cmpf oeq
# CHECK-DAG: arith.cmpf one
@run
def equal_and_not_equal_are_functions():
    def body(a, b):
        return air.ops.select(air.ops.equal(a[:], b[:]), a[:], b[:]) + air.ops.select(
            air.ops.not_equal(a[:], b[:]), a[:], b[:]
        )

    print(build(f32, body).mlir())


# CHECK-LABEL: TEST: select_integer_uses_signed_cmpi
# Integer buffers take arith.cmpi with the *signed* predicates -- an unsigned
# dtype never reaches the emitter's arith path at all (see unsigned.py).
# CHECK: arith.cmpi sgt, {{.*}} : vector<16xi32>
# CHECK: arith.select {{.*}} vector<16xi1>, vector<16xi32>
@run
def select_integer_uses_signed_cmpi():
    print(build(i32, lambda a, b: air.ops.select(a[:] > b[:], a[:], b[:])).mlir())


# CHECK-LABEL: TEST: select_against_a_scalar_broadcasts
# A scalar arm broadcasts to the vector width, same as any other scalar in an
# elementwise tree -- this is relu spelled the long way round.
# CHECK: arith.constant 0.0{{.*}} : f32
# CHECK: vector.broadcast {{.*}} : f32 to vector<16xf32>
# CHECK: arith.cmpf ogt, {{.*}} : vector<16xf32>
# CHECK: arith.select
@run
def select_against_a_scalar_broadcasts():
    print(build(f32, lambda a, b: air.ops.select(a[:] > 0.0, a[:], 0.0)).mlir())


# CHECK-LABEL: TEST: select_composes_inside_a_larger_tree
# A select is an ordinary value node: it can be an operand of the arithmetic
# operators, and the whole tree still lowers as one loop with one transfer_write.
# CHECK: arith.cmpf oge
# CHECK: arith.select
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write
# CHECK-NOT: vector.transfer_write
@run
def select_composes_inside_a_larger_tree():
    print(
        build(
            bf16,
            lambda a, b: 2.0 * air.ops.select(a[:] >= b[:], a[:], b[:]) + b[:],
        ).mlir()
    )


# CHECK-LABEL: TEST: scalar_fallback_keeps_the_predicate
# A tile that is not a multiple of the vector width falls back to a scalar
# memref.load/store loop. The comparison survives it as a scalar i1 -- there is
# no vector type anywhere -- which is what makes select usable on the narrow
# tiles the vector path cannot take.
# CHECK-NOT: vector.transfer_read
# CHECK: arith.cmpf oge, {{.*}} : f32
# CHECK: arith.select
# CHECK: memref.store
@run
def scalar_fallback_keeps_the_predicate():
    print(
        build(
            f32,
            lambda a, b: air.ops.select(a[:] >= b[:], a[:], b[:]),
            N=192,
            tile=24,
            vector=0,
            herd_shape=(2,),
        ).mlir()
    )
