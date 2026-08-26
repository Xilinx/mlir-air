# ./python/test/api/cast.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers air.ops.cast to the arith conversion ops.

A cast is the first node whose operand has a different element type from its
result, so it is the first thing that makes one assignment hold two element
types at once. Everything below it is read and computed in the source type;
only the conversion lands in the destination's. The two orderings below --
``cast(a[:] * 2.0, i32)`` against ``cast(a[:], i32) * 2`` -- are the whole
point of that rule, and they must emit different arithmetic.

The pairs checked here are the ones that were run on npu1 hardware against an
exact numpy reference: every conversion this file emits is one that computes.
"""

from air import api as air
from air.api.types import bf16, f32, i16, i32


def build(body, dt_in, dt_out, N=256, tile=128, vector=None):
    src = air.tensor([N], dt_in)
    out = air.tensor([N], dt_out)

    with air.launch(name="cast") as launch:

        @launch.body
        def _():
            with air.herd([range(0, N, tile)], shape=(2,)) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    a = air.alloc([tn], dt_in, scope=h.private(), vector=vector)
                    c = air.alloc([tn], dt_out, scope=h.private(), vector=vector)
                    air.ops.load(a, src[col : col + tn])
                    c[:] = body(a)
                    air.ops.store(c, out[col : col + tn])

    print(launch.build(target="npu1"))


# CHECK-LABEL: TEST: f32_to_i32
# A float source and an integer destination: arith.fptosi, on the vector type,
# with the read still at the destination's lane count.
# CHECK: vector.transfer_read {{.*}} vector<16xf32>
# CHECK: arith.fptosi %{{.*}} : vector<16xf32> to vector<16xi32>
# CHECK: vector.transfer_write {{.*}} vector<16xi32>
print("\nTEST: f32_to_i32")
build(lambda a: air.ops.cast(a[:], i32), f32, i32)


# CHECK-LABEL: TEST: i32_to_f32
# CHECK: arith.sitofp %{{.*}} : vector<16xi32> to vector<16xf32>
print("\nTEST: i32_to_f32")
build(lambda a: air.ops.cast(a[:], f32), i32, f32)


# CHECK-LABEL: TEST: widening_float_is_extf
# CHECK: arith.extf %{{.*}} : vector<16xbf16> to vector<16xf32>
print("\nTEST: widening_float_is_extf")
build(lambda a: air.ops.cast(a[:], f32), bf16, f32, vector=16)


# CHECK-LABEL: TEST: narrowing_float_is_truncf
# CHECK: arith.truncf %{{.*}} : vector<16xf32> to vector<16xbf16>
print("\nTEST: narrowing_float_is_truncf")
build(lambda a: air.ops.cast(a[:], bf16), f32, bf16, vector=16)


# CHECK-LABEL: TEST: widening_int_is_extsi
# CHECK: arith.extsi %{{.*}} : vector<16xi16> to vector<16xi32>
print("\nTEST: widening_int_is_extsi")
build(lambda a: air.ops.cast(a[:], i32), i16, i32)


# CHECK-LABEL: TEST: compute_below_the_cast_stays_in_the_source_type
# `cast(a[:] * 2.0, i32)` doubles in f32 -- an f32 constant, broadcast, and an
# arith.mulf on vector<16xf32> -- and converts once, at the end.
# CHECK: arith.constant 2.000000e+00 : f32
# CHECK: arith.mulf {{.*}} : vector<16xf32>
# CHECK: arith.fptosi %{{.*}} : vector<16xf32> to vector<16xi32>
print("\nTEST: compute_below_the_cast_stays_in_the_source_type")
build(lambda a: air.ops.cast(a[:] * 2.0, i32), f32, i32)


# CHECK-LABEL: TEST: compute_above_the_cast_is_in_the_target_type
# The same two operations in the other order: convert first, then add in i32.
# The constant is `3 : i32` rather than `3.0 : f32`, which is what proves the
# scalar took its type from the region it sits in rather than from the source
# buffer of the assignment.
# CHECK: arith.fptosi %{{.*}} : vector<16xf32> to vector<16xi32>
# CHECK: arith.constant 3 : i32
# CHECK: arith.addi {{.*}} : vector<16xi32>
print("\nTEST: compute_above_the_cast_is_in_the_target_type")
build(lambda a: air.ops.cast(a[:], i32) + 3, f32, i32)


# CHECK-LABEL: TEST: scalar_fallback_no_vector_width
# One of the two independent routes to the scalar loop: the caller turned
# vectorisation off. memref.load, a scalar arith.fptosi, memref.store.
# CHECK-NOT: vector.transfer_read
# CHECK: memref.load
# CHECK: arith.fptosi %{{.*}} : f32 to i32
# CHECK: memref.store
print("\nTEST: scalar_fallback_no_vector_width")
build(lambda a: air.ops.cast(a[:], i32), f32, i32, vector=0)


# CHECK-LABEL: TEST: scalar_fallback_tile_not_a_multiple
# The other route, and the one that matters for a cast: nothing in the source
# mentions a width, and the emitter picks the scalar loop because 100 is not a
# multiple of 16. A cast must mean the same thing on both routes -- which is
# exactly why narrowing between integer types is refused, since there it does
# not (the vector form saturates and the scalar form wraps).
# CHECK-NOT: vector.transfer_read
# CHECK: arith.fptosi %{{.*}} : f32 to i32
print("\nTEST: scalar_fallback_tile_not_a_multiple")
build(lambda a: air.ops.cast(a[:], i32), f32, i32, N=200, tile=100)


# CHECK-LABEL: TEST: cast_to_the_same_type_emits_nothing
# numpy's astype is a no-op when the type already matches, and so is this. It
# must not reach a builder: arith rejects a same-type extf with "operand type
# and result type are cast incompatible".
# CHECK-NOT: arith.extf
# CHECK-NOT: arith.truncf
# CHECK-NOT: arith.fptosi
# CHECK-NOT: arith.sitofp
# CHECK: vector.transfer_write
print("\nTEST: cast_to_the_same_type_emits_nothing")
build(lambda a: air.ops.cast(a[:], f32), f32, f32)


# CHECK-LABEL: TEST: ops_relu_over_a_cast_zeroes_in_the_target_type
# ops.relu picks its zero constant from the expression's element type. Those
# agree everywhere except across a cast, where the buffer under it is f32 and
# the result is i32 -- so the zero must be `0 : i32`, and arith.maxsi rather
# than arith.maximumf must do the clamping.
# CHECK: arith.fptosi %{{.*}} : vector<16xf32> to vector<16xi32>
# CHECK: arith.constant 0 : i32
# CHECK: arith.maxsi {{.*}} : vector<16xi32>
# CHECK-NOT: arith.maximumf
print("\nTEST: ops_relu_over_a_cast_zeroes_in_the_target_type")
build(lambda a: air.ops.relu(air.ops.cast(a[:], i32)), f32, i32)


# CHECK-LABEL: TEST: repr_names_the_target_type
# A cast node prints as the call the user wrote, naming the type it converts
# to. Without this it would fall through to the binary form and print as
# `(Buffer None Buffer)`, since a cast carries no operator key.
# CHECK: cast(Buffer(shape=(128,), dtype=air.api.f32, space=L1), air.api.i32)
print("\nTEST: repr_names_the_target_type")

_shown = []


def _show_then_cast(a):
    expr = air.ops.cast(a[:], i32)
    _shown.append(repr(expr))
    return expr


build(_show_then_cast, f32, i32)
print(_shown[0])


# CHECK-LABEL: TEST: relu_over_a_predicate_only_select
# The companion to the refusal in errors.py, and a regression guard. ops.relu
# asks the expression for its element type to decide whether to build 0.0 or
# 0; an expression whose buffers sit only in a predicate has no element type of
# its own, and relu must fall back to a leaf rather than reject it -- this
# lowers fine, and rejecting it would be a regression against a form ops.select
# explicitly permits.
# CHECK: arith.cmpf
# CHECK: arith.select
# CHECK: arith.maximumf
print("\nTEST: relu_over_a_predicate_only_select")
build(lambda a: air.ops.relu(air.ops.select(a[:] > 0.0, 1.0, 2.0)), f32, f32)
