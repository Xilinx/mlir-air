# ./python/test/api/shift.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers << and >> to arith.shli / shrsi.

Integer-only, like the bitwise operators beside them and for the same reason:
there is no floating-point counterpart in arith, so a float buffer reaching one
is refused by name rather than falling through the generic message.

`>>` is *arithmetic* (arith.shrsi, sign-replicating) rather than logical. That
matches Python, where `>>` on a negative int floors instead of filling with
zeros, and it matches air.api's integer dtypes, which are signed.

The last two tests pin the shift-clamp-truncate chain that
primitives/scalar_examples/scalar_shift_saturate is built on. The order of
those five ops is what mlir-aie's LowerScalarShiftClampTruncToSRS pattern
matches, so it is pinned here rather than left to the example's hardware lit --
which only runs on npu2.
"""

from air import api as air
from air.api.types import f32, i8, i16, i32


def build(dtype, body, N=65536, tile=1024, vector=16, out_dtype=None):
    out_dtype = out_dtype or dtype
    a = air.tensor([N], dtype)
    out = air.tensor([N], out_dtype)

    with air.launch(name="sh") as launch:

        @launch.body
        def _():
            with air.herd([range(0, N, tile)], shape=(2,)) as h:

                @h.body
                def _(tx):
                    col = tx * tile
                    a_buf = air.alloc([tile], dtype, scope=h.private(), vector=vector)
                    o_buf = air.alloc(
                        [tile], out_dtype, scope=h.private(), vector=vector
                    )
                    air.ops.load(a_buf, a[col : col + tile])
                    o_buf[:] = body(a_buf)
                    air.ops.store(o_buf, out[col : col + tile])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def clamp(x, lo, hi):
    return air.ops.minimum(air.ops.maximum(x, lo), hi)


# CHECK-LABEL: TEST: shl_and_shr
# CHECK: arith.shli {{.*}} : vector<16xi32>
# CHECK: arith.shrsi {{.*}} : vector<16xi32>
@run
def shl_and_shr():
    print(build(i32, lambda a: (a[:] << 2) >> 1).mlir())


# CHECK-LABEL: TEST: reflected_operands
# `1 << a[:]` takes __rlshift__, which puts the scalar on the left. A variable
# shift *amount* is as legal as a variable shift *operand* -- and is the one
# case the shift-count check cannot cover, since there is no constant to look
# at. An out-of-range value here still reaches the backend as poison; a
# constant one is refused at the call site (see errors.py).
# CHECK: vector.broadcast {{.*}} : i32 to vector<16xi32>
# CHECK: arith.shli {{.*}} : vector<16xi32>
@run
def reflected_operands():
    print(build(i32, lambda a: 1 << a[:]).mlir())


# CHECK-LABEL: TEST: narrower_integer
# CHECK: arith.shrsi {{.*}} : vector<16xi16>
@run
def narrower_integer():
    print(build(i16, lambda a: a[:] >> 3).mlir())


# CHECK-LABEL: TEST: scalar_fallback
# vector=0 is the path scalar_shift_saturate uses, and the path the SRS pattern
# matches against.
# CHECK-NOT: vector.transfer_read
# CHECK: arith.shrsi {{.*}} : i32
# CHECK: memref.store
@run
def scalar_fallback():
    print(build(i32, lambda a: a[:] >> 4, N=192, tile=24, vector=0).mlir())


# CHECK-LABEL: TEST: srs_chain
# The five ops scalar_shift_saturate emits, in the order the mlir-aie pattern
# expects: shift, clamp low, clamp high, narrow, widen back.
# CHECK: arith.shrsi {{.*}} : i32
# CHECK: arith.maxsi {{.*}} : i32
# CHECK: arith.minsi {{.*}} : i32
# CHECK: arith.trunci {{.*}} : i32 to i8
# CHECK: arith.extsi {{.*}} : i8 to i32
@run
def srs_chain():
    print(
        build(
            i32,
            lambda a: air.ops.cast(air.ops.cast(clamp(a[:] >> 4, -128, 127), i8), i32),
            N=192,
            tile=24,
            vector=0,
        ).mlir()
    )


# CHECK-LABEL: TEST: clamped_narrowing_is_order_insensitive
# max-then-min above; min-then-max here. Clamping is commutative, so both are
# accepted -- the rule is about the bounds, not the spelling.
# CHECK: arith.trunci {{.*}} : i32 to i8
@run
def clamped_narrowing_is_order_insensitive():
    print(
        build(
            i32,
            lambda a: air.ops.cast(
                air.ops.cast(
                    air.ops.maximum(air.ops.minimum(a[:] >> 4, 127), -128), i8
                ),
                i32,
            ),
            N=192,
            tile=24,
            vector=0,
        ).mlir()
    )
