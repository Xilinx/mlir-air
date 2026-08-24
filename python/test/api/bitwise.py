# ./python/test/api/bitwise.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers &, | and ^ to arith.andi / ori / xori.

These are the DSL's first integer-only operators. Every other binary operator
has both a float and an integer form, so `_FLOAT_OPS` and `_INT_OPS` are
parallel tables; the bitwise entries exist only in the integer one, and a float
buffer reaching them is refused by name rather than falling through the generic
"unsupported operator" message.
"""

from air import api as air
from air.api.types import i16, i32


def build(dtype, body, N=65536, tile=1024, vector=16, herd_shape=(2,)):
    a = air.tensor([N], dtype)
    b = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="bw") as launch:

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


# CHECK-LABEL: TEST: andi_ori_xori
# One op each, on vector<16xi32> -- what the three vector_examples emit. The
# order is the tree's, not the source line's: both operands of `|` are evaluated
# before it, so andi and xori come first and ori joins them.
# CHECK: arith.andi {{.*}} : vector<16xi32>
# CHECK: arith.xori {{.*}} : vector<16xi32>
# CHECK: arith.ori {{.*}} : vector<16xi32>
@run
def andi_ori_xori():
    print(build(i32, lambda a, b: (a[:] & b[:]) | (a[:] ^ b[:])).mlir())


# CHECK-LABEL: TEST: bitwise_with_a_scalar_mask
# A scalar operand broadcasts like any other, so a constant mask is ordinary.
# CHECK: arith.constant 255 : i32
# CHECK: vector.broadcast {{.*}} : i32 to vector<16xi32>
# CHECK: arith.andi {{.*}} : vector<16xi32>
@run
def bitwise_with_a_scalar_mask():
    print(build(i32, lambda a, b: a[:] & 0xFF).mlir())


# CHECK-LABEL: TEST: reflected_operands
# `0xff & a[:]` takes __rand__, which puts the scalar on the left -- the operand
# order is preserved rather than silently commuted.
# CHECK: arith.andi {{.*}} : vector<16xi32>
@run
def reflected_operands():
    print(build(i32, lambda a, b: 0xFF & a[:]).mlir())


# CHECK-LABEL: TEST: narrower_integer
# Nothing about these is i32-specific; i16 lowers the same way at its own width.
# CHECK: arith.xori {{.*}} : vector<16xi16>
@run
def narrower_integer():
    print(build(i16, lambda a, b: a[:] ^ b[:]).mlir())


# CHECK-LABEL: TEST: scalar_fallback
# A tile that is not a multiple of the vector width falls back to a scalar
# memref.load/store loop, and the bitwise op survives it as a scalar i32.
# CHECK-NOT: vector.transfer_read
# CHECK: arith.andi {{.*}} : i32
# CHECK: memref.store
@run
def scalar_fallback():
    print(build(i32, lambda a, b: a[:] & b[:], N=192, tile=24, vector=0).mlir())
