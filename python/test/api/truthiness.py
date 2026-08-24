# ./python/test/api/truthiness.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""A buffer expression has no truth value, and prints its operators.

`and`, `or` and `not` are the one part of Python's operator surface a library
cannot reach. There is no dunder for them: they coerce the operand through
__bool__ and short-circuit, so `a[:] and b[:]` evaluates bool(a[:]) -- default
object truthiness, always True -- and hands back b[:]. Nothing raises and no IR
is emitted, so the kernel silently computes half of what was written.

That is the same class of trap as `==` on a buffer expression, and the same
answer NumPy reached: refuse the coercion and name the operator to use instead.

The repr change is here for the same reason. A binary node stores an internal
key, not the source spelling, so it used to print `(Buffer add Buffer)` -- and
once the bitwise operators exist, `a[:] & b[:]` would print `(Buffer and
Buffer)`, naming the very operator the guard above rejects.
"""

from air import api as air
from air.api.types import f32


def trace(body):
    """Run ``body`` inside a herd, so allocation and scope rules are satisfied."""
    a = air.tensor([64], f32)
    out = air.tensor([64], f32)

    with air.launch(name="t") as launch:

        @launch.body
        def _():
            with air.herd([range(0, 64, 64)], shape=(1,)) as h:

                @h.body
                def _(tx):
                    x = air.alloc([64], f32, scope=h.private())
                    y = air.alloc([64], f32, scope=h.private())
                    body(x, y)
                    air.ops.load(x, a[0:64])
                    air.ops.store(x, out[0:64])

    launch.build(target="npu1")


def attempt(label, fn):
    print("\nTEST:", label)
    try:
        fn()
        print("ERROR: no exception raised")
    except TypeError as e:
        print(f"TypeError: {e}")


# CHECK-LABEL: TEST: and_on_a_buffer_expression
# CHECK: TypeError: cannot use a buffer expression as a truth value
attempt(
    "and_on_a_buffer_expression",
    lambda: trace(lambda x, y: x[:] and y[:]),
)


# CHECK-LABEL: TEST: or_on_a_buffer_expression
# CHECK: TypeError: cannot use a buffer expression as a truth value
attempt(
    "or_on_a_buffer_expression",
    lambda: trace(lambda x, y: x[:] or y[:]),
)


# CHECK-LABEL: TEST: not_on_a_buffer_expression
# CHECK: TypeError: cannot use a buffer expression as a truth value
attempt(
    "not_on_a_buffer_expression",
    lambda: trace(lambda x, y: not x[:]),
)


# CHECK-LABEL: TEST: if_on_a_buffer_expression
# The form a user is most likely to write by accident, and the one that used to
# take the `if` branch unconditionally.
# CHECK: TypeError: cannot use a buffer expression as a truth value
def _if_branch(x, y):
    if x[:]:
        pass


attempt("if_on_a_buffer_expression", lambda: trace(_if_branch))


# CHECK-LABEL: TEST: repr_uses_operator_symbols
# The internal key never reaches the user: `add` prints as `+`, and an operator
# with no infix spelling prints as the call it is.
# CHECK: (Buffer{{.*}} + Buffer{{.*}})
# CHECK: (Buffer{{.*}} * 2.0)
# CHECK: maximum(Buffer{{.*}}, 0.0)
# CHECK-NOT: add
print("\nTEST: repr_uses_operator_symbols")


def _show(x, y):
    print(repr(x[:] + y[:]))
    print(repr(x[:] * 2.0))
    print(repr(air.ops.maximum(x[:], 0.0)))


trace(_show)
