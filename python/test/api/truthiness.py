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


# CHECK-LABEL: TEST: advice_names_only_surface_that_exists
# The redirection is built from what the build has, not hardcoded. `&` and
# ops.select arrive in separate changes, and an error naming surface the caller
# does not have is worse than the bug it replaces -- while hedging every
# suggestion with "if available" would make the useful case vague.
#
# All four states are *constructed* rather than observed. An earlier version
# only printed the states the build happened to be in and skipped the rest,
# which meant its coverage silently shrank as the surface landed: once
# ops.select merged, two of the four lines stopped being produced at all and
# the test failed for a reason that had nothing to do with the guard.
# CHECK: neither: (no advice)
# CHECK: & only: Use the elementwise operators `&`, `|`, `^` for logic.
# CHECK: both: Use the elementwise operators `&`, `|`, `^` for logic or air.api.ops.select(cond, a, b) to choose between values.
# CHECK: select only: Use air.api.ops.select(cond, a, b) to choose between values.
print("\nTEST: advice_names_only_surface_that_exists")

from air.api import ops as _ops
from air.api._value import BufferExpr as _Expr


def _advice():
    try:
        bool(_Expr("scalar", scalar=1))
    except TypeError as e:
        return str(e).split("overloaded.")[1].strip() or "(no advice)"


_real_and = vars(_Expr).get("__and__")
_real_select = getattr(_ops, "select", None)


def _surface(has_and, has_select):
    """Put the build into one of the four states, whichever it started in."""
    if has_and:
        _Expr.__and__ = _real_and or (lambda self, o: None)
    elif "__and__" in vars(_Expr):
        del _Expr.__and__
    if has_select:
        _ops.select = _real_select or (lambda *a: None)
    elif hasattr(_ops, "select"):
        del _ops.select


for _label, _a, _s in (
    ("neither", False, False),
    ("& only", True, False),
    ("both", True, True),
    ("select only", False, True),
):
    _surface(_a, _s)
    print(f"{_label}: {_advice()}")

_surface(_real_and is not None, _real_select is not None)
