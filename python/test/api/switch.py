# ./python/test/api/switch.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers ops.switch to scf.index_switch in its value-returning form.

``ops.switch`` is the DSL's only N-way construct, and it is an expression: it
yields the chosen value. The N-way *statement* -- run one of N bodies for their
effects -- is nested ``ops.branch``, so what these pin is mostly the boundary
between the three constructs that all look like a choice: switch keys on an
integer, select keys on a per-element mask, branch keys on a condition and runs
statements.
"""

from air import api as air
from air.api import ops
from air.api.types import bf16


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(pick, n=64, vector=16):
    """A herd body whose accumulate adds a value chosen by ``pick(step)``."""
    A = air.tensor([n], bf16)
    OUT = air.tensor([n], bf16)

    with air.launch(name="sw") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([n], bf16, scope=h.private(), vector=vector)
                    ops.load(a, A)
                    for step in air.sequential(0, 2):
                        a[:] = a[:] + pick(step)
                    ops.store(a, OUT)

    return launch.mlir()


# CHECK-LABEL: TEST: switch_lowers_to_a_value_returning_index_switch
# One switch per assignment, hoisted above the elementwise loop it feeds: the
# choice does not depend on that loop's induction variable, so evaluating it per
# element would be N constants and N-1 dead arms per trip.
# CHECK: scf.index_switch %{{.*}} -> bf16
# CHECK: case 0
# CHECK: arith.constant 1.0
# CHECK: scf.yield
# CHECK: default
# CHECK: arith.constant 1.0{{0*}}e+01
# CHECK: scf.yield
# CHECK: scf.for
# CHECK: arith.addf
@run
def switch_lowers_to_a_value_returning_index_switch():
    print(build(lambda step: ops.switch(step, [1.0, 10.0])))


# CHECK-LABEL: TEST: a_constant_index_emits_no_switch
# `ops.switch(0, ...)` is a constant fold, not a one-armed switch: the same IR
# as writing the value out by hand. A Python `for` unrolls at trace time, so its
# counter is a constant and this is the path an unrolled loop takes.
# CHECK-NOT: scf.index_switch
# CHECK: arith.constant 1.0{{0*}}e+01
@run
def a_constant_index_emits_no_switch():
    A = air.tensor([64], bf16)
    OUT = air.tensor([64], bf16)

    with air.launch(name="swc") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], bf16, scope=h.private(), vector=16)
                    ops.load(a, A)
                    for step in range(1, 2):
                        a[:] = a[:] + ops.switch(step, [1.0, 10.0])
                    ops.store(a, OUT)

    print(launch.mlir())


def expect(what, build_fn):
    try:
        build_fn()
    except Exception as e:
        print(f"{what}: {type(e).__name__}: {e}")
        return
    raise AssertionError(f"{what}: expected a diagnostic, got none")


# CHECK-LABEL: TEST: the_three_choices_name_each_other
# Switch, select and branch are told apart by what they key on, and handing one
# the other's key is the likely mistake -- so each error names the neighbour it
# was probably meant to be, rather than complaining about a type.
# CHECK: a condition: TypeError: {{.*}}ops.select(cond, a, b)
# CHECK: a buffer: TypeError: {{.*}}ops.select(mask, a, b)
# CHECK: buffers to pick from: TypeError: {{.*}}ops.branch
# CHECK: one value: ValueError: {{.*}}nothing to choose
@run
def the_three_choices_name_each_other():
    A = air.tensor([64], bf16)
    A_out = air.tensor([64], bf16)

    with air.launch(name="swe") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], bf16, scope=h.private(), vector=16)
                    ops.load(a, A)
                    expect("a condition", lambda: ops.switch(tx == 0, [1.0, 2.0]))
                    expect("a buffer", lambda: ops.switch(a, [1.0, 2.0]))
                    expect("buffers to pick from", lambda: ops.switch(tx, [a, a]))
                    expect("one value", lambda: ops.switch(tx, [1.0]))
                    ops.store(a, A_out)

    # The body runs when the module is built, which is where the diagnostics
    # above are raised; the module itself is not what this test pins.
    launch.mlir()


# CHECK-LABEL: TEST: switch_builds_its_constants_in_the_destinations_type
# The values are kept exactly as written until the destination is known, because
# which type to build them in is the destination's and a switch is written
# before that. Coercing at construction -- everything to float, as this did
# first -- reached arith.ConstantOp with a Python float for an integer type and
# failed inside FloatAttr with "expected floating point type", naming neither
# the switch nor the value.
# CHECK: scf.index_switch %{{.*}} -> i32
# CHECK: arith.constant 1 : i32
# CHECK: arith.constant 10 : i32
@run
def switch_builds_its_constants_in_the_destinations_type():
    from air.api.types import i32

    A = air.tensor([64], i32)
    OUT = air.tensor([64], i32)

    with air.launch(name="swi") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], i32, scope=h.private(), vector=16)
                    ops.load(a, A)
                    for step in air.sequential(0, 2):
                        a[:] = a[:] + ops.switch(step, [1, 10])
                    ops.store(a, OUT)

    print(launch.mlir())


# CHECK-LABEL: TEST: a_fractional_value_in_an_integer_switch_is_refused
# A whole-number float is a harmless way to write an integer and the elementwise
# emitter accepts one; anything else would be truncated, so it is refused rather
# than rounded.
# CHECK: fractional: ValueError: {{.*}}not an integer
@run
def a_fractional_value_in_an_integer_switch_is_refused():
    from air.api.types import i32

    A = air.tensor([64], i32)
    OUT = air.tensor([64], i32)

    with air.launch(name="swf") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], i32, scope=h.private(), vector=16)
                    ops.load(a, A)
                    for step in air.sequential(0, 2):
                        expect(
                            "fractional",
                            lambda: a.__setitem__(
                                slice(None), a[:] + ops.switch(step, [1, 10.5])
                            ),
                        )
                    ops.store(a, OUT)

    launch.mlir()
