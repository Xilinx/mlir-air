# ./python/test/api/bitcast.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api reinterprets bits with ops.bitcast, and widens without a sign.

``cast`` preserves the value and changes the representation; ``bitcast``
preserves the representation and lets the value fall where it may. The pair
exists because a buffer's element type is sometimes a container rather than a
meaning -- bytes holding a packed float, or holding two quantised weights.

What these pin is mostly the int4 unpack, because its shape is not a matter of
taste: mlir-aie's LowerExtUIOfBitcastI4ToUnpackPattern rewrites exactly
`extui(vector.bitcast(<Nxi8> -> <2Nxi4>))` into one `aievec.unpack`, and any
other spelling computes the same numbers while missing the instruction.
"""

from air import api as air
from air.api import ops
from air.api.types import bf16, i4, i8, i16, i32


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: int4_unpack_emits_the_sequence_aievec_matches
# 32 bytes are 64 half-bytes: the read is at the operand's own lane count, the
# bitcast re-counts them, and the extui widens each nibble into a byte. The
# extui must be *unsigned* -- these are quantised magnitudes 0..15, and extsi
# would read 0x9 as -7.
# CHECK: vector.transfer_read %{{.*}} : memref<256xi8, 2 : i32>, vector<32xi8>
# CHECK-NEXT: vector.bitcast %{{.*}} : vector<32xi8> to vector<64xi4>
# CHECK-NEXT: arith.extui %{{.*}} : vector<64xi4> to vector<64xi8>
@run
def int4_unpack_emits_the_sequence_aievec_matches():
    P = air.tensor([256], i8)
    O = air.tensor([512], i8)

    with air.launch(name="unpack") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    p = air.alloc([256], i8, scope=h.private())
                    o = air.alloc([512], i8, scope=h.private(), vector=64)
                    ops.load(p, P)
                    for i in air.sequential(0, 8):
                        o[i * 64 : i * 64 + 64] = ops.cast(
                            ops.bitcast(p[i * 32 : i * 32 + 32], i4), i8, signed=False
                        )
                    ops.store(o, O)

    print(launch.mlir())


# CHECK-LABEL: TEST: a_same_width_bitcast_is_a_relabelling
# i16 bits read as bf16. Nothing about the shape moves, so unlike the unpack it
# applies to a computed value -- which is the point: an AWQ scale is assembled
# from two bytes with a shift and an or, and is a bf16 only once reinterpreted.
# CHECK: arith.shli
# CHECK: arith.ori
# CHECK: vector.bitcast %{{.*}} : vector<16xi16> to vector<16xbf16>
@run
def a_same_width_bitcast_is_a_relabelling():
    LO = air.tensor([64], i8)
    HI = air.tensor([64], i8)
    O = air.tensor([64], bf16)

    with air.launch(name="relabel") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    lo = air.alloc([64], i8, scope=h.private())
                    hi = air.alloc([64], i8, scope=h.private())
                    o = air.alloc([64], bf16, scope=h.private(), vector=16)
                    ops.load(lo, LO)
                    ops.load(hi, HI)
                    bits = ops.cast(lo[:], i16, signed=False) | (
                        ops.cast(hi[:], i16, signed=False) << 8
                    )
                    o[:] = ops.bitcast(bits, bf16)
                    ops.store(o, O)

    print(launch.mlir())


# CHECK-LABEL: TEST: signed_is_the_default_and_unsigned_is_asked_for
# The same widening, twice. Signedness belongs to the operation and not to the
# type -- MLIR's arith ops take signless integers, so a byte is eight bits until
# something widens it and has to decide what the top one meant.
# CHECK: arith.extsi %{{.*}} : vector<16xi8> to vector<16xi32>
# CHECK: arith.extui %{{.*}} : vector<16xi8> to vector<16xi32>
@run
def signed_is_the_default_and_unsigned_is_asked_for():
    A = air.tensor([64], i8)
    S = air.tensor([64], i32)
    U = air.tensor([64], i32)

    with air.launch(name="widen") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], i8, scope=h.private())
                    s = air.alloc([64], i32, scope=h.private(), vector=16)
                    u = air.alloc([64], i32, scope=h.private(), vector=16)
                    ops.load(a, A)
                    s[:] = ops.cast(a[:], i32)
                    u[:] = ops.cast(a[:], i32, signed=False)
                    ops.store(s, S)
                    ops.store(u, U)

    print(launch.mlir())


def expect(what, fn):
    try:
        fn()
    except Exception as e:
        print(f"{what}: {type(e).__name__}: {e}")
        return
    raise AssertionError(f"{what}: expected a diagnostic, got none")


# CHECK-LABEL: TEST: the_refusals_say_what_the_operation_means
# Each of these is a place where accepting the call would compile and compute
# something other than what was written, so each names the property that makes
# it wrong rather than reporting a type.
# CHECK: no i4 buffer: TypeError: {{.*}}whole bytes
# CHECK: repack of a computed value: TypeError: {{.*}}reinterprets memory rather than a value
# CHECK: unsigned float: TypeError: {{.*}}integer widening
# CHECK: unsigned narrowing: TypeError: {{.*}}no wider than
@run
def the_refusals_say_what_the_operation_means():
    expect("no i4 buffer", lambda: air.tensor([64], i4))

    A = air.tensor([64], i8)
    O = air.tensor([64], i8)

    with air.launch(name="refuse") as launch:

        @launch.body
        def _():
            with air.herd(range(1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], i8, scope=h.private(), vector=16)
                    o = air.alloc([64], i8, scope=h.private(), vector=16)
                    ops.load(a, A)
                    expect(
                        "repack of a computed value",
                        lambda: ops.bitcast(a[:] + 1, i4),
                    )
                    expect(
                        "unsigned float",
                        lambda: ops.cast(a[:], bf16, signed=False),
                    )
                    expect(
                        "unsigned narrowing",
                        lambda: ops.cast(ops.cast(a[:], i32), i16, signed=False),
                    )
                    o[:] = a[:]
                    ops.store(o, O)

    launch.mlir()
