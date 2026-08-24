# ./python/test/api/no_fp16.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api refuses arithmetic on f16, because neither NPU can do it.

AIE2 and AIE2P have no fp16 instruction, scalar or vector; bf16 is the 16-bit
float the hardware implements. What made this worth a guard is not the gap but
how the gap presented: nothing in the toolchain refused an f16 kernel. It
compiled, it ran, and it returned the result of having read the f16 bit
patterns as bf16 -- with no error, no warning, and no legalizer failure.

Measured on npu1 before this guard existed:

    c[:] = a[:] + b[:]   over f16 buffers   wrong on 2048 of 2048 elements
    f16 512.5 is 0x6001, and 0x6001 read as bf16 is 1.0078125 x 2^65
    the device returned                     3.718172e19

That is the failure mode this package exists to remove, so f16 now behaves the
way unsigned does: a buffer of it can be declared, allocated and *moved*, and
only arithmetic is refused. Movement is not refused because movement works --
an f16 DMA and an f16 elementwise copy are both exact on 2048 of 2048.
"""

from air import api as air
from air.api import ops
from air.api.types import bf16, f16, f32


def trace(body, dtype=f16):
    src = air.tensor([64], dtype)
    out = air.tensor([64], dtype)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd([range(0, 64, 64)], shape=(1,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc([64], dtype, scope=h.private())
                    c = air.alloc([64], dtype, scope=h.private())
                    ops.load(a, src[0:64])
                    body(c, a)
                    ops.store(c, out[0:64])

    return launch.build(target="npu1")


def expect(label, fn):
    print("\nTEST:", label)
    try:
        fn()
        print("no exception")
    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")


# CHECK-LABEL: TEST: f16_add
# The message has to say what the hardware cannot do and what to use instead;
# "not supported" alone would send the reader looking for a missing feature in
# air.api rather than a missing instruction in the core.
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar (a plain copy, dst[:] = src[:], is) is not supported for air.api.f16: neither NPU generation has an fp16 instruction
# CHECK-SAME: Use air.api.bf16
expect("f16_add", lambda: trace(lambda c, a: c.__setitem__(slice(None), a[:] + a[:])))


# CHECK-LABEL: TEST: f16_broadcast_scalar
# A scalar operand is an arith.constant, so this is refused for the same reason
# even though only one side is a buffer.
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar
expect(
    "f16_broadcast_scalar",
    lambda: trace(lambda c, a: c.__setitem__(slice(None), a[:] * 2.0)),
)


# CHECK-LABEL: TEST: f16_fill
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar
expect("f16_fill", lambda: trace(lambda c, a: c.__setitem__(slice(None), 1.0)))


# CHECK-LABEL: TEST: f16_named_op
# ops.relu composes down to arith.maximumf, so the guard has to catch it at the
# emitter rather than at each op's own front door.
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar
expect(
    "f16_named_op",
    lambda: trace(lambda c, a: c.__setitem__(slice(None), ops.relu(a[:]))),
)


# CHECK-LABEL: TEST: f16_extern_scalar
# An extern kernel's buffer arguments may be f16 -- moving the data is the
# point -- but a scalar argument is materialised by arith.constant.
# CHECK: NotImplementedError: an air.extern scalar argument is not supported for air.api.f16
expect("f16_extern_scalar", lambda: air.extern("k", object="k.o", scalars=[f16]))


# CHECK-LABEL: TEST: f16_copy_is_allowed
# The other half of the rule, and the half that keeps the type useful: a plain
# copy emits a read and a write and no arith op, so the bits arrive unchanged.
# Refusing it would make f16 undeclarable rather than uncomputable.
#
# The copy also *vectorises*, which is why f16's default_vector_width is live
# surface rather than dead: an unsigned buffer is forced onto the scalar path
# even for a copy, and f16 is not. Pinned here so the two restrictions do not
# get described as the same thing.
# CHECK: memref<64xf16
# CHECK: vector<16xf16>
# CHECK-NOT: arith.addf
print("\nTEST: f16_copy_is_allowed")
print(trace(lambda c, a: c.__setitem__(slice(None), a[:])))


# CHECK-LABEL: TEST: the_types_that_do_compute_are_untouched
# The guard keys on one flag, so the risk is that it catches more than f16.
# CHECK: bf16 arith.addf: yes
# CHECK: f32 arith.addf: yes
print("\nTEST: the_types_that_do_compute_are_untouched")
for name, dt in (("bf16", bf16), ("f32", f32)):
    ir = str(trace(lambda c, a: c.__setitem__(slice(None), a[:] + a[:]), dtype=dt))
    print(f"{name} arith.addf: {'yes' if 'arith.addf' in ir else 'NO'}")
