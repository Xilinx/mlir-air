# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The default lane count is a property of the target, and of the widest element
# in the assignment. Both halves are asserted here, because both are load-
# bearing and neither is visible in a test that pins `vector=` itself.
#
# RUN: %PYTHON %s | FileCheck %s

from air import api as air
from air.api import bf16, f32, ops

N, TILE = 4096, 1024


def build(body, dtype=bf16, target="npu2", vector=None):
    x = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="w", target=target) as launch:

        @launch.body
        def _():
            with air.herd(range(0, N, TILE), shape=(4,)) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    a = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    o = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    ops.load(a, x[col : col + tn])
                    body(a, o)
                    ops.store(o, out[col : col + tn])

    return launch.mlir()


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: bf16_is_512_bit_on_npu2
# npu2's backend legalizes a 512-bit bf16 vector, so that is what a bf16 tile
# gets by default. npu1's does not; it stays at 256.
# CHECK: arith.addf {{.*}} : vector<32xbf16>
@run
def bf16_is_512_bit_on_npu2():
    print(build(lambda a, o: o.__setitem__(slice(None), a[:] + a[:])))


# CHECK-LABEL: TEST: bf16_is_256_bit_on_npu1
# CHECK: arith.addf {{.*}} : vector<16xbf16>
@run
def bf16_is_256_bit_on_npu1():
    print(build(lambda a, o: o.__setitem__(slice(None), a[:] + a[:]), target="npu1"))


# CHECK-LABEL: TEST: an_f32_region_caps_the_whole_nest
# One assignment carries one lane count across every region in it, so a 32-lane
# bf16 destination with an f32 region inside would ask for a 1024-bit f32
# vector, which does not legalize. The nest drops to what the widest element
# allows -- 16 -- rather than the assignment having to spell `vector=` itself.
# CHECK: arith.addf {{.*}} : vector<16xf32>
# CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
@run
def an_f32_region_caps_the_whole_nest():
    print(
        build(
            lambda a, o: o.__setitem__(
                slice(None), ops.cast(ops.cast(a[:], f32) + 1.0, bf16)
            )
        )
    )


# CHECK-LABEL: TEST: an_explicit_width_still_wins
# `vector=` is an override, not a hint: it survives the target default.
# CHECK: arith.addf {{.*}} : vector<16xbf16>
@run
def an_explicit_width_still_wins():
    print(build(lambda a, o: o.__setitem__(slice(None), a[:] + a[:]), vector=16))


# CHECK-LABEL: TEST: fill_vectorises
# ops.fill is the elementwise store loop, not linalg.fill, so it moves a vector
# per trip rather than an element. linalg.fill lowers to the scalar loop here,
# which is what made an 8192-element bf16 fill 8192 stores.
# CHECK: vector.transfer_write {{.*}} vector<32xbf16>
# CHECK-NOT: linalg.fill
@run
def fill_vectorises():
    print(build(lambda a, o: ops.fill(o, 0.0)))


# CHECK-LABEL: TEST: a_transcendental_holds_the_narrow_width
# Only the arithmetic widths were measured for the wide default. bf16 math.tanh
# at 512 bits compiles and is quietly less accurate -- gelu, whose body is one
# tanh, goes from exact to 734 of 65536 elements outside its bf16 tolerance --
# so an expression containing one stays at the width its lowering is measured
# at. exp and rsqrt are held there too, as unmeasured rather than known-bad.
# CHECK: math.tanh {{.*}} : vector<16xbf16>
@run
def a_transcendental_holds_the_narrow_width():
    print(build(lambda a, o: o.__setitem__(slice(None), ops.tanh(a[:]))))


# CHECK-LABEL: TEST: arithmetic_beside_it_is_not_penalised
# The cap is per assignment, not per buffer: the same tile written by a plain
# arithmetic expression still gets the wide width.
# CHECK: arith.mulf {{.*}} : vector<32xbf16>
@run
def arithmetic_beside_it_is_not_penalised():
    print(build(lambda a, o: o.__setitem__(slice(None), a[:] * a[:])))
