# ./python/test/api/transcendental.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers exp and rsqrt, and stamps link_with on a herd with no call.

`exp` and `rsqrt` are unary nodes like `tanh`, and lower the same way -- one
math dialect op on the vector. What is new here is that on npu1 neither is an
instruction: the AIE lowering rewrites math.exp into a call to getExpBf16 and
math.rsqrt into getRsqrtBf16, both of which live in a hand-written object file.

That call appears several passes below the DSL, so air.extern -- whose whole
job is to emit a func.call and hang link_with off its declaration -- cannot
express the dependency. `air.herd(link_with=...)` declares it directly. The tests
below pin that it lands on the herd, that it is optional, and that it conflicts
with an air.extern call naming a different object.

The link_with checks are not decorative: dropping link_with= from vector_rsqrt on
npu1 fails to link with "undefined symbol: getRsqrtBf16", measured.
"""

from air import api as air
from air.api.types import bf16, f32, i32


def build(N, tile, body, dtype=bf16, herd_shape=None, vector=None, link_with=None):
    x = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="tr") as launch:

        @launch.body
        def _():
            with air.herd(
                range(0, N, tile), shape=herd_shape, link_with=link_with
            ) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    x_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    o_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    air.ops.load(x_buf, x[col : col + tn])
                    o_buf[:] = body(x_buf)
                    air.ops.store(o_buf, out[col : col + tn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: exp_vectorized
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: math.exp {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
@run
def exp_vectorized():
    print(build(65536, 1024, lambda b: air.ops.exp(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: rsqrt_vectorized
# CHECK: math.rsqrt {{.*}} : vector<16xf32>
@run
def rsqrt_vectorized():
    print(
        build(
            65536,
            1024,
            lambda b: air.ops.rsqrt(b[:]),
            dtype=f32,
            herd_shape=(4,),
            vector=16,
        ).mlir()
    )


# CHECK-LABEL: TEST: rsqrt_scalar
# Unlike tanh, rsqrt has a scalar form that legalizes -- on npu2, in f32. That
# is what vector_rsqrt's version 2 exists to exercise, so the emitter's scalar
# path must reach math.rsqrt on the element type rather than on a vector.
# CHECK-NOT: vector.transfer_read
# CHECK: memref.load
# CHECK: math.rsqrt {{.*}} : f32
# CHECK: memref.store
@run
def rsqrt_scalar():
    print(
        build(
            512, 64, lambda b: air.ops.rsqrt(b[:]), dtype=f32, herd_shape=(2,), vector=0
        ).mlir()
    )


# CHECK-LABEL: TEST: rsqrt_through_casts
# bf16 buffers, f32 arithmetic: everything below a cast is read and computed in
# the source type, so the widen happens on the loaded vector and the narrow on
# the result. vector_rsqrt's version 3, which the predecessor spelled with
# explicit arith.extf/truncf.
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: arith.extf {{.*}} vector<16xbf16> to vector<16xf32>
# CHECK: math.rsqrt {{.*}} : vector<16xf32>
# CHECK: arith.truncf {{.*}} vector<16xf32> to vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
@run
def rsqrt_through_casts():
    print(
        build(
            512,
            64,
            lambda b: air.ops.cast(air.ops.rsqrt(air.ops.cast(b[:], f32)), bf16),
            herd_shape=(2,),
        ).mlir()
    )


# CHECK-LABEL: TEST: herd_object_stamps_link_with
# The attribute is on the herd, and there is no func.call anywhere to have
# carried it -- which is the whole point of link_with=.
# CHECK: air.herd {{.*}} attributes {link_with = "extern_func.o"}
# CHECK-NOT: func.call
# CHECK: math.exp
@run
def herd_object_stamps_link_with():
    print(
        build(
            65536,
            1024,
            lambda b: air.ops.exp(b[:]),
            herd_shape=(4,),
            link_with="extern_func.o",
        ).mlir()
    )


# CHECK-LABEL: TEST: no_object_no_link_with
# link_with= is optional: on npu2 both ops are native, and stamping link_with there
# would link an aie2 object into an aie2p build.
# CHECK-NOT: link_with
# CHECK: math.exp
@run
def no_object_no_link_with():
    print(build(65536, 1024, lambda b: air.ops.exp(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: object_agrees_with_extern
# A herd links against one object file. Declaring the same one that an
# air.extern call needs is not a conflict -- it is the case where a kernel and
# a lowered math op share an object.
# CHECK: air.herd {{.*}} attributes {link_with = "kernels.o"}
# CHECK: func.call @scale
@run
def object_agrees_with_extern():
    scale = air.extern("scale", link_with="kernels.o", scalars=[i32])

    def body(buf):
        scale(2, buf)
        return air.ops.exp(buf[:])

    print(build(65536, 1024, body, herd_shape=(4,), link_with="kernels.o").mlir())


# CHECK-LABEL: TEST: object_conflicts_with_extern
# Two object files, one link_with slot. The message names both claims, and
# distinguishes the one that came from a call from the one that was declared.
# CHECK: herd 'herd_0' calls scale from 'kernels.o' and declares link_with='extern_func.o'
@run
def object_conflicts_with_extern():
    scale = air.extern("scale", link_with="kernels.o", scalars=[i32])

    def body(buf):
        scale(2, buf)
        return air.ops.exp(buf[:])

    try:
        build(65536, 1024, body, herd_shape=(4,), link_with="extern_func.o").mlir()
    except ValueError as e:
        print(e)
