# ./python/test/api/relu.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers ops.maximum / ops.minimum / ops.relu.

These are the first compute ops that are not Python operators, so they enter the
expression tree through `air.api.ops` rather than through `__add__` and friends,
and they must compose with the arithmetic operators in the same tree.
"""

from air import api as air
from air.api import ops
from air.api.types import bf16, f32


def build(N, tile, body, herd_shape=None, dtype=bf16, vector=None):
    x = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="relu") as launch:

        @launch.body
        def _():
            with air.herd(range(0, N, tile), shape=herd_shape) as h:

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


# CHECK-LABEL: TEST: relu_vectorized
# ops.relu is max(x, 0): a broadcast zero and a single arith.maximumf, which is
# exactly what the hand-written kernel emitted.
# CHECK: func.func @relu(%{{.*}}: memref<65536xbf16>, %{{.*}}: memref<65536xbf16>)
# CHECK: air.herd @herd_0 tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: vector.broadcast {{.*}} : bf16 to vector<16xbf16>
# CHECK: arith.maximumf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
@run
def relu_vectorized():
    print(build(65536, 1024, lambda b: ops.relu(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: clamp_composes
# maximum and minimum nest, and compose with the arithmetic operators, all in
# one tree lowered as a single loop.
# CHECK: arith.maximumf {{.*}} : vector<16xbf16>
# CHECK: arith.minimumf {{.*}} : vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
@run
def clamp_composes():
    print(
        build(
            65536,
            1024,
            lambda b: ops.minimum(ops.maximum(b[:], 0.0), 6.0) * 2.0,
            herd_shape=(4,),
        ).mlir()
    )


# CHECK-LABEL: TEST: relu_f32_scalar
# f32 relu runs scalar and the example pins it that way: mlir-aie's
# convert-vector-to-aievec rejects a vector maximumf on f32 outright with
# "aievec.max conversion fails due to unsupported element data type", on both
# generations. Scalar f32 max is fine.
# CHECK-NOT: vector.broadcast
# CHECK: memref.alloc() : memref<1024xf32, 2 : i32>
# CHECK: arith.maximumf {{.*}} : f32
# CHECK: memref.store
@run
def relu_f32_scalar():
    print(
        build(
            65536, 1024, lambda b: ops.relu(b[:]), herd_shape=(4,), dtype=f32, vector=0
        ).mlir()
    )
