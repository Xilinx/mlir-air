# ./python/test/api/leaky_relu.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api expresses leaky ReLU without a comparison or a select.

`select(x >= 0, x, alpha*x)` is `max(x, 0) + alpha * min(x, 0)`, which needs
only operators the DSL has. This pins the emitted shape so the rewrite cannot
silently change: two broadcasts, a maximumf, a minimumf, a mulf and an addf, all
in one loop over the tile.
"""

from air import api as air
from air.api.types import bf16, f32


def build(N, tile, alpha, herd_shape=None, dtype=bf16, vector=None):
    x = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="leaky_relu", target="npu1") as launch:

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
                    o_buf[:] = air.ops.maximum(x_buf[:], 0.0) + alpha * air.ops.minimum(
                        x_buf[:], 0.0
                    )
                    air.ops.store(o_buf, out[col : col + tn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: leaky_relu_vectorized
# No comparison and no select: the whole thing is max/min/mul/add. If a future
# change reintroduces arith.cmpf here, the rewrite has been undone.
# CHECK: func.func @leaky_relu(%{{.*}}: memref<65536xbf16>, %{{.*}}: memref<65536xbf16>)
# CHECK-NOT: arith.cmpf
# CHECK-NOT: arith.select
# CHECK: arith.maximumf {{.*}} : vector<16xbf16>
# CHECK: arith.minimumf {{.*}} : vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
@run
def leaky_relu_vectorized():
    print(build(65536, 1024, 0.01, herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: leaky_relu_f32_scalar
# f32 runs scalar: aievec has no f32 max, and the chained f32 multiply-add does
# not legalize at any vector width either.
# CHECK-NOT: vector.broadcast
# CHECK: memref.alloc() : memref<1024xf32, 2 : i32>
# CHECK: arith.maximumf {{.*}} : f32
# CHECK: arith.minimumf {{.*}} : f32
# CHECK: arith.addf {{.*}} : f32
# CHECK: memref.store
@run
def leaky_relu_f32_scalar():
    print(build(65536, 1024, 0.01, herd_shape=(4,), dtype=f32, vector=0).mlir())
