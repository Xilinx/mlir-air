# ./python/test/api/axpy.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers a nested elementwise expression: out = alpha * x + y.

Distinct from the eltwise_add tests in one way that matters: the right-hand side
is a *tree*, not a single operator. One arith builder's output feeds the next,
which is the case that requires each intermediate to be reduced to a Value --
the builders infer their result type from ``operands[0].type``, and an OpView
does not have one.
"""

from air import api as air
from air.api.types import bf16, f32


def build(N, tile, alpha, herd_shape=None, dtype=bf16, vector=None):
    x = air.tensor([N], dtype)
    y = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="axpy", target="npu1") as launch:

        @launch.body
        def _():
            with air.herd(range(0, N, tile), shape=herd_shape) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    x_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    y_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    out_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    air.ops.load(x_buf, x[col : col + tn])
                    air.ops.load(y_buf, y[col : col + tn])
                    out_buf[:] = alpha * x_buf[:] + y_buf[:]
                    air.ops.store(out_buf, out[col : col + tn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: axpy_vectorized
# The scalar is broadcast to the compute width once and multiplied in, then
# added: two chained arith ops on vector<16xbf16>. 64 tiles strip-mined onto 4
# cores gives 16 repeats, folded into one affine map: (tx*16 + i)*1024.
# CHECK: affine_map<()[s0, s1] -> (s0 * 16384 + s1 * 1024)>
# CHECK: func.func @axpy(%{{.*}}: memref<65536xbf16>, %{{.*}}: memref<65536xbf16>, %{{.*}}: memref<65536xbf16>)
# CHECK: air.herd @herd_0 tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1
# CHECK: scf.for
# CHECK: memref.alloc() : memref<1024xbf16, 2 : i32>
# CHECK: air.dma_memcpy_nd
# CHECK: arith.constant 2.000000e+00 : bf16
# CHECK: vector.broadcast {{.*}} : bf16 to vector<16xbf16>
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
# CHECK: air.dma_memcpy_nd
# CHECK: memref.dealloc
#
# Note this is no longer op-for-op what programming_examples/axpy emits: that
# example now spells the vectorised bf16 case air.ops.fma, which rounds once
# instead of twice. What is pinned here is the DSL property the file is about
# -- that a nested tree of operators lowers as one loop -- and `alpha * x + y`
# is still exactly what axpy emits on its scalar paths. See api/fma.py for the
# fused form.
@run
def axpy_vectorized():
    print(build(65536, 1024, 2.0, herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: axpy_scalar_fallback
# A 12-wide tile is not a multiple of the bf16 vector width, so the whole tree
# lowers scalar -- including the broadcast, which becomes a plain constant.
# CHECK-NOT: vector.broadcast
# CHECK: memref.load
# CHECK: arith.mulf {{.*}} : bf16
# CHECK: memref.load
# CHECK: arith.addf {{.*}} : bf16
# CHECK: memref.store
@run
def axpy_scalar_fallback():
    print(build(48, 12, 2.0, herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: axpy_f32_scalar
# f32 axpy runs scalar, and the example pins it that way. A chained f32
# multiply-add on 512-bit vectors does not legalize in the AIE2 backend
# ("unable to legalize instruction: <16 x s32> = G_FMUL"), at 8, 16 or 32 lanes,
# on either generation -- although each op on its own vectorises fine. The DSL
# cannot detect that, since it is a property of the pair, so the emitter still
# vectorises f32 on request and it is the caller that asks for scalar.
# CHECK-NOT: vector.broadcast
# CHECK: memref.alloc() : memref<1024xf32, 2 : i32>
# CHECK: arith.mulf {{.*}} : f32
# CHECK: arith.addf {{.*}} : f32
# CHECK: memref.store
@run
def axpy_f32_scalar():
    print(build(65536, 1024, 2.0, herd_shape=(4,), dtype=f32, vector=0).mlir())
