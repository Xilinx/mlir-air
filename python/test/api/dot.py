# ./python/test/api/dot.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers a tiled contraction: a K reduction accumulating into a buffer.

Two pieces of surface are exercised here that the elementwise examples do not
reach. Both are load-bearing for correctness on hardware, not conveniences:

``air.sequential`` emits an ``scf.for``. A plain Python ``for k in range(...)`` runs
at trace time and unrolls, so the AIE core comes out with no loop in it -- just
one straight-line copy of the body per trip over the same L1 buffers, with the
objectFifo acquire/release pairs stranded between the copies. That kernel builds
and runs and computes with stale operands. The loop has to reach the compiler as
a loop; see the ``dot_k_loop`` check that the DMAs and the matmul are *inside*
the ``scf.for``.

Scalar fill: ``acc[:] = 0.0`` has no buffer on the right-hand side, and the
destination alone supplies the shape. That is how an accumulator is zeroed
before a reduction, and for ``linalg.dot`` the destination is rank 0.

Rank dispatch: ``ops.dot`` is the *contraction*, in the ``numpy.dot`` /
``tl.dot`` sense rather than linalg's narrower one, so the operand ranks pick
which named op carries it. One rule covers all four -- ``a``'s last axis
contracts against ``b``'s first, and ``acc`` keeps what is left of each.
"""

from air import api as air
from air.api import ops  # noqa: F401
from air.api.types import bf16, f32


def build(M, N, K, tm, tn, tk, herd_shape=None):
    A = air.tensor([M, K], bf16)
    B = air.tensor([K, N], bf16)
    C = air.tensor([M, N], f32)

    with air.launch(name="gemm") as launch:

        @launch.body
        def _():
            # A list of ranges, not itertools.product: product materialises its
            # inputs, so a single-tile axis loses its step. See errors.py.
            with air.herd([range(0, M, tm), range(0, N, tn)], shape=herd_shape) as h:

                @h.body
                def _(tx, ty):
                    bm, bn = h.tile_sizes
                    row, col = tx * bm, ty * bn

                    acc = air.alloc([bm, bn], f32, scope=h.private())
                    a_buf = air.alloc([bm, tk], bf16, scope=h.private())
                    b_buf = air.alloc([tk, bn], bf16, scope=h.private())

                    acc[:] = 0.0
                    for k in air.sequential(0, K, tk):
                        air.ops.load(a_buf, A[row : row + bm, k : k + tk])
                        air.ops.load(b_buf, B[k : k + tk, col : col + bn])
                        air.ops.dot(a_buf, b_buf, acc=acc)

                    air.ops.store(acc, C[row : row + bm, col : col + bn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: dot_k_loop
# The accumulator is filled first, outside the K loop -- a vector broadcast of
# zero written across the tile, with no transfer_read, because there is no
# buffer on the right-hand side.
# CHECK: func.func @gemm(%{{.*}}: memref<64x64xbf16>, %{{.*}}: memref<64x64xbf16>, %{{.*}}: memref<64x64xf32>)
# CHECK: air.herd @herd_0
# CHECK: memref.alloc() : memref<32x32xf32, 2 : i32>
# CHECK: vector.broadcast %{{.*}} : f32 to vector<16xf32>
# CHECK: vector.transfer_write
#
# Then the K reduction: one scf.for whose *body* holds both loads and the
# matmul. The step is the K tile, so this is a real loop over 2 trips rather
# than two unrolled copies -- exactly one linalg.matmul appears in the module.
# CHECK: scf.for %{{.*}} = %c0{{.*}} to %c64{{.*}} step %c32
# CHECK: air.dma_memcpy_nd
# CHECK: air.dma_memcpy_nd
# CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<32x32xbf16, 2 : i32>, memref<32x32xbf16, 2 : i32>) outs(%{{.*}} : memref<32x32xf32, 2 : i32>)
# CHECK-NOT: linalg.matmul
# CHECK: air.dma_memcpy_nd
# CHECK: memref.dealloc
@run
def dot_k_loop():
    print(build(64, 64, 64, 32, 32, 32, herd_shape=(2, 2)).mlir())


# CHECK-LABEL: TEST: dot_single_core
# A 1-D physical shape still gets its K loop; nothing about the reduction is
# tied to the herd being 2-D.
# CHECK: air.herd @herd_0 tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c1{{.*}}, %{{.*}}=%c1
# CHECK: scf.for %{{.*}} = %c0{{.*}} to %c256{{.*}} step %c32
# CHECK: linalg.matmul
@run
def dot_single_core():
    print(build(32, 32, 256, 32, 32, 32, herd_shape=(1, 1)).mlir())


# CHECK-LABEL: TEST: dot_mixed_precision
# bf16 operands accumulating into f32 is the intended shape: linalg.matmul
# carries the mixed types straight through.
# CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<32x64xbf16, 2 : i32>, memref<64x32xbf16, 2 : i32>) outs(%{{.*}} : memref<32x32xf32, 2 : i32>)
@run
def dot_mixed_precision():
    print(build(64, 64, 128, 32, 32, 64, herd_shape=(2, 2)).mlir())


# CHECK-LABEL: TEST: dot_rank_dispatch
# Four contractions from one spelling. Ranks pick the op; nothing else changes.
# CHECK: memref.alloc() : memref<f32, 2 : i32>
# CHECK: linalg.dot ins(%{{.*}}, %{{.*}} : memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>) outs(%{{.*}} : memref<f32, 2 : i32>)
# CHECK: linalg.vecmat ins(%{{.*}}, %{{.*}} : memref<64xbf16, 2 : i32>, memref<64x32xbf16, 2 : i32>) outs(%{{.*}} : memref<32xf32, 2 : i32>)
# CHECK: linalg.matvec ins(%{{.*}}, %{{.*}} : memref<32x64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>) outs(%{{.*}} : memref<32xf32, 2 : i32>)
# CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<32x64xbf16, 2 : i32>, memref<64x32xbf16, 2 : i32>) outs(%{{.*}} : memref<32x32xf32, 2 : i32>)
@run
def dot_rank_dispatch():
    K, T = 64, 32
    v = air.tensor([K], bf16)
    m = air.tensor([T, K], bf16)
    n = air.tensor([K, T], bf16)
    out = air.tensor([T, T], f32)

    with air.launch(name="ranks") as launch:

        @launch.body
        def _():
            with air.herd(range(0, 1, 1), shape=(1,)) as h:

                @h.body
                def _(tx):
                    vb = air.alloc([K], bf16, scope=h.private())
                    mb = air.alloc([T, K], bf16, scope=h.private())
                    nb = air.alloc([K, T], bf16, scope=h.private())
                    air.ops.load(vb, v[0:K])
                    air.ops.load(mb, m[0:T, 0:K])
                    air.ops.load(nb, n[0:K, 0:T])

                    # (k,) . (k,) -> ()
                    s = air.alloc([], f32, scope=h.private(), vector=0)
                    s[:] = 0.0
                    air.ops.dot(vb, vb, acc=s)
                    # (k,) @ (k,n) -> (n,)
                    y = air.alloc([T], f32, scope=h.private(), vector=0)
                    y[:] = 0.0
                    air.ops.dot(vb, nb, acc=y)
                    # (m,k) @ (k,) -> (m,)
                    z = air.alloc([T], f32, scope=h.private(), vector=0)
                    z[:] = 0.0
                    air.ops.dot(mb, vb, acc=z)
                    # (m,k) @ (k,n) -> (m,n)
                    c = air.alloc([T, T], f32, scope=h.private(), vector=0)
                    c[:] = 0.0
                    air.ops.dot(mb, nb, acc=c)
                    air.ops.store(c, out[0:T, 0:T])

    print(launch.mlir())
