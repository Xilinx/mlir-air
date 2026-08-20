# ./python/test/api/eltwise_add.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api emits the same IR shape as the hand-written eltwise_add kernel."""

from itertools import product

from air import api as air
from air.api.types import bf16, f32


def build(M, N, tile, herd_shape=None, dtype=bf16):
    A = air.tensor([M, N], dtype)
    B = air.tensor([M, N], dtype)
    C = air.tensor([M, N], dtype)

    with air.launch(name="eltwise_add") as launch:

        @launch.body
        def _():
            grid = product(range(0, M, tile), range(0, N, tile))
            with air.herd(grid, shape=herd_shape) as h:

                @h.body
                def _(tx, ty):
                    tm, tn = h.tile_sizes
                    row, col = tx * tm, ty * tn
                    a = air.alloc([tm, tn], dtype, scope=h.private())
                    b = air.alloc([tm, tn], dtype, scope=h.private())
                    c = air.alloc([tm, tn], dtype, scope=h.private())
                    air.ops.load(a, A[row : row + tm, col : col + tn])
                    air.ops.load(b, B[row : row + tm, col : col + tn])
                    c[:] = a[:] + b[:]
                    air.ops.store(c, C[row : row + tm, col : col + tn])

    return launch


def build_1d(N, tile, herd_shape=None, dtype=bf16):
    A = air.tensor([N], dtype)
    B = air.tensor([N], dtype)
    C = air.tensor([N], dtype)

    with air.launch(name="eltwise_add_1d") as launch:

        @launch.body
        def _():
            with air.herd(range(0, N, tile), shape=herd_shape) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    a = air.alloc([tn], dtype, scope=h.private())
                    b = air.alloc([tn], dtype, scope=h.private())
                    c = air.alloc([tn], dtype, scope=h.private())
                    air.ops.load(a, A[col : col + tn])
                    air.ops.load(b, B[col : col + tn])
                    c[:] = a[:] + b[:]
                    air.ops.store(c, C[col : col + tn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: exact_fit
# The logical grid is 2x2 and fits the array, so the tile offset depends on the
# herd coordinate alone -- a one-symbol affine map, and no strip-mine loop.
# CHECK: affine_map<()[s0] -> (s0 * 64)>
# CHECK: func.func @eltwise_add(%[[A:.*]]: memref<128x128xbf16>, %[[B:.*]]: memref<128x128xbf16>, %[[C:.*]]: memref<128x128xbf16>)
# CHECK: air.herd @herd_0 tile (%[[TX:.*]], %[[TY:.*]]) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2
# CHECK: memref.alloc() : memref<64x64xbf16, 2 : i32>
# CHECK: affine.apply
# CHECK: air.dma_memcpy_nd
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
# CHECK: air.dma_memcpy_nd
# CHECK: memref.dealloc
@run
def exact_fit():
    print(build(128, 128, 64, herd_shape=(2, 2)).mlir())


# CHECK-LABEL: TEST: strip_mined
# A 16x16 logical grid on a 2x4 array becomes 8 repeats in x and 4 in y, and the
# tile offset folds into a single affine map: (tx*8 + i)*64 = tx*512 + i*64.
# CHECK-DAG: affine_map<()[s0, s1] -> (s0 * 512 + s1 * 64)>
# CHECK-DAG: affine_map<()[s0, s1] -> (s0 * 256 + s1 * 64)>
# CHECK: air.herd @herd_0 tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c4
# CHECK: scf.for
# CHECK: scf.for
# CHECK: memref.alloc() : memref<64x64xbf16, 2 : i32>
# CHECK: air.dma_memcpy_nd
# CHECK: arith.addf {{.*}} : vector<16xbf16>
@run
def strip_mined():
    print(build(1024, 1024, 64, herd_shape=(2, 4)).mlir())


# CHECK-LABEL: TEST: scalar_fallback
# A 12-wide tile is not a multiple of the bf16 vector width (16), so the emitter
# falls back to a scalar loop rather than emitting an illegal vector width.
# CHECK-NOT: vector.transfer_read
# CHECK: memref.load
# CHECK: memref.load
# CHECK: arith.addf {{.*}} : bf16
# CHECK: memref.store
@run
def scalar_fallback():
    print(build(24, 24, 12, herd_shape=(2, 2)).mlir())


# CHECK-LABEL: TEST: rank1
# A 1-D grid lays the herd out along x -- sizes [P, 1], the orientation the
# hand-written kernel uses. [1, P] does not place on npu2.
# CHECK: func.func @eltwise_add_1d(%{{.*}}: memref<65536xbf16>
# CHECK: air.herd @herd_0 tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1
# CHECK: memref.alloc() : memref<1024xbf16, 2 : i32>
# CHECK: air.dma_memcpy_nd
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
@run
def rank1():
    print(build_1d(65536, 1024, herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: f32_uses_512bit_vectors
# f32 defaults to 16 lanes, not 8: a 256-bit <8 x f32> add fails to legalize in
# the AIE backend on both npu1 and npu2.
# CHECK: memref.alloc() : memref<32x32xf32, 2 : i32>
# CHECK: vector.transfer_read {{.*}} vector<16xf32>
# CHECK: arith.addf {{.*}} : vector<16xf32>
@run
def f32_uses_512bit_vectors():
    print(build(128, 128, 32, herd_shape=(2, 2), dtype=f32).mlir())
