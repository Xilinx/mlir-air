//===- copy_to_dma_strided_subview_offset.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma | FileCheck %s

// A DMA counts an offset in units of its stride, element k of a dim at
// (offset + k) * stride. A subview's offset is in its source's coordinates,
// so with a step other than 1 it must be divided by the step: the subview
// below is elements 4, 6, 8, 10, which is [2] [4] [2], not [4] [4] [2]
// (elements 8, 10, 12, 14).

// CHECK-LABEL: func.func @strided_1d
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[2] [4] [2])
func.func @strided_1d(%m : memref<16xi32>, %d : memref<4xi32, 1>) {
  %s = memref.subview %m[4] [4] [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 4>>
  memref.copy %s, %d : memref<4xi32, strided<[2], offset: 4>> to memref<4xi32, 1>
  return
}

// Rows 1 and 3, columns 2-5: linear start 10. The row offset 1 is not a
// multiple of the step 2, so it is carried into the dim of stride 1.
// CHECK-LABEL: func.func @strided_2d
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 10] [2, 4] [16, 1])
func.func @strided_2d(%m : memref<4x8xi32>, %d : memref<2x4xi32, 1>) {
  %s = memref.subview %m[1, 2] [2, 4] [2, 1] : memref<4x8xi32> to memref<2x4xi32, strided<[16, 1], offset: 10>>
  memref.copy %s, %d : memref<2x4xi32, strided<[16, 1], offset: 10>> to memref<2x4xi32, 1>
  return
}

// A dynamic offset with a step other than 1 becomes a leading unit dim with
// the source's stride: elements %o + 2k.
// CHECK-LABEL: func.func @strided_dynamic
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, 0] [1, 4] [1, 2])
func.func @strided_dynamic(%m : memref<16xi32>, %d : memref<4xi32, 1>, %o : index) {
  %s = memref.subview %m[%o] [4] [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: ?>>
  memref.copy %s, %d : memref<4xi32, strided<[2], offset: ?>> to memref<4xi32, 1>
  return
}
