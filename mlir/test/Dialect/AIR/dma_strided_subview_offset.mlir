//===- dma_strided_subview_offset.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize="test-convergence=true" | FileCheck %s

// Folding a subview into a DMA. A DMA counts an offset in units of its stride, element k of a dim at
// (offset + k) * stride. A subview's offset is in its source's coordinates,
// so with a step other than 1 it must be divided by the step: the subview
// below is elements 4, 6, 8, 10, which is [2] [4] [2], not [4] [4] [2]
// (elements 8, 10, 12, 14).

// CHECK-LABEL: func.func @strided_1d
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[2] [4] [2])
func.func @strided_1d(%m : memref<16xi32>, %d : memref<4xi32, 1>) {
  %s = memref.subview %m[4] [4] [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 4>>
  air.dma_memcpy_nd (%d[] [] [], %s[] [] []) : (memref<4xi32, 1>, memref<4xi32, strided<[2], offset: 4>>)
  return
}

// Rows 1 and 3, columns 2-5: linear start 10. The row offset 1 is not a
// multiple of the step 2, so it is carried into the dim of stride 1.
// CHECK-LABEL: func.func @strided_2d
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 10] [2, 4] [16, 1])
func.func @strided_2d(%m : memref<4x8xi32>, %d : memref<2x4xi32, 1>) {
  %s = memref.subview %m[1, 2] [2, 4] [2, 1] : memref<4x8xi32> to memref<2x4xi32, strided<[16, 1], offset: 10>>
  air.dma_memcpy_nd (%d[] [] [], %s[] [] []) : (memref<2x4xi32, 1>, memref<2x4xi32, strided<[16, 1], offset: 10>>)
  return
}

// A dynamic offset with a step other than 1 becomes a leading unit dim with
// the source's stride: elements %o + 2k.
// CHECK-LABEL: func.func @strided_dynamic
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, 0] [1, 4] [1, 2])
func.func @strided_dynamic(%m : memref<16xi32>, %d : memref<4xi32, 1>, %o : index) {
  %s = memref.subview %m[%o] [4] [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: ?>>
  air.dma_memcpy_nd (%d[] [] [], %s[] [] []) : (memref<4xi32, 1>, memref<4xi32, strided<[2], offset: ?>>)
  return
}

// Chains the fold does not support are left as they are, and the rewrite
// bails out before touching the IR, so canonicalization converges.

// A step other than 1 after the front of the chain.
// CHECK-LABEL: func.func @cast_then_subview
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[S:.*]][] [] [])
// CHECK-SAME: strided<[2], offset: 4>
func.func @cast_then_subview(%u : memref<*xi32>, %d : memref<4xi32, 1>) {
  %r = memref.reinterpret_cast %u to offset: [0], sizes: [16], strides: [1] : memref<*xi32> to memref<16xi32>
  %s = memref.subview %r[4] [4] [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 4>>
  air.dma_memcpy_nd (%d[] [] [], %s[] [] []) : (memref<4xi32, 1>, memref<4xi32, strided<[2], offset: 4>>)
  return
}

// CHECK-LABEL: func.func @unit_then_strided
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[] [] [])
// CHECK-SAME: strided<[2], offset: 12>
func.func @unit_then_strided(%m : memref<32xi32>, %d : memref<4xi32, 1>) {
  %a = memref.subview %m[8] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 8>>
  %s = memref.subview %a[4] [4] [2] : memref<16xi32, strided<[1], offset: 8>> to memref<4xi32, strided<[2], offset: 12>>
  air.dma_memcpy_nd (%d[] [] [], %s[] [] []) : (memref<4xi32, 1>, memref<4xi32, strided<[2], offset: 12>>)
  return
}

// An explicit access pattern on a subview whose step is not 1.
// CHECK-LABEL: func.func @explicit_pattern
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0] [2] [2])
// CHECK-SAME: strided<[2], offset: 4>
func.func @explicit_pattern(%m : memref<16xi32>, %d : memref<2xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %s = memref.subview %m[4] [4] [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 4>>
  air.dma_memcpy_nd (%d[] [] [], %s[%c0] [%c2] [%c2]) : (memref<2xi32, 1>, memref<4xi32, strided<[2], offset: 4>>)
  return
}
