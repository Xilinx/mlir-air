//===- dma_compose_explicit_view_offset.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize="test-convergence" %s | FileCheck %s

// The offsets of an explicit access pattern on a view are in units of the
// pattern's strides, the view's offsets in units of its layout strides.
// Composing the view into the pattern converts them: o * L / s when the
// pattern's stride s divides it, else a leading unit dim of the layout stride.

// The view starts at element 4; [0] [2] [2] reads 4 and 6: offset 4 / 2 = 2.
// CHECK-LABEL: func.func @divides
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[2] [2] [2])
func.func @divides(%a: memref<16xi32>, %o: memref<2xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %v = memref.subview %a[4] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1], offset: 4>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0] [%c2] [%c2]) : (memref<2xi32>, memref<8xi32, strided<[1], offset: 4>>)
  return
}

// The view starts at element 3, which 2 does not divide: reads 3 and 5 through
// a leading unit dim of stride 1 and offset 3.
// CHECK-LABEL: func.func @leading_dim
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[3, 0] [1, 2] [1, 2])
func.func @leading_dim(%a: memref<16xi32>, %o: memref<2xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %v = memref.subview %a[3] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1], offset: 3>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0] [%c2] [%c2]) : (memref<2xi32>, memref<8xi32, strided<[1], offset: 3>>)
  return
}

// The pattern's strides are the view's layout strides: the offsets add as
// they are.
// CHECK-LABEL: func.func @same_strides
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[2, 4] [2, 16] [16, 1])
func.func @same_strides(%a: memref<8x16xi32>, %o: memref<32xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %v = memref.subview %a[2, 4] [4, 8] [1, 1] : memref<8x16xi32> to memref<4x8xi32, strided<[16, 1], offset: 36>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0, %c0] [%c2, %c16] [%c16, %c1]) : (memref<32xi32>, memref<4x8xi32, strided<[16, 1], offset: 36>>)
  return
}

// The view starts at row 2 of a 16-wide buffer (layout stride 16); the pattern
// steps by 32 elements in its outer dim, so the row offset becomes 2 * 16 / 32
// = 1: the pattern reads rows 2 and 4.
// CHECK-LABEL: func.func @outer_dim
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[1, 0] [2, 16] [32, 1])
func.func @outer_dim(%a: memref<8x16xi32>, %o: memref<32xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %v = memref.subview %a[2, 0] [6, 16] [1, 1] : memref<8x16xi32> to memref<6x16xi32, strided<[16, 1], offset: 32>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0, %c0] [%c2, %c16] [%c32, %c1]) : (memref<32xi32>, memref<6x16xi32, strided<[16, 1], offset: 32>>)
  return
}

// A view whose layout strides are not static: its offsets cannot be converted,
// so the view stays.
// CHECK-LABEL: func.func @dynamic_layout
// CHECK: %[[SV:.*]] = memref.subview
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[SV]][0] [2] [2])
func.func @dynamic_layout(%a: memref<16xi32, strided<[?], offset: ?>>, %o: memref<2xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %v = memref.subview %a[4] [8] [1] : memref<16xi32, strided<[?], offset: ?>> to memref<8xi32, strided<[?], offset: ?>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0] [%c2] [%c2]) : (memref<2xi32>, memref<8xi32, strided<[?], offset: ?>>)
  return
}
