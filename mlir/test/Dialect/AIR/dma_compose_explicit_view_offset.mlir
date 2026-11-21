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

// A reinterpret_cast's offset counts elements whatever its strides: the view
// starts at element 4, so [0] [2] [1] reads 4 and 5.
// CHECK-LABEL: func.func @reinterpret_cast_explicit
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[4] [2] [1])
func.func @reinterpret_cast_explicit(%a: memref<16xi32>, %o: memref<2xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %v = memref.reinterpret_cast %a to offset: [4], sizes: [4], strides: [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 4>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0] [%c2] [%c1]) : (memref<2xi32>, memref<4xi32, strided<[2], offset: 4>>)
  return
}

// The same view read whole: elements 4, 6, 8 and 10, offset 4 / 2 = 2.
// CHECK-LABEL: func.func @reinterpret_cast_implicit
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[2] [4] [2])
func.func @reinterpret_cast_implicit(%a: memref<16xi32>, %o: memref<4xi32>) {
  %v = memref.reinterpret_cast %a to offset: [4], sizes: [4], strides: [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 4>>
  air.dma_memcpy_nd (%o[] [] [], %v[] [] []) : (memref<4xi32>, memref<4xi32, strided<[2], offset: 4>>)
  return
}

// No stride of the pattern divides 3: a leading unit dim of stride 1.
// CHECK-LABEL: func.func @reinterpret_cast_leading_dim
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[3, 0] [1, 4] [1, 2])
func.func @reinterpret_cast_leading_dim(%a: memref<16xi32>, %o: memref<4xi32>) {
  %v = memref.reinterpret_cast %a to offset: [3], sizes: [4], strides: [2] : memref<16xi32> to memref<4xi32, strided<[2], offset: 3>>
  air.dma_memcpy_nd (%o[] [] [], %v[] [] []) : (memref<4xi32>, memref<4xi32, strided<[2], offset: 3>>)
  return
}

// A dynamic offset is added to the dim of stride 1.
// CHECK-LABEL: func.func @reinterpret_cast_dynamic
// CHECK-SAME: %[[OFF:[a-zA-Z0-9]+]]: index
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, %[[OFF]]] [2, 4] [16, 1])
func.func @reinterpret_cast_dynamic(%a: memref<64xi32>, %o: memref<8xi32>, %off: index) {
  %v = memref.reinterpret_cast %a to offset: [%off], sizes: [2, 4], strides: [16, 1] : memref<64xi32> to memref<2x4xi32, strided<[16, 1], offset: ?>>
  air.dma_memcpy_nd (%o[] [] [], %v[] [] []) : (memref<8xi32>, memref<2x4xi32, strided<[16, 1], offset: ?>>)
  return
}

// The pattern lists fewer dims than the view. The view starts at element 3,
// which 2 does not divide: [0] [2] [2] reads 3 and 5 through a leading unit
// dim, aligned with the pattern's sizes and strides.
// CHECK-LABEL: func.func @short_pattern
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[3, 0] [1, 2] [1, 2])
func.func @short_pattern(%a: memref<2x16xi32>, %o: memref<2xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %v = memref.subview %a[0, 3] [2, 8] [1, 1] : memref<2x16xi32> to memref<2x8xi32, strided<[16, 1], offset: 3>>
  air.dma_memcpy_nd (%o[] [] [], %v[%c0] [%c2] [%c2]) : (memref<2xi32>, memref<2x8xi32, strided<[16, 1], offset: 3>>)
  return
}

// A padded channel transfer through a view whose layout strides are not
// static: the view stays, and nothing is left behind for the next sweep (the
// RUN line tests convergence).
air.channel @chan [1]
// CHECK-LABEL: func.func @padded_dynamic_layout
// CHECK: %[[SV:.*]] = memref.subview
// CHECK: air.channel.put {{ *}}@chan[] (%[[SV]][0, 0] [4, 4] [8, 1])
func.func @padded_dynamic_layout(%a: memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %v = memref.subview %a[1, 2] [4, 4] [1, 1] : memref<?x?xi32> to memref<4x4xi32, strided<[?, 1], offset: ?>>
  air.channel.put @chan[] (%v[%c0, %c0] [%c4, %c4] [%c8, %c1]) {pad_before = array<i32: 0, 1>, pad_after = array<i32: 0, 0>} : (memref<4x4xi32, strided<[?, 1], offset: ?>>)
  return
}
