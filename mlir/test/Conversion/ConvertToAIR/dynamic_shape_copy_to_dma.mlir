//===- dynamic_shape_copy_to_dma.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma | FileCheck %s

// Dynamic-sized subview source, static L1 destination.
// CHECK-LABEL: func.func @dynamic_size_subview_src
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1{{.*}}])
func.func @dynamic_size_subview_src(%arg0: memref<?x64xf32>, %arg1: index) {
  %alloc = memref.alloc() : memref<16x64xf32, 2>
  %sv = memref.subview %arg0[%arg1, 0] [16, 64] [1, 1]
    : memref<?x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
  memref.copy %sv, %alloc
    : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, 2>
  memref.dealloc %alloc : memref<16x64xf32, 2>
  return
}

// Dynamic-sized reinterpret_cast with dynamic size.
// CHECK-LABEL: func.func @dynamic_size_reinterpret_cast
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}] [%{{.*}}] [%c1{{.*}}])
func.func @dynamic_size_reinterpret_cast(%arg0: memref<*xf32>, %off: index, %n: index) {
  %alloc = memref.alloc() : memref<32xf32, 2>
  %rc = memref.reinterpret_cast %arg0 to
    offset: [%off], sizes: [%n], strides: [1]
    : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
  memref.copy %rc, %alloc
    : memref<?xf32, strided<[1], offset: ?>> to memref<32xf32, 2>
  memref.dealloc %alloc : memref<32xf32, 2>
  return
}

// Both source and destination have dynamic shapes (no subview/reinterpret_cast).
// The copy should still be converted to air.dma_memcpy_nd.
// CHECK-LABEL: func.func @plain_dynamic_memref
// CHECK: air.dma_memcpy_nd
func.func @plain_dynamic_memref(%arg0: memref<?x64xf32>, %arg1: memref<16x64xf32, 2>) {
  memref.copy %arg0, %arg1 : memref<?x64xf32> to memref<16x64xf32, 2>
  return
}

// Negative test: both L3 should NOT be converted.
// CHECK-LABEL: func.func @both_l3_dynamic
// CHECK-NOT: air.dma_memcpy_nd
// CHECK: memref.copy
func.func @both_l3_dynamic(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
  return
}
