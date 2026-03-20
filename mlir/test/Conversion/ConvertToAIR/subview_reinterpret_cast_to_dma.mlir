//===- subview_reinterpret_cast_to_dma.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma | FileCheck %s

// Test that air-copy-to-dma correctly handles subview(reinterpret_cast) chains
// by placing the reinterpret_cast flat offset in the stride-1 dimension.

// CHECK-LABEL: func.func @transposed_a
// The transposed A has strides [1, 512]. The reinterpret_cast offset %arg1
// should be added to dim0 (stride=1), not dim1 (stride=512).
// CHECK: air.dma_memcpy_nd
// CHECK-SAME: %arg0[%arg1, %arg2]
// CHECK-SAME: [%c256, %c16]
// CHECK-SAME: [%c1, %c512]
func.func @transposed_a(%arg0: memref<*xf32>, %arg1: index, %arg2: index) {
  %alloc = memref.alloc() : memref<256x784xf32, 1>
  %rc = memref.reinterpret_cast %arg0 to
    offset: [%arg1], sizes: [256, 784], strides: [1, 512]
    : memref<*xf32> to memref<256x784xf32, strided<[1, 512], offset: ?>>
  %sv = memref.subview %rc[0, %arg2] [256, 16] [1, 1]
    : memref<256x784xf32, strided<[1, 512], offset: ?>>
      to memref<256x16xf32, strided<[1, 512], offset: ?>>
  %sv_dst = memref.subview %alloc[0, %arg2] [256, 16] [1, 1]
    : memref<256x784xf32, 1>
      to memref<256x16xf32, strided<[784, 1], offset: ?>, 1>
  memref.copy %sv, %sv_dst
    : memref<256x16xf32, strided<[1, 512], offset: ?>>
      to memref<256x16xf32, strided<[784, 1], offset: ?>, 1>
  return
}

// CHECK-LABEL: func.func @normal_layout
// Normal layout has strides [1024, 1]. The reinterpret_cast offset %arg1
// should be added to dim1 (stride=1).
// CHECK: air.dma_memcpy_nd
// CHECK-SAME: %arg0[%arg2, %arg1]
// CHECK-SAME: [%c16, %c256]
// CHECK-SAME: [%c1024, %c1]
func.func @normal_layout(%arg0: memref<*xf32>, %arg1: index, %arg2: index) {
  %alloc = memref.alloc() : memref<512x256xf32, 1>
  %rc = memref.reinterpret_cast %arg0 to
    offset: [%arg1], sizes: [512, 256], strides: [1024, 1]
    : memref<*xf32> to memref<512x256xf32, strided<[1024, 1], offset: ?>>
  %sv = memref.subview %rc[%arg2, 0] [16, 256] [1, 1]
    : memref<512x256xf32, strided<[1024, 1], offset: ?>>
      to memref<16x256xf32, strided<[1024, 1], offset: ?>>
  %sv_dst = memref.subview %alloc[%arg2, 0] [16, 256] [1, 1]
    : memref<512x256xf32, 1>
      to memref<16x256xf32, strided<[256, 1], offset: ?>, 1>
  memref.copy %sv, %sv_dst
    : memref<16x256xf32, strided<[1024, 1], offset: ?>>
      to memref<16x256xf32, strided<[256, 1], offset: ?>, 1>
  return
}
