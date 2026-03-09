//===- self_copy_elimination.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma | FileCheck %s

// Same-memory-space self-copy: legality predicate keeps it as memref.copy (not
// converted to DMA). The copy remains but is harmless and no invalid DMA is
// created.
// CHECK-LABEL: func.func @self_copy_same_memspace
// CHECK-NOT: air.dma_memcpy_nd
// CHECK: return
func.func @self_copy_same_memspace(%arg0: memref<64xbf16, 1>) {
  memref.copy %arg0, %arg0 : memref<64xbf16, 1> to memref<64xbf16, 1>
  return
}

// Normal cross-memory-space copy should still produce a DMA.
// CHECK-LABEL: func.func @cross_memspace_copy
// CHECK: air.dma_memcpy_nd
func.func @cross_memspace_copy(%arg0: memref<64x64xf32>) {
  %alloc = memref.alloc() : memref<16x16xf32, 2>
  %sv = memref.subview %arg0[0, 0] [16, 16] [1, 1]
    : memref<64x64xf32> to memref<16x16xf32, strided<[64, 1]>>
  memref.copy %sv, %alloc
    : memref<16x16xf32, strided<[64, 1]>> to memref<16x16xf32, 2>
  memref.dealloc %alloc : memref<16x16xf32, 2>
  return
}
