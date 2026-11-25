//===- overlapping_war_waw.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -split-input-file | FileCheck %s

// Accesses to an L2 buffer that start at different elements but share some
// conflict as much as accesses that start together. Reads after writes were
// ordered by their footprints already; writes after writes and after reads
// are too.

// A DMA writes rows 1 and 2 of %buf, a second writes row 2 only: a write after
// a write on row 2.

// CHECK-LABEL: func.func @waw_overlapping
// CHECK: %[[W0:.*]] = air.dma_memcpy_nd async [%{{.*}}] (%{{.*}}[%c1{{.*}}, %c0{{.*}}] [%c2
// CHECK: air.dma_memcpy_nd async [{{.*}}%[[W0]]{{.*}}] (%{{.*}}[%c2{{.*}}, %c0{{.*}}] [%c1
func.func @waw_overlapping(%src : memref<4x8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %buf = memref.alloc() : memref<4x8xi32, 1>
  air.dma_memcpy_nd (%buf[%c1, %c0] [%c2, %c8] [%c8, %c1], %src[%c1, %c0] [%c2, %c8] [%c8, %c1]) {id = 1 : i32} : (memref<4x8xi32, 1>, memref<4x8xi32>)
  air.dma_memcpy_nd (%buf[%c2, %c0] [%c1, %c8] [%c8, %c1], %src[%c0, %c0] [%c1, %c8] [%c8, %c1]) {id = 2 : i32} : (memref<4x8xi32, 1>, memref<4x8xi32>)
  return
}

// -----

// A DMA reads rows 1 and 2 of %buf, a second then writes row 2: a write after
// a read on row 2.

// CHECK-LABEL: func.func @war_overlapping
// CHECK: %[[R0:.*]] = air.dma_memcpy_nd async [%{{.*}}] (%{{.*}}[] [] [], %{{.*}}[%c1
// CHECK: air.dma_memcpy_nd async [{{.*}}%[[R0]]{{.*}}] (%{{.*}}[%c2{{.*}}, %c0{{.*}}] [%c1
func.func @war_overlapping(%src : memref<4x8xi32>, %out : memref<2x8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %buf = memref.alloc() : memref<4x8xi32, 1>
  air.dma_memcpy_nd (%out[] [] [], %buf[%c1, %c0] [%c2, %c8] [%c8, %c1]) {id = 1 : i32} : (memref<2x8xi32>, memref<4x8xi32, 1>)
  air.dma_memcpy_nd (%buf[%c2, %c0] [%c1, %c8] [%c8, %c1], %src[%c0, %c0] [%c1, %c8] [%c8, %c1]) {id = 2 : i32} : (memref<4x8xi32, 1>, memref<4x8xi32>)
  return
}

// -----

// Rows 0 and 2, then rows 1 and 3: no element in common, no edge.

// CHECK-LABEL: func.func @waw_disjoint
// CHECK: %[[W0:.*]] = air.dma_memcpy_nd async [%{{.*}}]
// CHECK-NOT: air.dma_memcpy_nd async [{{.*}}%[[W0]]
func.func @waw_disjoint(%src : memref<4x8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %buf = memref.alloc() : memref<4x8xi32, 1>
  air.dma_memcpy_nd (%buf[%c0, %c0] [%c2, %c8] [%c16, %c1], %src[%c0, %c0] [%c2, %c8] [%c8, %c1]) {id = 1 : i32} : (memref<4x8xi32, 1>, memref<4x8xi32>)
  air.dma_memcpy_nd (%buf[%c0, %c8] [%c2, %c8] [%c16, %c1], %src[%c0, %c0] [%c2, %c8] [%c8, %c1]) {id = 2 : i32} : (memref<4x8xi32, 1>, memref<4x8xi32>)
  return
}
